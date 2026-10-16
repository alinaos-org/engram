#!/usr/bin/env bash
# Engram GPU instances on AWS: launch, reach, run, pull results, terminate.
#
#   scripts/aws/gpu.sh stack --alert-email you@example.com [--budget 300]   # one-time setup
#   scripts/aws/gpu.sh up [--type g6e.xlarge] [--spot] [--hours 4] [--model Qwen/Qwen3-8B]
#                         [--ref main] [--dry-run]
#   scripts/aws/gpu.sh status
#   scripts/aws/gpu.sh wait                 # until vLLM is serving
#   scripts/aws/gpu.sh tunnel [8000]        # forward the instance's vLLM to localhost
#   scripts/aws/gpu.sh exec 'command'       # run as ubuntu in ~/engram, print output
#   scripts/aws/gpu.sh push-runs            # instance: sync ~/engram/runs to S3
#   scripts/aws/gpu.sh pull                 # local: sync S3 runs to ./runs/aws
#   scripts/aws/gpu.sh down                 # terminate every Project=engram instance
#
# Safety: instances have no SSH keys and no inbound ports (SSM only), are tagged Project=engram,
# and schedule their own shutdown after --hours with shutdown behaviour "terminate", so a
# forgotten instance cannot run past the cap. Nothing here runs unless you call it.
set -euo pipefail

PROFILE=${AWS_PROFILE:-alina-admin}
REGION=${AWS_REGION:-us-west-2}
STACK=${ENGRAM_STACK:-engram-gpu}
REPO=https://github.com/alinaos-org/engram
AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id
aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }
die() { echo "error: $*" >&2; exit 1; }

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

instance_ids() {
  aws ec2 describe-instances \
    --filters Name=tag:Project,Values=engram Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

one_instance() {
  local ids; ids=$(instance_ids)
  [[ -n $ids ]] || die "no running engram instance"
  [[ $ids != *[[:space:]]* ]] || die "more than one engram instance running: $ids"
  echo "$ids"
}

user_data() {  # $1 model, $2 git ref, $3 hours
  cat <<EOF
#!/bin/bash
set -euxo pipefail
shutdown -h +$(( $3 * 60 ))   # hard cap; the instance terminates on shutdown
apt-get update -y && apt-get install -y git
sudo -iu ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -iu ubuntu bash -c 'git clone --branch "$2" $REPO ~/engram'
sudo -iu ubuntu bash -c 'cd ~/engram && ~/.local/bin/uv sync --extra dev --extra ml --extra cu130'
sudo -iu ubuntu bash -c '~/.local/bin/uv venv ~/vllm --python 3.12 && ~/.local/bin/uv pip install --python ~/vllm/bin/python vllm'
cat > /etc/systemd/system/vllm.service <<UNIT
[Unit]
Description=vLLM serving $1 for engram
After=network-online.target
[Service]
User=ubuntu
Environment=HF_HOME=/home/ubuntu/.cache/huggingface
ExecStart=/home/ubuntu/vllm/bin/vllm serve $1 --host 127.0.0.1 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 64
Restart=on-failure
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now vllm
touch /var/lib/engram-ready
EOF
}

cmd_stack() {
  local email="" budget=300
  while [[ $# -gt 0 ]]; do
    case $1 in
      --alert-email) email=$2; shift 2 ;;
      --budget) budget=$2; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  [[ -n $email ]] || die "--alert-email is required"
  local vpc
  vpc=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
    --query 'Vpcs[0].VpcId' --output text)
  aws cloudformation deploy --stack-name "$STACK" \
    --template-file "$(dirname "$0")/../../infra/aws/engram-gpu.yaml" \
    --capabilities CAPABILITY_NAMED_IAM --tags Project=engram \
    --parameter-overrides VpcId="$vpc" AlertEmail="$email" MonthlyBudgetUsd="$budget"
}

cmd_up() {
  local type=g6e.xlarge spot=0 hours=4 model=Qwen/Qwen3-8B ref=main dry=()
  while [[ $# -gt 0 ]]; do
    case $1 in
      --type) type=$2; shift 2 ;;
      --spot) spot=1; shift ;;
      --hours) hours=$2; shift 2 ;;
      --model) model=$2; shift 2 ;;
      --ref) ref=$2; shift 2 ;;
      --dry-run) dry=(--dry-run); shift ;;
      *) die "unknown option $1" ;;
    esac
  done
  [[ $hours =~ ^[0-9]+$ && $hours -ge 1 && $hours -le 24 ]] || die "--hours must be 1-24"
  local ami profile sg market=()
  ami=$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
  if [[ ${#dry[@]} -eq 0 ]]; then
    [[ -z $(instance_ids) ]] || die "an engram instance is already running; 'down' it first"
    profile=$(stack_output InstanceProfileName)
    sg=$(stack_output SecurityGroupId)
    [[ -n $profile && -n $sg ]] || die "stack $STACK not deployed; run 'stack' first"
  fi
  [[ $spot -eq 1 ]] && market=(--instance-market-options \
    'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  local args=(
    --image-id "$ami" --instance-type "$type" --count 1
    --instance-initiated-shutdown-behavior terminate
    --metadata-options HttpTokens=required
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}'
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=engram},{Key=Name,Value=engram-gpu},{Key=Model,Value=$model}]" \
      "ResourceType=volume,Tags=[{Key=Project,Value=engram}]"
    --user-data "$(user_data "$model" "$ref" "$hours")"
  )
  if [[ ${#dry[@]} -eq 0 ]]; then
    args+=(--iam-instance-profile "Name=$profile" --security-group-ids "$sg")
  fi
  echo "launching $type (spot=$spot) for at most ${hours}h, model $model, ref $ref, AMI $ami"
  aws ec2 run-instances "${dry[@]}" "${market[@]}" "${args[@]}" \
    --query 'Instances[0].InstanceId' --output text
}

cmd_status() {
  aws ec2 describe-instances --filters Name=tag:Project,Values=engram \
    Name=instance-state-name,Values=pending,running,stopping,shutting-down \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,Tags[?Key==`Model`]|[0].Value]' \
    --output table
}

ssm_run() {  # $1 instance, $2 shell command (run as ubuntu); prints stdout/stderr
  local id=$1 cmd=$2 cid status params
  params=$(python3 -c 'import json,shlex,sys; print(json.dumps({"commands": [
      "sudo -iu ubuntu bash -lc " + shlex.quote(sys.argv[1])]}))' "$cmd")
  cid=$(aws ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
    --parameters "$params" --timeout-seconds 3600 --query Command.CommandId --output text)
  while :; do
    status=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$id" \
      --query Status --output text 2>/dev/null || echo Pending)
    case $status in Pending|InProgress|Delayed) sleep 5 ;; *) break ;; esac
  done
  aws ssm get-command-invocation --command-id "$cid" --instance-id "$id" \
    --query '[StandardOutputContent,StandardErrorContent]' --output text
  [[ $status == Success ]]
}

cmd_wait() {
  local id; id=$(one_instance)
  until ssm_run "$id" 'test -f /var/lib/engram-ready && curl -sf localhost:8000/v1/models' \
    >/dev/null 2>&1; do
    echo "waiting for vLLM on $id..."; sleep 30
  done
  echo "vLLM is serving on $id"
}

cmd_tunnel() {
  local port=${1:-8000} id; id=$(one_instance)
  aws ssm start-session --target "$id" --document-name AWS-StartPortForwardingSession \
    --parameters "portNumber=8000,localPortNumber=$port"
}

cmd_exec() { ssm_run "$(one_instance)" "cd ~/engram && $1"; }

cmd_push_runs() {
  local bucket; bucket=$(stack_output RunsBucketName)
  ssm_run "$(one_instance)" "cd ~/engram && aws s3 sync runs s3://$bucket/runs/"
}

cmd_pull() {
  local bucket; bucket=$(stack_output RunsBucketName)
  mkdir -p runs/aws
  aws s3 sync "s3://$bucket/runs/" runs/aws/
}

cmd_down() {
  local ids; ids=$(instance_ids)
  [[ -n $ids ]] || { echo "nothing to terminate"; return; }
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --instance-ids $ids \
    --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
}

sub=${1:-}; shift || true
case $sub in
  stack) cmd_stack "$@" ;;
  up) cmd_up "$@" ;;
  status) cmd_status ;;
  wait) cmd_wait ;;
  tunnel) cmd_tunnel "$@" ;;
  exec) cmd_exec "$1" ;;
  push-runs) cmd_push_runs ;;
  pull) cmd_pull ;;
  down) cmd_down ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
