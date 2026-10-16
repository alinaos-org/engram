# Running Engram on AWS GPUs

*Runbook, 2026-10-15. Account 794285265674, region us-west-2. Prices are from the AWS Pricing API on 2026-10-15 and will drift; spot prices change hourly.*

All infrastructure is code: [`infra/aws/engram-gpu.yaml`](../infra/aws/engram-gpu.yaml) (one-time support resources) and [`scripts/aws/gpu.sh`](../scripts/aws/gpu.sh) (launch, reach, run, terminate). The launch path has been validated with AWS's `--dry-run` ("request would have succeeded") for both on-demand and spot. Nothing has been created yet.

## Safety properties

- **No SSH and no inbound ports.** Access is through AWS Systems Manager (port forwarding and remote commands).
- **Hard time cap.** Every instance schedules its own shutdown after `--hours` (default 4, maximum 24), and its shutdown behaviour is *terminate*. A forgotten instance stops costing money at the cap.
- **One instance at a time.** `up` refuses to launch while another `Project=engram` instance is running.
- **Narrow permissions.** The instance role can use SSM and read and write one private, encrypted bucket. Nothing else.
- **Budget alerts** at 50% and 80% of actual spend, and at 100% of forecast spend.

## One-time setup (needs your approval: these change the account)

1. **Request GPU quota.** Every GPU quota is currently **0**, so no GPU instance can launch. Quotas are counted in vCPUs, and 16 covers one g6e.2xlarge or g6e.4xlarge with room to spare. Requests are free; approval can take hours to days.
   ```bash
   aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 16 --profile alina-admin --region us-west-2
   ```
   ```bash
   aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-3819A6DF --desired-value 16 --profile alina-admin --region us-west-2
   ```
   The first is on-demand G and VT instances; the second is G and VT spot requests.
2. **Deploy the support stack:** the bucket, instance role, security group and budget.
   ```bash
   scripts/aws/gpu.sh stack --alert-email you@example.com --budget 300
   ```
3. **Activate the `Project` cost-allocation tag** in Billing → Cost allocation tags. The tag only appears after a tagged resource exists, so do this after the first launch. Until it is active, the `engram-monthly` budget matches nothing; the account-wide $100 budget will alert first.

## Instance choice

| Instance | GPU | vCPU | On-demand $/h | Spot $/h (min, 10-15) | Fits |
|---|---|---|---|---|---|
| g6e.xlarge | 1× L40S 48 GB | 4 | 1.86 | 1.74 | 8B bf16 serving, or 8B LoRA (not both at once) |
| **g6e.2xlarge** (recommended) | 1× L40S 48 GB | 8 | 2.24 | 1.26 | same GPU, more CPU for embedding and data loading |
| g6e.12xlarge | 4× L40S | 48 | 10.49 | 9.27 | 32B models, parallel runs |
| p5.48xlarge | 8× H100 80 GB | 192 | 55.04 | 26.87 | Phase 1 training, not Phase 0 |

Memory estimates for each candidate base model are in [research/base-model-shortlist.md](../research/base-model-shortlist.md). Qwen3-8B needs about 19 GB to serve and about 26 GB for LoRA at 4K context, so one L40S covers all of Phase 0.

## A GPU day

```bash
scripts/aws/gpu.sh up --type g6e.2xlarge --spot --hours 5 --model Qwen/Qwen3-8B --ref main
```
```bash
scripts/aws/gpu.sh wait
```
```bash
scripts/aws/gpu.sh exec 'scripts/aws/gpu-day.sh'
```
```bash
scripts/aws/gpu.sh push-runs
```
```bash
scripts/aws/gpu.sh down
```
```bash
scripts/aws/gpu.sh pull
```

`wait` returns once the driver, uv, vLLM and the model are installed and vLLM is serving. [`gpu-day.sh`](../scripts/aws/gpu-day.sh) runs everything in Phase 0 that needs the base model:

1. Measure serving throughput (#12).
2. Run B0, B1 and B3 (hybrid retrieval) on `engram-bench` v0, 50 users (#3).
3. Run B3 on LongMemEval-S and LoCoMo.
4. Stop vLLM, time one LoRA sleep (#12), and run B4 per-user LoRA on the 10-user pilot.

Every model response is cached on the instance, so after a spot interruption you relaunch and re-run, and only the missing calls are made. `down` terminates the instance; the time cap is the backstop. `pull` copies the results back to `runs/aws/`.

**Estimated cost (assumption):** about 0.5 h of setup, about 1 h of vLLM baselines (≈12K requests at 16-way concurrency), about 1–1.5 h for B4. That is roughly **3 hours ≈ $4–7** on g6e.2xlarge, spot or on-demand. Measure, then update [experiment-plan.md](../research/experiment-plan.md).

**After pulling,** grade the free-form LongMemEval and LoCoMo answers with a judge on the subscription (no GPU needed):
```bash
uv run engram-bench judge runs/aws/gpu-day/b3-lme-s-Qwen-Qwen3-8B --bench data/longmemeval-s --model claude:claude-opus-5-5
```

## Interactive use

`scripts/aws/gpu.sh tunnel` forwards the instance's vLLM to `localhost:8000`. Local runs can then use it directly, for example `engram-bench run --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 …`.

## Known gaps

- The DLAMI is expected to ship the SSM agent and AWS CLI, as AWS's Ubuntu images do. If `wait` never sees the instance, check SSM registration in the console first.
- Gated models (Llama) need a Hugging Face token on the instance; the recommended Qwen3-8B is ungated.
- vLLM is installed unpinned at launch. Record the version from `measure-serve.json` or `pip show vllm` in the run notes.
