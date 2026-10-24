"""The launch guards in scripts/aws/gpu.sh, run against a stub ``aws`` (nothing reaches AWS)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "aws" / "gpu.sh"
# Records every call and answers only what `up` asks: an AMI, no running instance, a stack.
STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$AWS_LOG"
case "$*" in
  *"ssm get-parameter"*) echo ami-0 ;;
  *"ec2 describe-instances"*) ;;
  *"cloudformation describe-stacks"*) echo x ;;
  *"ec2 run-instances"*) echo i-0 ;;
  *) exit 1 ;;
esac
"""

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None, reason="needs bash"
)


def _up(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess, str]:
    """Run ``gpu.sh up`` with the stub first on PATH; returns the process and the stub's log.

    Every AWS setting is replaced, so a real ``aws`` that bash found instead of the stub (say,
    with tmp_path on a noexec mount) would have no profile, credentials or instance metadata.
    Nothing inherited may reroute ``aws`` either: bash sources ``BASH_ENV`` (``ENV`` in sh mode)
    before the script, and an exported ``aws`` function would shadow the stub."""
    stub, log = tmp_path / "aws", tmp_path / "aws.log"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log.touch()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AWS_", "BASH_FUNC_")) and k not in ("BASH_ENV", "ENV")
    }
    env |= {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
        "AWS_LOG": str(log),
        "AWS_PROFILE": "engram-test-no-such-profile",
        "AWS_CONFIG_FILE": str(tmp_path / "missing-aws-config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "missing-aws-credentials"),
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    if shutil.which("aws", path=env["PATH"]) != str(stub):
        pytest.skip("the stub is not the first executable aws on PATH (noexec tmp?)")
    cmd = ["bash", str(SCRIPT), "up", "--type", "g6e.2xlarge", *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    return proc, log.read_text()


@pytest.mark.parametrize(
    "args",
    [(), ("--approval",), *(("--approval", a) for a in ["", "a,b", "a b", "a}", "-x", "a" * 101])],
)
def test_up_without_a_valid_approval_fails_before_any_aws_call(tmp_path, args):
    proc, calls = _up(tmp_path, *args)
    assert proc.returncode != 0 and "--approval" in proc.stderr
    assert calls == ""


@pytest.mark.parametrize("approval", ["2026-10-24:chat", "a" * 100])
def test_up_tags_instance_and_volume_with_the_approval(tmp_path, approval):
    proc, calls = _up(tmp_path, "--approval", approval)
    assert proc.returncode == 0, proc.stderr
    tag = f"{{Key=Approval,Value={approval}}}"
    assert f"ResourceType=volume,Tags=[{{Key=Project,Value=engram}},{tag}]" in calls
    assert calls.count(tag) == 2 and "ec2 run-instances --image-id" in calls


def test_dry_run_needs_no_approval(tmp_path):
    proc, calls = _up(tmp_path, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "ec2 run-instances --dry-run" in calls and "describe-stacks" not in calls
