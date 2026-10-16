"""Frontier models through the Claude Code and Codex CLIs, billed to a personal subscription.

These run the official CLIs in non-interactive mode, so no API key is needed. Two caveats belong in
every report that uses them:

* **Claude** (``claude -p``) runs with tools, MCP servers and settings disabled and our own system
  prompt. The CLI still adds a short account note (the account email) to the context.
* **Codex** (``codex exec``) always runs inside Codex's own agent instructions; ours go at the
  top of the prompt. It runs from an isolated ``CODEX_HOME`` that holds only a symlink to the real
  ``auth.json`` (so token refreshes stay in one place) and a minimal config, so the user's own
  AGENTS.md, memories and MCP servers never reach the model. That matters in a memory benchmark.

Calls are slow (seconds each) and rate-limited, so wrap them in ``CachedModel``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from engram.models.chat import Message, strip_think

CACHE_ROOT = Path(os.environ.get("ENGRAM_CACHE", Path.home() / ".cache" / "engram"))


class CLIError(RuntimeError):
    pass


def _transcript(messages: list[Message]) -> tuple[str, str]:
    """Split messages into (system prompt, flattened conversation ending with the last turn)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1:
        return system, turns[0]["content"]
    lines = ["Conversation so far:"]
    for m in turns[:-1]:
        lines.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}")
    lines += ["", "Now reply to the user's latest message:", turns[-1]["content"]]
    return system, "\n".join(lines)


def _retrying(fn, retries: int, base_delay: float):
    for attempt in range(retries):
        try:
            return fn()
        except (CLIError, subprocess.TimeoutExpired):
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * 3**attempt)
    raise AssertionError("unreachable")


@dataclass
class ClaudeCLI:
    """Claude via ``claude -p`` on the user's subscription."""

    model: str = "claude-opus-5-5"
    effort: str = "low"
    timeout: float = 300.0
    retries: int = 3
    retry_delay: float = 10.0

    @property
    def name(self) -> str:
        return f"claude:{self.model}"

    def _call(self, system: str, prompt: str) -> str:
        cmd = [
            "claude", "-p", "--model", self.model, "--effort", self.effort,
            "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--setting-sources", "", "--no-session-persistence", "--output-format", "json",
            "--system-prompt", system or "You are a helpful assistant.", prompt,
        ]  # fmt: skip
        with tempfile.TemporaryDirectory(prefix="engram-claude-") as cwd:
            proc = subprocess.run(
                cmd, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=self.timeout,
            )  # fmt: skip
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise CLIError(f"claude exited {proc.returncode}: {proc.stderr[-500:]}") from e
        if data.get("is_error"):
            raise CLIError(f"claude error: {str(data.get('result'))[:500]}")
        return data.get("result") or ""

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        system, prompt = _transcript(messages)
        return strip_think(
            _retrying(lambda: self._call(system, prompt), self.retries, self.retry_delay)
        )


# Account-level apps and plugins would add tool definitions to the model's context, and agent
# features (browser, multi-agent, image generation) are irrelevant to answering a question.
CODEX_CONFIG = """# Managed by engram (engram/models/subscription.py); rewritten on every run.
# Isolated Codex home for engram-bench: no AGENTS.md, memories, apps, plugins or MCP servers.
[features]
apps = false
plugins = false
remote_plugin = false
browser_use = false
computer_use = false
multi_agent = false
image_generation = false
goals = false
"""


def isolated_codex_home(root: Path = CACHE_ROOT) -> Path:
    """A CODEX_HOME with only a symlink to the real auth.json and a minimal config."""
    home = root / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    real = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
    link = home / "auth.json"
    if not link.is_symlink():
        if link.exists():
            raise CLIError(f"{link} exists and is not a symlink; refusing to touch it")
        link.symlink_to(real)
    (home / "config.toml").write_text(CODEX_CONFIG)  # managed: rewritten on every run
    return home


@dataclass
class CodexCLI:
    """OpenAI models via ``codex exec`` on the user's ChatGPT subscription."""

    model: str = "gpt-6.1-sol"
    effort: str = "low"
    timeout: float = 300.0
    retries: int = 3
    retry_delay: float = 10.0
    home: Path = field(default_factory=isolated_codex_home)

    @property
    def name(self) -> str:
        return f"codex:{self.model}"

    def _call(self, prompt: str) -> str:
        env = {**os.environ, "CODEX_HOME": str(self.home)}
        with tempfile.TemporaryDirectory(prefix="engram-codex-") as cwd:
            out = Path(cwd) / "out.txt"
            cmd = [
                "codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
                "-m", self.model, "-c", f"model_reasoning_effort={self.effort}",
                "-c", "mcp_servers={}",
                "-o", str(out), prompt,
            ]  # fmt: skip
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=self.timeout,
            )  # fmt: skip
            if proc.returncode != 0 or not out.exists():
                raise CLIError(f"codex exited {proc.returncode}: {proc.stderr[-500:]}")
            return out.read_text(encoding="utf-8")

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        system, prompt = _transcript(messages)
        full = (
            f"Follow these instructions for your reply:\n{system}\n\n{prompt}\n\n"
            "Reply with your answer to the user only. Do not run commands or inspect files."
        )
        return strip_think(_retrying(lambda: self._call(full), self.retries, self.retry_delay))
