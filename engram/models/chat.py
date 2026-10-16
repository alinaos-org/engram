"""Minimal chat-model clients.

``OpenAIChat`` talks to any OpenAI-compatible ``/v1/chat/completions`` endpoint: a local vLLM
server, or a hosted API. It uses only the standard library. The test doubles at the bottom let the
whole benchmark run on CPU with no model at all.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str: ...


@dataclass
class OpenAIChat:
    """Client for an OpenAI-compatible chat endpoint.

    The API key is read from the environment variable named by ``api_key_env``, never passed in
    configuration files, so run manifests never contain secrets.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str | None = None
    temperature: float = 0.0
    timeout: float = 120.0
    retries: int = 3

    @property
    def name(self) -> str:
        return self.model

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                return data["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------
# Test doubles. They are deterministic and need no network.


@dataclass
class NullModel:
    """Always declines. Scores zero on recall and full marks on abstention."""

    name: str = "null"

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        return "I don't know."


@dataclass
class ParrotModel:
    """Answers by repeating every earlier message in its context.

    Useful for testing the plumbing: if a memory system puts the right text in context, the parrot
    "recalls" it; if a system leaks another user's text into context, the parrot exposes it.
    """

    name: str = "parrot"

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        context = [m["content"] for m in messages[:-1]]
        if not context:
            return "I don't know."
        return " | ".join(context)


def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks that some open reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
