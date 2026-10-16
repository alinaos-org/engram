"""Pluggable token counters.

Chunk sizes and the data card are in tokens of the base model (``Qwen/Qwen3-8B``). Tests and dry
runs use ``WhitespaceTokenizer``, a model-free approximation; ``tokens_per_word`` scales it (about
1.3 Qwen3 tokens per English word is a reasonable guess, unmeasured).

Specs for ``get_tokenizer``:
    whitespace          one token per whitespace-separated word
    whitespace:1.3      words x 1.3, rounded
    hf:Qwen/Qwen3-8B    the real tokenizer (needs ``transformers``; downloads tokenizer files)
    hf:<id>@<revision>  pinned revision
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_WORD = re.compile(r"\S+")


class Tokenizer(Protocol):
    name: str
    approximate: bool

    def count(self, text: str) -> int: ...


@dataclass
class WhitespaceTokenizer:
    tokens_per_word: float = 1.0
    approximate: bool = True

    @property
    def name(self) -> str:
        if self.tokens_per_word == 1.0:
            return "whitespace"
        return f"whitespace:{self.tokens_per_word:g}"

    def count(self, text: str) -> int:
        n = len(_WORD.findall(text))
        return n if self.tokens_per_word == 1.0 else round(n * self.tokens_per_word)


class HFTokenizer:
    approximate = False

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "hf: tokenizers need transformers (uv sync --extra ml); "
                "use --tokenizer whitespace for a model-free approximation"
            ) from e
        self.model_id = model_id
        self.revision = revision
        self._tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    @property
    def name(self) -> str:
        return f"hf:{self.model_id}" + (f"@{self.revision}" if self.revision else "")

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


def get_tokenizer(spec: str | None) -> Tokenizer:
    spec = spec or "whitespace"
    if spec == "whitespace":
        return WhitespaceTokenizer()
    if spec.startswith("whitespace:"):
        return WhitespaceTokenizer(float(spec.split(":", 1)[1]))
    if spec.startswith("hf:"):
        model, _, rev = spec[3:].partition("@")
        return HFTokenizer(model, rev or None)
    raise ValueError(f"unknown tokenizer spec {spec!r}")
