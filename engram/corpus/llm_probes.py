"""Stage 6-7: LLM probe generation and verification for natural text (§c).

This module is an interface. ``LLMProbeGenerator`` wraps any ``engram.models.chat.ChatModel``
(the plan names ``mistralai/Mistral-Small-3.2-24B-Instruct-2506`` behind vLLM); ``ChatVerifier``
wraps the open-book verifier (``allenai/Olmo-3.1-32B-Instruct``) and the closed-book floor
(``Qwen/Qwen3-8B-Base``). Nothing here runs a model unless the caller passes one.

``run_stage(..., dry_run=True)`` writes the exact requests (one per chunk, with prompt sha) and a
cost estimate, so a GPU day can be planned before any generation.

Checks applied without a model (drop on failure):
  1. ``evidence_span`` is an exact substring of the chunk, <= 40 words; the answer is <= 10 words
     and appears in the span (or a verifier says it is entailed);
  rules: allowed ``type``; no yes/no questions; the question does not copy the span verbatim;
  4. near-duplicate questions within a document (Jaccard >= 0.8 on word 5-grams) are dropped.
With a verifier: 2. open-book answer must match; 3. closed-book answer must *not* match.
Human spot-checks (5) are outside the script.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from engram.corpus.io import sha256, write_json, write_jsonl
from engram.corpus.text import jaccard, shingle_set, words
from engram.models.chat import ChatModel, strip_think

TYPES = ("fact", "attribute", "temporal_order", "cross_chunk", "abstain")
_YES_NO = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b", re.I
)

PROMPT = """You write reading-comprehension probes for a memory benchmark.

Read the passage and return a JSON list of {n_min} to {n_max} items. Each item is an object:
  {{"question": ..., "answer": ..., "evidence_span": ..., "type": ...}}
with "type" one of: fact, attribute, temporal_order, cross_chunk, abstain.

Rules:
- Each question must be answerable only from the passage, not from world knowledge.
- "answer" has at most 10 words.
- "evidence_span" is a verbatim quote from the passage of at most 40 words that supports the answer.
- Paraphrase the passage in the question; do not copy its wording.
- No yes/no questions.
- No questions about real living people, authors, or their affiliations.
- For "abstain" items, ask about an entity or attribute the passage plausibly does not state; the
  answer is "not stated" and the evidence_span is "".

Reply with the JSON list only.

Passage:
\"\"\"
{text}
\"\"\""""


class ProbeGenerator(Protocol):
    name: str

    def messages(self, chunk: dict[str, Any]) -> list[dict[str, str]]: ...

    def generate(self, chunk: dict[str, Any]) -> list[dict[str, Any]]: ...


class Verifier(Protocol):
    def open_book(self, passage: str, question: str) -> str: ...

    def closed_book(self, question: str) -> str: ...


@dataclass
class LLMProbeGenerator:
    model: ChatModel
    n_min: int = 3
    n_max: int = 5
    max_tokens: int = 1024

    @property
    def name(self) -> str:
        return self.model.name

    def messages(self, chunk: dict[str, Any]) -> list[dict[str, str]]:
        prompt = PROMPT.format(n_min=self.n_min, n_max=self.n_max, text=chunk["text"])
        return [{"role": "user", "content": prompt}]

    def generate(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        raw = self.model.complete(self.messages(chunk), max_tokens=self.max_tokens)
        return parse_items(raw)


@dataclass
class ChatVerifier:
    """Open-book verifier and closed-book floor model, each any ChatModel."""

    open_model: ChatModel | None = None
    closed_model: ChatModel | None = None

    def open_book(self, passage: str, question: str) -> str:
        assert self.open_model is not None
        msg = f"Answer briefly from the passage only.\n\nPassage:\n{passage}\n\nQ: {question}\nA:"
        out = self.open_model.complete([{"role": "user", "content": msg}], max_tokens=64)
        return strip_think(out)

    def closed_book(self, question: str) -> str:
        assert self.closed_model is not None
        msg = f"Q: {question}\nA:"
        out = self.closed_model.complete([{"role": "user", "content": msg}], max_tokens=64)
        return strip_think(out)


def parse_items(raw: str) -> list[dict[str, Any]]:
    text = strip_think(raw)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except ValueError:
        return []
    return [i for i in items if isinstance(i, dict)]


def _norm(s: str) -> str:
    return " ".join(words(s))


def answer_matches(answer: str, candidate: str) -> bool:
    a, c = _norm(answer), _norm(candidate)
    return bool(a) and (a in c or (bool(c) and c in a))


def check_item(item: dict[str, Any], chunk_text: str) -> str | None:
    """Model-free checks; returns the failure reason or None."""
    q = str(item.get("question", "")).strip()
    a = str(item.get("answer", "")).strip()
    span = str(item.get("evidence_span", ""))
    kind = item.get("type")
    if kind not in TYPES:
        return "type"
    if not q or not a:
        return "empty"
    if _YES_NO.match(q):
        return "yes_no"
    if len(a.split()) > 10:
        return "answer_length"
    if kind == "abstain":
        return None if _norm(a) == "not stated" else "abstain_answer"
    if not span or span not in chunk_text:
        return "span_not_substring"
    if len(span.split()) > 40:
        return "span_length"
    if _norm(q) and _norm(q) in _norm(chunk_text):
        return "copies_text"
    if _norm(a) not in _norm(span):
        return "answer_not_in_span"
    return None


def to_probe(item: dict[str, Any], chunk: dict[str, Any], generator: str, n: int) -> dict[str, Any]:
    """Convert a checked item into a document-level probe (spans in document offsets)."""
    cs, ce = chunk["char_offsets"]
    base = {
        "id": f"{chunk['id']}#q{n}",
        "source": chunk["source"],
        "doc_id": chunk["doc_id"],
        "question": item["question"].strip(),
        "attribute": None,
        "accept": [],
        "generator": generator,
    }
    if item["type"] == "abstain":
        return {**base, "type": "abstain", "facts": [], "anchor": [cs, ce]}
    i = chunk["text"].index(item["evidence_span"])
    start = cs + i
    end = start + len(item["evidence_span"])
    fact = {"value": item["answer"].strip(), "start": start, "end": end}
    return {**base, "type": item["type"], "facts": [fact], "anchor": None}


def verify(
    probes: list[dict[str, Any]],
    chunk: dict[str, Any],
    verifier: Verifier | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    kept, reasons = [], []
    for p in probes:
        if verifier is not None and p["facts"]:
            gold = p["facts"][-1]["value"]
            if not answer_matches(gold, verifier.open_book(chunk["text"], p["question"])):
                reasons.append("open_book")
                continue
            if answer_matches(gold, verifier.closed_book(p["question"])):
                reasons.append("closed_book_floor")
                continue
        kept.append(p)
    return kept, reasons


def dedup_questions(probes: list[dict[str, Any]], threshold: float = 0.8) -> list[dict[str, Any]]:
    """Drop near-duplicate questions within each document (step 4)."""
    seen: dict[tuple[str, str], list[set[int]]] = {}
    out = []
    for p in probes:
        sh = shingle_set(p["question"], 5, short_ok=True)
        prev = seen.setdefault((p["source"], p["doc_id"]), [])
        if any(jaccard(sh, o) >= threshold for o in prev):
            continue
        prev.append(sh)
        out.append(p)
    return out


def run_stage(
    chunks: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    generator: ProbeGenerator | None,
    verifier: Verifier | None = None,
    dry_run: bool = True,
    est_output_tokens: int = 400,
) -> dict[str, Any]:
    """Generate probes for ``chunks`` (or, with ``dry_run``, only write the requests)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(chunks)
    if dry_run or generator is None:
        gen = generator or LLMProbeGenerator(model=UnrunModel())
        requests = []
        for c in chunks:
            msgs = gen.messages(c)
            requests.append(
                {"chunk_id": c["id"], "prompt_sha256": sha256(json.dumps(msgs)), "messages": msgs}
            )
        write_jsonl(out_dir / "llm_requests.jsonl", requests)
        write_jsonl(out_dir / "llm.jsonl", [])
        summary = {
            "dry_run": True,
            "generator": gen.name,
            "requests": len(requests),
            "input_tokens_approx": sum(c.get("n_tokens", 0) for c in chunks) + 250 * len(requests),
            "output_tokens_estimate": est_output_tokens * len(requests),
        }
        write_json(out_dir / "llm_summary.json", summary)
        return summary
    probes: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for c in chunks:
        items = generator.generate(c)
        checked = []
        for n, item in enumerate(items):
            why = check_item(item, c["text"])
            if why:
                reasons[why] = reasons.get(why, 0) + 1
            else:
                checked.append(to_probe(item, c, generator.name, n))
        kept, failed = verify(checked, c, verifier)
        for why in failed:
            reasons[why] = reasons.get(why, 0) + 1
        probes.extend(kept)
    before = len(probes)
    probes = dedup_questions(probes)
    reasons["near_duplicate"] = before - len(probes)
    write_jsonl(out_dir / "llm.jsonl", probes)
    summary = {
        "dry_run": False,
        "generator": generator.name,
        "chunks": len(chunks),
        "probes": len(probes),
        "dropped": reasons,
    }
    write_json(out_dir / "llm_summary.json", summary)
    return summary


@dataclass
class UnrunModel:
    """Stands in for the configured model in dry runs: has its name, refuses to complete."""

    name: str = "unset (dry run)"

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        raise RuntimeError("dry run: no model is called")
