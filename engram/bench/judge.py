"""LLM judge for probes that string matching cannot score, plus judge calibration.

v0 probes are all exact-match scorable. The judge is used for preference-application and
implicit-constraint probes (LoCoMo-Plus style) as they are added. Before its scores are trusted,
it must be calibrated against human labels (issue #2), reported as Cohen's kappa.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.models.chat import ChatModel, strip_think

RUBRIC = """You are grading an AI assistant's answer for a memory benchmark.
Question: {question}
Correct current answer: {gold}
Outdated values (no longer true): {reject}
Assistant's answer: {answer}

Reply PASS if the answer gives the correct current answer (paraphrase is fine). Mentioning an
outdated value as history ("you switched from X") is fine; presenting an outdated value as the
current answer, or hedging between it and the correct one, is a FAIL. Reply with one word."""


@dataclass
class LLMJudge:
    model: ChatModel

    def grade(self, question: str, gold: str, reject: list[str], answer: str) -> bool:
        prompt = RUBRIC.format(
            question=question, gold=gold, reject=", ".join(reject) or "none", answer=answer
        )
        out = strip_think(self.model.complete([{"role": "user", "content": prompt}], max_tokens=8))
        return out.strip().upper().startswith("PASS")


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two binary labelers beyond chance."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
