"""Text normalisation, stable shingle hashes, Jaccard and MinHash.

Hashes are 64-bit BLAKE2b digests, so they are stable across processes (Python's ``hash`` is
salted per run) and an index built in one run can be compared with another.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Iterator

_PUNCT = re.compile(r"[^\w\s]+")
_WORD = re.compile(r"\S+")
_MASK = (1 << 64) - 1
_PRIME = (1 << 61) - 1


def words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words (research/phase1-corpus.md §d.4)."""
    return _PUNCT.sub(" ", text.lower()).replace("_", " ").split()


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def shingles(tokens: list[str], n: int) -> Iterator[int]:
    """Hashes of every n-word window. A text shorter than n yields nothing."""
    for i in range(len(tokens) - n + 1):
        yield hash64(" ".join(tokens[i : i + n]))


def shingle_set(text: str, n: int, *, short_ok: bool = False) -> set[int]:
    """Shingle hashes of ``text``. With ``short_ok``, a text shorter than n is one shingle."""
    toks = words(text)
    if short_ok and 0 < len(toks) < n:
        return {hash64(" ".join(toks))}
    return set(shingles(toks, n))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


class MinHasher:
    """MinHash signatures over word n-gram shingles, with LSH banding for candidate pairs.

    Long documents are sampled by content (only shingles whose hash is 0 mod ``sample``), so two
    copies of a book sample the same shingles even when their headers differ.
    """

    def __init__(self, num_perm: int = 64, bands: int = 16, n: int = 5, seed: int = 0) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        rng = random.Random(seed)
        self.n = n
        self.bands = bands
        self.rows = num_perm // bands
        self.params = [
            (rng.randrange(1, _PRIME), rng.randrange(0, _PRIME)) for _ in range(num_perm)
        ]

    def signature(self, text: str, sample: int = 1) -> tuple[int, ...]:
        hs = [h for h in shingle_set(text, self.n, short_ok=True) if h % sample == 0]
        if not hs:
            return tuple([_MASK] * len(self.params))
        return tuple(min((a * h + b) % _PRIME for h in hs) for a, b in self.params)

    def band_keys(self, sig: tuple[int, ...]) -> Iterable[tuple[int, tuple[int, ...]]]:
        for b in range(self.bands):
            yield b, sig[b * self.rows : (b + 1) * self.rows]

    @staticmethod
    def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def sample_rate(n_words: int, target: int = 4000) -> int:
    """Content-sampling modulus: 1 for short texts, 8 for anything book-length."""
    return 1 if n_words <= target else 8
