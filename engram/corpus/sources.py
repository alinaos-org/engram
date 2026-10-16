"""Source registry: the candidate-source table of research/phase1-corpus.md, as data.

Licence strings are verbatim from each card or licence file (fetched 2026-10-15). ``verdict``
decides what ``fetch`` will touch: ``use`` and ``holdout`` sources are fetched; ``validation``,
``optional`` and ``ablation`` sources need an explicit flag; lineage exclusions are refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    hf_id: str | None
    licence: str
    size: str
    verdict: str  # use | holdout | validation | optional | ablation | generated
    slice: str | None = None  # v0 mix slice (A-F)
    config: str | None = None
    split: str = "train"
    text_format: str = "text"  # text | pg19 | wikipedia | conversation_chronicles
    licence_field: str | None = None  # per-document licence path(s), "a.b|a.c" = first found
    real_people: bool = False
    generated: bool = False
    filters: tuple[str, ...] = ()  # ocr | biography | arxiv_denylist | pg19_holdout
    chunk_mode: str = "tokens"  # tokens | session
    probes: str = "llm"  # llm | template | none
    mirror: str | None = None  # dataset id actually streamed, when hf_id cannot be (scripts)
    notes: str = ""
    card_url: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.hf_id and not self.card_url:
            object.__setattr__(self, "card_url", f"https://huggingface.co/datasets/{self.hf_id}")


REGISTRY: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            "gutenberg",
            "common-pile/project_gutenberg_filtered",
            'No dataset licence tag. Books "marked as in the Public Domain according to the '
            'provided metadata"',
            "55,454 docs, 20.1 GB; 5.7B Comma tokens",
            "use",
            "A",
            filters=("pg19_holdout",),
        ),
        Source(
            "pg19",
            "deepmind/pg19",
            '"Apache License, Version 2.0"',
            "28,602 / 50 / 100 books (train/val/test)",
            "holdout",
            None,
            split="validation",
            text_format="pg19",
            mirror="emozilla/pg19",
            notes="val/test books are held out; their Gutenberg ids are removed from (1). "
            "deepmind/pg19 is a loading script, which datasets>=4 refuses; fetch streams the "
            "emozilla/pg19 parquet copy",
        ),
        Source(
            "pre1929",
            "common-pile/pre_1929_books_filtered",
            'No tag. "Books published in the US before 1929 passed into the public domain on '
            'January 1, 2024"',
            "124,898 docs, 46.3 GB; 12.4B Comma tokens",
            "use",
            "A",
            filters=("ocr",),
        ),
        Source(
            "arxiv",
            "common-pile/arxiv_papers_filtered",
            'Papers "uploaded under CC BY, CC BY-SA, and CC0 licenses"',
            "304,048 docs, 19 GB; 6.0B Comma tokens",
            "use",
            "B",
            licence_field="metadata.license",
            filters=("arxiv_denylist",),
        ),
        Source(
            "pes2o",
            "common-pile/peS2o",
            'Per-document `metadata.license`; the "Creative Common and public domain subset" of '
            "peS2o",
            "6,294,020 docs, 188.2 GB; 43.3B Comma tokens",
            "use",
            "B",
            # the card says metadata.license; rows fetched 2026-10-15 carry metadata.oa_license
            licence_field="metadata.license|metadata.oa_license",
            filters=("arxiv_denylist",),
        ),
        Source(
            "wikipedia",
            "wikimedia/wikipedia",
            '"GNU Free Documentation License (GFDL) and the Creative Commons '
            'Attribution-Share-Alike 3.0 License"',
            "6,407,814 articles, 11.6 GB parquet",
            "use",
            "C",
            config="20231101.en",
            text_format="wikipedia",
            filters=("biography",),
        ),
        Source(
            "conv_chronicles",
            "jihyoung/ConversationChronicles",
            "cc-by-4.0",
            "200K episodes, 1M sessions, 11.7M turns",
            "use",
            "E",
            text_format="conversation_chronicles",
            chunk_mode="session",
            notes="LLM-generated; full n-gram filter (multi-session shape like LoCoMo)",
        ),
        Source(
            "engram_bios",
            None,
            "Engram-generated (Apache-2.0)",
            "generated",
            "generated",
            "D",
            generated=True,
            probes="template",
            notes="bioS-style re-implementation (arXiv:2309.14316); bioS itself is not public",
        ),
        Source(
            "tofu",
            "locuslab/TOFU",
            "mit",
            "200 fictitious authors; full = 4,000 QA rows",
            "validation",
            config="full",
            probes="none",
        ),
        Source(
            "entigraph_quality",
            "zitongyang/entigraph-quality-corpus",
            "apache-2.0",
            "455M tokens, 263 articles",
            "optional",
            "F",
            notes="QuALITY per-article licences unverified",
        ),
        Source(
            "stackexchange",
            "common-pile/stackexchange",
            "Per-document `metadata.license`",
            "33,415,400 docs, 103.7 GB",
            "ablation",
            licence_field="metadata.license",
            real_people=True,
        ),
        Source(
            "github_archive",
            "common-pile/github_archive",
            'Per-document; threads kept only from repos with a "Blue Oak Council-approved license"',
            "30,318,774 docs, 54.7 GB",
            "ablation",
            real_people=True,
        ),
        Source(
            "ubuntu_irc",
            "common-pile/ubuntu_irc",
            'Logs "archived and released into the Public Domain"',
            "329,115 docs, 6.3 GB",
            "ablation",
            real_people=True,
        ),
    ]
}

# Whole sources excluded before download (§d.1): eval filler lineage or persona lineage.
LINEAGE_EXCLUSIONS: dict[str, str] = {
    "anon8231489123/ShareGPT_Vicuna_unfiltered": "ShareGPT: LongMemEval filler source",
    "RyokoAI/ShareGPT52K": "ShareGPT: LongMemEval filler source",
    "stingning/ultrachat": "UltraChat: LongMemEval filler source",
    "HuggingFaceH4/ultrachat_200k": "UltraChat: LongMemEval filler source",
    "lmsys/lmsys-chat-1m": "LMSYS-Chat-1M: PrefEval inter-turn filler; real users",
    "google/Synthetic-Persona-Chat": "Persona-Chat lineage (LoCoMo personas)",
    "bavard/personachat_truecased": "Persona-Chat lineage (LoCoMo personas)",
    "nayohan/multi_session_chat": "MSC lineage (LoCoMo personas)",
    "allenai/WildChat-1M": "real users with demographic data",
}

# arXiv ids of the benchmark papers reused in eval-design.md (§d.2).
ARXIV_DENYLIST = (
    "2410.10813",
    "2402.17753",
    "2502.09597",
    "2507.05257",
    "2510.27246",
    "2504.14225",
)

# Per-document licence allowlist (§e.2): PD, CC0, CC BY, CC BY-SA, GFDL.
LICENCE_ALLOWLIST = ("PD", "CC0", "CC-BY", "CC-BY-SA", "GFDL")


def normalise_licence(raw: str | None) -> str | None:
    """Map a free-text licence ("CC BY 4.0", "CCBY", "cc0-1.0", "Public Domain") to an
    allowlist class, or None if it is not openly licensed (NC and ND variants are not)."""
    if not raw:
        return None
    c = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if "PUBLICDOMAIN" in c or c in ("PD", "PDM"):
        return "PD"
    if c.startswith("CC0") or "CCZERO" in c:
        return "CC0"
    if "GFDL" in c or "GNUFREEDOCUMENTATION" in c:
        return "GFDL"
    if "NONCOMMERCIAL" in c or "NODERIV" in c or re.search(r"BY(SA)?(NC|ND)", c):
        return None
    if re.search(r"BYSA|SHAREALIKE", c):
        return "CC-BY-SA"
    if "CCBY" in c or "CREATIVECOMMONSATTRIBUTION" in c:
        return "CC-BY"
    return None


def get(key: str) -> Source:
    try:
        return REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"unknown source {key!r}; known: {', '.join(REGISTRY)}") from e
