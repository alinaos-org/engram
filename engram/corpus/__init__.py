"""Phase 1 training-corpus rebuild (research/phase1-corpus.md).

The ``engram-corpus`` command (``engram.corpus.cli:main``) runs the stages in order:

    fetch -> filter -> bios -> chunk -> decontam -> probes -> episodes -> pack -> card

Everything here uses the standard library. ``datasets`` (streaming fetch), ``transformers`` (the
Qwen3 tokenizer) and ``numpy`` (a compact n-gram index) are optional and imported lazily.
"""
