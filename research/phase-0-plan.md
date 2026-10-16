# Plan: from Phase 0 to Phase 1

*Draft, 2026-10-15. Covers how Engram finishes Phase 0 (evals and baselines) and what must be true before Phase 1 (the memory module) starts. Durations assume one lead plus coding agents; every compute and cost figure is an assumption until [#12](https://github.com/alinaos-org/engram/issues/12) measures it.*

## Where we are

- **Done:** literature survey, eval design, experiment plan, open questions, 12 Phase 0 issues.
- **Missing:** all code. No benchmark data, no baselines, no measurements.
- **Constraints:** no local GPU, so all GPU work runs on rented or cloud hardware. Phase 0 has a budget assumption of 50–150 GPU-hours plus $0.5–2K in API spend ([experiment plan](experiment-plan.md#phase-0-evals-and-baselines)).

## The Phase 1 entry gate

Phase 1 starts when **all** of these hold. Each maps to an issue.

| # | Gate item | Issue | Evidence |
|---|---|---|---|
| G1 | `engram-bench` v0 exists, is reproducible from a seed, and is tagged | #1, #2 | Release tag, data card, generator config |
| G2 | The benchmark discriminates: B0 (no memory) ≪ B1 (full context), and B2 (frontier + RAG) is well short of B1 on at least two categories | #3 | Baseline table with 95% CIs |
| G3 | The forgetting and leakage harness runs and catches planted leaks | #4 | Harness results on B3/B4, including a deliberately leaky control |
| G4 | The base model is chosen | #5 | Decision note |
| G5 | The fast-memory design is written, and one published retrofit is reproduced at small scale (or the failure is understood) | #6 | Design note + repro report |
| G6 | The container decision (LoRA, slots, or both) is made, or explicitly deferred into a Phase 1 ablation | #7 | Note with forgetting curves or reasons |
| G7 | The Phase 1 corpus has a data card and a rebuild script | #10 | Data card |
| G8 | Compute is measured and the Phase 1 budget is revised | #12 | Updated experiment plan |
| G9 | Threat model and shared-component training policy are written | #9 | Doc linked from SECURITY.md |
| G10 | Phase 1 claims are pre-registered: hypotheses, metrics and success thresholds, written before any Phase 1 run | #15 | Pre-registration doc |

If G2 fails because frontier + RAG already saturates the benchmark, we make the benchmark harder (longer histories, more implicit updates, more indirect probes) and repeat. We do not start Phase 1 on a benchmark that cannot show a difference.

## Workstreams

### A. Foundations (week 1)
- Python package `engram/`, managed with `uv`: `ruff`, `pytest`, and a CI workflow running both on every PR. (#13)
- One results format: every run writes a JSONL file of scored probes plus a manifest (config, seed, git commit, model IDs, hardware, wall-clock). (#14)
- A `MemorySystem` interface that every baseline and, later, Engram itself implements: load and unload a user, run a session turn, end the session (the context wipe), sleep, and forget. The benchmark only ever talks to this interface.

### B. Benchmark (weeks 1–3) — #1, #2
1. World-state schema and seeded event scheduler: facts, preferences, decisions, constraints and episodes, plus assert, update, forget (item, topic, all), incognito and canary events.
2. Session writers:
   - a **template writer** that is deterministic and needs no model, for tests and CI;
   - an **LLM writer** that calls any OpenAI-compatible endpoint (a vLLM server or an API).
3. Probe generator with gold answers and probe timing (before sleep, after sleep, after several sleeps).
4. Scorers: exact and fuzzy match, FAMA-style stale-value scoring, abstention, canary detection, bootstrap CIs, and an LLM-judge interface with a calibration workflow.
5. Leak-verified floor: drop probes that the no-memory model already answers.
6. Pilot with 5 users, spot-check 20 sessions by hand, then generate v0 (50 users × 30 sessions).

### C. Baselines (weeks 2–5) — #3
| Baseline | Needs | When |
|---|---|---|
| B0 no memory | any chat endpoint | week 2 (CPU, against a mock, then a real model) |
| B1 full context | long-context endpoint | week 3 |
| B3 same base + retrieval over the log | base on vLLM | week 3 |
| B2 frontier + retrieval | API key and budget | week 4 (**needs your decision**) |
| B4 per-user LoRA on transcripts | GPU, `peft` | week 4 |
| B5 Mem0, B6 Zep/Graphiti, B7 Letta | their open-source servers | week 5 |
| Reused suites (LongMemEval session-by-session with a wipe, LoCoMo, PrefEval) | dataset downloads | week 4–5 |

### D. Privacy harness (weeks 3–4) — #4
Module swap, canary sweep, retrieval-log ownership gating, and storage audit after forget requests. Validate it against a **deliberately leaky** control system, so we know the harness can fail something. Membership inference against adapters arrives with B4.

### E. Phase 1 groundwork (weeks 2–6, in parallel) — #5, #6, #7, #9, #10, #12
- **#5 Base model.** Shortlist dense 7–8B candidates already used in the retrofit literature (for example Llama-3.1-8B, Qwen3-8B, Mistral-7B-v0.3) plus one hybrid (Qwen3.5-9B; Nemotron-H-8B's licence is research-only). Score them on licence, tooling and B3/B4 results. Pick one primary and one fallback.
- **#6 Fast memory.** Read the full Titan-Llama, In-Place TTT and TTT-NTP papers. Write a design note. Reproduce one result at ≤1B scale, using released code where it exists and reimplementing otherwise.
- **#7 Container.** Estimate what it costs to retrofit memory layers at 8B, and run the slot-versus-LoRA forgetting curve at 0.5–1B first.
- **#9 Threat model.** Write the threat model, the key-custody design, and a list of every shared component with the data it may train on.
- **#10 Corpus.** Choose public-domain and synthetic sources, record licences, and write the chunking and probe pipeline.
- **#12 Compute.** Run a script that estimates costs from the plan's formulas, then replace the estimates with measured throughput on the chosen base.

### F. Move to Phase 1 (week 6–7)
- **#15 Pre-registration.** Write the Phase 1 hypotheses and the thresholds that count as success, before any Phase 1 run.
- **#11 and the 30-night simulation in #8** are experiments on the memory module itself, so they move to the **Phase 1** milestone. Their design notes stay in Phase 0.
- **Gate review:** walk the gate table above, then publish a short Phase 0 report in `research/`.

## Timeline

| Week | Milestone |
|---|---|
| 1 | Foundations merged; world-state and template writer; `MemorySystem` interface; CI green |
| 2 | Probes, scorers and floor; B0 and B3 running end to end on a mock model; base-model shortlist |
| 3 | LLM writer; 5-user pilot and hand check; leakage harness against a leaky control; B1 |
| 4 | v0 generated and tagged; B2 and B4; reused-suite adapters; compute measurements |
| 5 | B5–B7; full baseline table; fast-memory design note; threat model |
| 6 | Small-scale retrofit reproduction; container note; corpus data card |
| 7 | Pre-registration; gate review; Phase 0 report |

## Budget (assumptions)

| Item | Estimate |
|---|---|
| Baseline inference (B0, B1, B3, B4) on rented GPUs | 50–150 GPU-hours |
| Small-scale reproductions (#6, #7) | 50–200 GPU-hours |
| Frontier API for B2 and the LLM judge | $0.5–2K |
| **Total at $2–3 per GPU-hour** | **~$0.7–3K** |

## Decisions

**Made (2026-10-15):**
1. **Compute: AWS** (account 794285265674, us-west-2). The runbook is in [docs/aws-gpu.md](../docs/aws-gpu.md).
2. **Frontier models: Matt's Claude and ChatGPT subscriptions**, through the Claude Code and Codex CLIs, so there is no API spend. B2 and the LLM judge use `claude:claude-opus-5-5` and `codex:gpt-6.1-sol`. Caveats to state in every report:
   - Codex answers inside its own agent instructions.
   - Claude's CLI adds a one-line account note to the context.
   - Both run isolated from the user's own memories, AGENTS.md and MCP servers.
   - Rate limits are shared with Matt's interactive use, so runs use about 4 parallel workers.

**Still open:**
1. **GPU quota.** Every EC2 GPU quota on the account is 0. Requesting 16 vCPUs for G/VT on-demand and spot is free but changes the account. See the runbook.
2. **Support stack and budget.** Deploy `infra/aws/engram-gpu.yaml` (bucket, SSM-only instance role, no-ingress security group, $300 per month budget alerts) and choose the alert email.
3. **Session generator:** it must differ in model family from the base. If generated sessions will later train Engram (Phase 2), it should be open-weights. Candidate: a non-Qwen open model served on the GPU day.
4. **Name:** keep "Engram" despite DeepSeek's architecture of the same name, or rename now while it is cheap.

## Status (2026-10-15)

| Gate | State |
|---|---|
| G1 benchmark v0 | Generator, probes, scorers and floor done. LongMemEval-S/oracle and LoCoMo convert, and an LLM judge grades free-form answers. Sessions are short (template writer): still to do are long LLM-written sessions that exceed the context window |
| G2 discrimination | **Passes on the pilot** ([report](phase-0-pilot.md)): B0 0.28 vs B1 0.98. Frontier + retrieval trails full context on forget (0.42), update (0.70) and temporal (0.71–0.80) with both Claude and GPT. Still to do: repeat on long sessions with 50 users, B5–B7, and base-model runs on the GPU day |
| G3 leakage harness | Done: module swap, extraction sweep and **deletion checks**. It catches the leaky and no-delete controls, and found 0 leaks in the pilot |
| G4 base model | Recommendation: **Qwen3-8B** primary, Llama-3.1-8B-Instruct fallback ([shortlist](base-model-shortlist.md)). The hybrid comparator changes from Nemotron-H-8B (research-only licence) to **Qwen3.5-9B**. Final once B3/B4 run |
| G5 fast-memory design | [Design note](fast-memory-design.md) with three candidates. First reproduction: TTT-NTP on Qwen3-0.6B-Base (released script), ~40–70 GPU-h assumed |
| G7 corpus | [Data card](phase1-corpus.md) written; the rebuild script is still to do |
| G8 compute | `engram-measure` (serving throughput, sleep cost) is ready to run on the GPU day |
| G9 threat model | [Written](threat-model.md) and linked from SECURITY.md |
| B4 | Per-user LoRA baseline implemented and tested on CPU; real runs on the GPU day |

## Risks

| Risk | Mitigation |
|---|---|
| Frontier + RAG saturates the benchmark | Harden the categories where Engram should win (implicit preferences, silent updates, indirect probes); report saturation honestly |
| The LLM judge is noisy | Exact match wherever possible; judge calibrated against human labels; report agreement |
| Synthetic sessions are too easy for related models | Generator family differs from the base; include a human-written calibration subset |
| Retrofit papers have no code | Budget for reimplementation; reproduce at ≤1B first |
| Baseline systems (Mem0, Zep, Letta) cost a lot of setup time | Run them last; B2 and B3 already cover "memory in the harness" |
| Spend before decisions | Nothing that costs money runs until items 1–2 above are decided |
