# Phase 0 pilot: frontier baselines on subscription models

*2026-10-15. Raw results (answers, scores, leakage checks, manifests) are in [`results/pilot-2026-10-15/`](../results/pilot-2026-10-15/). Reproduce with `scripts/pilot.sh` and the judge commands below.*

## Setup

- **Benchmark:** `engram-bench` pilot: 10 synthetic users × 30 sessions, template writer, seed 0, sleep every 3 sessions. That gives 559 probes per system (hash in `bench-manifest.json`). The context is wiped after every session, and probes run in no-write sessions.
- **Models:** `claude:claude-opus-5-5` and `codex:gpt-6.1-sol`, both on personal subscriptions through their CLIs at low reasoning effort. See [phase-0-plan.md](phase-0-plan.md#decisions) for the isolation caveats.
- **Systems:**
  - B0: no memory.
  - B1: every earlier non-incognito user message in the prompt.
  - B2: hybrid BM25 + bge-small retrieval over the user's own log, top 5.
  - Session turns are logged, not answered (`--no-session-replies`); this doesn't change what these baselines remember.
- **Leakage:** for each user, module swap against 1 other user, 3 extraction prompts, then deletion checks on 3 users.
- **Judging:** answers that mention both the current and an outdated value were re-graded by an LLM judge from the *other* provider. Claude's answers were judged by GPT-6.1 (94 in B1, 70 in B2); OpenAI's by Claude Opus 5.5 (3 and 6). No judge call failed.
- **Cost:** 2,426 model calls (the rest were cache hits), about 40 minutes of wall-clock time per provider at 4 workers, plus about 175 judge calls. No API spend.

## Results

Accuracy with 95% bootstrap CIs; n is the number of probes.

**Claude Opus 5.5**

| category | B0 no memory | B1 full context | B2 frontier + retrieval |
|---|---|---|---|
| abstention | 1.00 (60) | 1.00 | 1.00 |
| decision | 0.00 (30) | 1.00 | 1.00 |
| fact_recall | 0.00 (117) | 1.00 | 1.00 |
| forget | 1.00 (45) | 0.91 [0.82, 0.98] | **0.42** [0.29, 0.58] |
| incognito | 1.00 (53) | 1.00 | 1.00 |
| preference | 0.00 (119) | 0.97 [0.94, 1.00] | 0.90 [0.84, 0.95] |
| temporal | 0.00 (35) | 0.97 [0.91, 1.00] | **0.80** [0.66, 0.91] |
| update | 0.00 (100) | 0.96 [0.92, 0.99] | **0.70** [0.61, 0.79] |
| **all** | 0.28 | 0.98 [0.97, 0.99] | 0.87 [0.84, 0.89] |

**GPT-6.1 (Codex)**

| category | B0 no memory | B1 full context | B2 frontier + retrieval |
|---|---|---|---|
| abstention | 1.00 (60) | 1.00 | 1.00 |
| decision | 0.00 (30) | 1.00 | 1.00 |
| fact_recall | 0.00 (117) | 1.00 | 1.00 |
| forget | 1.00 (45) | 1.00 | **0.42** [0.29, 0.58] |
| incognito | 1.00 (53) | 1.00 | 1.00 |
| preference | 0.00 (119) | 0.97 [0.94, 1.00] | 0.90 [0.84, 0.95] |
| temporal | 0.00 (35) | 0.91 [0.80, 1.00] | **0.71** [0.57, 0.86] |
| update | 0.00 (100) | 0.97 [0.93, 1.00] | **0.70** [0.61, 0.79] |
| **all** | 0.28 | 0.98 [0.97, 0.99] | 0.86 [0.83, 0.89] |

**Leakage:**
- B1: 0 of 199 checks leaked, for each provider.
- B2: 0 of 246 checks (module swap, extraction sweep and deletion), for each provider.
- Owner gating held, and deleting a user removed everything retrievable.

## What this says

1. **The Phase 0 discrimination gate (G2) passes on the pilot.**
   - B0 is far below B1 (0.28 vs 0.98).
   - Frontier + retrieval falls well short of full context on three categories with both providers: **forget** (0.42 vs 0.91–1.00), **update** (0.70 vs 0.96–0.97) and **temporal** (0.71–0.80 vs 0.91–0.97).
   - Fact, decision, abstention and incognito recall are saturated. The pilot was meant to check discrimination, not difficulty.
2. **The gap comes from the memory, not the model.** B2's per-category scores are almost identical across Claude and GPT. Retrieval returns the old value next to (or instead of) the new one, and keeps returning items the user asked to forget. Those are exactly the behaviours Engram's write, update and forget path must get right.
3. **Forgetting is where naive memory fails hardest.** The frontier models honour "please forget" when the request is in context (B1). They cannot honour it once it has been chunked away from the fact it refers to (B2). A text-memory system with explicit deletion (Mem0, B5) is the fair comparison and runs next.

## Caveats (why this is a pilot, not a result)

- **Too easy and too short.** Template sessions are a few sentences, so the whole history fits in context, and B1 is close to a perfect oracle. Engram's claim is about histories *beyond* the context window. G1's remaining work is long, LLM-written sessions with implicit updates, indirect probes and preference-application probes.
- **Small n.** With 10 users, temporal has n = 35 and wide CIs. The v0 run uses 50 users.
- **Single run, one prompt.** No variance across seeds or prompts yet.
- **Subscription harnesses are not raw models.** Codex answers inside its agent instructions. Claude's CLI injects the operator's account email into every call's context. Under the extraction-sweep prompts, Claude repeated that email in 4 answers in the B2 run. It is not benchmark data, so no leak counted, but it is real contamination of the baseline. Saved runs now redact email addresses (`redacted_emails` in each manifest), and published results contain none.

## Scoring bugs this pilot caught (all fixed, with tests)

| Symptom | Cause | Fix |
|---|---|---|
| Codex scored 0 on abstention | Curly apostrophe in "I don’t know" | Normalise quotes before matching |
| Correct update answers marked stale ("Feset. You switched after Ullio went downhill") | Any mention of the old value counted as asserting it | Stale only if it is not framed as history *and* comes before the current value; ambiguous answers go to the judge |
| "I don't know – metric or imperial?" counted as recall and as an incognito leak | Option lists treated as statements | Values only offered as "A or B" next to an abstention are neither hits nor leaks |
| Judge failed correct update answers | The rubric said "must not reveal any outdated value" | The rubric now allows history and fails only when the old value is presented as current |
| Claude judge call crashed the run | Transient CLI failure | Failed judge calls keep the string verdict and are counted (`judge.failed`) |

## Next

1. LLM-written long sessions (G1), then the 50-user v0 run.
2. Mem0, Zep and Letta baselines (B5–B7) on the same data.
3. GPU day ([docs/aws-gpu.md](../docs/aws-gpu.md)): Qwen3-8B B0, B1, B3 and B4, plus LongMemEval-S and LoCoMo.
4. Calibrate the judge against human labels (`engram-bench calibrate export`) before relying on judged categories.
