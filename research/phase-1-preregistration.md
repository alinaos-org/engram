# Phase 1 pre-registration

*Draft, 2026-10-15. This fixes Phase 1's claims, metrics and pass thresholds **before** any Phase 1 training run (issue #15). Changes after the first Phase 1 run must be logged in the [amendments](#amendments) section with a date and a reason. Results that are not pre-registered here are reported as exploratory.*

## What Phase 1 tests

Phase 1 asks whether a test-time-trained memory attached to a **frozen** base model can hold a user's information after the context is wiped. It then asks whether the full Engram stack beats the same base model with retrieval on the categories where retrieval is weak. The pilot showed retrieval is weak on forget, update and temporal ([phase-0-pilot.md](phase-0-pilot.md)). Candidate designs are in [fast-memory-design.md](fast-memory-design.md).

## Fixed setup

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-8B`, frozen, thinking disabled ([shortlist](base-model-shortlist.md)). Any change of base needs an amendment |
| Benchmark | `engram-bench` **v1**: v0 plus long LLM-written sessions that exceed a 128K context, 50 users, 30 sessions, seed fixed at generation and recorded by hash. Generated once and frozen before the first run. Not used for training or tuning. The test split is **sealed** per [sealed-split.md](sealed-split.md): the seed stays private, its commitment and the dataset hash are published before the first run, and the seed is revealed after the Phase 1 results |
| Tuning split | A separate `engram-bench` dev set (different seed, 20 users). All hyperparameter selection happens here |
| Comparators | B3: same base + hybrid retrieval. **B5: Mem0 on the same base** (extract-and-consolidate, with explicit update and delete). B4: same base + per-user LoRA on transcripts. B1: full context, for reference only |
| Systems under test | **E-fast**: memory module only, no adapter, no log. **E-full**: fast memory + sleep-consolidated adapter + retrieval log |
| Probe filter | The leak-verified floor: probes the no-memory base answers correctly are dropped before any analysis |
| Seeds | 3 training seeds for each trained system; report the mean and the range |

## Hypotheses

Each hypothesis has one **primary metric**. The test is a cluster bootstrap over users (10,000 resamples, one-sided) on the paired difference. Primary tests are Holm-corrected across H1–H5 at α = 0.05.

### H1: Fast memory holds information after a context wipe
- **Primary:** fact-recall accuracy at timing *a* (after the session, before any sleep), E-fast minus B4.
- **Pass:** difference ≥ +0.15 absolute, with E-fast ≥ 0.50 absolute.
- **Secondary:** the same at timings *b* and *c*; preference and decision categories.
- **Why this threshold:** B4 on Qwen3-0.6B recalled nothing in a smoke test. A memory that cannot clear naive LoRA by a wide margin does not justify its complexity.

### H2: Attaching memory does no harm
- **Primary:** change in general ability with an *empty* memory module attached, on a fixed suite (MMLU-Pro subset and GSM8K subset, both fixed before the run).
- **Pass:** |Δ| ≤ 1.0 point on each.
- **Secondary:** with an unrelated user's memory attached, Δ ≤ 2.0 points.

### H3: Sleep consolidation keeps what fast memory learned
- **Primary:** E-full recall at timing *c* (several sleeps later, fast memory reset after each sleep) as a fraction of E-fast recall at timing *a*.
- **Pass:** ≥ 0.80.
- **Secondary:** general-suite drift after 10 sleeps ≤ 2.0 points; KL to the base on held-out text reported per sleep.

### H4: Engram beats retrieval where retrieval is weak
- **Primary:** mean of update accuracy (FAMA-style: stale answers are wrong) and forget compliance, E-full minus B3.
- **Pass:** ≥ +0.10. Non-inferiority on fact recall is required too: E-full ≥ B3 − 0.03.
- **Co-primary (stronger comparator):** the same metric, E-full minus B5 (Mem0). Pass: non-inferior, ≥ −0.03. In a one-user smoke test Mem0 handled updates and forget requests well, unlike naive retrieval. So beating B3 alone would overstate the result, and Engram must at least match an explicit-update text memory.
- **Secondary:** temporal and preference categories, and the B1 comparison (reference only).

### H5: Privacy holds
- **Primary:** canary hits across the module-swap, sweep and deletion checks.
- **Pass:** exactly **0**. This is a gate, not a statistical test; any hit fails H5 and is filed under [SECURITY.md](../SECURITY.md).
- **Secondary:**
  - Membership-inference AUC on forgotten items versus never-seen items, after a forget request and the next sleep, reference-calibrated against the base. Target: AUC ≤ 0.55 with the CI upper bound ≤ 0.60.
  - The same AUC for current members, reported for context. We expect it to be high, which is the reason adapters must be encrypted.

## Judged categories

Categories scored by the LLM judge count toward a hypothesis only if the judge's agreement with human labels on that category's calibration set has Cohen's κ ≥ 0.70 (`engram-bench calibrate`). Otherwise they are reported as exploratory, and the hypothesis uses its string-scored categories only.

## Sample size (approximate)

v1 has about 2,900 probes over 50 users, roughly 300–600 per category after the floor. For a proportion near 0.7, the per-category standard error is about 0.02–0.03 before clustering. Users have different difficulty, so the cluster bootstrap will widen this. We expect to detect differences of about 0.08–0.10. The H1 and H4 thresholds were set above that on purpose.

## Decision rules

- **H1 fails** at 8B after the planned Phase 1 budget (≈1,500 GPU-hours; [experiment-plan.md](experiment-plan.md)): stop and review the design. Do not scale to 32B.
- **H1 passes and H3 fails:** the fast memory works but consolidation does not. Prioritise the slot-table and replay variants (#7, #8) before Phase 2.
- **H4 fails, but H1 and H3 pass:** report it honestly. Engram stores information, but does not yet beat retrieval where it should. Phase 2 (training on synthetic users) is then the planned remedy, not a reason to change the claim.
- **H5 fails:** stop all deployment work until it is fixed and re-tested.

## Not pre-registered (exploratory)

Ablations over placement, chunk size, write triggers and decay (#11); LongMemEval and LoCoMo scores; comparisons with B5–B7; any 32B run; cost and latency analyses.

## Amendments

*None yet.*
