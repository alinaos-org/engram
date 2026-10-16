# Experiment plan

*Draft, 2026-10-15. Every compute figure below is an **assumption** made before any measurement. Each one shows how it was derived, so it can be corrected once real runs exist.*

Engram moves through four phases. Each phase has a hypothesis, a decision gate, and an exit artifact. We do not start a phase until the previous gate passes, or until we have written down why we are skipping it.

| Phase | Question | Exit artifact |
|---|---|---|
| 0 · Evals | Can we measure memory across sessions in a way frontier systems do not already max out? | `engram-bench` v0, baseline table |
| 1 · Memory module | Does a frozen base plus a test-time-trained memory store and recall user facts after the context is wiped? | Memory-module checkpoint, ablations |
| 2 · Synthetic users | Does training on long simulated timelines teach the model to use its memory well: updates, preferences, forgetting? | SFT/RL recipe, sleep recipe, results on held-out users |
| 3 · Private deployment | Does it work for real people on private hardware, with deletion that holds up? | Pilot report, deletion audit |

---

## Shared assumptions

These numbers recur in the estimates below. All of them are assumptions.

- **Hardware.** H100-80GB-class GPUs. Training sustains about **400 TFLOP/s per GPU** in bf16, roughly 40% of dense peak. Cloud price is about **$2–3 per GPU-hour**.
- **Training FLOPs.** The standard estimate is about 6 × parameters × tokens for full training. With a **frozen base**, there are no weight gradients for the base. We still need the forward pass plus a backward pass through its activations to reach the memory. So we use about **4 × N_base × tokens**, plus about 6 × N_memory × tokens for the memory itself, which is small.
- **Test-time-training overhead.** Inner-loop memory writes during training, and backpropagating through them, add a multiplier of **1.5–3×**. The literature does not pin this down for a frozen 8–32B base, so this is the least certain number here.
- **Base models.** Start at **8B**; scale to **32B** only after a gate passes. A specific base model is chosen in Phase 0 (see [open questions](open-questions.md)).

---

## Phase 0: Evals and baselines

**Goal.** Build `engram-bench` v0 (see [eval-design.md](eval-design.md)) and measure how far today's systems get on it.

**Work.**
1. Synthetic-user generator: personas, a world timeline, and sessions with planted facts, preferences, decisions, updates, forget requests and adversarial cross-user probes.
2. Scorers: exact match where possible, an LLM judge with a rubric otherwise, and a human spot-check on 5% of items to calibrate the judge.
3. Baselines:
   - B0 no memory: lower bound.
   - B1 full history in context: upper bound while it fits.
   - B2 frontier model plus a retrieval memory, Mem0/Letta-style.
   - B3 the same open base model Engram will use, plus the same retrieval memory.
   - B4 per-user LoRA fine-tuned on raw transcripts, the naive in-weights baseline.
4. Saturation check: if B2 already scores ≥ 90% on a category, that category is too easy. Make it harder or drop it.

**Compute (assumption).** Inference only. With 200 synthetic users, 30 sessions each, and about 4K tokens per session, the session data alone is about 24M tokens, plus questions. Running B0/B1/B3/B4 on an 8B model via vLLM takes roughly **50–150 GPU-hours** including B4's small LoRA runs. The frontier API baseline (B2) is budgeted at **$500–2,000** in API spend, depending on model and judge choice.

**Gate.** The benchmark separates B0 from B1 by a wide margin, and at least two categories leave B2 well short of B1. Otherwise the benchmark isn't measuring memory and we iterate on it.

---

## Phase 1: Memory module plus continued pretraining

**Hypothesis.** A neural memory attached to a frozen base can be trained, with the base frozen, so that writing a document chunk into memory and then wiping the context still lets the model answer questions about it, better than per-user LoRA on the same text (B4).

**Design (initial; see [landscape.md](landscape.md) for the reasoning).**
- Memory modules in a small number of layers of the frozen base. The readout is injected as a residual or attention correction, following the placement in Titan-Llama.
- Large-chunk writes, about one per turn or per 1–4K tokens, as in LaCT and In-Place TTT, to keep GPUs busy.
- A write objective tied to next-token prediction (In-Place TTT, TTT-NTP), with learned momentum and learned decay (Titans). Robust inner losses (Miras) are an ablation.
- Meta-train the memory's initialization and write rule so test-time writes work well (the TTT-E2E lesson). An off-the-shelf frozen model was never trained to use such writes.

**Training data.** Continued pretraining on **chunked documents**. The model writes chunks 1..k into memory, the context is dropped, then it must predict chunk k+1 and answer questions about chunks 1..k. Sources are long public-domain documents, long multi-document threads, and synthetic "biography" corpora. Synthetic corpora give clean recall probes without real personal data.

**Experiments.**
1. Write-then-recall after a context wipe, against B4 and against the retrieval log alone.
2. Ablations: memory depth (linear vs MLP), number and placement of memory layers, chunk size, write objective, decay schedule, surprise gating on or off.
3. Interference: write N unrelated users' documents in sequence (resetting between users) and check that the base model's general benchmarks do not move. They shouldn't, because the base is frozen and the module is detached.
4. A first **sleep** recipe: distill the fast memory into a LoRA adapter overnight. The fast memory's behaviour is the teacher. Training data is self-generated question-answer pairs and rehearsal of the user's earlier material. Compare against direct LoRA on transcripts.

**Compute (assumption).** For an 8B base and 10B training tokens: 4 × 8e9 × 1e10 ≈ 3.2e20 FLOPs. At 400 TFLOP/s that is about 220 GPU-hours. Applying the TTT overhead (1.5–3×) and about 2× for ablations and failed runs gives **~700–1,500 GPU-hours** (about $1.5K–4.5K). Scaling the winning configuration to a 32B base costs about 4× per token: **~3,000–6,000 GPU-hours**. We run it only if the 8B gate passes.

**Gate.** On `engram-bench` fact and preference recall after a context wipe, the 8B memory model beats B4, and comes within reach of B3 (retrieval) on at least one category. General-capability benchmarks of the base are unchanged when the memory is empty.

---

## Phase 2: SFT and RL on synthetic users over long timelines

**Hypothesis.** Memory that stores information is not the same as memory that is *used well*. Training on simulated months of a person's life teaches the model to update stale facts, apply preferences without being asked, honour forget requests, and refuse cross-user probes.

**Data.** About 10K synthetic users, each with a timeline of about 100 sessions. Each timeline includes life events, changing facts (moves, job changes, new relationships), preference drift, explicit decisions, and forget requests. A generator model writes sessions from a structured world-state. The state is the ground truth for scoring.

**Training.**
1. **SFT** on trajectories where the correct answer depends on memory: fast memory carried across sessions, a sleep cycle every N sessions, and the context wiped between sessions.
2. **RL** (GRPO-style), with rewards from the structured world-state: correct recall, correct use of updated facts, correctly saying "I don't know" or "you asked me to forget that", and zero leakage on cross-user probes. Rollouts span several sessions, which is what makes this phase expensive.
3. Tune the sleep recipe: frequency, replay mix, rehearsal-generation prompts, and the LoRA rank.

**Compute (assumption).**
- Data generation: 10K users × 100 sessions × 3K tokens ≈ 3B tokens. With a 30B-class open model on vLLM at about 2,500 tokens/s per GPU, that is roughly **300–400 GPU-hours**. A frontier API would cost more but probably give higher quality; we decide in Phase 0.
- SFT on an 8B base: a few billion tokens with TTT overhead ≈ **300–800 GPU-hours**.
- RL on an 8B base: multi-session rollouts are long. We budget **2,000–8,000 GPU-hours**, the widest uncertainty in the plan.
- Total at 8B: **~3,000–9,000 GPU-hours** (about $6K–27K). A 32B run would be about 4× that, only if justified.

**Gate.** On held-out synthetic users, Engram beats the frontier-plus-retrieval baseline (B2) on at least preference application and fact updating, without regressing on exact-fact recall (with the log enabled), forgetting, or leakage.

---

## Phase 3: Private deployment

**Goal.** Run Engram for a small number of consenting people on private hardware and learn what synthetic users missed.

**Design.**
- One encrypted module per user: fast-memory state, LoRA adapter, and retrieval log. The key is held by the user, and the module is loaded only for that user's sessions.
- Deletion means destroying the module and its key. An audit runs the leakage and forgetting probes against the shared base model and the remaining modules.
- Sleep runs nightly on the user's hardware or a private node.

**Compute (assumption).**
- Serving: an 8B base quantized to 4–8 bits fits on one 24–48GB GPU. Memory writes add the TTT overhead per turn (measured in Phase 1).
- Sleep per user per night: training a LoRA on about 200K tokens of rehearsal data at 8B is about 6 × 8e9 × 2e5 ≈ 1e16 FLOPs, a few minutes of GPU time. Generating the rehearsal data dominates: **~0.05–0.2 GPU-hours per user-night**.

**Gate.** Before any pilot: an external review of the threat model in [SECURITY.md](../SECURITY.md), and a deletion audit that passes on synthetic users. During the pilot, users can inspect, correct and delete what Engram remembers.

---

## Summary of compute assumptions (8B path)

| Phase | GPU-hours (assumed) | Approx. cost at $2–3/GPU-h |
|---|---|---|
| 0 | 50–150 + $0.5–2K API | ~$0.6–2.5K |
| 1 | 700–1,500 | ~$1.5–4.5K |
| 2 | 3,000–9,000 | ~$6–27K |
| 3 | ~0.05–0.2 per user-night | small |

A full 32B path costs roughly 4× more in Phases 1–2. These figures will be replaced with measurements as runs land; see the issue tracker.
