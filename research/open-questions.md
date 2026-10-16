# Open questions

*Draft, 2026-10-15.* Each question is ordered by how much its answer could change Engram's design, with the evidence we have so far (see [landscape.md](landscape.md)) and how we plan to answer it. Questions marked **Phase 0** or **Phase 1** are tracked as GitHub issues.

## Architecture

### 1. Can a test-time-trained memory attached to a *frozen* 8–32B base store user facts usefully? (Phase 1)
**Why it matters:** it is the premise of the whole project.
**Evidence:** Titans-style memory has been retrofitted onto a frozen model only at 1B (Titan-Llama). Gradient-written fast weights help released 4–8B models modestly within a single context (In-Place TTT, TTT-NTP). TTT-E2E's strong results depend on a meta-learned initialisation, which a frozen base lacks.
**How to answer:** Phase 1 write-then-wipe recall experiments at 8B, against per-user LoRA (B4) and retrieval (B3).

### 2. Dense LoRA or sparse slots as the per-user container? (Phase 1)
**Why it matters:** it decides how well consolidation resists forgetting and whether single facts can be deleted exactly.
**Evidence:** sparse memory fine-tuning forgets far less than LoRA at equal learning (Lin et al. 2025), and *User as Engram* reports tiny, stackable per-user slot edits. No 8–32B open base has memory layers, and retrofitting is shown only at 0.5B.
**How to answer:** retrofit one memory layer into the 8B base, then repeat the SMF-versus-LoRA forgetting curve on synthetic-user facts.

### 3. Where should the memory sit in the network, and how should it be read?
Options: a few early layers with an attention correction (Titan-Llama), the MLP output projections (In-Place TTT), a separate memory-as-context stream (Titans MAC), or a residual adapter. We expect to answer this with ablations rather than theory.

### 4. Do we need a hybrid (recurrent) base?
Hybrid open models (Qwen3.5, Nemotron, Kimi Linear) already contain delta-rule memory within each context. Does that make the Titans-style module redundant within a session and more valuable across sessions? Or does it make LoRA and memory retrofitting harder?

### 5. What should the write objective be?
Candidates: key-value reconstruction (Titans), next-token-aligned targets (In-Place TTT, TTT-NTP), sliding-window objectives (ATLAS), and robust losses (Miras). Titan-Llama's authors themselves call reconstruction a surrogate.

## Learning dynamics

### 6. What triggers a write? (Phase 1)
Gradient-magnitude surprise, novelty weighting (aTTT), model-selected spans (S-TTT), or simply every turn? Writing irrelevant spans hurts (S-TTT), and hand-designed surprise-based *forgetting* underperforms learned decay (What Should Titans Forget?). Is surprise any better for *writing*?

### 7. Is the fast memory stable across many sessions without resets?
Repetitive self-training drifts (aTTT). How often should fast memory be reset to its last consolidated state, and does that lose information that sleep failed to capture?

### 8. What is the right sleep recipe? (Phase 1)
Synthesise-then-distil (Cartridges, Knowledge Modules, SDFT) is the best-supported recipe. Open details:
- What data does sleep generate? Q&A, restatements, conversations, or the fast memory's own outputs as a teacher?
- Does the teacher use the retrieval log in context, the fast memory, or both?
- How much replay of older material is needed to prevent forgetting inside the adapter?

### 9. Does a per-user adapter survive hundreds of sleep cycles?
No paper we found tracks both recall and general ability in one adapter over hundreds of consolidations. Sequential dense edits collapse (Gupta et al.), and self-distillation loops can amplify their own artifacts (Denser ≠ Better). Candidate mitigations: periodic rebuild from the log, KL bounds, normalised updates (StableEdit), and null-space projection (AlphaEdit).

### 10. Can memory make the model *worse*?
Always-on parametric memory may over-apply stored context or increase sycophancy (PersistBench). LoRA can lose the ability to say "you never told me" (Substrate Asymmetry). How do we detect and prevent this?

## Privacy and deletion

### 11. Can shared components be trained without any user data? (Phase 0)
Exact deletion holds only if nothing shared ever trains on users. That rules out meta-training the write rule on real users, shared adapter generators trained on user data, and cross-user merging. Is synthetic and public data enough to meta-train the memory well?

### 12. How do we forget one item inside a consolidated adapter?
Options: rebuild the adapter from the log minus the item (exact but costly), slot deletion (exact if facts live in slots), or targeted unlearning (approximate, and reversible per *Unlearning Isn't Deletion*). What latency is acceptable between "forget this" and the item being gone from weights?

### 13. What is the threat model for a stolen module?
An encrypted module is safe at rest. A decrypted adapter plus the public base allows membership inference (LoRA-Leak). Do we need differential privacy in consolidation, or is key custody enough?

## Evaluation and claims

### 14. What exactly does "better than a frontier model with retrieval" mean? (Phase 0)
On which `engram-bench` categories, at what history length, and at what cost? We expect Engram to win, if at all, on preference application, implicit constraints and updates, and not on verbatim recall, where the log does the work anyway. The claim should be stated before the results are in.

### 15. Is the benchmark measuring memory or the generator's style?
Synthetic sessions written by a model may be easy for a related model to "recall". Use a different generator family, plus human-written sessions in a calibration subset.

### 16. Which base model?
Licence (open weights with a permissive licence), size within 8–32B, dense or hybrid, quality of the instruct variant, and tooling support for LoRA and TTT. To be decided in Phase 0.
