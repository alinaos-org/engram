# Fast memory on a frozen base: Phase 1 design note

*Draft, 2026-10-15. Answers [#6](https://github.com/alinaos-org/engram/issues/6). It builds on [landscape.md](landscape.md) §1–5 and §12, [open-questions.md](open-questions.md) (OQ below), [experiment-plan.md](experiment-plan.md) Phase 1 and [base-model-shortlist.md](base-model-shortlist.md).*

**How this was written.** The full arXiv HTML of Titan-Llama, In-Place TTT, TTT-NTP, Titans, LaCT and TTT-E2E was read, including appendices. δ-mem and LSTMem were skimmed (method, setup, main tables). Each code repository was opened through the GitHub API (metadata, README, file list). Numbers come only from those pages and are the authors' claims; we have reproduced none of them. Section and table numbers refer to the HTML versions listed under Sources. Anything else is marked **our assumption** or **unverified**. Byte sizes for Qwen3-8B use its `config.json`: hidden 4,096, MLP 12,288, 36 layers, 8 KV heads, head dim 128.

## Summary

- **Only one of the gradient-written retrofits keeps the base frozen.** Titan-Llama freezes its base. TTT-NTP trains its fast-weight projection jointly with the backbone (§3.2, App. A). In-Place TTT never says the backbone is frozen, and its no-TTT baseline gets the same continual pretraining (§4.1, App. 9.2). We read that as full-model training, which is **unverified** at code level. δ-mem and LSTMem also freeze the base, but a learned recurrence writes their memory, not gradient descent. Design A below has therefore never been tested on a frozen base.
- **None of these papers tests recall after the source text is gone.** TTT-NTP reuses the prompt's KV cache when it decodes. Titan-Llama keeps full attention in its later layers. When Titan-Llama made memory the only path to earlier segments (App. C.1), accuracy stayed far below full attention. That is Engram's regime after a context wipe.
- **Next-token-aligned targets beat reconstruction targets when trained the same way.** In TTT-NTP's ablation, a next-position target beat convolutional targets by 5 or more RULER points (§4.5.1). Titan-Llama's authors call their key-value reconstruction loss a surrogate (§3.2). TTT-E2E's next-token loss beat the TTT-KVB starting point it was derived from (Table 1).
- **Chunk sizes of 512–1,024 tokens were used or found best at the scales tested.** See In-Place TTT §4.3, TTT-E2E §3.2, TTT-NTP App. B (fixed at 1,024) and Titan-Llama §4.4. Titan-Llama found no single best size across tasks.
- **Three candidate designs:** (A) in-place MLP down-projection deltas (about 604 MB per user at full rank, or about 13 MB at rank 64), (B) a Titans-style side memory (about 1.3 MB per user), and (C) a sparse slot table (about 8 KiB per touched slot). There is no winner yet. Two cheap reproductions decide between A and B. C belongs to [#7](https://github.com/alinaos-org/engram/issues/7).
- **First reproduction:** TTT-NTP on Qwen3-0.6B-Base with the released code. The target is RULER Full-13 average 68.55 → 71.43. We add two arms that matter for Engram: a frozen backbone, and recall after the context is wiped. Estimated cost: about 40–70 L40S GPU-hours (**our assumption**).

---

## 1. What the papers do

### 1.1 Titan-Llama (Pan, Behrouz; arXiv 2610.19210v1, 13 Oct 2026)
- **Where the memory sits.** The first 4 layers of a frozen Llama-3.2-1B-Instruct are converted to blockwise (segmented) attention, with one neural memory module (NMM) per KV head. The later layers keep full attention (§4).
- **Read path.** The query reads the NMM, and the result is mixed into the local attention output through a per-channel gate. The gate is a sigmoid of a linear map of RMSNorm(q) (§3.2). The derivation splits exact attention into a historical part and a local part (§3.1, eq. 1).
- **Inner loss.** Squared error between the NMM's output on a chunk's attention keys and that chunk's values. The authors say plainly that this is a surrogate for useful memory (§3.2).
- **Inner optimiser.** The Titans rule: one gradient step per chunk with learned momentum, decay and step size. An "offline" variant batches the gradient over several chunks (§3.2).
- **Chunk size.** 512 for the main results; 512, 1,024 and 2,048 were tested. Larger chunks help QuALITY and LongHealth, give mixed results on RULER, and slow prefill. 512 is best on most multi-key and multi-value settings (§4.4, App. C.2, Table 5).
- **What is trained offline, and on what.** Only the added parts, with the backbone frozen: 17.23M parameters, about 1% of the model. Each NMM is a 3-layer MLP with hidden size 64 and GELU (App. A.2). Training ran on 1B tokens of FineWeb at up to 16K tokens, then supervised fine-tuning on LongAlign and LoongRL, then the QuALITY training set (App. A.1). The outer loss is next-token prediction. Distilling the base's historical attention into the NMM was tried and dropped as hard to scale and ineffective (§4.1).
- **Headline results.** Against segmented attention: QuALITY 23.87 → 40.51 and LongHealth 14.50 → 40.75 (abstract). Full attention scores 43.34 and 54.25 (Table 3). RULER falls at 32K, beyond the 16K training length (§4.2).
- **Memory-only regime (App. C.1).** With every layer restricted to local windows, the NMM is the only path to earlier segments. QuALITY goes 26.8 → 35.2 and LongHealth 15.8 → 32.3, both well below full attention, and RULER stays low (Table 4).
- **Reported overhead.** Prefill more than 20% faster (with 5 converted layers, long prompts) and decode 37% faster at medium context. Steady-state decode memory is 24% lower, but peak prefill memory is higher for partial hybrids (A100-80GB; §4.3, Fig. 4). Mean relative decode time is 0.689 (Table 1). No GPU-hours are reported.
- **Code.** [sarahpannn/titans-pytorch](https://github.com/sarahpannn/titans-pytorch), MIT, a fork of lucidrains' unofficial Titans code. Its README points to `titan_llama.py` and `scripts/pretrain.py`. Last push 2026-10-08.

### 1.2 In-Place TTT (Feng et al.; arXiv 2604.06169v1, ICLR 2026)
- **Where the memory sits.** The MLP down-projection W_down. W_up and W_gate stay as slow weights (§3.1). In continual pretraining it applies to every sixth layer (App. 9.3).
- **What is updated, and the loss.** For each chunk, W_down is first applied to the activations Z and then updated with η·V̂ᵀZ. This is a gradient step on a negative inner-product loss (eq. 1). The target V̂ is a causal Conv1D (kernel 5) over token embeddings followed by a projection W_target. It carries future-token information (§3.2). Both parts are needed: the Conv1D for long context and W_target for short context (§4.3).
- **Chunk size.** 512 and 1,024 are best, and 1,024 is faster (1.7B ablation, §4.3).
- **Optimiser and stability.** Plain step, no momentum, no decay. Writes accumulate as a prefix sum, so context parallelism works (App. 8). Fast weights reset at document boundaries (§3.4). At inference, each update's Frobenius norm is clipped to τ = 1e-5 for Qwen3-4B (App. 9.2). The value of η is not given in the text we read.
- **Offline training.** Continual pretraining. The text never says the backbone is frozen, and the no-TTT baseline gets the identical curriculum (§4.1), so we read it as full-model training (**unverified** at code level). Qwen3-4B-Base: about 20B tokens at 32K, then about 15B at 128K with YaRN, AdamW at 5e-6. Llama-3.1-8B and Qwen3-14B-Base: about 20B tokens at 32K (Tables 6–7). The in-house data mixes books, repository code and synthetic long QA (App. 9.1). Conv1D is zero-initialised and W_target sparse-diagonal, so the model starts out identical to the base (App. 9.3).
- **Results.** RULER for Qwen3-4B: 64K 74.3 → 78.7, 128K 74.8 → 77.0 (Table 1). At 64K: Llama-3.1-8B 81.6 → 83.7 and Qwen3-14B 67.9 → 70.6 (Table 2). From-scratch models at 500M, 1.5B and 4B are in §4.2.
- **Reported overhead.** "Negligible" prefill throughput and memory overhead, shown only as a plot (Fig. 4). No GPU-hours are given.
- **Code.** [ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT), Apache-2.0, built on VeOmni. Per the README it covers training, checkpoint conversion, TTT inference and RULER evaluation, with recommended configs for **Qwen3-8B** and Llama-3.1-8B.

### 1.3 TTT-NTP (Ouyang, Cai, Hu; arXiv 2606.21803v2, EMNLP 2026 Findings)
- **Where the memory sits.** W_down as well. Layers {0, 6, …, 30} on the 32-layer and 36-layer models; {0, 4, …, 24} on Qwen3-0.6B (Table 6).
- **Inner loss and target.** Each write is rank-one: v·zᵀ, where z is the gated MLP activation. The target v is the same layer's hidden state at the next position, h_{t+1}, passed through a learned d×d projection W_proj. There is no convolution (§3.2, eqs. 5–7). Writes are summed per chunk and applied as a causal exclusive prefix sum (§3.3).
- **Chunk size and inner η.** Chunks are 1,024 tokens. η is 0.3 for Llama-3.1-8B, 0.15 for Mistral-7B-v0.3, 3.0 for Qwen3-4B and 2.6 for Qwen3-0.6B. Chunk writes are divided by their spectral norm for the 4B and 8B models (App. B).
- **Inference write.** One closed-form ridge solve per adapted layer, over the last 8,192 prompt tokens, applied before decoding. Settings: λ = 1, η = 0.1, a per-layer cap of 0.1 on relative Frobenius norm, and rows renormalised to their pretrained norms. The prompt's KV cache is reused, and the original weight is restored after each sample (§3.4, App. B). Using the training-time inner-product write at inference instead collapses performance: Llama-3.1-8B averages 12.61 against a base of 55.80 (Table 3).
- **Offline training.** Continual pretraining on Long-Data-Collections (without books) at 32K tokens, AdamW at 5e-6. Budgets: 0.4B tokens for Llama-3.1-8B, 0.1B for Mistral-7B-v0.3, 2B for Qwen3-4B and 0.2B for Qwen3-0.6B (App. A). W_proj is trained jointly with the backbone (§3.2).
- **Results (RULER Full-13 average, Table 1).** Llama-3.1-8B 55.80 → 59.70; Mistral-7B-v0.3 70.60 → 73.63; Qwen3-4B 81.92 → 85.98; Qwen3-0.6B 68.55 → 71.43. Plain CPT gives 68.86 on Qwen3-0.6B. In-Place TTT with matched settings falls *below* the base on all four models. LongBench-v2 rises +5.6 for Llama-3.1-8B (Table 2). General ability moves at most about 1 point (Table 4).
- **Reported overhead.** None reported.
- **Code.** [yancyou/TTT-NTP](https://github.com/yancyou/TTT-NTP), Apache-2.0, VeOmni-based. It has one-command reproduction scripts for all four backbones, including `scripts/repro_qwen3_0p6b_ttt.sh`, which runs 100 steps and then RULER at 4K–32K.

### 1.4 Titans (Behrouz, Zhong, Mirrokni; arXiv 2501.00663v1)
- **Memory.** A deep MLP. Memory as context (MAC), memory as a gate (MAG) and memory as a layer (MAL) are its three wirings, and it also uses learnable persistent-memory tokens (§3–4).
- **Update rule (eqs. 13–14).** The memory decays by a factor (1−α_t) and adds a momentum term S_t. S_t carries forward η_t of the previous momentum and subtracts θ_t times the gradient of the loss ‖M(k_t) − v_t‖², where k and v are linear projections of the input. α, η and θ depend on the data (computed per token in their experiments). Training runs in chunks using matrix multiplications and an associative scan (§3.2). The v1 text gives no chunk size.
- **Offline training.** Everything trained jointly from scratch. Models of 170M, 340M and 400M parameters on 15B FineWeb-Edu tokens, and 760M on 30B, at 4K length with AdamW at 4e-4 (§5.1).
- **Ablation (Table 5).** Removing weight decay hurts most (perplexity 27.01 → 29.04), then momentum (28.98), then using a linear memory (28.49).
- **Overhead.** The memory module trains slightly slower than Mamba2 and Gated DeltaNet (§5.8).
- **Code.** v1 says the authors intend to release it. The arXiv abstract page links no code. lucidrains/titans-pytorch describes itself as unofficial.

### 1.5 LaCT (Zhang et al.; arXiv 2505.23884v1)
- **Architecture.** A from-scratch hybrid. Each block has sliding-window attention and a fast-weight SwiGLU MLP that share the same Q, K and V, and their outputs are summed (§4.2, App. C.2).
- **Inner loss.** Negative dot product between f_W(k) and v, with per-token learning rates predicted from the input and initialised to 0.001 (§3.1, App. C.2).
- **Update.** Gradient descent, momentum, or Muon, each followed by L2 weight normalisation. There is no weight decay (§3.2). Muon did best (Fig. 7b).
- **Chunk size.** Equal to the window: 2,048 at 760M and 4,096 at 3B. Fast-weight state can reach 40% of model parameters (abstract, Fig. 7). A larger, nonlinear state helps. A linear large-chunk variant loses to GLA on language (Fig. 8).
- **Compute.** The 760M model took 40B tokens on 32 A100-40GB GPUs for about 20 hours. The 3B model took 60B tokens on 64 A100s for 50–60 hours. Training throughput at 3B and 32K: 4.1K tokens per second for full attention, 6.4K for SWA, and 4.3K–5.0K for LaCT (Table 7). GPU utilisation reaches up to 70% on A100s. Context-parallel overhead is 1–3%.
- **Code.** [a1600012888/LaCT](https://github.com/a1600012888/LaCT), MIT, with `lact_llm/` and minimal implementations.

### 1.6 TTT-E2E (Tandon et al.; arXiv 2512.23675v2)
- **Architecture.** A Transformer with sliding-window attention (window 8K). At test time only the MLPs of the last quarter of the blocks are updated, and those blocks gain a second, static MLP as "safe" storage. Embeddings, norms and attention stay frozen during TTT (§2.3.1).
- **Inner loss.** Ordinary next-token cross-entropy at the output, back-propagated only through the last L/4 blocks. One step per 1K-token mini-batch. Smaller mini-batches score better but fall below 1K become unstable (§3.2). The inner learning rate is not stated in what we read.
- **Meta-training.** The initial weights are learned by differentiating through the inner updates (gradients of gradients, §2.2). Pretraining is on DCLM at 8K, then extension on Books. Models span 125M to 3B (the 3B with 164B tokens). Updating 6 or 12 of 24 layers lets loss scale with context; updating 1 or 3 does not (§3.2.1).
- **Results.** Loss scales with context like full attention. Needle-in-a-haystack recall does not: at 128K, S-NIAH-1 is 0.06 versus 0.99 for full attention (Table 2).
- **Overhead.** Prefill is 2.7× faster than full attention at 128K on an H100. Training is 3.4× slower at 8K and 1.2× faster at 128K (§3.7).
- **Code.** [test-time-training/e2e](https://github.com/test-time-training/e2e), in JAX, with checkpoints at 125M, 1B and 3B in a requester-pays Google Cloud bucket. The GitHub API shows no licence.

### 1.7 Skimmed: δ-mem (2605.12357v2) and LSTMem (2609.33268v1)
- **δ-mem.** On a frozen base, each layer gets an r×r gated delta-rule state with r = 8. Reads from it make low-rank corrections to the attention query and output (§3). It is trained with SFT on 2,219 QASPER samples, 4.87M trainable parameters. Qwen3-8B's average rises from 47.20 to 50.86 (Table 2). Middle-layer insertion beats front or back layers, but all layers is best (Table 4). Code: [declare-lab/delta-Mem](https://github.com/declare-lab/delta-Mem), no licence shown.
- **LSTMem.** On a frozen Qwen3-4B-Instruct, each of 36 layers holds two 4×8×8 states (a cell state C and an expressed state D), written by a learned gated recurrence. At the end of each block, a reconstruction gradient also corrects the cells, at inference time as well (§3.3). The state is **36 KiB per sequence**. Training takes 96 updates. Decode speed is 92.10% of δ-mem's (§4). The ethics section asks for user-controlled reset. Code: [Longchentong/LSTMem](https://github.com/Longchentong/LSTMem), no licence shown.

### 1.8 Corrections for other notes (not edited here)
- landscape §1 calls In-Place TTT and TTT-NTP "drop-in" for released models. Both still continually pretrain on 0.1–35B tokens: TTT-NTP trains its backbone too, and In-Place TTT most likely does (§1.2–1.3 above).
- In-Place TTT's extension (§4.1, Table 7) uses **LLaMA-3.1-8B**; "Llama3-8B" appears only as a reference row in Table 1. (Corrected in base-model-shortlist.md.)

---

## 2. Candidate Phase 1 designs on a frozen Qwen3-8B

Common to all three (**our assumptions**, following experiment-plan Phase 1 and OQ11):
- The base stays frozen. Shared parameters are meta-trained only on public or synthetic chunked documents ([#10](https://github.com/alinaos-org/engram/issues/10)).
- The training task is write-then-wipe: write chunks 1..k, drop the context, then predict chunk k+1 and answer probes about 1..k.
- Each user's state is one tensor file, encrypted with a per-user data key under the user's key. It is loaded into the GPU only for that user's session and zeroed afterwards.
- Deleting a user means destroying the key and the file (crypto-shredding). Deleting one item means rebuilding the state from the retrieval log without that item, unless the design supports deleting a single slot.

### A. In-place MLP deltas (TTT-NTP-style)
- **Placement.** W_down in 6 layers, {0, 6, 12, 18, 24, 30}, the same set TTT-NTP uses on 36-layer Qwen3-4B.
- **Per-user state.** ΔW is 4,096 × 12,288 = 50.3M values per layer. Over 6 layers that is 302M values: **604 MB in bf16** (1.21 GB in fp32). A rank-64 factorisation is 1.05M values per layer, **12.6 MB** in total. Whether rank 64 keeps the gain is **unverified**.
- **Write rule.** At session end, the TTT-NTP closed-form ridge solve over the session's last ≤8,192 tokens, with targets W_proj·h_{t+1}. It is merged into the accumulated delta with a learned or tuned decay and TTT-NTP's norm caps. The decay, and accumulating across sessions at all, are **our additions**. TTT-NTP writes per sample and restores the weight afterwards.
- **Write trigger.** Session end, or each completed turn once 1,024 or more tokens have built up. Surprise gating is left to [#11](https://github.com/alinaos-org/engram/issues/11).
- **Read path.** No extra operations: the patched W_down applies to every token. With an empty delta, the base is bit-identical, which meets the experiment-plan gate.
- **Meta-training.** Only W_proj: 4,096² × 6 = 101M shared parameters, trained with the chunk-parallel prefix-sum write (no gradients of gradients needed). It has never been trained with a frozen backbone (§1.3).
- **Expected overhead (our assumption).** A dual-form ridge solve with T = 8,192 per layer at write time, which TTT-NTP does not time. Loading 604 MB per session, or 12.6 MB at rank 64. For batched serving, the delta should be applied as a per-request additive adapter (multi-LoRA style at rank r), never by mutating the shared W_down in place: a failed restore would leak one user's memory to the next.

### B. Titans-style side memory (Titan-Llama-style, full attention kept)
- **Placement.** One NMM per KV head in 4 layers. Titan-Llama used the first four. δ-mem's partial-insertion ablation favours middle layers. Placement is ablation OQ3. Unlike Titan-Llama, **attention stays full** within a session, so the NMM carries only what came before the wipe.
- **Per-user state.** A 3-layer MLP of 128 → 64 → 64 → 128 per NMM is 20,480 weights. That input and output width equal the head dim is **our assumption**. 8 KV heads × 4 layers gives 32 NMMs and 655K values: **1.31 MB in bf16**, or 2.6 MB with the momentum state saved. With hidden 256 it is 4.19M values, **8.4 MB**. LSTMem's 36 KiB state shows that much smaller states are possible.
- **Write rule.** The Titans update, once per chunk, with learned θ_c, η_c and α_c (§1.1). Inner loss: Titan-Llama's key-value reconstruction on the layer's own K and V as the reproducible default, with a TTT-NTP-style next-position target as the main ablation (OQ5).
- **Write trigger.** Each 512–2,048-token chunk or completed turn. The size is tuned per task, since Titan-Llama found no single best.
- **Read path.** The NMM's output is mixed into the attention output through Titan-Llama's gate. The gate's bias is initialised so it starts near 0, so an empty memory leaves the base unchanged (**our assumption**). The read adds a small MLP per token in 4 layers.
- **Meta-training.** The NMM initialisation, the gate, and the θ/η/α projections, trained with next-token loss through the inner updates while the base stays frozen. This is Titan-Llama's recipe applied to write-then-wipe data. TTT-E2E suggests the initialisation must be meta-learned (§1.6).
- **Expected overhead (our assumption).** A few percent at decode. The only nearby measurement is LSTMem at 5.9–6.5% end-to-end over δ-mem. Training costs within experiment-plan's 1.5–3× multiplier. The state is per sequence, like an RNN, so batching across users is natural.

### C. Sparse slot table (Fast-weight PKM plus sparse memory fine-tuning)
- **Placement.** One or two product-key memory layers added in parallel to the MLP at middle layers. The shared keys and value table are meta-trained into the frozen base first, which is the retrofit [#7](https://github.com/alinaos-org/engram/issues/7) must cost. Per landscape §4, read only at abstract level, this kind of retrofit is shown only at 0.5B.
- **Per-user state.** A sparse map from slot index to a value delta: 4,096 × 2 bytes per touched slot, so **8 KiB per slot**. 1K touched slots is about 8 MB and 10K about 82 MB. The shared table, assuming 512² = 262K slots × 4,096 dims (**our assumption**), is 2.1 GB, shared and never per-user.
- **Write rule.** A gradient step on the chunk loss, restricted to the top-t slots that the chunk activates far more than public data does. This is the selection rule from FwPKM and sparse memory fine-tuning (landscape §4, abstract level only, **unverified** in detail).
- **Write trigger.** Per turn, with t set by surprise ([#11](https://github.com/alinaos-org/engram/issues/11)).
- **Read path.** A top-k lookup per token (k = 32, **our assumption**), adding the user's delta to the shared value.
- **Deletion.** The one design with per-item deletion: each slot is tagged with the log message IDs that wrote it, and a "forget" zeroes or rebuilds those slots. Slots shared between facts must be rebuilt from the log, not zeroed (**our assumption**). See OQ12.
- **Expected overhead.** Unmeasured. The retrofit's training cost is the main unknown.

| | A in-place deltas | B side NMM | C slot table |
|---|---|---|---|
| Per-user state | 604 MB (full) / 12.6 MB (rank 64) | 1.3–8.4 MB | ~8 KiB × touched slots |
| Evidence on a frozen base | none (both papers train the backbone) | 1B (Titan-Llama) | none at 8B; 0.5B per landscape |
| Inference write | closed-form, no backprop | one gradient step per chunk | sparse gradient step |
| Per-item deletion | rebuild from log | rebuild from log | slot-level |

**Provisional ranking: none yet.** A has the best evidence at 8B for using the context (§1.3), but it has never been run on a frozen base or tested after a wipe. B has the frozen-base precedent and the smallest state, but App. C.1 warns that memory-only recall is weak. C belongs to [#7](https://github.com/alinaos-org/engram/issues/7).

---

## 3. First reproduction: TTT-NTP on Qwen3-0.6B-Base

**Why this one.** It is the cheapest published retrofit result that has released, one-command reproduction code. The model is the paper's own backbone and a sibling of Qwen3-8B (Apache-2.0 according to its Hugging Face tags, 596M parameters, same tokenizer). The same repository holds the 8B-class configs. And it lets us add the two arms Engram actually needs at almost no extra cost. One caveat: the paper calls its 0.6B run a smaller-scale consistency check (App. A). **Titan-Llama is second**: its code is days old, its base (Llama) is gated, and it needs about 5× the tokens. We run it next if Design B stays in contention, and reproduce its App. C.1 memory-only configuration.

**Arms.** Each is trained for 100 steps × 64 sequences × 32,768 tokens = 0.2B tokens, with the release config: layers {0, 4, …, 24}, η = 2.6, chunk 1,024, AdamW at 5e-6.
1. **Base:** RULER Full-13 at 4K/8K/16K/32K with no training. Paper: average 68.55.
2. **CPT:** `ttt_mode` off. Paper: 68.86.
3. **TTT-NTP:** the release recipe with the closed-form inference write. Paper: **71.43**, a gain of 2.88. General ability (Table 4): average 54.12 → 54.59.
4. **Engram arm, frozen backbone:** the same recipe, but only `ttt_proj` is trained (7 × 1,024² ≈ 7.3M parameters by our arithmetic). The released config shows no freeze option, so this needs a small code change (**unverified** effort).
5. **Engram arm, wipe probe:** write 100 synthetic biographies (from the #10 or engram-bench generator) with the closed-form write. Then ask questions from a fresh KV cache, without the document, for arms 3 and 4. No paper reports this number.

**Success and failure criteria.** Pre-register these in [#15](https://github.com/alinaos-org/engram/issues/15). The thresholds are our choice.
- *Harness check:* our Base average lands within ±1.5 points of 68.55. Otherwise, fix the evaluation before training anything.
- *Reproduction success:* TTT-NTP minus Base is at least +1.5 points and above CPT, with the general-ability average within 1 point of Base.
- *Failure:* a gain below +1.0, or TTT-NTP at or below CPT. Then check η, the closed-form settings and the data mix, and report the failure, which #6's "Done when" allows.
- *Design A signal (recorded, not gated):* the frozen arm keeps at least half of arm 3's gain, and the wipe probe scores above a no-memory control.

**Compute (assumptions, to be replaced by [#12](https://github.com/alinaos-org/engram/issues/12)).**
- *Hardware:* one L40S 48 GB (g6e.xlarge), assumed to sustain about 100 TFLOP/s in bf16.
- *FLOPs per training arm:* 6 × 0.6e9 × 0.2e9 ≈ 7e17 for the weights, plus causal attention at 32K. That attention is about 3.8e9 FLOPs per token forward over 28 layers, adding roughly 2.3e18 per arm with the backward pass. With recompute, the total is about 3–4e18 per arm: **about 8–12 hours each**. Arms 2–4 take about 25–35 hours.
- *Evaluation:* RULER for 4–5 checkpoints plus the probe, about 10–30 hours.
- *Total:* **about 40–70 GPU-hours**, inside Phase 0's 50–200 for #6 and #7 together.
- *Memory:* fits on one GPU only if the fused cross-entropy path (the repo installs liger-kernel) avoids materialising the 32,768 × 151,936 logits, which would be about 20 GB in fp32. Running VeOmni's FSDP2 on a single GPU with 64-step gradient accumulation is **unverified**.

---

## 4. Risks and open questions

- **OQ1 (frozen base).** The gradient-written gains at 4–14B all come with continual pretraining of the backbone (§1.2–1.3). Arm 4 is the cheapest test. If it fails, Design A needs Design B's meta-training, or an unfrozen base, which breaks Engram's premise.
- **OQ1 and OQ3 (recall after the wipe).** No paper measures recall with the source text removed. When Titan-Llama's memory was the only route to earlier segments (App. C.1), it fell well short of full attention. TTT-E2E is far below full attention on needle tests (Table 2). Expect the fast memory to store gist and preferences better than verbatim facts, which the retrieval log covers (landscape §12.5).
- **OQ5 (write objective).** Next-position targets beat convolutional ones (TTT-NTP §4.5.1). Reconstruction is a surrogate (Titan-Llama). The full next-token loss beat key-value binding (TTT-E2E Table 1). None of this has been compared on a frozen base.
- **OQ6 and #11 (write trigger).** 512–1,024-token chunks are the tested range. δ-mem's segment-level writes did best on Qwen3-8B (Table 2). Surprise gating is untested in all of these papers.
- **OQ7 (stability across sessions).** Every paper resets per document or sample. Stability tricks such as update clipping (τ = 1e-5), spectral normalisation, norm caps and L2 weight normalisation were tuned for one context. Accumulating writes for months is untested, and a write rule that is wrong at inference can collapse the model (TTT-NTP Table 3: 12.61 against 55.80).
- **OQ2 and OQ12 (deletion).** A and B can only delete by rebuilding from the log. C deletes per slot, but only after a retrofit nobody has shown at 8B.
- **OQ11 (shared parts).** W_proj, the NMM initialisation and gates, and the slot keys and table are shared across users. They must be trained only on public or synthetic data, so none of the papers' trained weights can carry user data.
- **OQ13 (stolen module).** Every state is a direct function of the user's text. A decrypted state plus the public base may invite inversion attacks (**our assumption**, by analogy with LoRA-Leak). Key custody is the main control.
- **Reproduction risks.** Single-GPU VeOmni; logits memory; differences in the RULER harness version; Titan-Llama's very new code; and no repository other than In-Place TTT and TTT-NTP measures overhead the same way.
- **Unknowns not settled here.** Titans' chunk size (absent from v1); the inner learning rates of TTT-E2E and In-Place TTT (not found in the text); and whether rank-limited deltas keep Design A's gain.

## Sources fetched (2026-10-15)

- arXiv HTML (full text): [2610.19210v1](https://arxiv.org/html/2610.19210), [2604.06169v1](https://arxiv.org/html/2604.06169), [2606.21803v2](https://arxiv.org/html/2606.21803), [2501.00663v1](https://arxiv.org/html/2501.00663), [2505.23884v1](https://arxiv.org/html/2505.23884), [2512.23675v2](https://arxiv.org/html/2512.23675). Skimmed: [2605.12357v2](https://arxiv.org/html/2605.12357), [2609.33268v1](https://arxiv.org/html/2609.33268).
- arXiv abstract pages for all eight, checked for code links. None on Titans' or LaCT's abstract page.
- LaCT project page: tianyuanzhang.com/projects/ttt-done-right, which links the LaCT repository.
- GitHub (API metadata, README, file list): sarahpannn/titans-pytorch, ByteDance-Seed/In-Place-TTT, yancyou/TTT-NTP (also `configs/pretrain/qwen3_0p6b.yaml` and `scripts/repro_qwen3_0p6b_ttt.sh`), test-time-training/e2e, a1600012888/LaCT, declare-lab/delta-Mem, Longchentong/LSTMem.
- Hugging Face `config.json` for Qwen/Qwen3-8B, Qwen/Qwen3-0.6B-Base and Qwen/Qwen3-1.7B-Base, and the model API entry for Qwen3-0.6B-Base.
