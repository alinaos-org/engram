# Base-model shortlist for Engram

*Draft, 2026-10-15. Decision support for [#5](https://github.com/alinaos-org/engram/issues/5) and [open question 16](open-questions.md#16-which-base-model). The recommendation at the end is provisional until the B3/B4 baselines run. Memory figures are estimates until [#12](https://github.com/alinaos-org/engram/issues/12) measures them.*

## Method

- **Facts** come from pages fetched on 2026-10-15: each model's Hugging Face card (`README.md`), its `config.json`, the Hugging Face model API (repo creation date, exact parameter count from the safetensors index, licence tag), and the licence pages the cards link to. Where a card is silent, the note says where the value came from, or marks it **unverified**.
- **vLLM support** means the model's architecture class is registered in vLLM's model registry, and the class declares `SupportsLoRA` (vLLM `main` at commit `8f93c64866`, 2026-10-16 UTC). This shows the code path exists. It does not show that it is fast or bug-free.
- **LoRA usage** is the number of Hugging Face repos tagged as adapters of that exact repo (`base_model:adapter:<id>`). The query returns at most 1,000, so "≥1,000" means the cap was hit. This is a popularity signal, not a quality signal.
- **Release date** is the date printed on the card where there is one. Otherwise it is the Hugging Face repo creation date, labelled "HF repo".

## Shortlist at a glance

| # | Repo id | Params (safetensors) | Architecture | Native context | Licence | Released | Base variant on HF |
|---|---|---|---|---|---|---|---|
| 1 | `meta-llama/Llama-3.1-8B-Instruct` | 8.03B | Dense transformer, GQA (32 layers, per the Llama 3 paper) | 128K | Llama 3.1 Community | 2024-07-23 (card) | `meta-llama/Llama-3.1-8B` |
| 2 | `Qwen/Qwen3-8B` | 8.19B | Dense transformer, GQA 32Q/8KV, 36 layers | 32,768 (131,072 with YaRN) | Apache-2.0 | 2025-04-27 (HF repo) | `Qwen/Qwen3-8B-Base` |
| 3 | `Qwen/Qwen3-32B` | 32.76B | Dense transformer, GQA 64Q/8KV, 64 layers | 32,768 (131,072 with YaRN) | Apache-2.0 | 2025-04-27 (HF repo) | none found (`Qwen/Qwen3-32B-Base` not found) |
| 4 | `mistralai/Mistral-7B-Instruct-v0.3` | 7.25B | Dense transformer, 32 layers, 8 KV heads (config) | 32,768 (config; the card does not say) | Apache-2.0 | 2024-05-22 (HF repo) | `mistralai/Mistral-7B-v0.3` |
| 5 | `nvidia/Nemotron-H-8B-Base-8K` / `-Reasoning-128K` | 8.10B | Hybrid Mamba-2 + MLP with 4 attention layers, 52 layers | 8K (Base) / 128K (Reasoning) | NVIDIA Internal Scientific R&D (**research only**) | 2025-04-14 / 2025-06-06 (cards) | Base is the 8K model; no instruct, only a reasoning model |
| 6 | `Qwen/Qwen3.5-9B` | 9.65B incl. vision encoder | Hybrid: 8 × (3 Gated DeltaNet + 1 gated attention), dense FFN, 32 layers; vision encoder | 262,144 (extensible to ~1.01M) | Apache-2.0 | 2026-02-27 (HF repo) | `Qwen/Qwen3.5-9B-Base` |
| 7 | `google/gemma-4-12B-it` | 11.96B | Dense; 40 sliding-window (1,024) + 8 global attention layers, 48 layers; encoder-free text, image and audio input | 256K | Apache-2.0 (licence link resolves to Apache 2.0) | 2026-05-23 (HF repo) | `google/gemma-4-12B` |

## Candidate notes

**1. Llama-3.1-8B-Instruct.** Download is gated behind manual approval on Hugging Face. The licence allows research and commercial use and creating derivatives. If you distribute the model, or a product or model that contains it, you must include the agreement and show "Built with Llama". A model trained from Llama outputs and then distributed must have a name that starts with "Llama". Use must follow Meta's Acceptable Use Policy. Licensees above 700M monthly active users need a separate licence. Private deployment with per-user adapters is allowed. Distributing adapters would trigger the attribution and naming terms. *Tooling:* registered in vLLM with LoRA support, ≥1,000 adapter repos. *Literature:* TTT-NTP (abstract) uses Llama-3.1-8B; In-Place TTT's paper body reports an extension with LLaMA-3.1-8B (§4.1, Table 7). Titan-Llama uses the same family at 1B (Llama-3.2-1B-Instruct).

**2. Qwen3-8B.** Ungated and Apache-2.0. One checkpoint switches between thinking and non-thinking modes. Thinking is on by default, so benchmark runs must set `enable_thinking=False`. The card warns against greedy decoding in thinking mode. The card names `vllm>=0.8.5` and `sglang>=0.4.6.post1` for serving. Registered in vLLM with LoRA support, ≥1,000 adapter repos. *Literature:* δ-mem attaches to Qwen3-8B (arXiv HTML). Several papers use smaller Qwen3 models: TTT-NTP (Qwen3-4B and 0.6B), LSTMem (Qwen3-4B-Instruct), In-Place TTT (Qwen3-4B-Base main, Qwen3-14B-Base extension), and Prometheus Mind (Qwen3-4B, per landscape.md).

**3. Qwen3-32B.** The same architecture as Qwen3-8B, scaled up. The card describes the same thinking switch, the same vLLM and SGLang versions, and the same context. There is no public base checkpoint, so any experiment that needs a base model at 32B cannot use this family. 489 adapter repos. *Literature:* none at 32B.

**4. Mistral-7B-Instruct-v0.3.** Ungated and Apache-2.0. Compared with v0.2, it extends the vocabulary to 32,768, uses the v3 tokenizer and supports function calling. The card has no moderation mechanism and states no release date or context length. The card recommends `mistral-inference` and also gives `transformers` snippets. Registered in vLLM with LoRA support, 896 adapter repos. *Literature:* TTT-NTP (Mistral-7B-v0.3, abstract).

**5. Nemotron-H-8B.** Of its 52 layers, 24 are Mamba-2 layers, 24 are MLP layers, and 4 are attention layers (from `hybrid_override_pattern`). Both cards say the model is "for research and development only". The governing licence limits use to internal scientific research in a non-production environment. It bans production use and use to generate works for sale, and bars distributing, deploying or sublicensing the model or its derivatives. **This rules it out as Engram's deployed base.** The cards name NeMo as the runtime and NeMo for PEFT/LoRA. The Base card does not mention vLLM, while the Reasoning card mentions it once. `NemotronHForCausalLM` is registered in vLLM with LoRA support. There are no adapter repos for either 8B repo. *Literature:* none in landscape.md §5.

**6. Qwen3.5-9B (newest Qwen in the 8–10B class).** Ungated and Apache-2.0. It is a vision-language model, and the parameter count includes the vision encoder. vLLM can skip the encoder with `--language-model-only`. The Gated DeltaNet recurrent state is held in fp32 (`mamba_ssm_dtype`). The card says to use SGLang or vLLM built from `main` (that note may be dated) and that thinking is on by default. It advises keeping at least 128K context to preserve thinking quality. Registered in vLLM (`Qwen3_5ForConditionalGeneration`, through a class that declares `SupportsLoRA`), 816 adapter repos. *Literature:* none at this version, but it is the closest in-range model to the Qwen3.5 hybrids discussed in landscape.md §6. The card's general highlights mention "sparse Mixture-of-Experts", but the 9B config has no experts.

**7. Gemma-4-12B-it.** Ungated. The card labels the licence Apache 2.0, and its link resolves to the Apache 2.0 text. The Gemma site also lists a "Prohibited use policy" and "Terms of use". **Whether these bind Gemma 4 is unverified** (the fetched page did not include their text). "Unified" means it has no separate vision or audio encoder: image patches and audio waveforms are projected straight into the LLM. Global layers use 1 KV head with head dim 512 and shared K/V (`attention_k_eq_v`). Registered in vLLM (`Gemma4UnifiedForConditionalGeneration` subclasses a class with `SupportsLoRA`). The card shows no vLLM snippet. 122 adapter repos. *Literature:* none.

### Also checked, not shortlisted

| Repo id | Facts read | Why not on the shortlist |
|---|---|---|
| `Qwen/Qwen3.8-27B` | 27.78B, 64 layers, same 3:1 Gated DeltaNet layout as Qwen3.5, vision encoder; 262,144 context; Apache-2.0; HF repo 2026-08-05; 198 adapter repos; Qwen calls it "dense" (no MoE) | Newest Qwen in range. No base checkpoint found. Hybrid plus multimodal. The best 27–32B upgrade if the hybrid track wins |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 8.89B, 56 layers, Mamba-2 hybrid with 4 attention layers; 128K; NVIDIA Open Model License (commercial use allowed; rights end automatically if guardrails are bypassed without a similar substitute); card release 2025-08-18; vLLM snippet on card; 0 adapter repos | A deployable Mamba-2 alternative to Nemotron-H-8B. The guardrail clause needs a legal read before Engram relies on it |
| `ibm-granite/granite-4.2-8b` | 8.79B dense, 40 layers; 128K native; Apache-2.0; card release 2026-08-25; base model `granite-4.1-8b-base`; 7 adapter repos | Recent and permissively licensed, but it has almost no LoRA ecosystem and no retrofit literature |
| `allenai/Olmo-3-7B-Instruct` | 7.30B; 32 layers (24 sliding + 8 full); 65,536 context; Apache-2.0 "intended for research and educational use" under Ai2 guidelines; HF repo 2025-11-19; 49 adapter repos | Fully open data and checkpoints, which helps attribution studies. Weaker tooling signal. A good secondary for interpretability work |
| `google/gemma-4-31B-it` | 30.7B dense, 60 layers; 256K; text and image; Apache-2.0; HF repo 2026-03-11 | The 31B counterpart of #7. Consider only if Gemma wins at 12B |

No new Llama or Mistral text model in the 7–32B range appeared in a listing of either organisation's newest Hugging Face repos (2026-10-15). Mistral's newest ones in this range are the `Ministral-3-8B/14B-2512` repos, created 2025-10-31; their cards were not fetched, so they are left out.

## Overlap with the retrofit literature

| Paper (landscape.md) | Backbone(s) | Source checked |
|---|---|---|
| TTT-NTP | Llama-3.1-8B, Mistral-7B-v0.3, Qwen3-4B, Qwen3-0.6B | arXiv abstract 2606.21803 |
| In-Place TTT | Qwen3-4B-Base (main); Qwen3-14B-Base and LLaMA-3.1-8B in extensions | arXiv HTML 2604.06169 (abstract says only "4B") |
| Titan-Llama | Llama-3.2-1B-Instruct | landscape.md §5 |
| LSTMem | Qwen3-4B-Instruct | arXiv abstract 2609.33268 |
| δ-mem | Qwen3-8B, Qwen3-4B-Instruct, SmolLM3-3B | arXiv HTML 2605.12357 |
| Prometheus Mind | Qwen3-4B | landscape.md §5 (not re-checked) |

Llama-3.1-8B is used by two gradient-written TTT retrofits (TTT-NTP and In-Place TTT). The Qwen3 family is the most-used family overall, and its 0.6B and 4B siblings share a tokenizer and architecture with Qwen3-8B. That matters for the ≤1B reproduction in #6 and #7. landscape.md §1 has been updated to note the larger models in In-Place TTT's paper body.

## GPU memory estimates (assumptions)

**Assumptions.**
- Weights take 2 bytes per parameter in bf16. At 4-bit they take about 0.55 bytes per parameter (0.5 plus about 10% for scales and layers kept at higher precision). Large embedding tables, which are often left unquantized, make the 4-bit figure optimistic for Qwen3.5 and Gemma.
- The KV cache is bf16 and counted only for attention layers, at 4,096 tokens per sequence.
- Inference adds a flat 2 GB of runtime overhead.
- LoRA training uses a frozen bf16 base, batch 1 × 4,096 tokens, gradient checkpointing, FlashAttention, and rank-16 adapters on all linear layers (LoRA weights, gradients and Adam states ≈ 1 GB). Activations are:
  - checkpoints: L·T·d·2 bytes;
  - one recomputed layer: 34·T·d bytes (Korthikanti et al.'s per-layer formula without the attention-score term, which FlashAttention removes);
  - unfused fp32 logits and their gradient: 8·T·V bytes.
  
  Training adds 2 GB of overhead. QLoRA swaps the bf16 weights for the 4-bit figure. Mamba and DeltaNet layers do not follow the 34·T·d rule exactly, so treat hybrid training numbers as rough.

**Worked example (Qwen3-8B).**
- Weights: 8.19B × 2 = 16.4 GB.
- KV per token: 2 (K, V) × 36 layers × 8 heads × 128 dim × 2 B = 147,456 B = 144 KiB. At 4K: 0.60 GB.
- Inference at batch 1: 16.4 + 0.6 + 2 = 19.0 GB.
- LoRA: 16.4 + 1 + (36 × 4096 × 4096 × 2 = 1.21 GB) + (34 × 4096 × 4096 = 0.57 GB) + (8 × 4096 × 151,936 = 4.98 GB) + 2 = **26.1 GB**.

**Hybrid KV.**
- Nemotron-H-8B has 4 attention layers × 8 KV heads × 128 dim, which is 16 KiB per token. Its Mamba state is about 24 layers × 128 heads × 64 × 128 × 4 B ≈ 0.10 GB per sequence.
- Qwen3.5-9B has 8 attention layers × 4 KV heads × 256 dim, which is 32 KiB per token. Its DeltaNet state is 24 × 32 × 128 × 128 × 4 B ≈ 0.05 GB per sequence.
- Gemma-4-12B's sliding layers cap at 1,024 tokens: 40 × 2 × 8 × 256 × 2 B × 1,024 = 0.34 GB. Its global layers add 8 × 2 × 1 × 512 × 2 B × 4,096 = 0.07 GB. That totals 0.40 GB at 4K, counting K and V separately even though they are shared.

| Model | bf16 weights | KV @4K | bf16 inf. b=1 / b=16 | 4-bit inf. b=1 / b=16 | LoRA bf16 @4K | QLoRA @4K |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 16.1 | 0.54 | 18.6 / 26.7 | 7.0 / 15.0 | 24.9 | 13.3 |
| Qwen3-8B | 16.4 | 0.60 | 19.0 / 28.0 | 7.1 / 16.2 | 26.1 | 14.3 |
| Qwen3-32B | 65.5 | 1.07 | 68.6 / 84.7 | 21.1 / 37.2 | 76.9 | 29.4 |
| Mistral-7B-v0.3 | 14.5 | 0.54 | 17.0 / 25.1 | 6.5 / 14.6 | 20.2 | 9.7 |
| Nemotron-H-8B | 16.2 | 0.07 (+0.1 state) | 18.3 / 19.3 (+state) | 6.5 / 7.5 | 25.8 | 14.1 |
| Qwen3.5-9B | 19.3 | 0.13 (+0.05 state) | 21.4 / 23.5 (+state) | 7.4 / 9.5 | 32.1 | 18.1 |
| Gemma-4-12B | 23.9 | 0.40 | 26.3 / 32.3 | 9.0 / 15.0 | 37.5 | 20.2 |
| Qwen3.8-27B (ref.) | 55.6 | 0.27 | 57.8 / 61.9 | 17.5 / 21.6 | 70.1 | 29.8 |

All figures are in GB. Unfused logits are a large share of LoRA memory for the big-vocabulary models (about 8 GB for Qwen3.5 and Gemma). A fused or chunked cross-entropy kernel removes most of it.

**Mapping to AWS** (instance specs as given in the #5 brief, not re-checked against AWS):

| Workload | g6e.xlarge (1×L40S 48 GB) | g6e.12xlarge (4×L40S) | p4d (8×A100 40 GB) | p5 (8×H100 80 GB) |
|---|---|---|---|---|
| 7–9B: bf16 inference, LoRA | Fits (≤32 GB) | 4 independent jobs | Fits per GPU; 8 jobs | Fits; overkill |
| Gemma-4-12B LoRA bf16 | Fits (~38 GB) | 4 jobs | Tight on 40 GB: fused CE or QLoRA | Fits |
| Qwen3-32B bf16 inference | No (69–85 GB) | TP=2 or TP=4 | TP=4 (TP=2 leaves little room for KV) | 1 GPU at b=1; TP=2 for batching |
| Qwen3-32B 4-bit inference / QLoRA | Fits (21–37 / ~29 GB) | Fits | Fits on 1 GPU (QLoRA tight) | Fits |
| Qwen3-32B LoRA bf16 | No | FSDP over 4 GPUs | FSDP over ≥4 GPUs | 1 GPU marginal (~77 GB); 2 with FSDP |

All of Phase 0's 8B-class baseline and LoRA work fits on a g6e.xlarge. Only a 27–32B scale-up needs a multi-GPU instance, unless it uses QLoRA.

## Recommendation (pending the B3/B4 baseline results)

**Primary: `Qwen/Qwen3-8B`** (with `Qwen/Qwen3-8B-Base` for experiments that need a base).
- Apache-2.0 and ungated: no attribution, naming or user-count conditions on per-user adapters or private deployment.
- Dense, standard attention: the simplest target for Titan-Llama-style memory layers, In-Place TTT's MLP-projection fast weights, and per-user LoRA.
- The literature is densest here. δ-mem uses this exact model, and four more retrofits use its 0.6B–14B siblings. That lets #6 and #7 reproduce at ≤1B on Qwen3-0.6B and scale up without changing tokenizer or architecture.
- A same-architecture 32B (Qwen3-32B) is available for scaling inside the 8–32B range. The catch: it has no base checkpoint.
- Tooling: vLLM with LoRA support, ≥1,000 adapter repos, fits a g6e.xlarge for all Phase 0 work.
- Caveats:
  - Thinking mode must be turned off for benchmarks.
  - Native context is 32K. B1 (full context) may need YaRN or a long-context comparator.
  - The session generator must then be a non-Qwen family (phase-0-plan decision 3).

**Fallback: `meta-llama/Llama-3.1-8B-Instruct`.**
- The only candidate used by two gradient-written TTT retrofits (TTT-NTP, In-Place TTT).
- 128K native context suits B1. Mature tooling.
- Costs: gating, Llama licence attribution and naming terms if adapters or models are ever distributed, and an older (2024) model.
- If licence simplicity matters more than literature overlap, Mistral-7B-Instruct-v0.3 (Apache-2.0, used by TTT-NTP) is the alternative fallback.

**Hybrid comparator for open question 4: `Qwen/Qwen3.5-9B`, not Nemotron-H-8B.**
- Nemotron-H-8B's licence bans production use and deployment, which conflicts with Engram's private-deployment goal. The phase-0 plan's hybrid example should change.
- Qwen3.5-9B is Apache-2.0, has a base variant, has 262K native context, and is supported by vLLM LoRA. Its vocabulary (248,320) differs from Qwen3-8B's (151,936), so adapters and memory modules will not transfer between them.
- If a Mamba-2 hybrid specifically is wanted, use NVIDIA-Nemotron-Nano-9B-v2 after a legal read of its guardrail clause.

**Decision rule.** Run B3 and B4 on the primary and the fallback. Keep the primary unless the fallback beats it by more than the 95% CI on at least two `engram-bench` categories. Run the hybrid only on B3/B4 and the leakage harness, to answer question 4, not as a contender for primary.

| Role | Repo id | Licence | Why | Main risk |
|---|---|---|---|---|
| Primary | `Qwen/Qwen3-8B` | Apache-2.0 | Clean licence; densest retrofit literature (family); 0.6B→32B ladder | 32K native context; thinking default; no 32B base |
| Fallback | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Community | Used by TTT-NTP and In-Place TTT; 128K context | Licence conditions on distribution; gated |
| Alt. fallback | `mistralai/Mistral-7B-Instruct-v0.3` | Apache-2.0 | Used by TTT-NTP; clean licence | 2024-era; 32K context |
| Hybrid comparator | `Qwen/Qwen3.5-9B` | Apache-2.0 | Gated DeltaNet hybrid, 262K, LoRA in vLLM | Multimodal wrapper; hybrid LoRA less trodden |
| Dropped (deployment) | `nvidia/Nemotron-H-8B-*` | NVIDIA internal R&D | n/a | Licence bans production and deployment |

## Open items and unverified points

- Whether Gemma 4's Prohibited Use Policy and Terms of Use bind the Apache-licensed weights.
- Whether PEFT LoRA on Gated DeltaNet or Mamba-2 projections trains stably. vLLM's `SupportsLoRA` covers serving only.
- Measured throughput and memory: replace the table above in #12.
- Ministral-3 8B/14B (2025-10-31 repos) were not assessed.

## Sources (fetched 2026-10-15)

- Hugging Face cards, configs and model API for every repo id named above (`https://huggingface.co/<repo id>` and `https://huggingface.co/api/models/<repo id>`).
- Licences: [Llama 3.1 licence](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) (as embedded in the gated card), [NVIDIA Internal Scientific R&D Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/), [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (version 2025-10-24), [Gemma 4 licence](https://ai.google.dev/gemma/docs/gemma_4_license).
- Llama 3 architecture table: [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).
- vLLM model registry: `vllm/model_executor/models/registry.py` on `main` at `8f93c64866`.
- Retrofit backbones: arXiv [2606.21803](https://arxiv.org/abs/2606.21803), [2604.06169](https://arxiv.org/html/2604.06169), [2609.33268](https://arxiv.org/abs/2609.33268), [2605.12357](https://arxiv.org/html/2605.12357).
