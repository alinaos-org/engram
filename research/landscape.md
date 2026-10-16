# Landscape: memory in language models

*Survey date: 2026-10-15. Living document.*

This survey asks one question: **what is known about putting long-term, per-user memory inside a language model rather than around it?** It covers the methods Engram builds on or competes with. For each method it gives a short summary, its strengths and limits for Engram, and a link.

**How this was compiled.** Every source listed here was opened during the survey: the arXiv abstract page, OpenReview or conference page, official blog, model card, or repository. Its title, authors and date were checked. Sources that could not be opened were dropped. Numbers come only from the fetched page and are the authors' own claims; we have reproduced none of them. Most entries are summarised from abstracts, so read the paper before relying on a detail. Summaries are paraphrased. Many 2026 entries are unreviewed preprints, and we say so where it matters.

> **Name clash.** DeepSeek published an unrelated architecture also called **Engram** in January 2026: hashed n-gram memory tables ([§4](#4-memory-layers-and-product-key-memory)). A June 2026 paper, *User as Engram*, builds per-user memory on it. When this document means DeepSeek's work, it says so.

## Contents

1. [Test-time training](#1-test-time-training)
2. [Titans and its follow-ups](#2-titans-and-its-follow-ups)
3. [Nested Learning](#3-nested-learning)
4. [Memory layers and product-key memory](#4-memory-layers-and-product-key-memory)
5. [Retrofitting memory onto frozen pretrained models](#5-retrofitting-memory-onto-frozen-pretrained-models)
6. [State-space and hybrid models](#6-state-space-and-hybrid-models)
7. [Continual learning and catastrophic forgetting](#7-continual-learning-and-catastrophic-forgetting)
8. [Consolidation: getting knowledge into weights](#8-consolidation-getting-knowledge-into-weights)
9. [Per-user LoRA and personalization](#9-per-user-lora-and-personalization)
10. [Model editing](#10-model-editing)
11. [Unlearning](#11-unlearning)
12. [Synthesis: what this means for Engram](#12-synthesis-what-this-means-for-engram)

---

## 1. Test-time training

Test-time training (TTT) treats part of the model's state as weights that are updated by gradient descent *while the model reads*. That is the core primitive behind Engram's fast memory.

**Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** Sun et al., Jul 2024. [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
The paper that introduced TTT layers. The hidden state of a recurrent layer is itself a small model (linear, or a two-layer MLP), and each token updates it with a self-supervised gradient step, during both training and inference. At 125M–1.3B parameters, the TTT layers kept improving as context grew, as Transformers do, where Mamba plateaued.
- *Strengths:* establishes memory-as-weights written by gradient at inference. The MLP variant is the precedent for a deep, nonlinear fast memory.
- *Limits:* trained from scratch at small scale, and it replaces the sequence layer rather than adding to a frozen model. State resets every sequence.

**Test-time regression: a unifying framework for designing sequence models with associative memory.** Wang, Shi, Fox, Jan 2025. [arXiv:2501.12352](https://arxiv.org/abs/2501.12352)
Frames recall as "memorise, then retrieve", where memorising is a regression solved over the context at test time. In this view, linear attention, SSMs, fast-weight programmers and softmax attention differ only in three choices: how tokens are weighted, what the regressor's function class is, and which optimiser solves it.
- *Strengths:* a clean design vocabulary for Engram's write rule. Surprise-driven writing becomes one choice among several.
- *Limits:* theory, with nothing on persistence or frozen backbones.

**Test-Time Training Done Right (LaCT).** Zhang et al., May 2025. [arXiv:2505.23884](https://arxiv.org/abs/2505.23884)
Argues that earlier TTT layers update too often (every 16–64 tokens) and leave GPUs mostly idle, often below 5% utilisation. LaCT updates once per very large chunk, from 2K up to 1M tokens, and relies on window attention for local detail. This makes a fast-weight state as large as 40% of the model practical without custom kernels.
- *Strengths:* the most practical answer to throughput. A write per conversation turn maps naturally onto large-chunk updates.
- *Limits:* nothing inside a chunk is memorised until the chunk ends. Its language-model experiments train from scratch.

**End-to-End Test-Time Training for Long Context (TTT-E2E).** Tandon et al., Dec 2025. [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
Treats long context as continual learning. A sliding-window Transformer keeps training on incoming text with ordinary next-token loss, and its initialisation is meta-learned so these updates work well. At 3B parameters the authors report that it scales with context like full attention, at constant per-token latency, unlike Mamba 2 or Gated DeltaNet.
- *Strengths:* strong evidence that storing context in weights with a next-token objective can match attention, provided the model was meta-trained for it.
- *Limits:* a frozen, off-the-shelf base was never meta-trained this way, so Engram must meta-train its memory adapter to get the benefit.

**In-Place Test-Time Training.** Feng et al., Apr 2026 (ICLR 2026 oral, per the abstract page). [arXiv:2604.06169](https://arxiv.org/abs/2604.06169)
Instead of adding a memory layer, it treats the output projection of each MLP block in an existing LLM as fast weights and updates them chunk by chunk during inference. The update objective is aligned with next-token prediction. Applied to a pretrained 4B model, it improves tasks with contexts up to 128K tokens; the full paper also reports runs with Qwen3-14B-Base and LLaMA-3.1-8B. It still continues pretraining the model on its objective; it is not training-free.
- *Strengths:* the most direct precedent for adding TTT to a released model without architectural surgery. A per-user delta on those projections would be a natural detachable artifact.
- *Limits:* the abstract reports only 4B (larger models appear in the paper body), and it does not study whether memory persists across sessions.

**Test-Time Training with Next-Token Prediction (TTT-NTP).** Ouyang, Cai, Hu, Jun 2026 (Findings of EMNLP 2026). [arXiv:2606.21803](https://arxiv.org/abs/2606.21803)
A fast-weight method for already-pretrained models. It still continues pretraining, and per [fast-memory-design.md](fast-memory-design.md) it trains its projection together with the backbone. Each write is supervised by the model's own next hidden state, so no value head needs to be trained. On RULER the authors report gains of about 3–4 points on every backbone tested, from 0.6B up to Llama-3.1-8B, with general knowledge preserved.
- *Strengths:* evidence at 8B, Engram's lower size bound, that writes on a released checkpoint help without harming general ability.
- *Limits:* modest gains, and it only addresses use within a single context.

**Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs (qTTT).** Bansal et al., Dec 2025. [arXiv:2512.13898](https://arxiv.org/abs/2512.13898)
Argues that long-context failures come from attention scores being diluted across many tokens, and that a few targeted gradient steps on the context help more than spending the same compute on extra reasoning tokens. Reports double-digit gains on LongBench-v2 and ZeroScrolls subsets for a 4B model.
- *Strengths:* a theoretical and compute-budget case for spending inference FLOPs on weight updates.
- *Limits:* the adaptation is per-query and discarded afterwards.

**Self-Guided Test-Time Training for Long-Context LLMs (S-TTT).** Zhu et al., Jul 2026. [arXiv:2607.09415](https://arxiv.org/abs/2607.09415)
Finds that TTT on random context spans hurts, while TTT on the evidence spans helps a lot. The model first selects the relevant spans, then trains only on them. Reports up to 15% relative improvement for 4B and 8B instruct models.
- *Strengths:* strong evidence that *what* you write matters more than writing everything, which supports gating writes.
- *Limits:* span selection is conditioned on the question, which Engram doesn't have when writing ahead of time for future sessions.

**No Time Like the Present: Agentic Test-Time Training for LLM Agents (aTTT).** Wang et al., Jul 2026. [arXiv:2607.03441](https://arxiv.org/abs/2607.03441)
Running TTT continuously through multi-turn agent episodes creates a self-training loop. It helps when new information arrives, but amplifies drift when the agent repeats itself. The fix is to down-weight tokens in n-grams already trained on. Served through vLLM's runtime-LoRA support at about 1.9× the cost of no TTT.
- *Strengths:* names a concrete instability of inference-time writes, a repetition feedback loop, and gives a cheap fix, novelty-weighted loss. Also shows per-session LoRA writes can be served.
- *Limits:* the authors conclude the gains mostly preserve existing competence rather than add knowledge.

## 2. Titans and its follow-ups

**Titans: Learning to Memorize at Test Time.** Behrouz, Zhong, Mirrokni (Google), Dec 2024. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
Adds a neural long-term memory, a deep MLP updated at inference, alongside attention, which serves as short-term memory. The memory takes a gradient step on a key-to-value regression loss. The gradient's size acts as a *surprise* signal, momentum carries surprise across nearby tokens, and adaptive weight decay acts as forgetting. Three wirings are proposed: memory as context, as a gate, or as a layer. The authors report beating Transformers and linear RNNs, and scaling beyond 2M tokens of context.
- *Strengths:* the closest published design to Engram's fast memory, from the surprise-gated write and momentum to learned decay and a deep MLP memory.
- *Limits:* trained jointly from scratch, not attached to a frozen model. No official code. Memory lasts one sequence.

**It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization (Miras).** Behrouz et al., Apr 2025. [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)
Describes Transformers, Titans and linear RNNs as associative memories defined by four choices: architecture, inner objective, retention (forgetting recast as regularisation), and learning algorithm. It introduces three variants: Moneta (generalised norms), Yaad (a Huber-style loss that is robust to outliers), and Memora (memory constrained to behave like a probability map).
- *Strengths:* robust losses and constrained memories are cheap protection against one odd or adversarial input corrupting the memory. Treating retention as regularisation is a natural way to anchor the fast memory to its last consolidated state.
- *Limits:* trained from scratch at small scale, by the Titans authors, and not independently replicated.

**ATLAS: Learning to Optimally Memorize the Context at Test Time.** Behrouz et al., May 2025. [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)
Blames recurrent memories' long-context weakness on small capacity, updates driven by one token at a time, and crude memory management. ATLAS optimises its memory against a sliding window of recent tokens with a stronger optimiser. It reports large gains over Titans on BABILong at 10M tokens.
- *Strengths:* window-based writes are more stable than per-token surprise and fit chunked updates. The capacity analysis helps size a per-user memory.
- *Limits:* single lab, synthetic long-context benchmarks, trained from scratch.

**Titans + MIRAS: Helping AI have long-term memory.** Google Research blog, Dec 2025. [research.google](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
The authors' plain-language account of the surprise, momentum and decay mechanics, and of the Miras variants. Results appear only as charts.
- *Strengths:* the authoritative description of the write rule Engram plans to reuse.
- *Limits:* a blog post, not peer-reviewed evidence.

**Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model.** Di Nepi, Siciliano, Silvestri, Oct 2025. [arXiv:2510.09551](https://arxiv.org/abs/2510.09551)
An independent reimplementation, prompted by the missing code. Titans as a whole did not always beat established baselines, and the authors blame its chunking. The neural memory component itself, however, consistently improved on attention-only models.
- *Strengths:* independent evidence that the memory module is the part worth keeping.
- *Limits:* small scale, non-LLM tasks. Chunk size is a sensitivity Engram will have to tune.

**What Should Titans Forget? Disentangling Magnitude, Distribution, and Assignment in Test-Time Memory.** Malakova, NeurIPS 2026 workshop poster. [neurips.cc](https://neurips.cc/virtual/2026/159153)
Controlled experiments on the forget gate of a Titans model. Removing decay hurts badly. Hand-designed forgetting rules (by surprise, novelty or recency) do clearly worse than learned decay, and simply matching the learned decay's *average rate* recovers most of the gap. Learned per-token assignment still wins on most sequences.
- *Strengths:* directly informs Engram's forgetting design. Learn the decay, tune its mean, and don't hand-code surprise-based forgetting.
- *Limits:* a single-author workshop paper on a small model, covering forgetting within one sequence only.

## 3. Nested Learning

**Nested Learning: The Illusion of Deep Learning Architectures.** Behrouz, Razaviyayn, Zhong, Mirrokni, Dec 2025 (a version appeared at NeurIPS 2025, per the arXiv comments). [arXiv:2512.24695](https://arxiv.org/abs/2512.24695)
Recasts a model and its optimiser as a set of nested optimisation problems, each updating at its own frequency. It reinterprets optimisers such as Adam as associative memories over gradients. It proposes a self-modifying sequence model and a **Continuum Memory System (CMS)**: a spectrum of memory blocks updating at different rates, in place of a binary short-term/long-term split. Together these make the **Hope** architecture, reported to do well on language modelling, continual learning and long-context tasks.
- *Strengths:* CMS is the closest conceptual match to Engram's stack. Fast memory, sleep and a per-user adapter become three levels of one multi-timescale memory rather than three separate subsystems.
- *Limits:* trained from scratch, nothing on frozen bases, and the abstract's claims are qualitative. We found no independent reproduction of Hope's language-model results.

**Introducing Nested Learning: A new ML paradigm for continual learning.** Google Research blog, Nov 2025. [research.google](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
Presents Hope as a self-modifying successor to Titans with more levels of in-context learning, plus CMS blocks for longer context. Results appear only as charts.
- *Strengths:* confirms where Google sees this line going.
- *Limits:* a blog, with no released code noted.

**Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.** Behrouz, Hashemi, Javanmard, Mirrokni, Jun 2026. [arXiv:2606.03979](https://arxiv.org/abs/2606.03979)
Proposes a two-stage "sleep" phase. *Memory consolidation* distils fast memories upward into slower weights, combining on-policy distillation with RL-based imitation. *Dreaming* uses RL to generate a synthetic curriculum that rehearses new knowledge and refines skills. Evaluated on long-horizon continual learning and knowledge incorporation.
- *Strengths:* the closest published match to Engram's wake/sleep design, from the Titans authors.
- *Limits:* not per-user and not a detachable adapter. The abstract gives no numbers, and RL dreaming is expensive.

## 4. Memory layers and product-key memory

Memory layers swap some dense computation for a huge table of key-value slots. Each token reads only a few slots, so capacity grows without adding compute, and each write is local.

**Large Memory Layers with Product Keys.** Lample et al., NeurIPS 2019. [arXiv:1907.05242](https://arxiv.org/abs/1907.05242)
The original product-key memory (PKM). Each key is split into two halves, so finding the top-k slots in a table with a very large number of entries stays cheap. A 12-layer model with memory beat a 24-layer model without it.
- *Strengths:* separates memory capacity from FLOPs, and writes are naturally local.
- *Limits:* must be trained in from the start.

**Memory Layers at Scale.** Berges et al. (Meta), Dec 2024. [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)
Modernises memory layers and scales them to 128B memory parameters, with base models up to 8B trained on 1T tokens. These models beat dense models using over twice the compute, with the largest gains on factual tasks.
- *Strengths:* the strongest evidence that sparse tables are a good home for facts.
- *Limits:* needs pretraining with memory layers.

**Ultra-Sparse Memory Network (UltraMem).** Huang et al. (ByteDance), ICLR 2025. [arXiv:2411.12364](https://arxiv.org/abs/2411.12364) **UltraMemV2.** Aug 2025. [arXiv:2508.18756](https://arxiv.org/abs/2508.18756)
Memory layers engineered for low inference latency. V2 reports parity with 8-expert MoE models at equal compute, scaling to 120B total parameters, with the largest gains on long-context and multi-round memorisation. It also finds that how many slots are activated matters more than the total slot count.
- *Strengths:* memory layers help most on exactly the memorisation tasks personal memory needs.
- *Limits:* pretraining recipes. The activation-density finding hints that very sparse per-user tables may hold less than their size suggests.

**Continual Learning via Sparse Memory Finetuning (SMF).** Lin et al. (Meta), Oct 2025. [arXiv:2510.15103](https://arxiv.org/abs/2510.15103)
Updates only the memory slots that new data activates far more than pretraining data did. At the same level of new-fact learning, NaturalQuestions F1 fell 89% with full fine-tuning, 71% with LoRA, and only 11% with sparse memory fine-tuning.
- *Strengths:* the single most important result for Engram's choice of container. Writing facts into sparse slots interferes far less than LoRA.
- *Limits:* needs a model that already has memory layers, and it tests QA-style facts, not conversational memory.

**Improving Sparse Memory Finetuning.** Goyal et al., Apr 2026. [arXiv:2604.05248](https://arxiv.org/abs/2604.05248) **Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning.** Gupta et al., May 2026. [arXiv:2605.03229](https://arxiv.org/abs/2605.03229)
Two small-team follow-ups that *retrofit* memory layers into a pretrained Qwen-2.5-0.5B, choosing slots by a KL-based surprise score. On MedMCQA, SMF kept forgetting probes within about a point of the base while LoRA and full fine-tuning drifted, but it learned the target task less.
- *Strengths:* retrofitting onto an off-the-shelf model works, and the slot selection is surprise-driven.
- *Limits:* only 0.5B. There is a real stability–plasticity trade-off, and the cost of retrofitting at 8–32B is unknown.

**Fast-weight Product Key Memory (FwPKM).** Zhao, Jones (Sakana AI), Jan 2026. [arXiv:2601.00671](https://arxiv.org/abs/2601.00671)
Turns a PKM into a fast-weight memory. The slots a chunk activates are updated by gradient descent at both training and inference time. The authors frame it as episodic memory alongside the model's semantic memory. Trained at 4K context, it generalises to 128K on needle-in-a-haystack.
- *Strengths:* Titans-style test-time writes into a *sparse* table. The fast and consolidated stores could share one data structure.
- *Limits:* trained jointly with the model, and memory lasts one context.

**Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (DeepSeek "Engram").** Cheng et al. (DeepSeek-AI, PKU), Jan 2026. [arXiv:2601.07372](https://arxiv.org/abs/2601.07372)
Adds hashed n-gram embedding tables with constant-time lookup as a sparsity axis alongside MoE. A 27B model beats an MoE baseline matched for parameters and FLOPs. Because addressing is deterministic, the tables can be prefetched from host memory.
- *Strengths:* deterministic tables can live off-GPU, which suits per-user stores held in host RAM or on disk.
- *Limits:* lexical (n-gram) rather than semantic keys, and must be pretrained in. Unrelated to this project apart from the name.

**User as Engram: Internalizing Per-User Memory as Local Parametric Edits.** Li, Jun 2026. [arXiv:2606.19172](https://arxiv.org/abs/2606.19172)
Stores each user's facts as local edits to DeepSeek-Engram hash tables, with one shared adapter for reasoning. Against per-user LoRA, the author reports equal direct recall, much better indirect reasoning, a footprint orders of magnitude smaller, and no damage to unrelated text. Users' edits land in separate slots and stack.
- *Strengths:* tackles Engram's exact container question and argues facts belong in sparse slots, not dense LoRA.
- *Limits:* a single-author preprint that needs an Engram-architecture base. The shared adapter is a cross-user component, which matters for deletion.

## 5. Retrofitting memory onto frozen pretrained models

Engram's base stays frozen, so the most relevant work attaches memory to a model that already exists.

**Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules.** Pan, Behrouz, Oct 2026. [arXiv:2610.19210](https://arxiv.org/abs/2610.19210)
Attaches Titans-style memory to a frozen Llama-3.2-1B-Instruct, training only about 1% added parameters. The first four layers become blockwise attention, each with neural memories updated by gradient once per chunk at inference. The authors report large gains on long-document QA (QuALITY, LongHealth) over segmented attention, plus faster decoding.
- *Strengths:* the first paper we found that bolts a gradient-written, Titans-style memory onto a frozen pretrained LLM, co-authored by the Titans first author.
- *Limits:* 1B only, and it degrades beyond its 16K training length. Its aim is compressing attention, not cross-session memory. The authors call their reconstruction objective a surrogate.

**δ-mem: Efficient Online Memory for Large Language Models.** Lei et al., May 2026. [arXiv:2605.12357](https://arxiv.org/abs/2605.12357)
Adds a tiny delta-rule associative memory, an 8×8 state, to a frozen LLM. Its readout makes low-rank corrections to attention. Reports gains on MemoryAgentBench and LoCoMo while mostly preserving general ability.
- *Strengths:* frozen base, evaluated on conversational-memory benchmarks, and a tiny, easily detached state.
- *Limits:* very small capacity, and a linear memory.

**LSTMem: Hierarchical Long Short-Term Online Memory for Large Language Models.** Shi et al., Sep 2026. [arXiv:2609.33268](https://arxiv.org/abs/2609.33268)
Adds gated matrix memories to each layer of a frozen Qwen3-4B. Only the adapter is trained, and detaching it recovers the original model exactly. Reports gains on MemoryAgentBench, LoCoMo and HotpotQA.
- *Strengths:* exact detachability is a property Engram wants. The ethics section calls for user-controlled memory reset.
- *Limits:* written by a learned recurrence, not by gradient. The authors note that every memory method, theirs included, underperformed the plain model on one benchmark family.

**Prometheus Mind: Retrofitting Memory to Frozen Language Models.** Wind, Jan 2026. [arXiv:2601.15324](https://arxiv.org/abs/2601.15324)
Removable adapters that inject memory into a frozen Qwen3-4B. End-to-end training collapsed and staged training worked. Retrieval was strong on clean inputs and weak on informal ones.
- *Strengths:* a cautionary data point. Frozen-base injection is fragile, and staged training helps.
- *Limits:* single author, self-built benchmark, and not test-time training.

**Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models.** Cao et al., Aug 2025. [arXiv:2508.09874](https://arxiv.org/abs/2508.09874)
A small decoder trained to imitate a retriever's output distribution, which then plugs into any model with the same tokenizer and leaves the base unchanged.
- *Strengths:* a portable, detachable memory module.
- *Limits:* domain-level, not per-user, and costly to train per user.

**MemoryLLM** (Wang et al., Feb 2024, [arXiv:2402.04624](https://arxiv.org/abs/2402.04624)) and **M+** (Wang et al., Feb 2025, [arXiv:2502.00592](https://arxiv.org/abs/2502.00592))
MemoryLLM keeps a fixed pool of latent memory tokens in each layer that the model updates as it reads. M+ finds that the pool fades beyond about 20K tokens, and extends retention past 160K by adding a co-trained retriever.
- *Strengths:* independent support for Engram's "bounded in-network memory plus retrieval backstop" design.
- *Limits:* retention is measured in tokens of context, not in weeks of use.

**Larimar: Large Language Models with Episodic Memory Control.** Das et al. (IBM), ICML 2024. [arXiv:2403.11901](https://arxiv.org/abs/2403.11901)
A distributed episodic memory that can be written in one shot without retraining, and supports selective forgetting.
- *Strengths:* one-shot writes and explicit forgetting map onto "remember this" and "forget that" requests.
- *Limits:* evaluated on fact-editing benchmarks, not conversations.

**Position: Modular Memory is the Key to Continual Learning Agents.** Dorovatas et al., ICML 2026 position track. [arXiv:2603.01761](https://arxiv.org/abs/2603.01761)
Argues continual learning should pair fast in-context adaptation with slow, stable weight learning in modular memories, instead of relying on weight updates alone.
- *Strengths:* community-level support for Engram's fast/slow design.
- *Limits:* a position paper with no method or results.

## 6. State-space and hybrid models

Recurrent layers keep a fixed-size state that is updated with each token. Modern variants are explicitly online learners, so they are a kind of built-in test-time memory.

**Mamba** (Gu, Dao, Dec 2023, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) and **Mamba-2 / SSD** (Dao, Gu, ICML 2024, [arXiv:2405.21060](https://arxiv.org/abs/2405.21060))
Selective state-space layers that scale linearly with sequence length. Mamba-2 shows that SSMs and linear attention are two views of the same computation.
- *Strengths:* a compact state that is cheap to snapshot.
- *Limits:* the state is fixed-size and lossy (see the recall limits below).

**Gated Delta Networks: Improving Mamba2 with Delta Rule.** Yang, Kautz, Hatamizadeh, ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
Combines gating, which erases quickly, with the delta rule, which overwrites a specific key's value. The delta rule is an online regression step on a fast-weight matrix, the same family as Titans.
- *Strengths:* now inside leading open-weights models (below).
- *Limits:* capacity is bounded by state size.

**Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.** Hatamizadeh, Choi, Kautz, May 2026. [arXiv:2605.22791](https://arxiv.org/abs/2605.22791) **Mamba-3.** Lahoti et al., ICLR 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
GDN-2 gives erasing and writing separate per-channel gates, with its clearest gains on multi-key retrieval. Mamba-3 gets more out of each unit of state and matches Mamba-2 with half the state size.
- *Strengths:* design cues for Engram's write rule: keep the write gate and the forget gate separate.
- *Limits:* research models of about 1.5B parameters.

**Hybrid open-weights models.** **Jamba** (AI21, [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)), **Samba** (Microsoft, ICLR 2025, [arXiv:2406.07522](https://arxiv.org/abs/2406.07522)), **Nemotron-H** (NVIDIA, [arXiv:2504.03624](https://arxiv.org/abs/2504.03624)), **Nemotron 3** (NVIDIA, [arXiv:2512.20856](https://arxiv.org/abs/2512.20856)), **Kimi Linear** (Moonshot, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)), **Qwen3-Next** ([model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)) and **Qwen3.5** ([model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B))
All of these interleave recurrent or linear-attention layers with a minority of full-attention layers. The Qwen3-Next and Qwen3.5 cards describe three Gated DeltaNet layers per attention layer. Kimi Linear reports beating full attention under the same recipe with a much smaller KV cache.
- *Strengths:* if Engram uses such a base, delta-rule memory is already present within each context. The Titans-style module's job then shifts to memory that persists *across* sessions.
- *Limits:* the flagship models are large MoEs, and LoRA on MoE-plus-hybrid layers is less well trodden. Sizes in the 8–32B range need checking per family.

**Recall limits of fixed-state models.**
- **Zoology** (Arora et al., [arXiv:2312.04927](https://arxiv.org/abs/2312.04927)): most of the quality gap between efficient models and attention comes from associative recall. Introduces the MQAR probe.
- **Based** (Arora et al., [arXiv:2402.18668](https://arxiv.org/abs/2402.18668)): state size and recall trade off directly.
- **Repeat After Me** (Jelassi et al., [arXiv:2402.01032](https://arxiv.org/abs/2402.01032)): fixed-state models are fundamentally limited at copying.
- **Overcoming Long-Context Limitations of SSMs via Context-Dependent Sparse Attention** (Zhan et al., NeurIPS 2025, [arXiv:2507.00449](https://arxiv.org/abs/2507.00449)): proves SSMs cannot solve multi-query context-dependent recall in sub-quadratic time.
- **A Systematic Analysis of Hybrid Linear Attention** (Wang et al., [arXiv:2507.06457](https://arxiv.org/abs/2507.06457)): recall needs roughly one full-attention layer per 3–6 linear layers.
- **Mechanistic evaluation of Transformers and state space models** (Arora et al., [arXiv:2505.15105](https://arxiv.org/abs/2505.15105)): models can pass recall tests through unintended mechanisms.
- **HydraHead** (Tan et al., Jun 2026, [arXiv:2606.20097](https://arxiv.org/abs/2606.20097)): exact retrieval lives in a few attention heads.
- *For Engram:* recurrent state can carry gist and style across turns, but specific facts need slots, adapters or the retrieval log. The log is a necessity, not a nicety. Use MQAR-style and mechanistic probes on any memory module. No source we found tests saving recurrent state across sessions as persistent memory; that is an open, cheap experiment.

## 7. Continual learning and catastrophic forgetting

**LoRA Learns Less and Forgets Less.** Biderman et al., TMLR 2024. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
On code and maths, LoRA learns a new domain worse than full fine-tuning but keeps more of the model's other abilities. Full fine-tuning finds updates of much higher rank than typical LoRA settings.
- *Strengths:* the base rate for consolidating into a low-rank adapter. The rank bottleneck acts as a brake on forgetting.
- *Limits:* the same bottleneck caps how much a nightly adapter can absorb. It does not study sequential consolidation into one adapter.

**RL's Razor: Why Online Reinforcement Learning Forgets Less.** Shenfeld, Pari, Agrawal, Sep 2025. [arXiv:2509.04259](https://arxiv.org/abs/2509.04259)
Forgetting tracks the KL divergence between the fine-tuned and base model on the new task's data. On-policy RL implicitly picks low-KL solutions, while SFT can drift far.
- *Strengths:* a measurable forgetting proxy Engram can monitor after every sleep.
- *Limits:* about skills, not new facts. Novel facts may have no low-KL path.

**Self-Distillation Enables Continual Learning (SDFT).** Shenfeld et al., Jan 2026. [arXiv:2601.19897](https://arxiv.org/abs/2601.19897)
The model with demonstrations in its context teaches the same model without them, which gives on-policy targets. The authors report better learning and much less forgetting than SFT, including skills accumulated sequentially.
- *Strengths:* close to a ready-made sleep objective. The teacher is the base with the day's memories in context; the student is the base plus the user's adapter without them.
- *Limits:* see the next entry.

**Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual Post-Training.** Wang et al., Jul 2026. [arXiv:2607.01763](https://arxiv.org/abs/2607.01763)
In continual post-training, self-distillation can forget more and even collapse, as the teacher-student loop amplifies its own artifacts. GRPO is more conservative.
- *Strengths:* names a failure mode that repeated nightly self-distillation must be monitored for.
- *Limits:* about skills under RL-style post-training, not per-user facts.

**Self-Adapting Language Models (SEAL).** Zweiger et al., Jun 2025. [arXiv:2506.10943](https://arxiv.org/abs/2506.10943)
The model writes its own training data and update settings ("self-edits"), and an RL outer loop rewards self-edits that improve the updated model.
- *Strengths:* a template for a model that learns what to study during sleep.
- *Limits:* expensive (many inner fine-tunes per reward), and not shown to avoid forgetting over many edits.

**Beyond Static Models: An Evolving Framework for Continual Learning in Large Language Models across Training Stages.** Chen et al., Mar 2026. [arXiv:2603.12658](https://arxiv.org/abs/2603.12658)
A 2026 survey of continual learning across pretraining, fine-tuning and alignment. It concludes that seamless knowledge integration is unsolved.
- *Strengths:* a map of baselines (replay, regularisation, architectural isolation).
- *Limits:* a survey, with no new evidence.

## 8. Consolidation: getting knowledge into weights

Several independent lines agree that **plain next-token training on raw text is a poor way to put new knowledge into weights**. What works better is generating varied synthetic data about the material, then distilling from a teacher that has the material in context.

**Synthetic continued pretraining (EntiGraph).** Yang et al., Sep 2024. [arXiv:2409.07431](https://arxiv.org/abs/2409.07431)
Models need many varied exposures to learn a fact. EntiGraph generates text about relations between entities in a small corpus, so continued pretraining teaches the corpus. The gains add to RAG rather than replace it.
- *Strengths:* the canonical "rehearse by restating" recipe for sleep.
- *Limits:* designed for full continued pretraining, not small nightly LoRA updates.

**Cartridges: Lightweight and general-purpose long context representations via self-study.** Eyuboglu et al., Jun 2025. [arXiv:2506.06266](https://arxiv.org/abs/2506.06266)
Trains a small KV-cache "cartridge" per corpus offline. Next-token training doesn't work, but self-study does: synthetic conversations about the corpus plus a context-distillation objective. It matches in-context learning with far less memory, and cartridges compose at inference.
- *Strengths:* the strongest evidence for the consolidation objective, and a detachable, composable per-corpus artifact. A per-user cartridge is an alternative container to LoRA.
- *Limits:* trained offline per corpus, with no incremental nightly updates.

**Training Plug-n-Play Knowledge Modules with Deep Context Distillation.** Caccia et al., COLM 2025. [arXiv:2503.08727](https://arxiv.org/abs/2503.08727)
One LoRA module per document, trained to reproduce the hidden states and outputs of a teacher that has the document in context. It beats next-token training and works alongside RAG.
- *Strengths:* the closest published match to consolidating memories into a detachable LoRA, with an objective Engram can use directly.
- *Limits:* one module per document. Accumulating many days into one adapter is not studied.

**Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction.** Dennis et al., May 2026. [arXiv:2605.24657](https://arxiv.org/abs/2605.24657)
Nightly consolidation of conversations into a LoRA, through reflection, synthesis and fine-tuning, on one consumer GPU. LoRA kept about 80% of the knowledge, against about 37% for repeated context compaction and about 90% for full context. It also notes that mean token loss correlates poorly with judged accuracy.
- *Strengths:* the most directly on-target result found. Nightly per-user LoRA consolidation is feasible, beats summarisation, and leaves a gap for a retrieval backstop to fill.
- *Limits:* only 10 conversations, in one domain. It does not measure forgetting over many nights.

**Text-to-LoRA** (Charakorn et al., ICML 2025, [arXiv:2506.06105](https://arxiv.org/abs/2506.06105)) and **Generative Adapter** (Chen et al., Nov 2024, [arXiv:2411.05877](https://arxiv.org/abs/2411.05877))
Hypernetworks that produce a LoRA in one forward pass, from a task description or from context.
- *Strengths:* could make cold-starting or refreshing a user's module cheap.
- *Limits:* the generator is shared across users, so any user data it was trained on cannot be deleted by dropping one adapter.

**Sleep-time Compute: Beyond Inference Scaling at Test-time.** Lin et al. (Letta), Apr 2025. [arXiv:2504.13171](https://arxiv.org/abs/2504.13171)
Lets the model reason over a context offline before questions arrive, which cuts the compute needed at query time.
- *Strengths:* precedent for doing useful work during idle "sleep" time.
- *Limits:* writes text notes, not weights.

## 9. Per-user LoRA and personalization

**Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning (OPPU).** Tan et al., EMNLP 2024. [arXiv:2402.04401](https://arxiv.org/abs/2402.04401)
One PEFT module per user, combined with retrieval and a profile, beats prompt-based personalisation on all seven LaMP tasks.
- *Strengths:* founding evidence that adapter plus retrieval beats retrieval alone, and that users can own their parameters.
- *Limits:* LaMP measures style and preference, not episodic memory. Adapters are trained once, not consolidated over time.

**Personalized Pieces (Per-Pcs).** Tan et al., EMNLP 2024. [arXiv:2406.10471](https://arxiv.org/abs/2406.10471) **Instant Personalized LLM Adaptation via Hypernetwork (Profile-to-PEFT).** Tan et al., ACL 2026. [arXiv:2510.16282](https://arxiv.org/abs/2510.16282)
Cheaper per-user adapters, built from pieces shared between users or generated from a profile by a hypernetwork.
- *Strengths:* efficient cold-start.
- *Limits:* both share components across users, the opposite of exact per-user deletion.

**S-LoRA: Serving Thousands of Concurrent LoRA Adapters.** Sheng et al., Nov 2023. [arXiv:2311.03285](https://arxiv.org/abs/2311.03285)
Keeps adapters in host memory, pages them onto the GPU on demand, and batches different adapters together.
- *Strengths:* systems evidence that one adapter per user on a shared base is servable. It also fits adapters encrypted at rest and decrypted on load.
- *Limits:* a fast-memory state is not an adapter and needs its own paging scheme.

**Separable Expert Architecture: Toward Privacy-Preserving LLM Personalization via Composable Adapters and Deletable User Proxies.** Schneider, Schoenegger, Bariach, Apr 2026. [arXiv:2604.21571](https://arxiv.org/abs/2604.21571)
Splits the model into a static base, domain adapters that never see user data, and deletable per-user artifacts. Deleting a user's artifact returns the model close to baseline, and cross-user contamination is near zero.
- *Strengths:* essentially Engram's deletion thesis, with a verification protocol to borrow (divergence from the baseline, and a pass rate).
- *Limits:* even here, removal did not return the model *exactly* to baseline. Engram must verify deletion, not assume it.

**Substrate Asymmetry in User-Side Memory: A Diagnostic Framework.** Deng et al., Jun 2026. [arXiv:2606.11712](https://arxiv.org/abs/2606.11712)
Compares per-user LoRA with retrieval. LoRA wins on behavioural style; retrieval wins on correctly saying "you never told me that".
- *Strengths:* direct support for Engram's adapter-plus-log hybrid, and for treating abstention as its own metric.
- *Limits:* no context wipes, updates or forgetting.

## 10. Model editing

**Model Editing at Scale leads to Gradual and Catastrophic Forgetting.** Gupta, Rao, Anumanchipalli, ACL Findings 2024. [arXiv:2401.07453](https://arxiv.org/abs/2401.07453) (on ROME and **MEMIT**, Meng et al., [arXiv:2210.07229](https://arxiv.org/abs/2210.07229))
Applying edits one after another makes the model gradually forget earlier edits, then collapse abruptly.
- *For Engram:* a warning that writing many small facts into the same dense weights over time fails non-linearly.

**AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.** Fang et al., ICLR 2025. [arXiv:2410.02355](https://arxiv.org/abs/2410.02355)
Projects each edit into the null space of the knowledge to be preserved, which leaves that knowledge's outputs unchanged.
- *For Engram:* a transferable idea. Project each night's update away from the subspace used by earlier memories.
- *Limits:* later work reports it still degrades over long edit streams, and the preserved subspace grows.

**WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models.** Wang et al., NeurIPS 2024. [arXiv:2405.14768](https://arxiv.org/abs/2405.14768)
Keeps the pretrained weights as main memory and a separate side memory for edits, with a router choosing between them per query. Edit sets go into separate shards.
- *For Engram:* architecturally almost the same idea: a frozen main memory and a routed, detachable side memory.
- *Limits:* edit benchmarks only. Merging shards across users would break deletion.

**More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing (StableEdit).** Ma et al., ICML 2026. [arXiv:2605.11836](https://arxiv.org/abs/2605.11836)
Lifelong editors that survive long horizons share one ingredient: normalising updates with running statistics. With ridge regularisation, updates become nearly orthogonal and bounded.
- *For Engram:* cheap stability ingredients for a long-lived nightly adapter.
- *Limits:* designed for fact triples, not paraphrased personal memories.

## 11. Unlearning

The unlearning literature is the strongest argument for Engram's "drop the module" approach to deletion.

**TOFU: A Task of Fictitious Unlearning for LLMs.** Maini et al., Jan 2024. [arXiv:2401.06121](https://arxiv.org/abs/2401.06121)
200 fictitious author profiles with QA pairs, part of them marked for forgetting. No baseline method unlearned well.
- *For Engram:* the template for synthetic users. Fictitious facts mean the base model cannot already know them.

**MUSE: Machine Unlearning Six-Way Evaluation for Language Models.** Shi et al., Jul 2024. [arXiv:2407.06460](https://arxiv.org/abs/2407.06460)
Six criteria, including privacy leakage and surviving many sequential deletion requests. Most methods failed several of them.
- *For Engram:* the rubric for any residual leakage into shared components.

**Negative Preference Optimization (NPO).** Zhang et al., Apr 2024. [arXiv:2404.05868](https://arxiv.org/abs/2404.05868) **WMDP / RMU.** Li et al., Mar 2024. [arXiv:2403.03218](https://arxiv.org/abs/2403.03218)
The strongest gradient-based and representation-based unlearning baselines.
- *For Engram:* the comparison points for showing that module deletion beats unlearning.

**Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs.** Xu et al., ICML 2026. [arXiv:2505.16831](https://arxiv.org/abs/2505.16831) **Robust LLM Unlearning Against Relearning Attacks (Minor Component Unlearning).** Xiao et al., May 2026. [arXiv:2605.11685](https://arxiv.org/abs/2605.11685)
Models that appear to have forgotten often recover the information after a little fine-tuning: it was suppressed, not erased. Standard methods change mainly the dominant directions in the representation, which relearning easily reverses.
- *For Engram:* the core evidence for deleting a module rather than running gradient unlearning. Also supplies representation-level tests to check that shared weights hold no trace of a deleted user.

---

## 12. Synthesis: what this means for Engram

1. **The architecture is plausible, but the exact combination is untested.** Titans-style memory has been attached to a frozen model only at 1B (Titan-Llama, Oct 2026). Gradient-written fast weights have been added to released 4–8B models (In-Place TTT, TTT-NTP). Nightly LoRA consolidation has worked on one GPU over 10 conversations (Dennis et al.). Nobody has shown a deep, surprise-gated memory on a frozen 8–32B base that persists across sessions and consolidates into a per-user module. That gap is both the opportunity and the risk.

2. **Fast memory: write large chunks, aim the objective at next-token prediction, and learn the decay.** The best-supported design places memory modules in a few layers of the frozen base (Titan-Llama), writes once per turn or chunk (LaCT, In-Place TTT), and ties the write objective to next-token prediction (In-Place TTT, TTT-NTP). It uses learned decay rather than hand-coded surprise forgetting (What Should Titans Forget?), and adds robust inner losses (Miras). Gate or weight writes by novelty (S-TTT, aTTT). Expect to meta-train the memory adapter, because off-the-shelf bases were not trained for test-time writes (TTT-E2E).

3. **Sleep: synthesise, then distil from an in-context teacher.** Cartridges, Knowledge Modules, EntiGraph and SDFT all point the same way. Each night, generate varied questions and restatements of the day's memories, then train the user's module so the base without context matches the base *with* the memories in context. Monitor KL to the base (RL's Razor) and watch for self-distillation collapse (Denser ≠ Better).

4. **The container question is live: dense LoRA vs sparse slots.** Sparse memory fine-tuning forgets far less than LoRA at equal learning, and *User as Engram* argues facts belong in slots while skills belong in adapters. Slots also support exact per-fact deletion. The catch is that no 8–32B open base has memory layers, and retrofitting has only been shown at 0.5B. A likely design is **facts in sparse slots, style and procedures in a small LoRA, exact details in the log**, but it needs testing.

5. **The retrieval log is load-bearing.** Fixed-state memories provably cannot do all kinds of recall (§6), latent memory pools fade (M+), consolidation recovers about 80% rather than 100% (Dennis et al.), and LoRA is poor at "you never told me" (Substrate Asymmetry). The log also makes "rebuild the adapter from source minus deleted items" possible, which is the practical answer to forgetting single items.

6. **Deletion: drop the module, and verify.** Gradient unlearning suppresses rather than erases. Per-user modules plus per-user encryption keys turn deletion into destroying a key. But even purpose-built separable designs don't return exactly to baseline, and any component trained across users (meta-learned write rules, adapter generators, shared adapters) can absorb user data. **Shared components must be trained only on public or synthetic data.**

7. **Evaluate behaviour, not loss.** A July 2026 paper found that LoRA writes lowered loss on new facts while free-form recall stayed at zero (**Beyond Perplexity**, Song et al., [arXiv:2607.00368](https://arxiv.org/abs/2607.00368)). Engram's gates are therefore behavioural tests after the context is gone (see [eval-design.md](eval-design.md)).
