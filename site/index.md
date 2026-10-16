Canonical: https://alinaos-org.github.io/engram/

Engram — memory inside the model                 

 Research preview · Phase 0 

# Engram

A language model that remembers you in its weights, not in a prompt.

Engram is an open research project to put long-term memory inside the network: a fast memory that learns while you talk, a nightly “sleep” that consolidates it into your own adapter, and a log for exact facts. Your memory lives in a module you can detach, encrypt, or delete.

 [View on GitHub](https://github.com/alinaos-org/engram) [Read the research](https://github.com/alinaos-org/engram/blob/main/research/landscape.md)  

 license: Apache-2.0 status: research preview base: open weights, 8–32B    

Why

## Retrieval remembers text. People expect a model to remember them.

Today’s assistants “remember” by pasting notes back into the prompt. That recalls quoted facts well, but it does not let the model learn how you think, what you’ve already decided, or which of your facts have changed. Engram asks whether memory written into the network can do better.

### Memory in the harness
  
- Retrieves snippets; the model itself never changes.
 
- Recall depends on the search query matching how you said it.
 
- Every remembered item spends context on every turn.
 
- Preferences and habits have to be written down to be used.

### Memory in the network
  
- The model learns from each session, so it adapts with use.
 
- Knowledge is used implicitly, without being retrieved first.
 
- Constant per-token cost, however long the history grows.
 
- A per-user module: drop it and the memory is gone.

How it works

## Two speeds of memory around a frozen model.

Inspired by how the brain pairs fast hippocampal learning with slow cortical consolidation. The base model never changes, so everything Engram learns about you sits in parts that belong to you.

-  

### Frozen base model

An open-weights model in the 8–32B range supplies language and world knowledge. Its weights are shared by everyone and never trained on any user, which keeps users isolated from each other.

-  

### Fast memory

A small neural memory, in the style of Titans and test-time training, is updated by gradient steps while the model reads. Writes are driven by surprise: what the model failed to predict is what it stores.

-  

### Sleep consolidation

Periodically, offline, what the fast memory has learned is distilled into a per-user LoRA adapter, using replay and self-generated rehearsal so new knowledge settles without overwriting the old.

-  

### Retrieval log

An append-only record of sessions acts as a backstop for exact details (numbers, names, dates) and as an audit trail of where each memory came from.

-  

### Detachable, per-user modules

Fast memory, adapter and log form one encrypted module per user. Forgetting someone means deleting their module rather than trying to scrub weights.

Roadmap

## Measure first, then build.

Each phase ends with a decision. Compute figures in the experiment plan are assumptions until measured.
  
-  Phase 0 · now 

### Evals and baselines
  
- Multi-session benchmark with context wiped between sessions
 
- Recall, updates, forgetting, cross-user leakage
 
- Frontier model + retrieval baselines

-  Phase 1 

### Memory module
  
- Attach a neural memory to a frozen base
 
- Continued pretraining on chunked documents
 
- First sleep-consolidation recipe

-  Phase 2 

### Synthetic users
  
- Simulated people over months-long timelines
 
- SFT, then RL on memory-dependent rewards
 
- Ablations of each memory tier

-  Phase 3 

### Private deployment
  
- Encrypted per-user modules on local hardware
 
- Deletion and audit tooling
 
- Small opt-in pilot

Research

## What we have read, and what we will test.

Every source in these notes was opened and checked. Summaries are our own words.

 [ research/landscape.md 

### Landscape

About 90 verified sources: test-time training, Titans, Nested Learning, memory layers, SSMs, continual learning, LoRA, editing and unlearning.
 ](https://github.com/alinaos-org/engram/blob/main/research/landscape.md) [ research/eval-design.md 

### Eval design

A draft benchmark for memory across sessions, with the context wiped each time.
 ](https://github.com/alinaos-org/engram/blob/main/research/eval-design.md) [ research/experiment-plan.md 

### Experiment plan

Phases 0–3 with hypotheses, decision gates and compute assumptions.
 ](https://github.com/alinaos-org/engram/blob/main/research/experiment-plan.md) [ research/open-questions.md 

### Open questions

What we don’t know yet, ranked by how much it could change the design.
 ](https://github.com/alinaos-org/engram/blob/main/research/open-questions.md)  

Where the evidence points (October 2026). Titans-style memory has just been retrofitted onto a frozen pretrained model, but only at 1B. Distilling rehearsal data from a teacher with the memories in context beats training on raw text. Sparse memory slots forget far less than LoRA. Gradient unlearning hides data rather than removing it, so dropping a per-user module is the stronger deletion. No one has yet combined these on a frozen 8–32B model across sessions. That gap is what Engram sets out to test.

Across about 100 sources, every citation was opened and checked against its source. See the [synthesis](https://github.com/alinaos-org/engram/blob/main/research/landscape.md#12-synthesis-what-this-means-for-engram).

Status

## Research preview. Benchmark code, no model weights yet.

Engram is at Phase 0. The repository includes the engram-bench synthetic-user generator, probes, scorers, baseline runners, and cross-user leakage checks. The neural-memory architecture remains proposed; no trained model weights are released. Follow the [current implementation and experiment status](https://github.com/alinaos-org/engram/blob/main/README.md#status).

 Landscape survey Eval design draft Benchmark generator Baseline runners Memory module prototype   

Contributing

## Help us find out if this works.

The most useful contributions now are careful reading, sharp questions, and small reproducible experiments. Never contribute real personal data; Engram uses synthetic users.

 [Phase 0 issues](https://github.com/alinaos-org/engram/issues?q=is%3Aopen+milestone%3A%22Phase+0%22) [Contributing guide](https://github.com/alinaos-org/engram/blob/main/CONTRIBUTING.md) [Report a privacy issue](https://github.com/alinaos-org/engram/blob/main/SECURITY.md)   

Citation

## Cite Engram

 @misc{engram2026, title = {Engram: Language Models with Memory in the Network}, author = {Mangano, Matthew and {Engram contributors}}, year = {2026}, howpublished = {\url{https://github.com/alinaos-org/engram}}, note = {Research preview} } Copy
