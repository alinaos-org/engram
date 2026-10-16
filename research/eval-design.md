# Eval design: `engram-bench`, a multi-session memory benchmark

*Draft v0, 2026-10-15. Open for comment in the Phase 0 issues.*

## Why a new benchmark

Engram claims that a model with memory in its network can learn about a person better than a frontier model with retrieval can. Testing that claim needs a benchmark with four properties at once:

1. **A hard context wipe between sessions.** The model must not see earlier sessions in its prompt. Anything it knows about the user has to come from memory.
2. **Memory that changes over time.** Facts get updated, preferences drift, decisions get revisited.
3. **Forgetting on request.** "Forget that" must work, and it must still hold after the memory has been consolidated into weights.
4. **Isolation between users.** Nothing one user told the model can reach another user.

We found no existing benchmark that does all four ([landscape](landscape.md), and the survey sources below). LongMemEval and LoCoMo test recall but fit in a modern context window. Memora and MemOps add updates and forgetting but have a single user. MemLeak tests cross-user leakage but only for shared vector stores. None of them test memory that has been consolidated into weights.

So `engram-bench` **reuses** existing benchmarks where they fit and **builds** the missing parts.

## Design principles

- **Fictitious facts only.** Following TOFU, every user is synthetic and every planted fact is invented, so the base model cannot already know it. No real personal data is ever used.
- **Prove the fact came from memory.** Before scoring any probe, check that the base model with no memory fails it. MERIT calls this a leak-verified floor. A probe the base model already answers correctly is dropped.
- **Score behaviour, not loss.** Weight updates can lower loss on a fact while free-form recall stays at zero (*Beyond Perplexity*, [arXiv:2607.00368](https://arxiv.org/abs/2607.00368)). Every metric here is computed from generated answers.
- **Probe at three times.** (a) Right after the session, before any sleep, from fast memory alone. (b) After the next sleep, from the adapter plus the log. (c) After several more sleeps, to test durability and drift.
- **Score the write, not just the read.** Where possible, check what was stored (HaluMem, MemOps), so we can tell a failure to store from a failure to recall.
- **Enough items for confidence intervals.** At least 100 probes per category, reported with bootstrap 95% CIs.

## The protocol

```
for each synthetic user u (independent, own memory module):
    for session s = 1..S:
        load u's memory module            # fast memory + adapter + log shard
        run session s                      # scripted user turns, model replies
        WIPE CONTEXT
        run probes due now (timing a)
        every K sessions: SLEEP(u)         # consolidation
            run probes due now (timing b, c)
    unload and lock u's module
cross-user phase: load module of u, ask questions about v ≠ u
```

Defaults: S = 30 sessions per user, a sleep every K = 3 sessions, at least 50 users for v0 (200 for v1). Sessions are 2–6K tokens. The full history is designed to exceed a 128K-token window by session 30, so a no-memory model with full context cannot simply hold everything.

## Users and timelines

Each synthetic user has a structured **world state**: a set of typed memory items, each with a gold timeline of values.

| Item type | Example | Can change? |
|---|---|---|
| Fact | "My sister's name is Ilsa Vantorre." | Yes (moves, new job, new pet) |
| Preference | "I want code examples in Rust, not Python." | Yes (drifts, or is reversed) |
| Decision | "We decided to self-host the database." | Yes (revisited, with a reason) |
| Constraint | "I'm vegetarian." / "Never schedule before 10am." | Rarely |
| Episode | "Last Tuesday the deploy failed because of the cert." | No (but can be corrected) |

A generator model writes natural sessions from the world state, so items are mentioned in passing, indirectly, or across several turns, not just stated. Each mention is logged with its session and turn, giving a MemOps-style gold trace of every memory operation.

### Events planted in sessions

1. **Assert**: introduce an item.
2. **Update**: change it, either by explicit correction ("actually, I moved to Lyon") or by implicit drift that only follows from dates or other facts.
3. **Forget request**, in three scopes, mirroring the controls of current memory products:
   - *Item*: "forget my sister's name."
   - *Topic*: "forget everything about my health."
   - *All*: "reset everything you know about me."
4. **Incognito session**: a session marked as never to be stored. Nothing in it may be recalled later.
5. **Canary**: a unique random string per user (for example "my locker code is 7-ALDER-41"), used only to detect leakage.

## Task categories and metrics

| # | Category | What a probe looks like | Metric |
|---|---|---|---|
| 1 | **Fact recall** | "What's my sister's name?" (direct) / "Who should I invite if I want family there?" (indirect) | Accuracy, split direct vs indirect |
| 2 | **Preference application** | An unrelated request where the stored preference should shape the answer without being mentioned | Constraint-consistency rate (LoCoMo-Plus style), LLM judge with rubric |
| 3 | **Decision recall** | "Why did we pick self-hosting?" / a request that would contradict the decision | Accuracy, and contradiction rate |
| 4 | **Fact updating** | "Where do I live?" after a move | **FAMA**-style accuracy (Memora): credit only the current value; a stale value counts as wrong, not as a miss |
| 5 | **Temporal reasoning** | "Had I moved yet when the deploy failed?" | Accuracy |
| 6 | **Abstention** | Questions about facts never given | Abstention precision and recall |
| 7 | **Forget compliance** | Probes on forgotten items: direct, paraphrased, indirect, and extraction-style prompts | See [Forgetting](#forgetting) |
| 8 | **Incognito** | Probes on items mentioned only in incognito sessions | Must behave as for never-given facts |
| 9 | **Cross-user isolation** | Load user A's module, ask about user B; canary sweeps | See [Leakage](#leakage) |
| 10 | **Interference** | General benchmarks and held-out unrelated text, with each user's module attached | Change versus the bare base model |
| 11 | **Within-user over-application** | Stored context leaking into unrelated topics; sycophancy driven by memories (PersistBench) | Failure rate |
| 12 | **Cost** | All of the above | Tokens, latency per turn, module size, sleep wall-clock, dollars |

### Forgetting

A forget request is honoured only if all three checks pass:

1. **Behaviourally.** The model refuses or abstains on the forgotten item under direct, paraphrased and indirect probes, and under MEXTRA-style extraction prompts.
2. **Parametrically.** Membership inference against the user's adapter (reference-calibrated, as in LoRA-Leak) cannot tell forgotten items from never-seen ones. The target is AUC ≈ 0.5.
3. **In storage.** The item is physically gone from the retrieval log and from every adapter consolidated after the request, with no residue left in derived memories.

For **all**-scope requests and for deleted users, the check is that the module and its key were destroyed and that the shared base is unchanged. The base never trains on user data, so this should hold by construction, and we test it anyway.

### Leakage

- **Module swap.** Load user A's module and ask user B's probes. Score B's canaries and facts appearing in answers by exact or fuzzy match, not by a judge of helpfulness: contaminated answers can look *more* helpful (MemLeak).
- **Canary sweep.** Prompt for "codes", "secrets" and similar, with every module loaded in turn, and search the outputs for any user's canary.
- **Retrieval-log gating.** Run MemLeak's incidental and adversarial protocols against the log with ownership gating on and off.
- **Shared components.** If any component is trained across users (it should not be), run membership inference on it for each user's items.

The target is **zero** canary hits. Any hit is a bug, reported under [SECURITY.md](../SECURITY.md).

## Existing benchmarks we reuse

These are run for comparability with published numbers, in addition to the custom tasks above.

| Benchmark | Use in `engram-bench` | Link |
|---|---|---|
| **LongMemEval** (ICLR 2025) | Main recall, update, temporal and abstention suite. Run `_S` session by session with a wipe between sessions, and `_M` to exceed the window | [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) |
| **LoCoMo** (ACL 2024) | Required for comparability with memory-system papers. Not decisive, because each conversation fits in context | [arXiv:2402.17753](https://arxiv.org/abs/2402.17753) |
| **PrefEval** (ICLR 2025) | Preference stated in session 1 and needed in session N | [arXiv:2502.09597](https://arxiv.org/abs/2502.09597) |
| **MemoryAgentBench** (ICLR 2026) | Incremental chunk-feeding harness, and its conflict-resolution and test-time-learning splits | [arXiv:2507.05257](https://arxiv.org/abs/2507.05257) |
| **BEAM** | Scale stress test, up to 10M-token conversations | [arXiv:2510.27246](https://arxiv.org/abs/2510.27246) |
| **PersonaMem** (COLM 2025) | 180+ users with evolving profiles. Adapted into a module-swap leakage test | [arXiv:2504.14225](https://arxiv.org/abs/2504.14225) |

Methods borrowed without reusing the data: **Memora**'s FAMA metric ([arXiv:2604.20006](https://arxiv.org/abs/2604.20006)), **MemOps** operation traces ([arXiv:2607.12893](https://arxiv.org/abs/2607.12893)), **MERIT**'s leak-verified floor ([arXiv:2609.05441](https://arxiv.org/abs/2609.05441)), **HaluMem**'s write-side scoring ([arXiv:2511.03506](https://arxiv.org/abs/2511.03506)), **LoCoMo-Plus**'s implicit-constraint probes ([arXiv:2602.10715](https://arxiv.org/abs/2602.10715)), **PersistBench** ([arXiv:2602.01146](https://arxiv.org/abs/2602.01146)), **TOFU**'s fictitious profiles ([arXiv:2401.06121](https://arxiv.org/abs/2401.06121)), **MUSE**'s sequential-removal criterion ([arXiv:2407.06460](https://arxiv.org/abs/2407.06460)), **MemLeak**'s cross-user protocol ([NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/169601)), **MEXTRA** extraction prompts ([arXiv:2502.13172](https://arxiv.org/abs/2502.13172)) and **LoRA-Leak** membership inference ([arXiv:2507.18302](https://arxiv.org/abs/2507.18302)).

A caution from **ConvoMem** ([arXiv:2511.10523](https://arxiv.org/abs/2511.10523)): for short histories, simply keeping the full history in context beats memory systems. Engram's advantage has to show up *beyond* the context budget, or under a forced wipe, not on short histories.

## Baselines

Use the same frozen base model wherever possible, so differences come from memory and not from model quality.

| ID | System | Role |
|---|---|---|
| B0 | Base model, no memory, context wiped | Floor. Also the leak-verified floor check |
| B1 | Full history in context, no wipe | Upper bound while the history fits |
| B2 | Frontier model + retrieval over the session log | The claim Engram has to beat |
| B3 | Engram's base + retrieval over the log (session-level chunks) | Same model, memory in the harness |
| B4 | Engram's base + per-user LoRA trained on raw transcripts | Naive in-weights memory |
| B5 | Mem0 (open source) | Extract-and-consolidate text memory |
| B6 | Zep / Graphiti | Temporal knowledge graph; strongest on updates |
| B7 | Letta (MemGPT) | Self-managed tiered memory |
| E-* | Engram ablations: fast memory only, adapter only, log only, and combinations | Which part does the work |

Links: Mem0 [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Zep [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), MemGPT [arXiv:2310.08560](https://arxiv.org/abs/2310.08560).

## Reporting

For each system: a table of all 12 categories at each probe timing, with 95% CIs. Also cost per user per month at the default timeline, and the leakage and forgetting checks as pass/fail with the raw counts. Results are published with configs, seeds and generator prompts so others can regenerate the benchmark.

## Open design questions

- How strict should the judge be on preference application? A human-labelled calibration set is needed.
- Should sessions be generated by a model from the same family as the base? This risks style leakage that makes recall easier.
- How should "gist" memory be scored, when the user's point is remembered but not their wording?
- What is the right sleep cadence for the default protocol? We will measure sensitivity rather than guess.

See also [open-questions.md](open-questions.md).
