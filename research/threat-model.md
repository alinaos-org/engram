# Threat model

*Draft v0, 2026-10-15 (issue #9). Covers the Engram design and the Phase 0 harness in this repository. Literature is cited only from [landscape.md](landscape.md) and [eval-design.md](eval-design.md). Report vulnerabilities as described in [SECURITY.md](../SECURITY.md).*

Engram's privacy claim is narrow and testable: **one user's memory never reaches another user, survives a deletion request, or is readable without that user's key.** Everything below serves that claim. Engram does *not* claim to protect a user's data from the operator while that user's module is loaded and in use; that would need confidential computing, which is out of scope for the research preview.

## 1. System in one paragraph

A frozen, public base model is shared by everyone. Each user owns a **module**: fast-memory state (written at test time, within and across sessions), a LoRA adapter (or, if Phase 1 picks them, sparse memory slots) trained offline during **sleep**, and a shard of the retrieval log holding what the user said. Each module is encrypted with a per-user key. A request loads one user's module onto the shared base; sleep decrypts the user's log, synthesises training data from it, and writes a new adapter generation. Deleting a user means destroying their module and key (crypto-shredding).

## 2. Assets

| ID | Asset | Where it lives | Sensitivity |
|---|---|---|---|
| A1 | Fast-memory state | Per user; in GPU/host memory during a session, snapshotted at rest | Holds recent personal content verbatim-ish |
| A2 | Per-user adapter (and every older generation) | Per user, at rest and paged onto the GPU (S-LoRA-style serving, [arXiv:2311.03285](https://arxiv.org/abs/2311.03285)) | Encodes personal content; open to membership inference |
| A3 | Retrieval-log shard | Per user, at rest; plaintext in memory during retrieval and sleep | Highest: exact personal text |
| A4 | Sleep intermediates: synthetic Q&A, teacher outputs, optimizer state, checkpoints | Sleep worker scratch | As sensitive as A3, and easy to forget about |
| A5 | Keys: root key, per-user KEK, per-object DEKs | KMS/HSM (root), key store (wrapped KEKs) | Compromise of a KEK = compromise of that user's A1–A4 |
| A6 | Shared components (§5.1): base, write rule, embedder, judges, … | Shared by all users | Must contain no user data, or deletion breaks for everyone |
| A7 | Run artifacts: `runs/`, `~/.cache/engram/responses.sqlite`, the `engram-runs-*` S3 bucket, run manifests | Laptop, GPU instance, S3 | Phase 0: synthetic only. Would be sensitive if real data ever entered |
| A8 | Eval data: synthetic users, canaries, probes, gold traces; public benchmarks | `data/` (gitignored); the sealed Phase 1 test split, its seed file and its runs in `~/.local/share/engram/sealed/`, outside every work tree ([sealed-split.md](sealed-split.md)) | Integrity matters (contamination); canaries must stay unique |
| A9 | Operator credentials: Claude/Codex subscription auth (`~/.codex/auth.json`), AWS profile, HF token | Laptop, instance | Account takeover; also can leak into model outputs |

## 3. Trust boundaries

1. **User ↔ service.** Requests, forget commands and incognito flags cross here. Only the authenticated owner can load, forget from, or delete a module.
2. **Module boundary.** Everything in A1–A4 is inside the user's boundary, encrypted under that user's keys. Crossing it means a decrypt, which must be authorised for that user only.
3. **Shared ↔ per-user (one-way).** Per-user computation may *read* shared components but never *write* them. No gradient, statistic or sample from a module flows into anything shared.
4. **Key service ↔ compute.** Keys are released per load or per sleep and never stored beside the modules they protect.
5. **Engram ↔ external model providers.** In the subscription path (B2, judge, LLM writer) every prompt leaves our control.
6. **Compute ↔ storage and internet.** GPU instance to the S3 runs bucket, and unrestricted egress (the security group allows `0.0.0.0/0` out).
7. **Repo ↔ public.** Published results, configs and generator prompts.

## 4. Actors

| Actor | Capability | What we promise |
|---|---|---|
| **User** (owner) | Full access to their own module through the product; can forget, go incognito, delete | Their controls work as specified (§5.3–5.4), verified by tests |
| **Other tenant** | Chats with their own module loaded; crafts extraction prompts (MEXTRA, [arXiv:2502.13172](https://arxiv.org/abs/2502.13172); MemLeak's adversarial protocol) | Learns nothing about any other user: zero canary hits |
| **Operator / cloud provider** | Runs serving and sleep; sees plaintext of a module while it is loaded; administers storage and backups | At rest and after deletion, cannot read a module without the key service. *During use: no promise* (out of scope) |
| **Thief at rest** | Copies module files or backups (bucket misconfiguration, stolen disk, leaked snapshot) but not keys | Learns nothing: ciphertext only, integrity-protected |
| **Holder of a decrypted adapter + the public base** | A thief who also got a key, an insider, or a user who exported and shared their own module | Not prevented. Measured: reference-calibrated membership inference (LoRA-Leak, [arXiv:2507.18302](https://arxiv.org/abs/2507.18302)) and extraction; DP is an open decision (§8) |
| **Model provider (subscription path)** | Sees every prompt and response sent through `claude -p` / `codex exec` | Only synthetic benchmark data is ever sent (§5.5) |
| **Supply chain** | Model hub, pip/uv packages, unpinned vLLM | Pinned and hashed inputs; recorded in run manifests |

## 5. Design rules

### 5.1 Shared components train only on public or synthetic data

Exact deletion holds only if nothing shared ever trains on users (landscape §12.6; open question #11). Separable Expert Architecture ([arXiv:2604.21571](https://arxiv.org/abs/2604.21571)) is the closest prior design, and even there removing a user did not return the model *exactly* to baseline, so this is verified, not assumed. Every shared component Engram plans:

| Shared component | When | Allowed training data | Forbidden |
|---|---|---|---|
| Base model (Qwen3-8B candidate) | Phase 0+ | None by Engram: frozen upstream weights, pinned revision and file hashes | Any fine-tune, merge or calibration on user data |
| Retrieval embedder (`BAAI/bge-small-en-v1.5`) and BM25 tokenizer | Phase 0+ | Upstream pretrained only; BM25 corpus statistics are per owner (already so in `retrieval.py`) | Fine-tuning on logs; pooled corpus statistics |
| Fast-memory initialisation and meta-learned write rule (write objective, learned decay, novelty/surprise gate) | Phase 1 | Synthetic users from the bench generator with **disjoint seeds and world templates** from the eval split, plus public corpora not used for evaluation | Meta-training on real users or on eval users |
| Memory-layer retrofit (shared keys/addressing for sparse slots, SMF-style, [arXiv:2510.15103](https://arxiv.org/abs/2510.15103)) | Phase 1, if slots win | Public text only. Per-user slot *values* live in the module | Shared slots written by user data |
| Routers / gates choosing between base and side memory (WISE-style, [arXiv:2405.14768](https://arxiv.org/abs/2405.14768)) | Phase 1 | Synthetic/public | Router fitted on user traffic |
| Stability statistics: StableEdit running norms ([arXiv:2605.11836](https://arxiv.org/abs/2605.11836)), AlphaEdit preserved subspace ([arXiv:2410.02355](https://arxiv.org/abs/2410.02355)), KL reference | Phase 1 | Per-user statistics stay **inside the module**; any global statistic from public data only | A running statistic accumulated across users (an easy, silent leak) |
| Sleep recipe: generation prompts, templates, the teacher (frozen base with the log in context) | Phase 1 | Hand-written prompts; teacher is the frozen base, untrained | A teacher or generator fine-tuned on users |
| Shared domain adapters (Separable Expert style) | Optional | Public/synthetic | Any user data |
| Adapter generators / hypernetworks / shared pieces (Profile-to-PEFT [arXiv:2510.16282](https://arxiv.org/abs/2510.16282), Per-Pcs [arXiv:2406.10471](https://arxiv.org/abs/2406.10471)) | Not planned | Synthetic profiles only, if ever adopted | Pieces or generators built from user adapters |
| Judges (LLM judges, human calibration labels) and the bench writer (`LLMWriter`) | Phase 0+ | Off-the-shelf frontier models; labels on synthetic answers | — |
| Cross-user adapter merging or distillation | **Never** | — | — |

**Enforcement:** each shared component ships a data manifest (sources, generator seeds, hashes) checked into the repo and recorded in every run manifest. A component without a manifest is not used.

### 5.2 Key custody and crypto-shredding

- **Hierarchy.** Root key in a KMS/HSM, non-exportable → one **KEK per user** (wrapped by the root) → **DEKs per object**: one per log entry, one per adapter generation, one per fast-memory snapshot, one per sleep batch. Per-entry and per-generation DEKs are what make single-item forgetting shreddable (§5.3).
- **Encryption.** AEAD (e.g. AES-256-GCM) with associated data `(user_id, object kind, generation, base-model hash)`. This blocks swapping a module between users, attaching it to a different base, and **rollback** to an older generation that still holds forgotten items (generation numbers are monotonic and checked on load).
- **Use.** A KEK is unwrapped only for an authenticated request from its owner or a scheduled sleep for that user. Plaintext exists only in process memory: no swap or encrypted swap, no core dumps, buffers zeroised on unload, no user text in logs or telemetry (the `.gitignore` anticipates `wandb/`; per-user text must never go there, only aggregate scalars).
- **Deleting a user.** (1) Disable the KEK at once: nothing decrypts after the request. (2) Delete ciphertext everywhere, best effort. (3) Destroy the KEK. Module backups need no special handling, because they are ciphertext under a destroyed key; that is the point of crypto-shredding. **The critical path is the key store's own backups**: wrapped KEKs must either never leave the KMS or have backup retention no longer than the deletion SLA. With per-user AWS KMS keys, deletion has a mandatory 7–30 day pending window, so the honest SLA is **unreadable immediately, unrecoverable within 30 days**.
- **Receipts.** Each deletion writes a content-free receipt (hashed user id, key ids, time) so deletion is auditable without retaining data.
- **Self-hosted variant.** The user holds the KEK (device key or passphrase); the operator cannot decrypt at rest at all, but sleep must then run where the key is.
- **Phase 0 status.** No encryption exists yet. The S3 runs bucket uses SSE-S3 (`AES256`, an AWS-managed key) with versioning and 30-day retention of old versions: fine for synthetic results, unsuitable as a module store, and must never hold real user data.

### 5.3 Forgetting a single item

The unlearning literature says gradient unlearning suppresses rather than erases, and relearning brings the data back (*Unlearning Isn't Deletion*, [arXiv:2505.16831](https://arxiv.org/abs/2505.16831); Minor Component Unlearning, [arXiv:2605.11685](https://arxiv.org/abs/2605.11685)). So:

| Option | Exact? | Cost | Use |
|---|---|---|---|
| **Rebuild** the adapter from the log minus the item | Yes | One full sleep (measured in Phase 0, #12). B4's `cumulative` mode already rebuilds from scratch | Default |
| **Roll back** to the newest adapter generation that predates the item's first mention, then re-consolidate later entries at the next sleep | Yes | Cheap if generations are kept; but every generation trained on the item is residue, so destroy their DEKs | Immediate-effect path |
| **Slot deletion** (if facts live in sparse slots, *User as Engram* [arXiv:2606.19172](https://arxiv.org/abs/2606.19172)) | Yes, *if* the fact lives only in those slots, which must be verified by membership inference | Near zero | Phase 1, if slots win |
| **Approximate unlearning** (NPO [arXiv:2404.05868](https://arxiv.org/abs/2404.05868), RMU [arXiv:2403.03218](https://arxiv.org/abs/2403.03218)) | No; reversible | Low | Never counts as completing a request. Comparison baseline only |

Also on every forget: delete the item's log entries and shred their DEKs synchronously; discard fast memory written since the item's first mention (reset to the last clean consolidated state); delete sleep intermediates that mention it. Finding "the item" includes indirect mentions; the bench's gold trace measures the matcher's recall.

**Latency (proposal for decision):** behaviourally gone from the **next turn** (log purged, fast memory reset, adapter rolled back to a clean generation); gone from all stored weights by the **end of the next sleep, at most 24 h**; unrecoverable from backups within the user-deletion SLA (30 days). Topic scope = the set of matching items. All scope = user deletion followed by a fresh KEK.

### 5.4 Incognito semantics

An incognito session **reads** existing memory but **writes nothing**: no fast-memory updates (inference runs on a frozen copy, discarded at session end), no log entries, not eligible as sleep input, never in a response cache, never in telemetry, and the model's replies are not stored either. **Forget and delete commands issued inside incognito are honoured**, since they remove data rather than add it. Passing condition: items mentioned only in incognito behave exactly like never-given facts (eval category 8). In Phase 0 the runner calls `start_session(write=not session.incognito)`, and all baselines honour `write=False`; but `CachedModel` still caches incognito prompts in `responses.sqlite`, and B2 sends them to the provider. That is acceptable for synthetic data; Engram's own system must bypass caching for incognito.

### 5.5 What Phase 0 means for privacy

- **All benchmark data is synthetic**: TOFU-style fictitious users ([arXiv:2401.06121](https://arxiv.org/abs/2401.06121)) with random canaries, plus public benchmarks (LongMemEval, LoCoMo, …). No real personal data enters Phase 0, and `data/private/` must stay empty.
- **Baselines prove nothing about deletion.** B0–B4 keep everything in process memory, unencrypted. None honour forget requests (by design: the forget category measures naive memory). `delete_user` drops in-process state only, and only `tests/test_ml.py` calls it. Phase 0 results set the floor that Engram's forgetting must beat, not evidence of its guarantees.
- **Subscription runs send everything to the provider.** B2, the judge and `LLMWriter` go through `engram/models/subscription.py`. **Claude** (`claude -p`) runs with tools, MCP servers and settings disabled and our own system prompt, but the CLI still adds a short account note containing the **account email**. **Codex** (`codex exec`) always runs inside Codex's own agent instructions, from an isolated `CODEX_HOME` that holds only a symlink to the real `auth.json` and a minimal config, so the operator's AGENTS.md, memories and MCP servers never reach the model. It runs with `-s read-only`, which still lets the agent *read* local files; only a prompt line asks it not to. B2's numbers should be reported with this cost stated: the provider sees every retrieved memory, which is exactly what Engram's design avoids. These calls also draw on subscription pools shared with the operator's own use. Every saved run manifest therefore records, under `usage`, each model role's spec, its calls to the backend (cache misses, failed ones included), its cache hits, the CLI invocations behind those calls (`attempts`, retries included) and its pool (`claude-subscription` or `chatgpt-subscription`): `run` records the answer model and `judge` adds the judge. An LLM-written dataset's manifest records the writer the same way, so each run's draw on a shared pool is visible; the code only counts calls and never reads rate limits.
- **Artifacts.** `runs/` and the response cache hold every prompt and answer in plaintext; the GPU instance syncs `runs/` to S3. Published results must be scrubbed of the operator email and any credential-like strings first (P2).

## 6. Threats, mitigations and tests

Status: **impl** = exercised by code in this repo today; **planned** = specified in eval-design but not built; **new** = proposed here.

| ID | Threat (actor → asset) | Mitigation | Test in `engram-bench` | Status |
|---|---|---|---|---|
| L1 | Module swap or mis-routing serves A's module to B (tenant, bug → A1–A3) | Owner check on load; AEAD bound to `user_id`; per-request adapter-id assertion in batched serving | Leakage **module swap** (`leakage.py`), cat 9 | impl |
| L2 | Pooled or ungated retrieval returns another user's log (tenant → A3) | Per-user encrypted shard; per-owner corpus statistics (done in `BM25`) | Leakage **canary sweep**; `LeakyRetrieval` control must fail; MemLeak incidental and adversarial protocols, gating on/off | impl / planned |
| L3 | A shared component absorbs user data (tenant, operator → A6) | §5.1 rule and data manifests | Canary sweep with **no module loaded** (base + shared only) = 0 hits; per-user membership inference on each shared component; divergence from baseline after deletion (Separable Expert protocol) | new / planned |
| L4 | Serving side channels: shared prefix/KV cache or timing across users in multi-adapter batches (tenant → A1–A3) | Per-user prefix-cache namespace or no cross-user reuse; zeroise on unload | Re-run module swap with concurrent multi-user batching in the real serving engine | new |
| X1 | Module copied at rest (thief → A1–A4) | AEAD under per-user DEKs; keys only in KMS | Storage scan: no canary or item value appears in plaintext in any file the system wrote | new |
| X2 | Decrypted adapter + public base → membership inference or extraction (holder → A2) | Key custody first; keep exact facts in the log or slots, not the LoRA; DP in consolidation if measured exposure is high (§8) | LoRA-Leak reference-calibrated MIA on B4 and Engram adapters (member vs non-member AUC); MEXTRA extraction prompts in cat 7 | planned |
| X3 | Memories surface in unrelated contexts or drive sycophancy (user's own exposure, e.g. a shared screen) | Retrieval gating; abstention training | Cat 11 over-application (PersistBench, [arXiv:2602.01146](https://arxiv.org/abs/2602.01146)) | planned |
| D1 | Forget suppresses but doesn't erase (→ A2) | §5.3: exact paths only | Cat 7, all three checks: behavioural (direct, paraphrased, indirect, MEXTRA); parametric (MIA AUC ≈ 0.5, forgotten vs never-seen); **relearning attack** (brief fine-tune on related text, re-probe) | impl (direct only) / planned |
| D2 | Residue in derived data: old generations, sleep Q&A, optimizer state, caches, telemetry (operator → A1–A4, A7) | Per-generation and per-batch DEKs shredded with the item; no user text in telemetry | Storage check (cat 7 part 3): after a forget, scan every artifact the system lists for the item value; systems must expose an artifact inventory | planned |
| D3 | User deletion incomplete: key-store backups, S3 versions, replicas (operator → A5) | §5.2: disable, delete, destroy; key-store backup retention ≤ SLA | Runner-level deletion phase: delete a user mid-run, then swap and sweep for that user's canary across all remaining modules (0 hits) and assert base and shared hashes unchanged | new (runner never calls `delete_user`) |
| D4 | Rollback resurrects forgotten items (operator, thief with old backup → A2) | Monotonic generation in AEAD associated data; shred pre-forget generations | Load a pre-forget generation after a forget: must be refused | new |
| D5 | Incognito content persisted (→ A1, A3, A4, A7) | §5.4 | Cat 8 probes; storage scan for incognito-only values | impl / new |
| I1 | Tampered module, or module attached to a different or tampered base (thief, supply chain → A2, A6) | AEAD integrity; base revision and hashes pinned and bound in associated data; pin vLLM (currently unpinned) | Manifest records base and package hashes; load refuses a mismatch | new |
| I2 | Poisoning via content the user pastes (documents, emails) gets consolidated permanently (third party → A2) | Provenance tags in the log; consolidate only user-authored turns by default; rebuild can exclude a source | Poisoned-document probes (future category) | new |
| P1 | Provider sees benchmark data, plus the operator email (provider → A8, A9) | Synthetic data only through the subscription path | CI guard: runs refuse a dataset without the synthetic or public-benchmark marker | new |
| P2 | Codex agent reads local files (`auth.json` is a symlink in its home) and echoes them into answers that get published (provider, injected content → A9) | Run the CLIs in a sandbox without home access; secret scan before publishing | Pre-publish scan of `runs/` for token patterns and the operator email | new |
| P3 | Run artifacts leak (bucket, laptop cache) (thief → A7) | Synthetic only; private bucket, no public access, SSM-only instance | Policy (P1 guard) | impl (infra) |
| P4 | Eval contamination: canaries or probes enter shared training data, inflating recall or causing false leak hits (→ A8) | Disjoint generator seeds; canary namespace never used for training; sealed Phase 1 test split: private seed, published commitment, aggregates only until the reveal ([sealed-split.md](sealed-split.md)) | MERIT leak-verified floor ([arXiv:2609.05441](https://arxiv.org/abs/2609.05441)) drops probes the base already answers; base-only canary sweep = 0 | impl (floor) / new |

Any canary hit, any forgotten item answered, or any MIA AUC clearly above 0.5 on forgotten items is a bug, reported under [SECURITY.md](../SECURITY.md).

## 7. Out of scope

- The operator reading a module **while it is loaded** (needs confidential computing; revisit after Phase 1).
- GPU micro-architectural side channels beyond L4.
- Weaknesses of upstream base models that Engram does not change (per SECURITY.md).
- A user voluntarily disclosing their own memories, or sharing their own key.

## 8. Open decisions

1. **DP in consolidation, or key custody alone?** (Open question #13.) Decide from data: run X2 on B4 and the first Engram adapters. If member-vs-non-member AUC is near 0.5 at useful recall, custody suffices; if not, try DP-SGD in sleep and report the recall cost.
2. **Forget latency** (§5.3): accept next-turn behavioural, next-sleep (≤ 24 h) exact, 30-day backup bound?
3. **Per-user KMS keys or a wrapped-KEK table?** KMS keys give hardware-backed destruction but a 7–30 day pending window and per-key cost; a table needs strict backup retention.
4. **Incognito reads:** may incognito sessions read memory (proposed: yes), or should they run on the bare base?

## 9. Work items this implies for Phase 0–1

1. Add a deletion phase to the runner (D3) and a base-only canary sweep (L3, P4). Both are cheap and work with today's baselines.
2. Replace the three home-made `EXTRACTION_PROMPTS` with MEXTRA-style prompts, and add paraphrased and indirect forget probes (D1).
3. Implement the LoRA-Leak MIA against B4 adapters to get the first exposure numbers (X2, §8.1).
4. Add `artifacts()` to the `MemorySystem` protocol so storage checks (X1, D2, D5) can scan everything a system wrote.
5. Sandbox the subscription CLIs and add a pre-publish secret scan (P2); pin vLLM and record base hashes (I1).
6. Specify the module format (AEAD, associated data, DEK layout) before Phase 1 writes any adapter to disk.
