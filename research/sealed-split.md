# Sealed test split for `engram-bench` v1

*2026-10-24. Implemented in `engram/bench/seal.py`, with the changes to `dataset.py`, `runner.py` and `cli.py` listed under [Code changes](#code-changes), and tested in `tests/test_seal.py`; the commitment format below is normative. It applies to the Phase 1 test split in [phase-1-preregistration.md](phase-1-preregistration.md) and addresses assets A7–A8 and threat P4 in [threat-model.md](threat-model.md).*

The pilot published its seed and every per-probe answer, which was fine for a pilot. If the v1 test split were made the same way, anyone could regenerate it from the public manifest, and any future training corpus that crawls this repository could memorise its answers. Sealing the split changes four things:

- The seed is drawn at random and kept in a private file outside the repository.
- Before any run, the repository publishes a commitment to the seed and the hash of the dataset.
- Until the reveal, only aggregate scores are published.
- After the reveal, anyone can check that the split published then is the one committed to before the runs.

The 20-user dev split (the pre-registration's tuning split) stays public and unsealed, like the pilot.

## What would leak today

| Artefact | Written by | Holds | Public today? |
|---|---|---|---|
| Dataset `manifest.json` | `dataset.save` | Config **including the seed**, writer, content hash, counts | Yes (`results/pilot-2026-10-15/bench-manifest.json`) |
| `users.jsonl` | `dataset.save` | The whole split | No (`data/` is gitignored) |
| Run `manifest.json` | `runner.run` | A copy of the bench config, **seed included** | Yes |
| `results.jsonl` | `save_run` | Probe id, category and answer per probe. Answers repeat the gold values | Yes |
| `leakage.jsonl` | `save_run` | Probe questions, answers, and other users' leaked values and canaries | Yes |
| Calibration CSV | `calibrate export` | Question, gold, rejected values, answer | Wherever it is written |
| `summary.txt`, `report`, `compare`, `membership.json` | runner, CLI | Category and timing tables, leak counts, pooled AUCs | Yes, and safe |
| `~/.cache/engram/responses.sqlite` | `CachedModel` | Every model output: LLM-written sessions and every answer. Prompts are stored only as hashes | No |
| Corpus `contamination_report.json` | `engram-corpus` | Matched text from each reference set | Published with the data card |

`report`, `compare`, `summary.txt` and the tables that `run`, `judge` and `rescore` print hold only aggregates (checked in `runner.report`, `cli.cmd_report` and `cli.cmd_compare`). The `-v` progress lines print user ids only.

**The seed alone reveals the answers, even for LLM-written splits.** `generate` draws every random choice from one `random.Random(seed)`. The writers' model calls draw no random numbers: `LongWriter.write` draws its choices before it calls the model, and `LLMWriter.rewrite` uses none. So the seed and the public config reproduce every user's planted values, canaries, probes and gold answers. Only the wording of the sessions is lost. v1 uses the long writer, so its seed must stay private as well.

## Seed, salt and commitment

- **Seed and salt.** Each is 32 bytes from `secrets.token_hex(32)`, stored as 64 lowercase hex characters. The generator uses `random.Random(int(seed, 16))`. A 256-bit seed cannot be found by brute force against the public content hash, so publishing that hash reveals nothing. The salt keeps the commitment hiding whatever the seed is, and makes two splits with the same seed and spec commit differently.
- **Spec hash.** This is a hash of the manifest's config with the `seed` key removed, plus the writer name:
  ```
  spec_sha256 = sha256(json.dumps({"config": <manifest config without "seed">, "writer": <writer>},
                                  sort_keys=True, separators=(",", ":"), ensure_ascii=True))
  ```
  It hashes the config dict as stored in the manifest, not one rebuilt from `BenchConfig`. A field added to `BenchConfig` later then cannot break an old commitment.
- **Commitment.** The SHA-256, in lowercase hex, of the ASCII string below. Each hex field is exactly 64 characters, so the encoding is unambiguous:
  ```
  engram-seal-v1|<salt>|<seed>|<spec_sha256>
  ```

Worked example, for the default config with the template writer, salt `11…11` and seed `22…22` (each 64 characters):

| Step | Value |
|---|---|
| Spec preimage | `{"config":{"abstain_per_user":3,"application_probes":false,"items_per_user":10,"late_sleeps":3,"sessions":30,"sleep_every":3,"users":50},"writer":"template"}` |
| `spec_sha256` | `1a301dc710f5d2f013b3a53411a3304f4312bbf64abbb8f6d65f66afb8d8a26d` |
| `seed_commitment` | `82dc41127dcaa0e4a7aa6a1ed76c014c27edc176b2b3b95510ff7a92932268e1` |

The same check by hand, with no engram code (jq's sorted compact output matches Python's here because every string is ASCII):

```sh
spec=$(jq -jcS '{config: (.config | del(.seed)), writer}' manifest.json | sha256sum | cut -d' ' -f1)
printf '%s' "engram-seal-v1|$(jq -r .salt seed.json)|$(jq -r .seed seed.json)|$spec" | sha256sum
```

## Seed file

The file holds one JSON object with exactly these fields. It is written with mode `0600` and is never overwritten.

| Field | Value |
|---|---|
| `scheme` | `"engram-seal-v1"` |
| `seed` | 64 hex characters |
| `salt` | 64 hex characters |
| `spec_sha256` | 64 hex characters, as above |
| `seed_commitment` | 64 hex characters, as above |

The last two repeat the manifest. That lets `verify-seal` say "this seed file belongs to a different split" instead of failing with a bare hash mismatch.

`generate --seal` refuses a seed-file path in three cases, with no override:

- the path is inside any git work tree;
- the path is inside the dataset directory, which gets copied to run hosts;
- the file already exists.

Seed files are named `*.seed.json` by convention, and `.gitignore` ignores that pattern as a second guard. Reading a seed file is allowed anywhere, because at the reveal it lives in the repository.

Back up the seed file privately. If it is lost, the commitment can never be opened.

The dataset is never overwritten either. `generate --seal` refuses an `--out` that already holds a `manifest.json` or `users.jsonl`, before the seed is drawn, and plain `generate` refuses an `--out` whose manifest has a `sealed` block. The old seed could rebuild the plan of a lost split, but never its LLM-written sessions.

## Sealed dataset manifest

`config.seed` is present and **`null`**. Keeping the key leaves the config's keys equal to `BenchConfig`'s fields, so `BenchConfig(**manifest["config"])` still works. It also tells a reader the seed was withheld; a missing key would look like a manifest from before the field existed. A `sealed` object is appended:

```json
{
  "engram_version": "0.0.1",
  "config": {"users": 50, "sessions": 30, "seed": null, "sleep_every": 3, "...": "..."},
  "writer": "long:<writer model>",
  "sha256": "<sha256 of users.jsonl, unchanged>",
  "n_users": 50,
  "n_probes": 2900,
  "usage": {
    "writer": {"model": "<writer model>", "calls": 1500, "cache_hits": 0, "attempts": 1502,
               "pool": "claude-subscription", "fallbacks": 0, "failed_sessions": 0}
  },
  "sealed": {
    "scheme": "engram-seal-v1",
    "spec_sha256": "<hex>",
    "seed_commitment": "<hex>",
    "git_commit": "<HEAD of the engram checkout when generation started, or null>",
    "python": "<platform.python_version() at generation>"
  }
}
```

`usage` is written for the `llm` and `long` writers only. It counts the writer's model calls (see [threat-model.md](threat-model.md) §5.5) and is in neither `sha256` nor `spec_sha256`.

`git_commit` and `python` are informational and not part of the commitment. `git_commit` is read from the checkout that holds the engram code, not from the current directory (`GIT_*` variables such as `GIT_DIR` are ignored), before the first writer call, and ends in `-dirty` if that checkout had uncommitted changes. It is null when engram does not run from a git checkout. Regenerating a split at the reveal needs the same generator code. It also needs the same Python, because Python guarantees a stable sequence only for `random()`, not for `choice` or `randint`. Generate from a clean checkout.

`load(path, seed_file=None)` keeps checking the content hash. For a sealed manifest it also checks three things:

- `config.seed` is null;
- `scheme` is known;
- `spec_sha256` matches the stored config and writer. This catches a hand-edited config.

Given a seed file, it also checks the commitment. `run`, `judge`, `rescore` and `calibrate` never need the seed, so run hosts never get the seed file.

## Where sealed content may be written

**Rule:** per-probe content of a sealed split is never written inside a git work tree unless `--allow-in-repo` is passed. Gitignored directories (`data/`, `runs/`) count as inside. One rule is easier to check than a gitignore audit, and it also covers `git add -f`, edits to `.gitignore`, and tools that read the whole checkout.

A path is inside a work tree when it, or any parent of its resolved path, contains a `.git` entry (a directory, or the file that worktrees and submodules use). No `git` binary is needed, and a symlink from outside into a repository is caught. The guard does not expand `~`. The CLI expands it in the path options of these commands (shells leave `--out=~/x` as typed), so the guard checks exactly the path that is then written. A library caller's literal `~/x` is checked, and written, under the current directory.

The rule covers:

- the sealed dataset directory (`generate --seal --out`);
- `results.jsonl`, `leakage.jsonl` and `manifest.json` from `run`, `judge` and `rescore`;
- the `calibrate export` CSV.

`run` and `judge` check the output directory **before** the first model call, so a refusal never wastes a run. `save_run` checks again for library callers. `judge`, `rescore` and `calibrate` decide from the bench they load, not only from the run manifest, and `judge` and `rescore` copy its `sealed` block into the run manifest, so a run saved without one (by engram before sealed splits) is guarded too. Unsealed datasets are unaffected.

Suggested layout, outside every repository:

```
~/.local/share/engram/sealed/
  bench-v1-test.seed.json    # private until the reveal; never copied to run hosts
  bench-v1-test/             # manifest.json, users.jsonl
  runs/<system>/             # per-probe results
```

## Runs and published results

- **Run manifest.** For a sealed split, `bench.config.seed` is forced to `null` and `bench.sealed` is a copy of the dataset's `sealed` block. Unsealed run manifests keep their current `bench` block. Every run manifest, sealed or not, also gains the `usage` accounting (`{}` when no model is called). Judging a run saved before `usage` existed records `"answer": null` (not recorded) beside `judge`. Its `git_commit`, read the same way as the dataset's, is read when the run starts. `bench_sha256` is already recorded, and ties each run to the committed content.
- **Aggregate export.** `engram-bench report RUN [--floor B0RUN] --json PATH` writes one JSON object, safe to commit:
  - `manifest`: the run manifest;
  - `floor`: `null`, or `{"system_name", "bench_sha256", "dropped"}`;
  - `by_category` and `by_timing`: `asdict` of each `scoring.Summary`;
  - `leakage`: `{kind: {"checks": n, "leaks": n}}`.

  It holds no probe ids, questions, answers or values.
- **Hypothesis tests.** The H1–H5 cluster bootstraps need per-user data. They run locally on the sealed per-probe files, and only their statistics are published.

## Lifecycle

1. **Generate once**, from a clean checkout:
   ```
   engram-bench generate --seal ~/.local/share/engram/sealed/bench-v1-test.seed.json \
       --out ~/.local/share/engram/sealed/bench-v1-test --users 50 --sessions 30 \
       --writer long --writer-model <model> [other v1 options]
   engram-bench verify-seal ~/.local/share/engram/sealed/bench-v1-test \
       --seed-file ~/.local/share/engram/sealed/bench-v1-test.seed.json
   ```
   The seed file is written before the first writer call, so a long generation never loses it. If generation fails, delete the orphan seed file and start again. Nothing has been published yet. Back up the dataset directory privately as well: an LLM-written split cannot be regenerated.
2. **Commit.** Copy `manifest.json` to `results/bench-v1/test-manifest.json`, then commit and push it before any system runs on the split, smoke tests included. The date of that commit is the commitment time.
3. **Run.** Copy the dataset directory to the run host (not the seed file), and write runs outside the checkout. `scripts/aws/gpu.sh push-runs`/`pull` sync only the checkout's `runs/`, so sealed runs need their own sync to the private bucket (see follow-ups).
4. **Publish aggregates.** Write `report --json` files and the `compare` table into `results/bench-v1/test/`, and publish the H1–H5 statistics.
5. **Reveal.** Do this after the frozen run set is published: the pre-registered Phase 1 runs on this split and their H1–H5 results.
   - Commit the seed file as `results/bench-v1/test.seed.json`, with `git add -f`, since `*.seed.json` is ignored.
   - Publish `users.jsonl` as a release asset. It is tens of MB; its hash is already in git.
   - Per-probe results may follow.
   - From then on the split is public, and any later result on it is reported as not held out.
6. **Verify (anyone).** Run `engram-bench verify-seal <dir with manifest.json and users.jsonl> --seed-file test.seed.json`, or use the shell check above. Then check that each published summary's `manifest.bench_sha256` equals the committed `sha256` and that its `bench.sealed` matches. Finally, check in git history that the commitment commit predates every result commit.

Replacing a published commitment (say, a generator bug found before the frozen runs) is an amendment to the pre-registration, with a date and a reason. The old seed is revealed in the same commit, so the replacement cannot be read as cherry-picking.

## `verify-seal`

`engram-bench verify-seal PATH --seed-file SEED_FILE`. PATH is either a dataset directory (`manifest.json`, plus `users.jsonl` if present) or a bare manifest file. The command prints each line below as its check passes. On the first failure it raises `ValueError` (exit status 1), after the lines of the checks that passed, so a verifier can tell a seed file that does not open the commitment from content that does not match it.

| Check | When | Line printed |
|---|---|---|
| Sealed manifest: `seed` null, scheme known, spec hash matches config and writer | always | — |
| The seed file opens the commitment | always | `commitment: ok` |
| `users.jsonl` matches `sha256` (through `load`) | PATH is a directory with `users.jsonl` | `content sha256: ok` |
| Regenerate with the revealed seed; `n_users` and `n_probes` match | always | `regenerated counts: ok (<n> users, <m> probes)` |
| Regenerated `users.jsonl` hash equals `sha256` | writer `template` | `regenerated content: ok` |
| Every user except `sessions` (name, canary, items, abstain slots, probes, gold) equals the published one | writer `llm:*` or `long:*`, with `users.jsonl` | `regenerated plan: ok` |
| — | writer `llm:*` or `long:*`, no `users.jsonl` | `regenerated plan: not checked (no users.jsonl)` |

Regeneration uses `TemplateWriter()` for `template` and `llm:*`, and `LongWriter(NullModel())` for `long:*`. These draw the same random numbers as the original writers and call no model, so the check runs offline in seconds. On a mismatch, the error names the manifest's `git_commit` and `python`.

## Code changes

A small new module, `engram/bench/seal.py`, holds the scheme. The commitment format is public, so a verifier should find it in one short file that imports nothing from engram. The work-tree guard is used by `dataset`, `runner` and `cli` alike. Keeping both out of `dataset.py` also avoids an import cycle through `runner`.

| File | Change |
|---|---|
| `engram/bench/seal.py` (new) | Defines `SCHEME`, `PER_PROBE` (the refusal wording) and these functions: `spec_sha256(config, writer)`, `commitment(salt, seed, spec)`, `new_record(spec)`, `seed_int(record)`, `sealed_block(record)`, `write_seed_file(path, record, *, dataset_dir)`, `read_seed_file(path)`, `check_manifest(manifest)`, `verify(sealed, record)`, `work_tree(path)` and `refuse_in_repo(path, what, *, allow=False)`. Errors are `ValueError`, as in `dataset.load` |
| `engram/bench/dataset.py` | `BenchConfig.seed: int \| None`. `Benchmark.sealed: dict \| None = None` and `Benchmark.usage: dict \| None = None`. `generate` refuses `seed=None`. `save` writes `seed: null`, the `sealed` block and `usage`. `load(path, seed_file=None)` makes the checks above and reads `usage`. New `verify_seal(path, seed_file, report=None) -> list[str]`, which also passes each line to `report` as its check passes |
| `engram/bench/runner.py` | The run manifest nulls the seed, copies `sealed` and records `usage` (`cache` is kept for older readers). `_git_commit` now reads the engram checkout's HEAD, with `-dirty`, in every run manifest (it used the cwd's repository and a bare SHA), ignores `GIT_*` variables, and is read when the run starts. `save_run(..., *, allow_in_repo=False)` guards sealed runs. New `summary_json(manifest, scores, leaks, floor=None) -> dict` |
| `engram/cli.py` | The flags below, a `verify-seal` subcommand, `~` expansion in the path options of `generate`, `run`, `judge`, `calibrate`, `rescore` and `verify-seal`, and two new usage lines in the module docstring. `generate` never overwrites a sealed split. Usage accounting: `--no-cache` becomes `CachedModel(path=None)`, which stores nothing but counts calls; `generate` records the writer's `usage`, `judge` adds its own, and `calibrate score` prints the judge's |
| `engram/models/cache.py`, `engram/models/subscription.py` | Usage accounting: `CachedModel` counts calls under its lock and accepts `path=None`; new `usage(model)`; the CLI adapters count `attempts` (retries included) and name their `pool`. Documented in [threat-model.md](threat-model.md) §5.5 |
| `scripts/aws/gpu.sh`, `docs/aws-gpu.md` | A separate guard in the same change: a real `up` needs `--approval REF`, kept as the `Approval` tag on the instance and its volume (tested in `tests/test_aws_gpu.py` against a stub `aws`) |
| `.gitignore` | Adds `*.seed.json` |
| `tests/test_seal.py` (new) | The tests below. Usage accounting is tested in `tests/test_cli.py` and `tests/test_models_runner.py` |

| Command | New flag or behaviour |
|---|---|
| `generate` | `--seal SEED_FILE`, in an argparse mutually exclusive group with `--seed`; an empty SEED_FILE is an error, not an unsealed split. `--allow-in-repo` applies to `--out` only, never to the seed file |
| `run`, `judge`, `rescore`, `calibrate` | `--allow-in-repo`. It matters only for sealed splits: `run`'s `--out`, `judge`'s `--out` (or the run), `rescore`'s run directory, and the `calibrate export` CSV |
| `report` | `--json PATH`, the aggregate export |
| `verify-seal` | New: `PATH --seed-file SEED_FILE` |

## Tests (`tests/test_seal.py`, CPU only, no model or network)

- **Known answer.** `spec_sha256` and `commitment` reproduce the worked example above.
- **Sealed generate.** Run `generate --seal` (template writer, 3 users, 9 sessions). Then check:
  - the manifest has `config.seed is None` and a `sealed` block with exactly the five keys;
  - the seed file has exactly the five fields and mode `0600`;
  - neither the seed's hex nor its decimal form appears in any file under the dataset directory;
  - `load` works with and without the seed file.
- **Seed-file refusals.** The seed file is refused inside a fake work tree (`tmp/repo/.git` as a directory, and as a file), even with `--allow-in-repo`. It is also refused inside `--out`, when it already exists (the original stays unchanged), and through a symlink into the fake repo. In each case no dataset directory is created.
- **`--out` in a fake repo.** `--out` inside the fake repo is refused without `--allow-in-repo` and accepted with it. Unsealed `generate` and `run` inside the fake repo still work.
- **`~` paths.** From a cwd inside the fake repo, `--out=~/…` and `--seal=~/…` are written under `$HOME`, and nothing appears under `./~`. `save_run` given a literal `~/…` is refused there.
- **Flag errors.** `--seed 5 --seal …`, `--seed 0 --seal …` (Python 3.11 missed it while 0 was the default), `--seal ""` and an unexpandable `~user/…` exit with an argparse error.
- **No overwrite.** `generate`, sealed or not, refuses to write over a sealed split, and `--seal` refuses any existing dataset; the files and the would-be seed file are untouched.
- **`load` rejects:**
  - a sealed manifest with a non-null seed;
  - an edited config (spec mismatch);
  - an unknown scheme;
  - a seed file from another sealed split ("different split");
  - a seed file with an altered seed or salt (commitment);
  - a seed file with an extra field or another scheme;
  - a seed file passed for an unsealed dataset.

  `dataset.generate` rejects `seed=None`.
- **Sealed runs.** For `run --system oracle` on a sealed split:
  - the run manifest has `bench.config.seed is None` and `bench.sealed` equal to the dataset's block, and no file in the run directory contains the seed;
  - `run --out` inside the fake repo is refused with no run directory created and no model call made; it works with `--allow-in-repo`;
  - `rescore` and `judge` on an in-repo sealed run are refused without the flag, the judge before any model call, even when the run manifest has no `sealed` block;
  - `calibrate export` to an in-repo CSV is refused;
  - `save_run` called directly is refused too.
- **`report --json`.** It has exactly the five top-level keys and records the floor, with `dropped` checked against an independent count from a floor that solves gold probes. No probe id (`u\d{4}-p\d+`), canary or answer text from `results.jsonl` appears in it.
- **`verify-seal`:**
  - a template split passes all lines, both from the directory and from a copied bare manifest (no content line);
  - editing `users.jsonl` fails, and a contents mismatch is printed after the lines of the checks that passed;
  - `--writer long` and `--writer llm` splits written by a faked `claude:` model, whose session text differs from the offline regeneration, pass `regenerated plan: ok`;
  - changing one gold answer and updating `sha256` to match fails the plan check, while the commitment still passes.
- **Regression.** Existing tests pass unchanged: the pilot and dev flows are untouched.

## What sealing does not protect against

- **Local access.** Any process running as the operator can read the split, including agent CLIs. `codex exec` runs in a read-only sandbox, which can still read files. The places it can read:
  - the seed file;
  - the dataset and run directories;
  - `~/.cache/engram/responses.sqlite` (session text and every answer);
  - anything a model CLI logs locally.

  Real isolation needs a separate OS user (or host) that owns these files and runs the benchmark. That is outside this change.
- **Model providers.** Writing v1 with a hosted model sends every planted fact to that provider. Running B2 or the judge sends every probe and answer. Sealing keeps the split out of git, not away from a provider that keeps or trains on traffic. Phase 1's systems run on a local frozen base, but the writer and the judge may not.
- **Proof of randomness.** The commitment proves the split was fixed before the runs. It does not prove the seed was drawn only once. Someone could draw many seeds and keep a convenient split before committing. The defence is procedural: commit before any system runs on the split, under a protocol fixed in the pre-registration.
- **Deliberate publication.** `--allow-in-repo`, `git add -f` or a manual copy can still put sealed content in git. The guard stops accidents, not intent. It also does not detect a copy of the repository without `.git`, or a `GIT_DIR` setup.
- **Aggregates.** Category counts and accuracies are published. They reveal no content.
- **After the reveal.** The split is public, and future training corpora may contain it, canaries included. The held-out guarantee ends at the reveal. Phase 2 needs a new sealed split.

## Follow-ups (not in this change)

- **Corpus decontamination.** If `engram-corpus` decontaminates against the sealed split (`--bench-dir`), its `contamination_report.json` can quote matched text from the split. Until the reveal, treat that report as per-probe content, or redact matches from sealed sources.
- **GPU sync.** `scripts/aws/gpu.sh` needs an out-of-tree sync target for sealed runs.
- **CI guard.** A CI check could fail when a committed run directory has a `results.jsonl` next to a manifest whose `bench.sealed` is set.
