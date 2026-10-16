# Contributing to Engram

Thanks for your interest. Engram is an early research project, so the most useful contributions right now are careful reading, sharp questions, and small, reproducible experiments.

## Ways to help

- **Research notes.** Extend [`research/landscape.md`](research/landscape.md) with a method we missed, or correct a summary we got wrong.
- **Evaluation.** Help build the multi-session memory benchmark described in [`research/eval-design.md`](research/eval-design.md): task generators, scorers, baselines.
- **Experiments.** Pick an open issue labelled `research`, `eval`, `architecture`, or `data`, and comment before you start so work isn't duplicated.
- **Open questions.** If you think something in [`research/open-questions.md`](research/open-questions.md) is already answered, show us where.

## Ground rules for research writing

1. **Cite only what you have opened.** Every paper, result, or number needs a link to a source you read. If you cannot verify a claim, leave it out.
2. **Paraphrase.** Summarise in your own words; quote sparingly and attribute.
3. **Separate evidence from opinion.** Mark speculation as such ("we expect", "untested").
4. **State assumptions.** Compute estimates, data sizes, and costs are assumptions until measured; say so.

## Ground rules for experiments

- Record the base model and revision, data source, seeds, hardware, and wall-clock time.
- Commit configs, not checkpoints. Large artifacts belong in external storage, linked from the issue.
- Report negative results. A clean "this doesn't work, and here is why" is a real contribution.

## Data and privacy

Engram is about models that remember people, so we hold a strict line:

- **Never commit real user data**, chat logs, or personal information, including your own. Use synthetic users.
- Never commit secrets, API keys, or tokens.
- Report privacy problems privately (see [SECURITY.md](SECURITY.md)), not in issues.

## Workflow

1. Fork the repo and create a branch from `main`.
2. Keep pull requests focused: one idea, one experiment, or one document per PR.
3. Fill in the pull request template, including how you checked the change.
4. A code owner reviews and merges.

## Code of conduct

Be kind and assume good faith. Critique ideas, not people. Maintainers may remove comments or contributors that make the project unwelcoming.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
