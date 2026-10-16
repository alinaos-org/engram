# Security and privacy policy

Engram builds models that store information about the people who use them. We treat privacy failures (one user's memory leaking to another, memory surviving a deletion request, memory modules readable without their key) as security vulnerabilities.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through GitHub's [private vulnerability reporting](https://github.com/alinaos-org/engram/security/advisories/new).

Include what you found, how to reproduce it, and the impact you expect. We aim to acknowledge reports within 7 days and to agree a disclosure timeline with you.

## In scope

- Cross-user leakage: a model or memory module revealing information from another user's memory.
- Deletion failures: information recoverable after a user's memory module was dropped or a forget request was honoured.
- Weaknesses in how memory modules are encrypted, stored, or attached.
- Extraction or membership-inference attacks against released adapters or memory modules.
- Vulnerabilities in code, workflows, or the GitHub Pages site in this repository.

## Out of scope

- Weaknesses of upstream base models that Engram does not change (report those upstream).
- Findings that need access the attacker should never have, such as the user's own decryption key.

## Supported versions

Engram is a research preview. Only the `main` branch is supported, and nothing here is production-ready.
