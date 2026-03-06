# Security Policy

## Reporting a vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

Instead, report privately to the maintainers with:

- affected component/file,
- reproduction steps or proof-of-concept,
- potential impact,
- suggested mitigation (if available).

If a dedicated security email is not yet published, open a private communication channel with the corresponding author listed in `CITATION.cff`.

## Response targets

Best-effort targets for maintainers:

- initial acknowledgment: within 7 days,
- triage decision: within 14 days,
- fix timeline: depends on severity and reproducibility.

## Scope notes

This is a research repository. Main risk areas include:

- accidental credential leaks in scripts/configs,
- unsafe execution paths in code evaluation tooling,
- dependency vulnerabilities in pinned environments.

Please include exact environment information (OS, Python, dependency versions, GPU/runtime stack) in your report.
