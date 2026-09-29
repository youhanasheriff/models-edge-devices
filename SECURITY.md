# Security Policy

## Supported versions

This is a research and deployment repository without versioned releases. Only the
latest commit on the default branch (`main`) is maintained.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub's private vulnerability reporting instead: open the repository's
**Security** tab and choose **Report a vulnerability**. If that option is not
available, contact the repository owner through the email address on their GitHub
profile.

Please include:

- a description of the issue and its potential impact,
- the affected file(s) or component (for example `llm/`, `vision/`, `orchestration/`),
- steps to reproduce.

## Scope and handling of secrets

- Do not commit credentials, API tokens, or `.env` files. `.env*` files are
  git-ignored.
- If you find a committed secret, report it privately as described above so it
  can be rotated and removed from history.
- Model weights, datasets, and training outputs in this repository are research
  artifacts. Treat third-party weights and datasets according to their own licenses
  and terms.
