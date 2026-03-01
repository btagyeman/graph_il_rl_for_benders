# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Standards

- Keep modules import-safe; use `main()` for executable scripts.
- Avoid hardcoded absolute paths.
- Prefer small functions with explicit inputs/outputs.
- Add only concise comments that clarify non-obvious logic.
- Run `make lint` and `make test` before opening a PR.

## Pull Requests

- Keep changes scoped to one concern.
- Document behavior changes in the PR description.
- If training or solver behavior changes, include before/after metrics.
