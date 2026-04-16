# Repository Guidelines

## Project Structure & Module Organization
`src/lerobot/` contains the Python package. Core areas include `robots/`, `teleoperators/`, `policies/`, `datasets/`, `envs/`, `processor/`, and `scripts/` for CLI entry points such as `lerobot-train` and `lerobot-eval`. Keep new library code under `src/lerobot/...`, examples in `examples/`, and user-facing docs in `docs/source/`. Tests live in `tests/` and generally mirror the feature or CLI they cover. Container definitions live in `docker/`; shared project config is centered in `pyproject.toml`, `Makefile`, and `.pre-commit-config.yaml`.

## Build, Test, and Development Commands
Use Python 3.10. CI uses `uv`, so prefer:

```bash
uv sync --extra test
uv sync --extra all
```

Use `--extra test` for the fast local environment and `--extra all` to match fuller CI coverage. Install hooks once with `pre-commit install`, then run `pre-commit run --all-files` before pushing. Run the main test suite with `uv run pytest tests -vv --maxfail=10`. For end-to-end coverage, run `uv run make test-end-to-end`. Build images with `make build-user` or `make build-internal`.

## Coding Style & Naming Conventions
Formatting is enforced by Ruff: 4-space indentation, double quotes, and a 110-character line limit. Use `snake_case` for modules, functions, and config fields; use `PascalCase` for classes. Let Ruff handle import ordering. Markdown and MDX are formatted through Prettier in pre-commit. Prefer type annotations in touched code; mypy is already stricter in modules such as `configs`, `envs`, `optim`, `model`, `cameras`, and `transport`.

## Testing Guidelines
Tests use `pytest`. Name files `tests/test_*.py` and keep reusable fixtures in `tests/conftest.py` or `tests/utils.py`. Add or update tests for every behavior change. There is no explicit coverage threshold in the repo, but `pytest-cov` is available in the `test` extra. Some suites depend on Git LFS artifacts, so run `git lfs install` and `git lfs pull` before the full suite. During development, narrow scope first, for example `uv run pytest tests/test_cli_peft.py -vv`.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects. Recent history mixes plain summaries with conventional prefixes, so `fix(scope): ...` or `feat(scope): ...` is preferred when helpful. Open PRs against `main`, rebase on `upstream/main`, and avoid working directly on `main`. Follow `.github/PULL_REQUEST_TEMPLATE.md`: explain motivation, list concrete changes, link related issues, and document how the change was tested. Run `pre-commit` and the relevant `pytest` commands before requesting review.
