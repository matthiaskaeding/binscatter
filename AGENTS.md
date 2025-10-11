# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `binscatter/`, with `core.py` housing the dataframe‑agnostic `binscatter` implementation and helper utilities. Public exports are wired through `binscatter/__init__.py`.
- Automated checks reside under `tests/`, combining integration coverage (cross‑backend checks in `tests/test_binscatter.py`) and focused unit helpers (see `tests/test_fallback.py` and `tests/test_unit.py`).
- Usage samples and benchmarking scripts live in `examples/`; `examples/time_pyspark.py` demonstrates large‑scale Spark execution.

## Build, Test, and Development Commands
- Install dependencies for development: `uv pip install -e .[dev]`.
- When a command would normally use `python`, invoke it as `uv run python …` (e.g., `uv run python examples/demo.py`). Other tooling such as `pytest` continues to run directly.
- We manage common workflows through the `justfile`; run `just test` or `just lint` instead of remembering raw commands.
- Formatting and linting rely on `just lint`, which wraps `ruff format` and `ruff check --fix`; use it before reviews even though CI does not enforce it automatically.
- Manual style checks are editor‑driven; no repo‑managed formatter is enforced, so run `pytest` before submitting to guard functionality.

## Coding Style & Naming Conventions
- Python code follows PEP 8 defaults: 4‑space indentation, `snake_case` for functions/variables, `CamelCase` for classes, and module‑level constants in `UPPER_SNAKE_CASE`.
- Keep functions small and backend‑agnostic by routing backend‑specific logic through dedicated helpers (see `configure_quantile_handler` in `binscatter/core.py`).
- Prefer explicit imports from third‑party backends inside try/except blocks so optional dependencies fail gracefully.

## Testing Guidelines
- Primary framework is `pytest`; execute suites directly with `pytest` so failures surface quickly. Assertions lean on `numpy.testing` for numeric comparisons across distributed engines.
- When adding features, extend parametrized cases in `tests/test_binscatter.py` to cover every supported backend (`pandas`, `polars`, `duckdb`, `dask`, `pyspark`).
- Name new tests descriptively (`test_<behavior>`) and colocate fixtures alongside their usage.

## Commit & Pull Request Guidelines
- Follow the active Git history style: brief, imperative commit subjects (e.g., “Add dask”) with optional detail in the body.
- Before opening a PR, ensure `pytest` passes locally and mention which backends were exercised.
- Include concise summaries, reference related issues, and add screenshots or HTML links if visual outputs (Plotly renders) changed.

## Security & Configuration Tips
- Keep optional backends isolated by lazy importing; never assume availability of Spark, DuckDB, or Dask in core paths.
- For Spark work, set `SPARK_LOG_LEVEL=ERROR` (as in `examples/demo.py`) to keep logs manageable during tests and demos.
