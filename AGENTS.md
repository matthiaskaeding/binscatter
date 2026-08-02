# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `src/binscatter/`, with `core.py` housing the dataframe-agnostic `binscatter` implementation and helper utilities. Public exports are wired through `binscatter/__init__.py`.
- Automated checks reside under `tests/`, combining integration coverage (cross-backend checks in `tests/test_binscatter.py`) and focused unit helpers.
- Usage samples and benchmarking scripts live in `examples/`; `examples/time_pyspark.py` demonstrates large-scale Spark execution.

## Build, Test, and Development Commands

```bash
# Install for development
uv pip install -e .[dev]

# Run scripts (use uv run instead of python directly)
uv run python examples/demo.py

# Linting and formatting
make lint                    # runs ruff format + ruff check --fix

# Type checking
make ty                      # runs ty check src

# Both lint and type check
make ok
```

## Testing

```bash
# Representative sample -- check with this first, while working
make test-fast

# Whole suite, PySpark skipped -- the gate, run before committing
make test

# Whole suite including PySpark
make test-spark

# Run a single test
uv run pytest tests/test_binscatter.py::test_name -v

# Run tests for a specific backend
uv run pytest tests -k "polars"
```

**Always check with `make test-fast` while iterating.** It is the representative
sample -- tests marked `@pytest.mark.quick` on the exact backends only -- and takes
about 18 seconds, against roughly two minutes for `make test` and considerably
longer for `make test-spark`, which starts a JVM and reruns the whole backend matrix
through it. Run `make test` before committing; leave `make test-spark` for changes
that touch the PySpark paths.

The sample is meant to hit each concern once: agreement with `binsreg`, control
partialing, dummy encoding across backends, fixed-effect absorption, the inference
intervals, quantile edge cases and the input-validation errors. Mark a new test
`quick` when it covers a concern nothing else in the sample covers, not merely
because it is fast. `--strict-markers` is on, so a misspelled marker fails rather
than silently selecting nothing.

PySpark tests are skipped by default. Use `--run-pyspark` flag to include them.

`tests/test_fe_contract.py` is the portable half of the fixed-effect tests: what any
implementation must satisfy, judged only through `binscatter()` against `binsreg` —
the authors' own package, which takes fixed effects as dummy columns in `w` and has
no notion of absorbing anything. It imports nothing private, patches nothing, and
asserts nothing about which route ran, so it stays valid when the fixed-effect
machinery is replaced; it is verified against three implementations (as shipped,
absorption disabled, absorption forced everywhere) and only the cost test tells them
apart. `test_fixed_effects.py` is the other kind: it forces this design's two routes
and compares them, and would need rewriting alongside a new one. Put a new
fixed-effect guarantee in the contract file unless it is genuinely about how this
implementation works.

## Architecture

### Core Flow

The main `binscatter()` function in `src/binscatter/core.py` orchestrates the entire pipeline:

1. **Input normalization**: Converts any supported dataframe to a narwhals LazyFrame via `clean_df()`
2. **Quantile computation**: Backend-specific quantile calculation via `configure_quantile_computer()` in `quantiles.py`
3. **Bin assignment**: Backend-specific bin assignment via `configure_add_bins()` in `quantiles.py`
4. **Aggregation**: Either simple bin means (`compute_bin_means()`) or control partialing (`partial_out_controls()`)
5. **Output**: Returns either a Plotly figure or native dataframe

### Backend Strategy Pattern

The codebase uses factory functions that return backend-specific implementations based on `narwhals.Implementation`:

- `configure_quantile_computer()` → Returns a function that computes quantiles for the specific backend
- `configure_add_bins()` → Returns a function that assigns bin labels for the specific backend

Supported backends: pandas, polars, duckdb, dask, pyspark. Unsupported backends fall back to generic narwhals operations.

### Key Data Structures

- `Profile` (NamedTuple in core.py): Carries configuration through the pipeline (bin count, column names, regression features, etc.)
- `QuantileCollection` (dataclass in quantiles.py): Holds computed quantile edges and max feasible bins

### Control Partialing

When controls are specified, `partial_out_controls()` implements the Cattaneo et al. (2024) method:
- Builds normal equations from bin-level aggregates
- Solves for bin effects and control coefficients jointly
- Avoids materializing per-row residuals

## Testing Conventions

Tests are parametrized across backends using `@pytest.fixture` from `conftest.py`. The `convert_to_backend()` helper converts pandas DataFrames to each backend type. When adding features, extend parametrized cases in `tests/test_binscatter.py` to cover all backends. Use `numpy.testing` for numeric comparisons across distributed engines.

## Coding Conventions

- **Run `make ok` before committing** to ensure code is formatted and type-checked
- Use lazy imports (try/except blocks) for optional backend dependencies—never assume Spark, DuckDB, or Dask availability in core paths
- Commit style: brief, imperative subjects (e.g., "Add dask support")
- For Spark work, set `SPARK_LOG_LEVEL=ERROR` to reduce log noise

## Example Notebooks

**IMPORTANT: When adding new features that affect plots or user-facing behavior, update the example notebooks.**

The repository maintains comprehensive example notebooks in `examples/` to showcase all library features:

- `examples/demo.ipynb` - Demonstrates all binscatter features with readily available datasets

### When to Update Examples

Update the example notebooks when you:
- Add new parameters to the `binscatter()` function
- Implement new plot customization options
- Add support for new data types or backends
- Change the visual output or plotting behavior
- Add new automatic bin selection methods

### How to Update

1. Add a new cell demonstrating the feature with clear explanatory text
2. Use readily available datasets (preferably from `plotly.express.data`)
3. Test that the entire notebook executes without errors: `make make-nb`. The rendered copy is written under `artifacts/notebooks/`.
4. Include the output-free notebook source in the same PR as the feature. Do not commit rendered outputs; after the PR reaches `main`, GitHub Actions publishes them to the orphan `notebooks` branch.

## Commit & Pull Request Guidelines

- Follow the active Git history style: brief, imperative commit subjects (e.g., "Add dask") with optional detail in the body.
- Before opening a PR, ensure `make ok` and tests pass locally.
- Include concise summaries, reference related issues, and add screenshots or HTML links if visual outputs (Plotly renders) changed.
- **Update `CHANGELOG.md` in the same commit or PR as the change** (see Changelog section below).
- **When adding new user-facing features (especially those that change plots), update the example notebooks** (see Example Notebooks section above).

## Changelog

**IMPORTANT: Update `CHANGELOG.md` on every branch, in the same commit or PR as the change itself.**

Add entries under `## Unreleased`, in the appropriate subsection — `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`. Create the subsection if it is not there yet.

This applies to more than library features. Anything a user or contributor would be surprised to discover on their own belongs here:

- New or changed behaviour in `binscatter()` and its helpers
- Bug fixes, including ones only reachable on a specific backend
- Build, packaging and release changes (a new publish workflow, dropping a lockfile)
- Tooling changes that alter what is enforced (a linter upgrade that changes the rules a contributor must satisfy)

Purely internal churn — a refactor with no observable effect, a test-only rename — does not need an entry.

Write entries so they still make sense months later, without the PR open next to them: say what changed and why it matters, not just which symbol moved. Match the level of detail in the existing sections.

Do not defer this to release time. An empty `## Unreleased` section under a series of merged behaviour changes means the history was lost, and reconstructing it from the commit log afterwards is guesswork.
