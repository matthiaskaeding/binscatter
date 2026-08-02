# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

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
# Representative sample across every module -- a fast confidence check
make qtest

# Fast tests (excludes PySpark)
make ftest

# Full test suite including PySpark
make test

# Run a single test
uv run pytest tests/test_binscatter.py::test_name -v

# Run tests for a specific backend
uv run pytest tests -k "polars"
```

PySpark tests are skipped by default. Use `--run-pyspark` flag to include them.

`make qtest` (`pytest tests --quick`) runs only tests marked `@pytest.mark.quick`,
and only on pandas and polars. It is the check to run while iterating: roughly
13s against 95s for the full suite, covering every test module. It is *not* a
substitute for `make ftest` before pushing -- it deliberately skips the duckdb,
dask and PySpark parametrizations, which is where backend-specific bugs live.

The `quick` marker means "representative of this module", not "important". Add one
when a new test file appears, or when a new area of behaviour is not reachable from
any currently marked test; do not mark a test simply because it is a good test.
Mark the test function rather than individual `pytest.param` entries -- `--quick`
cuts the backend axis itself, so a marked test keeps working when a backend is
added. `--quick` fails loudly if it selects nothing.

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
