# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install for development
uv pip install -e .[dev]

# Run scripts (use uv run instead of python directly)
uv run python examples/demo.py

# Linting and formatting
just lint                    # runs ruff format + ruff check --fix

# Type checking
just ty                      # runs ty check src

# Both lint and type check
just ok
```

## Testing

```bash
# Fast tests (excludes PySpark)
just ftest

# Full test suite including PySpark
just test

# Run a single test
uv run pytest tests/test_binscatter.py::test_name -v

# Run tests for a specific backend
uv run pytest tests -k "polars"
```

PySpark tests are skipped by default. Use `--run-pyspark` flag to include them.

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

- **Run `just ok` before committing** to ensure code is formatted and type-checked
- Use lazy imports (try/except blocks) for optional backend dependencies—never assume Spark, DuckDB, or Dask availability in core paths
- Commit style: brief, imperative subjects (e.g., "Add dask support")
- For Spark work, set `SPARK_LOG_LEVEL=ERROR` to reduce log noise

## Planning

Store the active plan in `PLAN.md`. Keep the current feature plan at the top; archive completed plans below a horizontal rule.
