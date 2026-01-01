# PR Review Issues: feature/perf-investigation

## Critical Issues (Must Fix Before Merge)

### 1. Code Duplication in `maybe_add_regression_features`

**Location:** `src/binscatter/core.py`

**Description:** There are two implementations of the categorical dummy variable creation logic:

1. **Old implementation** (lines 930-953): Uses narwhals generic approach with `df.select(c).unique().collect()`
2. **New implementation** (lines 970-1076): Uses backend-specific dummy builders via `configure_dummy_builder()`

**Problem:**
- Code duplication leads to maintenance burden
- Risk of divergence between the two implementations
- Unclear which code path is actually executed
- The old implementation still has the performance issues that the PR aims to fix

**Action Required:**
- [ ] Determine which implementation is active
- [ ] Remove the inactive implementation completely
- [ ] If both are somehow used, consolidate into a single code path
- [ ] Add comments explaining the removal

**Files to modify:**
- `src/binscatter/core.py`

---

### 2. Missing Test Coverage

**Description:** The PR introduces significant new functionality but no test file changes were included.

**Missing tests:**

#### 2.1 PySpark Caching Behavior
- [ ] Test that `_maybe_cache_pyspark` actually caches the dataframe
- [ ] Test that unpersist is called on context exit
- [ ] Test that caching is a no-op for non-PySpark backends
- [ ] Test behavior when caching fails

**Test file:** `tests/test_performance.py` (created) or `tests/test_binscatter.py`

**Implementation notes:**
```python
def test_pyspark_caching_cleanup():
    """Verify that PySpark cache is properly cleaned up."""
    # Check SparkContext storage levels before/after
    # Verify no dangling cached RDDs
```

#### 2.2 Dummy Variable Naming Consistency
- [ ] Test that `_format_dummy_alias` produces consistent names across backends
- [ ] Test handling of special characters in category values (spaces, slashes, etc.)
- [ ] Test that PySpark, pandas, and polars produce identical dummy column names
- [ ] Test very long category names (edge case)

**Test file:** `tests/test_performance.py` (partially added)

#### 2.3 PySpark Dummy Builder Edge Cases
- [ ] Test with null values in categorical columns
- [ ] Test with high-cardinality categorical (1000+ unique values)
- [ ] Test with categorical column that has only one unique value
- [ ] Test with empty categorical column
- [ ] Test with categorical values containing underscores (potential name collision)

**Test file:** `tests/test_binscatter.py`

#### 2.4 Backend-Specific Dummy Builders
- [ ] Add parametrized tests covering all three dummy builder implementations
- [ ] Test that pandas builder uses `pd.get_dummies` correctly
- [ ] Test that polars builder uses `to_dummies` correctly
- [ ] Test that fallback builder works for unsupported backends

**Test file:** `tests/test_binscatter.py`

#### 2.5 Performance Regression Tests
- [ ] Baseline tests to ensure optimizations persist over time
- [ ] Tests that verify PySpark doesn't fall back to slow path
- [ ] Tests for memory usage (ensure caching doesn't OOM)

**Test file:** `tests/test_performance.py` (created)

---

### 3. Performance Claims Need Validation

**Description:** The PR claims significant speedups but these are not verifiable in the code.

**Claims from PLAN.md:**
- `maybe_add_regression_features`: 2.15s → 0.38s (5.7x faster)
- `_fit_polynomial_line`: 1.14s → 0.71s (1.6x faster)

**Questions to address:**
- [ ] What dataset size were these measurements taken on?
- [ ] Were these measured with caching alone, dummy builder alone, or both combined?
- [ ] What's the memory overhead of caching? (especially for large datasets)
- [ ] Do the improvements hold for different data sizes (10k, 100k, 1M, 10M rows)?
- [ ] What's the breakeven point where caching helps vs hurts?

**Action Required:**
- [ ] Document the benchmark methodology in `docs/perf_notes.md`
- [ ] Add dataset size and hardware specs to performance claims
- [ ] Create reproducible benchmark script (or enhance `scripts/debug_binscatter.py`)
- [ ] Test with multiple dataset sizes to understand scaling behavior

**Files to modify:**
- `docs/perf_notes.md`
- `scripts/debug_binscatter.py`
- New file: `scripts/run_benchmarks.py` (optional)

---

## Major Issues (Should Fix Before Merge)

### 4. PySpark Dummy Builder Scalability Concerns

**Location:** `src/binscatter/core.py:1049`

**Code:**
```python
distinct_row = native.agg(*agg_exprs).collect()[0]
```

**Description:**
The optimized PySpark implementation still collects all unique category values to the driver. While this is better than N separate scans, it has scalability limitations:

**Problems:**
- High-cardinality categoricals (e.g., user IDs, product SKUs) could cause driver OOM
- Doesn't leverage Spark's distributed nature for dummy variable creation
- All category values must fit in driver memory

**Alternatives to consider:**
- Use `pyspark.ml.feature.StringIndexer` + `OneHotEncoder` (stays distributed)
- Use Spark SQL's `pivot` operation
- Add cardinality check and warning before collecting

**Action Required:**
- [ ] Document cardinality limitations (e.g., "supports up to 10,000 unique values per categorical")
- [ ] Add validation check and raise informative error for high-cardinality categoricals
- [ ] Consider adding alternative implementation for high-cardinality case
- [ ] Add test case with high-cardinality categorical

**Suggested validation:**
```python
if len(categories) > 10_000:
    raise ValueError(
        f"Categorical column '{column}' has {len(categories)} unique values. "
        f"PySpark backend supports up to 10,000 levels per categorical. "
        f"Consider using StringIndexer or reducing cardinality."
    )
```

**Files to modify:**
- `src/binscatter/core.py` (add validation)
- `tests/test_binscatter.py` (add test)
- `docs/perf_notes.md` (document limitation)

---

### 5. Caching Design: Automatic vs. Opt-in

**Location:** `src/binscatter/core.py:195` (wraps entire pipeline)

**Description:**
The PR implements automatic caching for all PySpark operations. This has trade-offs:

**Pros:**
- Transparent to users
- Simplifies the API
- Works great for small-to-medium datasets

**Cons:**
- May cause memory pressure with large datasets (10M+ rows, 100+ columns)
- Users have no control over caching behavior
- Could cause OOM errors in memory-constrained environments
- Caches even when data is accessed only once (wasted overhead)

**Alternatives:**
1. **Make caching opt-in:** Add `cache=True` parameter to `binscatter()`
2. **Smart caching:** Only cache if multiple operations are detected (e.g., controls + poly_line)
3. **Configurable cache level:** Allow users to specify Spark storage level (MEMORY_ONLY, MEMORY_AND_DISK, etc.)

**Action Required:**
- [ ] Test memory usage with large datasets (1M, 10M, 100M rows)
- [ ] Document memory implications in function docstring
- [ ] Consider adding `cache` parameter (default: True for backwards compatibility)
- [ ] Add warning log when caching large dataframes

**Suggested API:**
```python
def binscatter(
    df,
    x,
    y,
    controls=None,
    num_bins=None,
    cache="auto",  # "auto", True, False, or StorageLevel
    ...
):
```

**Files to modify:**
- `src/binscatter/core.py`
- `docs/perf_notes.md`
- `README.md` (if API changes)

---

## Minor Issues (Nice to Have)

### 6. Incomplete Type Hints

**Location:** Various functions in `src/binscatter/core.py`

**Description:**
The new dummy builder functions could benefit from more explicit type hints:

```python
# Current
def _dummy_builder_pyspark(
    df: nw.LazyFrame, categorical_controls: Tuple[str, ...]
) -> Tuple[nw.LazyFrame, Tuple[str, ...]]:

# Could use the type alias
def _dummy_builder_pyspark(
    df: nw.LazyFrame, categorical_controls: Tuple[str, ...]
) -> Tuple[nw.LazyFrame, Tuple[str, ...]]:
```

**Action Required:**
- [ ] Consider using `DummyBuilder` type alias in function signatures
- [ ] Add type hints to internal helper functions
- [ ] Run mypy/pyright to check for type issues

**Priority:** Low (code is already fairly well-typed)

---

### 7. `_format_dummy_alias` Edge Cases

**Location:** `src/binscatter/core.py:983`

**Code:**
```python
def _format_dummy_alias(column: str, value: Any) -> str:
    safe_value = str(value).replace(" ", "_").replace("/", "_")
    return f"__ctrl_{column}_{safe_value}"
```

**Potential issues:**
- What if `value` contains `__`? Could cause parsing issues
- What if `value` is very long? Column name limits in some backends
- What if two different values become the same after sanitization? (e.g., "foo/bar" and "foo_bar")

**Action Required:**
- [ ] Add more robust sanitization (replace all non-alphanumeric except underscore)
- [ ] Add length limit and truncation with hash suffix if needed
- [ ] Add collision detection
- [ ] Add test cases for edge cases

**Suggested improvement:**
```python
import re
import hashlib

def _format_dummy_alias(column: str, value: Any) -> str:
    safe_value = re.sub(r'[^a-zA-Z0-9]', '_', str(value))
    if len(safe_value) > 50:
        # Truncate and add hash to avoid collisions
        hash_suffix = hashlib.md5(str(value).encode()).hexdigest()[:8]
        safe_value = safe_value[:40] + "_" + hash_suffix
    return f"__ctrl_{column}_{safe_value}"
```

**Files to modify:**
- `src/binscatter/core.py`
- `tests/test_binscatter.py` (add edge case tests)

---

### 8. Debug Script Dependency Documentation

**Location:** `scripts/debug_binscatter.py`

**Description:**
The PR adds `pyarrow` to the commented dependency list but doesn't explain why.

**Action Required:**
- [ ] Document why `pyarrow` is needed (if it is)
- [ ] Update the script's docstring to explain dependencies
- [ ] Consider adding a check with helpful error message if pyarrow is missing

**Priority:** Low

---

### 9. Polars Dummy Builder Inefficiency

**Location:** `src/binscatter/core.py:1018-1037`

**Code:**
```python
if isinstance(native, pl.LazyFrame):
    dataset = native.collect()
else:
    dataset = native
```

**Description:**
The Polars path collects a lazy frame, creates dummies, then converts back to lazy. This seems inefficient.

**Action Required:**
- [ ] Investigate if `to_dummies` can work on LazyFrame directly
- [ ] If not, document why the collect is necessary
- [ ] Consider caching if the same lazy frame is used multiple times

**Priority:** Low (Polars is typically fast anyway)

---

### 10. Missing Logging for Cache Decisions

**Location:** `src/binscatter/core.py`

**Description:**
The caching context manager logs when it caches/unpersists, but there's no log indicating:
- The size of the dataframe being cached
- Whether caching was skipped (for non-PySpark backends)
- How long the cache operation took

**Action Required:**
- [ ] Add log statement with dataframe size before caching
- [ ] Log cache operation time
- [ ] Add debug log for non-PySpark backends (currently silent)

**Example:**
```python
logger.debug("[binscatter] caching PySpark dataframe (estimated size: X MB)")
logger.debug("[binscatter] skipping cache for non-PySpark backend: %s", implementation)
```

**Priority:** Low (nice for debugging but not critical)

---

## Documentation Issues

### 11. Performance Notes Need Expansion

**Location:** `docs/perf_notes.md`

**Missing information:**
- [ ] Hardware specs for benchmarks (CPU, RAM, Spark config)
- [ ] Dataset characteristics (number of columns, data types, nulls percentage)
- [ ] Comparison with/without optimizations (not just final numbers)
- [ ] Memory usage measurements
- [ ] Scaling behavior (how does performance change with 10x more data?)

**Action Required:**
- Expand `docs/perf_notes.md` with comprehensive benchmark methodology

---

### 12. PLAN.md Formatting

**Location:** `PLAN.md`

**Description:**
The plan file has "Remaining optimizations" but it's not clear if these are planned for this PR or future work.

**Action Required:**
- [ ] Clarify that optimizations #3-4 are future work
- [ ] Move completed items to an "Archive" section
- [ ] Add success metrics for each completed optimization

**Priority:** Low (cosmetic)

---

## Testing Checklist

Before merging, ensure all these tests pass:

### Unit Tests
- [ ] All existing tests still pass
- [ ] New tests for `_maybe_cache_pyspark`
- [ ] New tests for `configure_dummy_builder`
- [ ] New tests for `_format_dummy_alias`
- [ ] New tests for each dummy builder implementation

### Integration Tests
- [ ] Full binscatter with PySpark + caching
- [ ] Full binscatter with PySpark + categorical controls
- [ ] Full binscatter with PySpark + controls + poly_line

### Performance Tests
- [ ] Benchmark suite shows improvement (run `pytest tests/test_performance.py`)
- [ ] No performance regression for other backends
- [ ] Memory usage is acceptable

### Edge Case Tests
- [ ] High-cardinality categoricals
- [ ] Null values in categoricals
- [ ] Special characters in category values
- [ ] Very large datasets (if possible)

---

## Review Approval Checklist

- [ ] Code duplication removed (#1)
- [ ] Test coverage added (#2)
- [ ] Performance claims validated (#3)
- [ ] Scalability concerns addressed (#4)
- [ ] Caching design documented (#5)
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No performance regressions for existing code
- [ ] Memory usage tested and documented

---

## Notes for Reviewers

### Testing the PR Locally

```bash
# Checkout the branch
git fetch origin
git checkout feature/perf-investigation

# Install with dev dependencies
uv pip install -e .[dev]

# Run fast tests (excludes PySpark)
just ftest

# Run full test suite including PySpark
just test --run-pyspark

# Run performance benchmarks
uv run pytest tests/test_performance.py --run-pyspark -v

# Run the debug script to see timings
uv run python scripts/debug_binscatter.py
```

### Questions to Ask

1. Is automatic caching the right default? Should it be opt-in?
2. What's the maximum cardinality we support for categoricals?
3. Should we add progress indicators for long-running operations?
4. Do we need different optimization strategies for different dataset sizes?
