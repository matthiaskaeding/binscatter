# PLAN.md

## Active Plan: PySpark ML-native regression implementation

### Objective
Leverage PySpark's ML library (VectorAssembler + LinearRegression) to handle control partialing instead of manually building normal equations. This is PySpark-specific and applies only when controls are present.

### Current Implementation (All Backends)
The `partial_out_controls()` function:
1. Aggregates data to bin level (counts, sums per bin)
2. Manually computes feature cross-products via `_ensure_feature_moments()`
3. Manually builds XTX matrix from cached aggregates
4. Solves normal equations on driver using NumPy
5. Computes adjusted bin means using coefficients

This approach is efficient because it works with small aggregated data (typically 20-50 bins), but requires collecting all feature moments to the driver.

### Proposed PySpark ML Implementation
When `implementation == PYSPARK` and `controls` are present:

1. **After binning**: Keep row-level data (don't aggregate immediately)
2. **Create bin dummies**: One-hot encode the bin column using PySpark's `StringIndexer` + `OneHotEncoder` or manual approach
3. **Assemble features**: Use `VectorAssembler` to combine:
   - Bin dummy columns (num_bins columns, drop-first to avoid collinearity)
   - Control features (numeric controls + categorical dummies already created)
4. **Fit regression**: Use `LinearRegression` to fit: `y ~ bin_dummies + controls`
   - This is equivalent to current approach: y = sum(beta_i * bin_i) + sum(gamma_j * control_j)
5. **Extract coefficients**:
   - First (num_bins - 1) coefficients → bin effects (relative to dropped first bin)
   - Remaining k coefficients → control effects
   - Intercept → effect of the dropped first bin
6. **Compute adjusted means**: Same formula as current:
   - For dropped bin: fitted = intercept + mean_controls @ gamma
   - For other bins: fitted = intercept + bin_coef + mean_controls @ gamma
7. **Aggregate bin centers**: Still need x means per bin for plotting

### Implementation Plan

#### Phase 1: Core PySpark ML regression path
- [ ] Create `_partial_out_controls_pyspark()` function in core.py
  - Takes binned dataframe (row-level, not aggregated)
  - Creates bin dummy columns (one-hot encoding)
  - Uses VectorAssembler to combine bin dummies + control features
  - Fits LinearRegression model
  - Extracts coefficients and computes adjusted bin means
  - Returns same output format as current `partial_out_controls()`

#### Phase 2: Integration with main flow
- [ ] Modify `binscatter()` to use factory pattern:
  - Add `configure_partial_out_controls(implementation)` factory
  - Returns `_partial_out_controls_pyspark` for PySpark
  - Returns current `partial_out_controls` for other backends
- [ ] Ensure x means per bin are computed for plotting

#### Phase 3: Handle polynomial overlay
- [ ] Verify polynomial overlay still works with PySpark path
  - Polynomial features are already created before binning
  - May need separate path for `_fit_polynomial_line()` when using PySpark ML
  - Or keep current aggregation-based approach for polynomials

#### Phase 4: Testing
- [ ] Add tests for PySpark ML path with controls
- [ ] Verify numerical equivalence with current implementation
- [ ] Test edge cases (no controls, categorical controls, polynomials)
- [ ] Performance benchmarking vs current approach

### Technical Considerations

**Benefits:**
- Uses Spark's distributed regression solver
- Avoids collecting all feature cross-products to driver
- May scale better for very large datasets with many controls
- Cleaner separation of concerns (Spark handles regression mechanics)

**Challenges:**
- Requires row-level data (higher memory than aggregates for small num_bins)
- Need to create bin dummy columns (not currently done)
- More complex integration (PySpark-specific path diverges from other backends)
- May be slower for typical use cases (20-50 bins) due to ML overhead
- Need to ensure coefficient ordering matches expected structure

**When PySpark ML helps:**
- Large datasets (millions of rows) with many controls (10+)
- Cases where collecting aggregates to driver is a bottleneck
- When feature cross-product computation is expensive

**When current approach is better:**
- Small to medium datasets
- Few controls (< 10)
- Small number of bins (20-50)

### Files to Modify
- `src/binscatter/core.py`: Add `_partial_out_controls_pyspark()` and factory function
- `tests/test_binscatter.py`: Add PySpark ML-specific tests
- `docs/perf_notes.md`: Document performance comparison

---

## Archived Plan: PySpark performance optimizations

### Completed
1. **Categorical dummy caching** (optimization #1): Implemented backend-specific dummy builders via `configure_dummy_builder()`:
   - `_dummy_builder_pyspark`: Batches all categorical discovery into a single `agg(*collect_set(...))` call
   - `_dummy_builder_pandas_polars`: Uses native `pd.get_dummies()` / `pl.to_dummies()`
   - `_dummy_builder_fallback`: Generic narwhals implementation for other backends

2. **Automatic PySpark caching** (optimization #4): Added `_maybe_cache_pyspark` context manager:
   - Caches dataframe after `clean_df` to avoid repeated scans
   - Automatically unpersists on exit
   - Results: `maybe_add_regression_features` 2.15s → 0.38s (5.7x faster), `_fit_polynomial_line` 1.14s → 0.71s (1.6x faster)

### Remaining optimizations
3. **Reuse aggregated moments**: Push `_ensure_feature_moments` to request sums/cross-products in the same grouping job that produces `per_bin`
4. **Spark-native regression solve**: Use VectorAssembler + Spark ML linear regression instead of shipping moments to driver

---

## Archived Plan: enforce equidistant uniqueness in compute_max_bins tests

**Status: Complete**

## Objectives
1. Inspect the existing compute_max_bins tests to ensure there are no lingering test classes and understand how the helper is used.
2. Update the shared helper (and any straggler tests) so that every compute_max_bins test also asserts equidistant quantile uniqueness.
3. Run the focused pytest subset covering compute_max_bins to confirm the updated assertions pass.

---

## Archived Plan: refactor quantiles module to pre-compute quantiles and enforce uniqueness before bin assignment

**Status: Complete**

## Objectives
1. Make a function factory that configures a function that computes quantiles. Should takes as input: num_bins and df.Implementation.
This can be based on existing configure_add_bins logic, each of these functions has already a way to compute quantiles.
2. After calling the quantile computation function, check if the quantiles are unique. If not, we can compute the maximum number of bins possible. If num_bins is user_inputm throw error with this info. If num_bins is auto, set to highest possible and continue.
3. Redesign `configure_add_bins` so that it has as input Collection of quantiles input (instead of computing quantiles internally)
4. Update `binscatter` (and tests) to use the new quantile workflow: compute quantiles once per iteration, handle auto-bin fallback by retrying with reduced counts, and ensure backend-specific assigners consume the shared quantile data structure.

---

## Archived Plan: if we set bins automatically, have fallback for case when quantiles are not unique

**Status: Complete**

## Objectives
1. Inspect existing automatic bin creation flow (`add_bins`, quantile handling, `compute_bin_means`/`partial_out_controls`) to understand where bin uniqueness should be validated and how reruns can be triggered.
2. Implement a fallback mechanism that detects duplicate `(xname, binname)` combinations after initial binning, recalculates the feasible `num_bins`, and re-executes the binning pipeline with this adjusted value while preserving control handling and caches.
3. Extend tests (likely in `tests/test_binscatter.py`) with scenarios exhibiting low `x` variation to verify that automatic bin counts adjust downward gracefully across relevant backends.

---

## Archived Plan: separate compute_max_bins tests

**Status: Complete**

## Objectives
1. Capture all existing `compute_max_bins` helper/tests currently embedded in `tests/test_binscatter.py`.
2. Move that suite into a new `tests/test_compute_max_bins.py` module that imports only what it needs.
3. Confirm the relocated tests pass via the focused pytest invocation.

---

## Archived Plan: enforce real quantile uniqueness in compute_max_bins tests

**Status: Complete**

## Objectives
1. Update the `compute_max_bins` tests to derive equidistant quantile values (not just probabilities) and assert their uniqueness.
2. Adjust helper(s) and any other affected code to satisfy the stronger assertion while keeping the suite green.
3. Rerun the focused pytest subset to confirm the relocated tests still pass with the new checks.
