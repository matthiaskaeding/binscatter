# PLAN.md

## Active Plan: PySpark performance optimizations

### Completed
1. **Categorical dummy caching** (optimization #1): Implemented backend-specific dummy builders via `configure_dummy_builder()`:
   - `_dummy_builder_pyspark`: Batches all categorical discovery into a single `agg(*collect_set(...))` call
   - `_dummy_builder_pandas_polars`: Uses native `pd.get_dummies()` / `pl.to_dummies()`
   - `_dummy_builder_fallback`: Generic narwhals implementation for other backends

### Remaining optimizations
2. **Reuse aggregated moments**: Push `_ensure_feature_moments` to request sums/cross-products in the same grouping job that produces `per_bin`
3. **Spark-native regression solve**: Use VectorAssembler + Spark ML linear regression instead of shipping moments to driver
4. **Optional caching**: Add `cache()`/`persist()` option for lazy backends to avoid repeated scans

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
