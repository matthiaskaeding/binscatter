# Changelog

## Unreleased

## 0.4.0 - 2026-08-01

### Added
- `categorical=` parameter to treat a control as categorical regardless of dtype. Integer-coded identifiers (`firm_id`, `zip`) were previously classified numeric and silently entered as a single linear term; listing them in `categorical` gives them fixed-effect semantics. It is an override, not a declaration — string columns are still detected automatically. Must be a subset of `controls`; floating-point columns are rejected, since grouping on floats is unreliable.
- Absorption of high-cardinality categorical controls via the Mundlak / within transform, in the new `fixed_effects.py` module (#69). For any two matrices and the group-dummy matrix `D`, `(M_D A)'(M_D B) = A'B - S_A' N^-1 S_B`, so only group counts, group sums, and the bin-by-group crosstab are needed — two aggregations, independent of the number of levels, with no per-row residuals materialized. A categorical with 50,000 levels previously required ~1.25 billion sum-product aggregations and a dense 50k x 50k solve; it now costs the same as a handful of levels.
- `tests/test_fixed_effects.py`: equivalence against the one-hot path on every backend, an independent statsmodels oracle, ordering and exotic-label handling, degenerate structure (singleton groups, levels confined to one bin, disconnected bin-group blocks), level recovery, and unit tests for the within algebra itself.
- `tests/test_benchmarks_fixed_effects.py`: benchmarks asserting cost is flat in cardinality, that absorption is not silently falling back to one-hot, and that cardinality discovery stays a bounded share of the aggregation cost. Marked `benchmark` and skippable with `--skip-benchmarks`. Measured on pandas at n=20,000: 50 levels 1.08s one-hot vs 0.03s absorbed; 100 levels 5.92s vs 0.03s; 200 levels 46.35s vs 0.03s; 400 levels exceeded 8 minutes on the one-hot path while absorbed stayed at 0.03s.

### Changed
- `add_regression_features` returns a third element naming the absorbed column, and `Profile` carries `fe_name`.
- The rule-of-thumb and DPI bin selectors, and the `poly_line` overlay, all apply the within correction so the absorbed control is not silently dropped from bin selection.
- The pandas dummy builder now passes `columns=` to `pd.get_dummies`, which otherwise silently skips non-object columns and encoded nothing for an integer-coded control.

### Notes
- Absorption applies at `ABSORB_MIN_LEVELS` (100) or more. Below that the existing one-hot path is kept, so existing results do not move. The threshold exists because the DPI selector's sandwich variance is not invariant to the reparameterization: the one-hot design is full rank (drop_first pins the level) while the within system is rank-deficient by one, so per-bin coefficients — and hence their variances — are identified only up to a shift.
- Only one categorical is absorbed, the highest-cardinality one; the rest are one-hot encoded into the control block and ride along at no extra cost. Absorbing two is the two-way fixed effects problem: `D'D` stops being diagonal and the closed form does not exist.

## 0.3.1 - 2026-01-29

### Changed
- Adopted `prek`-managed pre-commit hooks (ruff format/check, ty, nbstripout, and core sanity checks) and run the same tooling directly in CI for faster feedback.
- Updated local and CI type-check targets to `ty@latest` and removed stale `type: ignore` suppressions flagged by the newer release.

### Fixed
- Fixed y-axis scaling issue with `poly_line` parameter (#65): When polynomial overlay was enabled, the y-axis would auto-scale to include both scatter points and polynomial line, causing unexpected rescaling. Now the y-axis range is always explicitly set based on binned scatter points (with Plotly-style padding), ensuring identical scaling whether `poly_line` is present or not.

## 0.3.0 - 2026-01-02

### Added
- Backend-specific dummy variable builders in new `dummy_builders.py` module.
- Hash-based dummy variable naming to prevent collisions (e.g., "foo/bar" vs "foo_bar").
- Performance benchmarks in `tests/test_performance.py`.
- Comprehensive tests for individual `build_dummies` functions covering edge cases, lazy evaluation, and multiple categorical columns.
- Cross-backend regression coefficient test (`test_partial_out_controls_coefficients_across_backends`) to ensure consistent results with categorical variables across all backends.

### Changed
- Reimplemented the direct plug-in (DPI) selector using the SA-4.2 IMSE formulas, matching `binsreg` within a bin across the tested scenarios.
- Refactored dummy variable builders: split `build_dummies_pandas_polars` into separate `build_dummies_pandas` and `build_dummies_polars` functions for cleaner backend-specific logic.
- Optimized Polars dummy builder to preserve lazy evaluation by only collecting categorical columns instead of entire dataframe.
- Extracted rename mapping logic into `build_rename_map` helper function to reduce code duplication.
- Replaced internal `df._compliant_frame.native` with public narwhals API `nw.to_native(df)` across all dummy builders.
- Optimized PySpark categorical handling with batched `collect_set()` aggregation (5.7x speedup).
- Renamed `maybe_add_regression_features` to `add_regression_features`.
- Simplified quantile deduplication logic using `dict.fromkeys` instead of iterative reduction.

### Fixed
- **Critical**: Fixed categorical variable dummy encoding inconsistency across backends. Pandas and Polars were dropping different reference categories (first alphabetically vs first in appearance), causing regression coefficients to differ. Now all backends consistently drop the first category alphabetically, ensuring identical results across pandas, polars, duckdb, and dask.
- Fixed rule-of-thumb bin selector to match Cattaneo et al. (2024) SA-4.1 exactly: corrected bias constant (1/12 vs 1/3), use squared inverse density, and added density trimming at 2.5th percentile.
- Capped rule-of-thumb bins at n/10 to ensure ~10 observations per bin, fixing issues with heavy-tailed data (e.g., GDP).
- Fixed `pd.cut` bin assignment to use `right=False` for correct handling of boundary values.
- Fixed `PerformanceWarning` when passing Polars LazyFrame by avoiding eager schema resolution.

### Added
- Warning when user-specified `num_bins` is reduced due to non-unique quantiles.

## 0.2.0 - 2025-12-25

### Added
- ``poly_line`` argument to overlay degree-1–3 polynomial fits computed from the raw ``x`` and all supplied controls.
- Ensure the Plotly-based binscatter output always applies the ``simple_white`` template so figures look consistent across environments.
- Document feature additions and template tweak in CHANGELOG.
- Automatic rule-of-thumb bin selection for the canonical binscatter implementation.
- Plotly-friendly x-axis padding to keep the rightmost point away from the edge.
- CI workflow plus optional PySpark tests, enabling PR checks.
- ``just`` targets for lint, test, and plot replication, and README documentation improvements.

### Changed
- Refactored ``partial_out_controls`` internals into reusable helpers so future regression overlays can share the same cached cross-moments.
- Refactored the control/partialling pipeline into smaller helpers and improved validation.
- Switched regression data cleaning to use ``drop_nulls``.
- Updated README metadata and clarified usage examples.
- Added optional slow-test opt-in gate.

## 0.1.0

- Initial release.
