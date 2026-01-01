# Changelog

## Unreleased

### Added
- Backend-specific dummy variable builders in new `dummy_builders.py` module.
- Hash-based dummy variable naming to prevent collisions (e.g., "foo/bar" vs "foo_bar").
- Performance benchmarks in `tests/test_performance.py`.
- Comprehensive tests for individual `build_dummies` functions covering edge cases, lazy evaluation, and multiple categorical columns.

### Changed
- Refactored dummy variable builders: split `build_dummies_pandas_polars` into separate `build_dummies_pandas` and `build_dummies_polars` functions for cleaner backend-specific logic.
- Optimized Polars dummy builder to preserve lazy evaluation by only collecting categorical columns instead of entire dataframe.
- Extracted rename mapping logic into `build_rename_map` helper function to reduce code duplication.
- Replaced internal `df._compliant_frame.native` with public narwhals API `nw.to_native(df)` across all dummy builders.
- Optimized PySpark categorical handling with batched `collect_set()` aggregation (5.7x speedup).
- Renamed `maybe_add_regression_features` to `add_regression_features`.
- Fixed rule-of-thumb bin selector to match Cattaneo et al. (2024) SA-4.1 exactly: corrected bias constant (1/12 vs 1/3), use squared inverse density, and added density trimming at 2.5th percentile.
- Simplified quantile deduplication logic using `dict.fromkeys` instead of iterative reduction.
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
