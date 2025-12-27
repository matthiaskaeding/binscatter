# Changelog

## Unreleased

### Changed
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
