# Changelog

## Unreleased

### Added
- Ensure the Plotly-based binscatter output always applies the ``simple_white`` template so figures look consistent across environments.
- Document feature additions and template tweak in CHANGELOG.
- Automatic rule-of-thumb bin selection for the canonical binscatter implementation.
- Enforcement of keyword-only arguments for ``binscatter`` to avoid positional mistakes.
- Plotly-friendly x-axis padding to keep the rightmost point away from the edge.
- CI workflow plus optional PySpark tests, enabling PR checks.
- ``just`` targets for lint, test, and plot replication, and README documentation improvements.

### Changed
- Refactored the control/partialling pipeline into smaller helpers and improved validation.
- Switched regression data cleaning to use ``drop_nulls``.
- Updated README metadata and clarified usage examples.
- Added optional slow-test opt-in gate.

## 0.1.0

- Initial release.
