# Changelog

## Unreleased

### Added
- `ci=` and `ci_level=` on `binscatter()`, adding confidence intervals to the dots, in the new `inference.py` module (#39). Two styles, because they answer different questions: `ci="pointwise"` is the heteroskedasticity-robust (HC1) interval for the binscatter estimator itself and is centred on the dot, while `ci="rbc"` is the robust bias-corrected interval for the underlying regression function — the `p=0` dots are flat within a bin and therefore biased for the true conditional mean, so the interval comes from the next-order (`p+1=1`, `s+1=1`) fit and is deliberately *not* centred on the dot. `ci="rbc"` reproduces `binsreg`'s `ci=(1,1)` bounds to six decimals on its own simulation data, with and without controls; `ci="pointwise"` matches a statsmodels HC1 dummy regression to machine precision. Both are computed from bin-level aggregates in two passes, so nothing is materialized per row, and both work on every backend. With `return_type="native"` the bounds arrive as `ci_lower`, `ci_upper` and `ci_std_error`; Plotly figures grow interval bars and widen the y-axis so the bars are never clipped.
- Requesting intervals while the bin count comes from an automatic selector now warns. ROT and DPI minimise IMSE, which is exactly the regime where these intervals under-cover, so a reliable interval needs `num_bins` passed explicitly. `binsreg` raises the same caveat.
- GitHub Actions now executes `examples/demo.ipynb` after notebook-relevant pushes to `main` and force-publishes the rendered result to the orphan `notebooks` branch only when its tracked inputs changed. The README links to that rendered copy while `main` keeps the output-free source notebook, avoiding generated Plotly output in the main branch history.
- Benchmark tests comparing binscatter's bin dot values and automatic bin counts (ROT and DPI, with and without controls) directly against the `binsreg` Python package's own reference output, using its canonical simulation dataset (checked in at `data/binsreg_sim.csv`) (#71).
- Property-based tests in `tests/test_properties.py` using Hypothesis, covering structural invariants (bin count, ordering, range containment), invariance under row permutation and control rescaling, equivariance under affine maps of `x` and `y`, dropping of null/non-finite rows, and label-independence of categorical controls (#53). `hypothesis>=6.100` is now a `dev` and `ci` dependency.
- `.github/workflows/publish.yaml`: publishing is driven by GitHub Releases, so cutting a release is the single action that ships a version and pushes to `main` never publish (#72). The release tag is checked against the version in `pyproject.toml` before anything is built, so a release tagged `v0.1.0` cannot ship a `0.4.0` artifact, and upload uses PyPI trusted publishing rather than a stored API token. `workflow_dispatch` runs the same build and `twine check` as a dry run that stops short of publishing.
- Cross-backend regression coverage for the default DPI selector with controls on
  discrete `x`, ensuring duplicate pilot quantile boundaries produce the two
  feasible bins and the correct control-adjusted estimates (#70).

### Changed
- The contributing guides (`AGENTS.md`, `CLAUDE.md`) now require a `CHANGELOG.md` entry in the same commit or PR as the change itself, where they previously asked only for one "before merging", and say what qualifies: not just library features, but bug fixes reachable on a single backend, build and packaging changes, and tooling changes that alter what CI enforces. Entries deferred to release time are the reason the 0.3.0–0.4.0 sections had to be reconstructed from the commit log afterwards; those sections are now backfilled.
- Upgraded ruff to 0.16.1 and fixed the 143 findings its newer default rule set reports, rather than staying on a release old enough not to raise them. The version is named in four places that are now kept in step — `ci.yaml`, `.pre-commit-config.yaml`, the `Makefile`, and `required-version` in `pyproject.toml` — so CI, contributors' hooks, `make ok`, and a stray global install cannot lint under different rules; the Makefile now requests the pinned tool instead of reusing whichever Ruff uv previously cached. `BLE001` and `S110` are ignored in `pyproject.toml`, since the broad `except` in the optional-backend import guards is the intended behaviour rather than an oversight.

### Known limitations
- `ci=` is not available when a categorical control is absorbed as a fixed effect (50 or more levels, see `ABSORB_MIN_LEVELS`). The interval needs that control's contribution to the sandwich variance, and absorption exists precisely to never form it. The call raises `NotImplementedError` naming the column rather than returning an interval that quietly ignores the fixed effect.

### Removed
- Stopped committing `uv.lock`; dependency versions are no longer pinned in the repo.
- Dropped the requirement that contributors keep a written plan in `PLAN.md`, and deleted the file. Its contents were entirely archived plans for work that has already shipped, and the instruction is gone from both `AGENTS.md` and `CLAUDE.md`, so no contributor guide asks for a plan file any more.

### Fixed
- DPI automatic bin selection is now invariant to whether a categorical control is one-hot encoded or absorbed, and to which category is used as the omitted dummy. Controls and fixed effects are evaluated at their sample-average levels, so crossing the absorption threshold no longer changes the selected bin count for the same model.
- `_quantile_edges` (`core.py`, `bin_selectors.py`) no longer carries a `numpy<1.22` fallback passing the `interpolation=` keyword that NumPy removed in 2.0. The project requires `numpy>=2.3`, so the `except TypeError` branch was unreachable and its `type: ignore` was masking a real overload error.
- The fallback quantile path in `quantiles.py` re-raises with a bare `raise`, preserving the original traceback instead of restarting it at the handler.

## 0.4.0 - 2026-08-01

### Added
- `categorical=` parameter to treat a control as categorical regardless of dtype. Integer-coded identifiers (`firm_id`, `zip`) were previously classified numeric and silently entered as a single linear term; listing them in `categorical` gives them fixed-effect semantics. It is an override, not a declaration — string columns are still detected automatically. Must be a subset of `controls`; floating-point columns are rejected, since grouping on floats is unreliable.
- Absorption of high-cardinality categorical controls via the Mundlak / within transform, in the new `fixed_effects.py` module (#69). For any two matrices and the group-dummy matrix `D`, `(M_D A)'(M_D B) = A'B - S_A' N^-1 S_B`, so only group counts, group sums, and the bin-by-group crosstab are needed — two aggregations, independent of the number of levels, with no per-row residuals materialized. A categorical with 50,000 levels previously required ~1.25 billion sum-product aggregations and a dense 50k x 50k solve; it now costs the same as a handful of levels.
- `tests/test_fixed_effects.py`: equivalence against the one-hot path on every backend, an independent statsmodels oracle, ordering and exotic-label handling, degenerate structure (singleton groups, levels confined to one bin, disconnected bin-group blocks), level recovery, and unit tests for the within algebra itself.
- `scripts/benchmark_fixed_effects.py` (`make benchmark-fe`): measures absorption against one-hot encoding by cardinality, absorbed scaling in rows, and the cost of the cardinality lookup that routes between the two. Measured on pandas at n=20,000: 50 levels 1.08s one-hot vs 0.03s absorbed; 100 levels 5.92s vs 0.03s; 200 levels 46.35s vs 0.03s; 400 levels exceeded 8 minutes on the one-hot path while absorbed stayed at 0.03s. It is a script rather than a test because it measures rather than asserts — the guard against silently falling back to one-hot is a direct assertion on the routing decision in `test_fixed_effects.py`, which needs no timing.

### Changed
- `add_regression_features` returns a third element naming the absorbed column, and `Profile` carries `fe_name`.
- The rule-of-thumb and DPI bin selectors, and the `poly_line` overlay, all apply the within correction so the absorbed control is not silently dropped from bin selection.
- The pandas dummy builder now passes `columns=` to `pd.get_dummies`, which otherwise silently skips non-object columns and encoded nothing for an integer-coded control.

### Notes
- Absorption applies at `ABSORB_MIN_LEVELS` (50) or more. Below that the existing one-hot path is kept, so existing results do not move. The threshold exists because the DPI selector's sandwich variance is not invariant to the reparameterization: the one-hot design is full rank (drop_first pins the level) while the within system is rank-deficient by one, so per-bin coefficients — and hence their variances — are identified only up to a shift. Crossing it can therefore change the number of bins the `dpi` selector picks. 50 is where the one-hot path begins costing real time (0.74s at 50 levels, 3.52s at 100, 31s at 200, ~6 min at 400) without yet being unusable.
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
