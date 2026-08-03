# Changelog

## Unreleased

### Added
- `test_binscatter_matches_binsreg_sim_no_controls` and `test_binscatter_matches_binsreg_sim_with_controls` now also request `ci="rbc"` and assert the bounds against `binsreg_fit(..., ci=(1,1))`, at the same `num_bins=20` / controls configuration already used there for the point-estimate check. Previously CI parity with `binsreg` was only checked in `tests/test_inference.py`, at `num_bins=5` and a different controls list, so a CI regression scoped to the point-estimate benchmark's own configuration had no test that would catch it (#98).
- `test_matches_binsreg_reference_with_categorical_controls` in `tests/test_inference.py`, extending the `binsreg` comparison to categorical controls: one categorical alone, a numeric plus one categorical, two categoricals together, and a numeric plus two categoricals. `binsreg` has no notion of a categorical control, so its reference `w` is built by dummy-encoding each categorical column the same way `binscatter` does internally (`drop_first=True`, alphabetically sorted levels) (#98).
- `ci="pointwise"` is now checked against `binsreg` as well as against statsmodels. It has an exact counterpart there that the tests had assumed did not exist: `binsreg`'s `ci=(p, s)` names the fit an interval is built from, so `ci=(0, 0)` is the interval for the piecewise-constant estimator the dots show, just as `ci=(1, 1)` is the robust bias-corrected one. The two comparisons in `tests/test_inference.py` that were `rbc`-only are now parametrized over both styles and renamed accordingly (`test_matches_binsreg_reference`, `test_matches_binsreg_reference_with_categorical_controls`). Agreement is ~1e-15. The statsmodels HC1 oracle is kept: it spells the estimator out as an explicit dummy regression, so it pins the standard errors themselves rather than only the bounds, and it would not move with us if `binsreg` and this package ever drifted together.
- Ten further `binsreg` comparisons in `tests/test_inference.py`, each parametrized over both interval styles, covering input cases the previous tests reached only at their defaults. `ci_level` was checked for monotonicity but never against `binsreg`, so 0.80, 0.90 and 0.99 now are; `ci_std_error` was never compared to `binsreg` at all and is now checked against the error implied by its bounds, which catches a scaled or stale column that leaves the bounds themselves correct. Also added: bin counts at the extremes (2, 3, 20, 40, where 2 leaves the spline basis a single interior knot); automatically selected bin counts (`dpi`, `rot`), where the selectors had their own comparisons but nothing checked the count reached the interval machinery intact; six numeric controls, against a previous maximum of two; deliberately heteroskedastic data, which is what tells HC1 apart from a classical variance estimate and which no other test used; mass points in `x`, against `binsreg` with `masspoints="off"` so both cut the same bins; frames with nulls in `x`, `y` or a control, against `binsreg` on the frame with those rows already dropped; an integer-coded control declared through `categorical=`; a categorical with `ABSORB_MIN_LEVELS - 1` levels, the widest dummy block that still reaches this code path and the other side of the boundary from the absorption error; and each exact-quantile backend (pandas, polars, duckdb) individually, where they were previously only pinned to each other. A final test asserts `poly_line` leaves the bounds and standard errors untouched, the overlay being fitted from the raw rows.
- `test_rank_deficient_controls_differ_from_binsreg_only_in_the_dof_factor`, pinning the one input case where the two packages do not agree to machine precision. Two perfectly collinear controls leave the design singular; both packages still return the interval, since the fitted value and its variance are identified where the coefficients are not, but they count the redundant column differently in the HC1 finite-sample correction `n / (n - k)` — `binsreg` drops it, `binscatter` keeps it. The standard errors therefore sit at exactly `sqrt((n - k_binsreg) / (n - k_ours))`, which is 1.0005 at the tested `n = 1000` and shrinks with the sample. The test asserts that exact ratio rather than a loose tolerance, so any other divergence in the rank-deficient path fails it.
- Three consistently named test targets replace the ambiguous `ftest`/`test` pair, whose names implied the opposite of what they did — `make test` was the *slower* one, since it added `--run-pyspark`. Now `make test-fast` runs a representative sample (tests marked `@pytest.mark.quick`, exact backends only, ~18s), `make test` runs the whole suite with PySpark skipped and is the gate, and `make test-spark` runs the full backend matrix. `--strict-markers` means a misspelled or stale marker fails loudly rather than silently selecting nothing. CI invokes `pytest` directly and is unaffected. The contributor guides now direct you to check with `make test-fast` while iterating and `make test` before committing.
- `tests/test_fe_contract.py`, checking agreement with `binsreg` — the paper authors' own package — across a wide sweep of fixed-effect specifications: one, two and three factors at cardinalities from 5 to 200 levels, with none, one and two numeric controls, from 2 to 20 bins, plus skewed and singleton-heavy level distributions, nested factors, integer identifiers via `categorical=`, and every backend. Fixed effects reach `binsreg` as dummy columns in `w`, which is what makes it a fair judge of an implementation that absorbs them instead; agreement is ~1e-15. The file imports only `binscatter()`, patches nothing and asserts nothing about which route ran, so it stays valid across implementations — verified by running it against three (as shipped, absorption disabled, absorption forced everywhere), where only the cost test tells them apart. `test_fixed_effects.py` remains the regression test for the current design, which it tests by forcing its two routes and comparing them.

### Fixed
- A control column named after an internal aggregation temporary (`__fe_count`, `__fe_resp_0`) crashed with `cannot insert __fe_count, already exists`, which named neither the column nor the cause. The fixed-effect collector now steps aside from names the frame already carries, the way `binscatter` already suffixes its bin column (#95).
- A column with no usable values — empty, or every entry null or non-finite — reached the quantile coercion as `float(None)` and raised `TypeError: float() argument must be a string or a real number, not 'NoneType'`. Unusable quantiles are now dropped, so the caller sees too few bin edges and gets the existing error about the distribution of `x` (#95). All five backend quantile paths share one coercion, where three previously rolled their own.
- An x or y column named `bin` is rejected up front instead of being silently overwritten by the bin index in the returned frame, and a non-integer `num_bins` is rejected rather than truncated (#95).

## 0.4.0 - 2026-08-02

### Added
- `ci=` and `ci_level=` on `binscatter()`, adding confidence intervals to the dots, in the new `inference.py` module (#39). Two styles, because they answer different questions: `ci="pointwise"` is the heteroskedasticity-robust (HC1) interval for the binscatter estimator itself and is centred on the dot, while `ci="rbc"` is the robust bias-corrected interval for the underlying regression function — the `p=0` dots are flat within a bin and therefore biased for the true conditional mean, so the interval comes from the next-order (`p+1=1`, `s+1=1`) fit and is deliberately *not* centred on the dot. `ci="rbc"` reproduces `binsreg`'s `ci=(1,1)` bounds to six decimals on its own simulation data, with and without controls; `ci="pointwise"` matches a statsmodels HC1 dummy regression to machine precision. Both are computed from bin-level aggregates in two passes, so nothing is materialized per row, and both work on every backend. With `return_type="native"` the bounds arrive as `ci_lower`, `ci_upper` and `ci_std_error`; Plotly figures grow interval bars and widen the y-axis so the bars are never clipped.
- Requesting intervals while the bin count comes from an automatic selector now warns. ROT and DPI minimise IMSE, which is exactly the regime where these intervals under-cover, so a reliable interval needs `num_bins` passed explicitly. `binsreg` raises the same caveat.
- `categorical=` parameter to treat a control as categorical regardless of dtype. Integer-coded identifiers (`firm_id`, `zip`) were previously classified numeric and silently entered as a single linear term; listing them in `categorical` gives them fixed-effect semantics. It is an override, not a declaration — string columns are still detected automatically. Must be a subset of `controls`; floating-point columns are rejected, since grouping on floats is unreliable.
- Absorption of high-cardinality categorical controls via the Mundlak / within transform, in the new `fixed_effects.py` module (#69). For any two matrices and the group-dummy matrix `D`, `(M_D A)'(M_D B) = A'B - S_A' N^-1 S_B`, so only group counts, group sums, and the bin-by-group crosstab are needed — two aggregations, independent of the number of levels, with no per-row residuals materialized. A categorical with 50,000 levels previously required ~1.25 billion sum-product aggregations and a dense 50k x 50k solve; it now costs the same as a handful of levels.
- `tests/test_fixed_effects.py`: equivalence against the one-hot path on every backend, an independent statsmodels oracle, ordering and exotic-label handling, degenerate structure (singleton groups, levels confined to one bin, disconnected bin-group blocks), level recovery, and unit tests for the within algebra itself.
- `scripts/benchmark_fixed_effects.py` (`make benchmark-fe`): measures absorption against one-hot encoding by cardinality, absorbed scaling in rows, and the cost of the cardinality lookup that routes between the two. Measured on pandas at n=20,000: 50 levels 1.08s one-hot vs 0.03s absorbed; 100 levels 5.92s vs 0.03s; 200 levels 46.35s vs 0.03s; 400 levels exceeded 8 minutes on the one-hot path while absorbed stayed at 0.03s. It is a script rather than a test because it measures rather than asserts — the guard against silently falling back to one-hot is a direct assertion on the routing decision in `test_fixed_effects.py`, which needs no timing.
- GitHub Actions now executes `examples/demo.ipynb` after notebook-relevant pushes to `main` and force-publishes the rendered result to the orphan `notebooks` branch only when its tracked inputs changed. The README links to that rendered copy while `main` keeps the output-free source notebook, avoiding generated Plotly output in the main branch history.
- Benchmark tests comparing binscatter's bin dot values and automatic bin counts (ROT and DPI, with and without controls) directly against the `binsreg` Python package's own reference output, using its canonical simulation dataset (checked in at `data/binsreg_sim.csv`) (#71).
- Property-based tests in `tests/test_properties.py` using Hypothesis, covering structural invariants (bin count, ordering, range containment), invariance under row permutation and control rescaling, equivariance under affine maps of `x` and `y`, dropping of null/non-finite rows, and label-independence of categorical controls (#53). `hypothesis>=6.100` is now a `dev` and `ci` dependency.
- `.github/workflows/publish.yaml`: publishing is driven by GitHub Releases, so cutting a release is the single action that ships a version and pushes to `main` never publish (#72). The release tag is checked against the version in `pyproject.toml` before anything is built, so a release tagged `v0.1.0` cannot ship a `0.4.0` artifact, and upload uses PyPI trusted publishing rather than a stored API token. `workflow_dispatch` runs the same build and `twine check` as a dry run that stops short of publishing.
- Cross-backend regression coverage for the default DPI selector with controls on
  discrete `x`, ensuring duplicate pilot quantile boundaries produce the two
  feasible bins and the correct control-adjusted estimates (#70).

### Changed
- `add_regression_features` returns a third element naming the absorbed column, and `Profile` carries `fe_name`.
- The rule-of-thumb and DPI bin selectors, and the `poly_line` overlay, all apply the within correction so the absorbed control is not silently dropped from bin selection.
- The pandas dummy builder now passes `columns=` to `pd.get_dummies`, which otherwise silently skips non-object columns and encoded nothing for an integer-coded control.
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
