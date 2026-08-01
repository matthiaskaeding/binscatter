# PLAN.md

## Active Plan: Parameterization-Invariant DPI With Fixed Effects

**Status: Complete**

### Problem

The DPI selector currently uses the covariance of raw bin coefficients. With
reference-coded categorical controls that system is full rank, while the absorbed
fixed-effect path fully demeans the bin basis and then uses a pseudoinverse of a
rank-deficient system. The plotted point estimates agree, but their DPI variance
constants can differ solely because cardinality routed the same model through a
different parameterization.

### Design

Evaluate the nonparametric component at the sample mean of every control, matching
the plotted curve:

1. Center all ordinary regression controls before the DPI pilot regressions.
2. For an absorbed fixed effect, project onto the **centered** group-effect
   subspace: `a - mean(a | group) + mean(a)`. Unlike full demeaning, this retains
   the intercept and leaves the bin basis full rank.
3. Apply the same centered projection to `y`, the spline basis, bin indicators and
   controls. Extend the HC1 sandwich blocks to the centered bin basis without
   materializing an `n x J` matrix.
4. Compute the DPI variance from the resulting bin-coefficient covariance, which
   now represents the actual mean-adjusted curve under either dummy encoding or
   absorption.

### Verification

1. Force the one-hot and absorbed paths on the same panels and require identical
   DPI constants, selected bin counts and final plotted values.
2. Check invariance to the omitted categorical reference level and category label
   ordering.
3. Retain the existing `binsreg` DPI comparisons for models with and without
   numeric controls.
4. Run the fixed-effect and DPI suites across backends, then `make ok`, the fast
   suite, and the example notebook.

---

## Archived Plan: Resolve PR #80 Merge Conflicts

**Status: Complete.**

Current `main` was merged into `claude/issue-71-binsreg-benchmark`. Both sets of
changelog entries and test imports were preserved, while `uv.lock` remains deleted
per current repository policy. Focused benchmark/property tests, the fast suite,
the affected PySpark tests, formatting, linting, and type checking all pass.

1. Merge current `main` into `claude/issue-71-binsreg-benchmark` in an isolated
   checkout.
2. Preserve both changelog histories and all compatible test imports.
3. Keep `uv.lock` deleted, matching current repository policy.
4. Run focused tests and repository checks, then push the resolution.

---

## Archived Plan: Verify Default DPI With Controls on Discrete `x` (#70)

**Status: Complete** — closes #70.

### Outcome note

The reported binary/discrete case already passes on `main`: the DPI helper
deduplicates the ROT pilot's quantile edges, and the public pipeline caps the result
to the feasible raw-`x` bins. The regression case deliberately makes the ROT pilot
request more bins than binary `x` can support, supplies a numeric control, omits
`num_bins` to exercise the public DPI default, and matches both coordinates against
an independent least-squares reference on pandas, Polars, DuckDB, Dask, and PySpark.

### Objective

Close the coverage gap around the default automatic selector: when `x` has only a
few distinct values and controls are supplied, DPI must complete successfully,
collapse duplicate quantile boundaries to the feasible bin count, and return the
same finite result on every supported dataframe backend.

### Steps

1. Reproduce the issue with discrete `x`, a numeric control, and the implicit
   `num_bins="dpi"` default; trace both the DPI pilot and final quantile fallback.
2. Add a cross-backend regression test that exercises the public default rather
   than calling the selector helper in isolation.
3. If the regression test exposes a selector or bin-assignment defect, fix the
   narrowest shared path and add focused edge-case coverage.
4. Record the user-visible guarantee in `CHANGELOG.md` and run focused tests,
   formatting, linting, type checking, and the fast suite.

---

## Archived Plan: Absorb Fixed Effects (Mundlak) + `categorical=` Parameter

**Status: Complete, released as 0.4.0** — closes #69. Multi-way absorption is
tracked separately in #73.

### Outcome note

Absorption turned out **not** to be a free swap at every cardinality in the initial
implementation. The bin estimates were exact either way (equivalence holds to
~5e-15, confirmed against a statsmodels oracle), but the DPI selector's sandwich
variance depended on the parameterization. Hence `ABSORB_MIN_LEVELS = 50` initially
kept small categoricals on the existing path. The active plan above supersedes that
limitation by evaluating both parameterizations at the same centered target.

The threshold was set from measurement rather than guessed (`make benchmark-fe`).
One-hot scales roughly cubically in levels — 0.74s at 50, 3.52s at 100, 31s at 200,
~6 minutes at 400 (pandas, n=20,000) — so 50 sits at the knee: past it users pay
seconds for an estimator difference they cannot see, before it the existing path is
genuinely cheap.

### Overview

Categorical controls are one-hot encoded today, so `_ensure_feature_moments` builds
`k(k+1)/2` sum-product expressions and `partial_out_controls` solves a dense
`(J+k)x(J+k)` system. At a few thousand levels this dies while building the query
plan. Replace that with the Mundlak / within projection applied at the moment level:

```
(M_D A)'(M_D B) = A'B - S_A' N^-1 S_B
```

which needs only group counts, group sums, and the bin x group crosstab — two extra
`group_by` calls, independent of level count. Absorb the highest-cardinality
categorical; one-hot the rest into the existing control block (the correction is
generic in `W`, so they ride along free). `D'D` is diagonal only for a single
categorical, so absorbing two is out of scope.

Add `categorical=` so integer-coded IDs (`firm_id`, `zip`) can reach that path at
all — today they are classified numeric and silently become a single linear term.

### Steps

1. **`categorical=` parameter**: validate (subset of `controls`, not `x`/`y`, no
   duplicates, reject floats), thread through `clean_df` to override
   `split_columns`, and keep reclassified columns in the finite/NaN filter.
2. **`fixed_effects.py`**: `FEMoments` (counts, bin x group crosstab, feature sums,
   y sums) + `compute_fe_moments()` doing the two aggregations + a generic
   `within_correct()` helper. Map group labels to positions explicitly — `group_by`
   order is not stable across backends.
3. **Absorb selection**: one `n_unique()` pass over categorical controls, absorb the
   argmax, dummy the rest.
4. **`partial_out_controls`**: apply the correction to the four blocks, solve with
   `lstsq` (the within system is rank-deficient by exactly 1), recover the level via
   `fitted = beta + mean_controls @ gamma + (counts_g @ alpha) / n`.
5. **Other three consumers** of `regression_features`: `_fit_polynomial_line` and
   `_select_rule_of_thumb_bins` take the same moment algebra; `_select_dpi_bins`
   already materializes to pandas so it can demean numerically.
6. **Tests**: equivalence absorbed vs. one-hot across all backends is the gate; then
   ordering/label handling, degenerate structure (singletons, single-bin levels,
   disconnected bin x group), level recovery, statsmodels oracle, interactions with
   `poly_line`/`rot`/`dpi`, nulls, and a high-cardinality case that would explode
   today. See #69 for the full plan.
7. **Release**: version bump, CHANGELOG entry, update `examples/demo.ipynb`.

---

## Archived Plan: Replace justfile With Makefile

**Status: Complete**

### Overview

Move the developer workflow from the previous `justfile` commands to a standard `Makefile`, ensure all documentation references are updated, and provide a default help target that explains the available commands.

### Steps

1. **Switch Branches**: Move to `main`, create a feature branch, and ensure the workspace is clean.
2. **Introduce Makefile**: Recreate the existing developer targets in a new `Makefile`, add a user-friendly default help target, and remove the old `justfile`.
3. **Update References**: Replace every instruction that mentions the old command runner (README, contributor guides, CHANGELOG, etc.) with the appropriate `make` invocations and add a CHANGELOG entry for the switch.
4. **Verification**: Run a representative target (e.g., `make help`) to confirm the Makefile works and search the repo to confirm no lingering references remain.

---

## Archived Plan: DPI (Direct Plug-In) Bin Selector

**Status: Deferred**

### Overview

Implement the DPI (Direct Plug-In) bin selector from Cattaneo et al. (2024) as an alternative to the current ROT (Rule-of-Thumb) selector. DPI is more data-driven and doesn't rely on parametric assumptions for the density.

### Background

From binsreg results on gapminder:
- ROT: 21 bins
- DPI: 35 bins

The DPI method typically recommends more bins than ROT because it uses pilot estimates of the IMSE components rather than asymptotic approximations.
In testing each time the ROT method recommended way to few bins, in doubbt more bins is better than too few.

### Research Needed

1. **Read SA-4.2 of Cattaneo et al. (2024)** - DPI selector formula. Stay extremly close to that paper.
2. **Identify key differences from ROT**:
   - How is the bias term B estimated?
   - How is the variance term V estimated?
   - What pilot bandwidths/preliminary estimates are needed?

### Implementation Steps

1. **Understand DPI Formula**
   - Extract DPI formula from paper's supplementary appendix
   - Compare to ROT formula to identify differences
   - Document the mathematical approach

2. **Implement Core DPI Logic**
   - Add `_select_dpi_bins()` function in core.py
   - Implement pilot estimators for bias/variance
   - Add numerical safeguards (similar to ROT)

3. **Integrate with API**
   - Add `"dpi"` as option for `num_bins` parameter
   - Update type hints and docstrings
   - Consider making DPI the default (if it performs better)

4. **Testing**
   - Add tests comparing to binsreg DPI output
   - Test on various data distributions
   - Verify numerical stability

### Open Questions

1. Should DPI become the default selector?
2. What tolerance is acceptable vs binsreg? (ROT is within 0-2 bins for symmetric data)
3. Are there performance considerations? (DPI may require more computation)

### References

- Cattaneo, M. D., Crump, R. K., Farrell, M. H., & Feng, Y. (2024). On Binscatter. American Economic Review, 114(5), 1488-1514.

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
