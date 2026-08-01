# PLAN.md

## Active Plan: Absorb Fixed Effects (Mundlak) + `categorical=` Parameter

**Status: Implemented, released as 0.4.0** — closes #69. Remaining: update
`examples/demo.ipynb` with a `categorical=` / high-cardinality example.

### Outcome note

Absorption turned out **not** to be a free swap at every cardinality. The bin
estimates are exact either way (equivalence holds to ~5e-15, confirmed against a
statsmodels oracle), but the DPI selector's sandwich variance is not invariant to
the reparameterization: the one-hot design is full rank because `drop_first` pins
the level, while the within system is rank-deficient by one, so `pinv` returns a
min-norm inverse and the per-bin variances differ (~0.4x on the test panel, moving
the selected bin count from 24 to 30). Hence `ABSORB_MIN_LEVELS = 100`: small
categoricals stay on the existing path so no current result moves, and large ones
get a path that works at all.

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
