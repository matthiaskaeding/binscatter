## Plan: `add_polynomial` argument and regression overlay

## Objectives
- Add an argument add_poynomial. The polynomial is the regression line of the regresison y=poly(x) * alpha + controls * gamma + u. The regression line will be poly(x) * alpha.
- Make the control machinery flexible enough to drop or add subsets of controls without recomputing the whole dataset, anticipating a future stepwise-controls toggle.
- Allow the regression layer to fit polynomial-in-x specifications (e.g., quadratic) on demand.
- Write function testable. So in the end we will always get regression coefficients as outputs. So we can always test if those are equal to a benchmark.

1. **Study and reuse `partial_out_controls` internals**
   - Trace how `partial_out_controls` computes `X'X`, `X'y`, and the cached aggregates that bins reuse.
   - Document which intermediates we can expose (e.g., design matrices, control means) so the polynomial regression shares them rather than recomputing from raw frames.
   - Confirm how raw `x`/`y` and controls are available before partialling so we can build polynomial columns directly off the unmodified data as required.

2.a **Split `partial_out_controls` into reusable helpers**
   - Introduce focused helpers such as `build_regression_inputs(df, x, y, controls)` and `solve_regression(design, response)` so polynomial regressions just add columns before solving.
   - Keep a thin orchestrator `partial_out_controls` that wires these helpers together for existing behavior while returning a structured object (dataclass) storing the reusable matrices.
   - Ensure new helpers return the raw moments (means, sums, xtx, xty) so additional polynomial degrees can be solved without another pass through the dataframe.

3. **Design the `add_polynomial` API surface**
   - Add an optional integer parameter (e.g., `add_polynomial: int | None = None`) to `binscatter` and propagate it to the plotting layer.
   - Interpret `add_polynomial=d` as “fit a degree-`d` polynomial in `x` using the raw data plus any controls”; default `None` keeps current behavior.
   
4. **Implement polynomial regression leveraging the refactors**
   - Use the reusable design-building helper to append `[x, x², …, xᵈ]` columns before solving the regression; include controls unchanged so coefficients align with the regression described in the request.
    - Evaluate the polynomial on a dense grid across the observed x-range (not needed if pokynomial is 1) and add the resulting line to the plot when `add_polynomial` is set.

5. **Testing and documentation pass**
   - Extend `tests/test_binscatter.py` with parametrized cases covering degrees 1–3 with and without controls, checking coefficients against `statsmodels`.
   - Update README + CHANGELOG to describe the new argument and emphasize that it regresses on the raw data with all specified controls.
   

## Future hook: stepwise controls
- Once the control-spec abstraction exists, a future `stepwise_controls` feature can iterate through predefined subsets (baseline, +1 control, etc.) without re-scanning the dataframe; the cached matrices from Workstreams 1–2 become the shared foundation.
- Consider exposing a lightweight cache object (e.g., `DesignCache`) from `partial_out_controls` that stores `xtx`, `xty`, bin sums, and control cross-products. Reusing this cache keeps future stepwise regressions and inference fast.
