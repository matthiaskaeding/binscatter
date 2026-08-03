# Dataframe agnostic binscatter plots

**TL;DR:** Fast binscatter plots for all kinds of dataframes.

- Built on the `narwhals` dataframe abstraction, so pandas, Polars, DuckDB, Dask, and PySpark inputs all work out of the box.
- Uses `plotly` as graphics backend - because: (1) it's great (2) it uses `narwhals` as well, minimizing dependencies
- Lightweight - little dependencies
- Just works: by default picks the number of bins automatically via the DPI (Direct Plug-In) selector from Cattaneo et al. (2024) - no manual tuning

## What are binscatter plots?

Binscatter plots group the x-axis into bins and plot average outcomes for each bin, giving a cleaner view of the relationship between two variables—possibly controlling for confounders. They show an estimate of the conditional mean, rather than all the underlying data as in a classical scatter plot.

## Installation

```bash
pip install binscatter
```

---

## Example

```python
import plotly.express as px
from binscatter import binscatter

df = px.data.gapminder()
binscatter(df, "gdpPercap", "lifeExp")
```

<img src="https://raw.githubusercontent.com/matthiaskaeding/binscatter/images/images/readme/gapminder_gdp_lifeexp_dpi.png" alt="Binscatter: GDP per capita vs Life Expectancy (DPI selector)" width="640" />

By default binscatter chooses bins via the DPI (Direct Plug-In) selector. Often we want more bins for a rawer look—use `num_bins` to specify the bin count:
```python
binscatter(df, "gdpPercap", "lifeExp", num_bins=120)
```

<img src="https://raw.githubusercontent.com/matthiaskaeding/binscatter/images/images/readme/gapminder_gdp_lifeexp_fixed.png" alt="Binscatter: GDP per capita vs Life Expectancy (120 bins)" width="640" />

### Fixed effects

Categorical controls are absorbed as fixed effects rather than one-hot encoded, so
cardinality costs nothing. Pass as many as you like:

```python
binscatter(
    df,
    "wage",
    "tenure",
    controls=["age", "firm_id", "year"],
    categorical=["firm_id", "year"],
)
```

String columns are detected automatically; `categorical=` is the override for
integer-coded identifiers like `firm_id`, which would otherwise enter as a single
linear term. Everything is computed from group-level aggregates — the number of
levels never enters the cost, and no per-row residuals are formed — so absorbing
several high-cardinality factors stays cheap on lazy backends.

With one factor the projection is a closed form. With several it is solved
iteratively, to a relative residual of `1e-11`; if a design is too poorly connected
to reach that, the call raises rather than returning a number that depends on the
iteration budget.

`ci=` works alongside absorbed fixed effects at any cardinality. The interval's
sandwich variance is built from the same group-level aggregates, so it never forms
the fixed-effect block either.

### Confidence intervals

Pass `ci` to draw an interval around each dot:

```python
binscatter(df, "gdpPercap", "lifeExp", num_bins=120, ci="rbc")
```

There are two styles, and they answer different questions:

- `ci="pointwise"` is the heteroskedasticity-robust interval for the binscatter estimate itself — how precisely each bin's mean is pinned down. It is centred on the dot.
- `ci="rbc"` is the robust bias-corrected interval for the underlying regression function. Because the dots are flat within a bin they are biased for the true conditional mean, so the interval is built from the next-order fit; it is a valid interval for the true curve, and is deliberately *not* centred on the dot. This matches `binsreg`'s `ci=(1, 1)`.

Use `ci_level` to change the level (default `0.95`). With `return_type="native"` the bounds come back as `ci_lower`, `ci_upper` and `ci_std_error` columns.

Intervals are only trustworthy when the number of bins is well above the IMSE-optimal choice, so pass `num_bins` explicitly rather than relying on the automatic selectors — binscatter warns if you don't.

See the [rendered demo notebook](https://github.com/matthiaskaeding/binscatter/blob/notebooks/demo.ipynb) for more examples. Its lightweight source lives at [`examples/demo.ipynb`](examples/demo.ipynb); GitHub Actions executes it after relevant pushes to `main` and publishes the output separately so generated plots do not inflate the main branch. This package implements binscatter following [Cattaneo et al. (2024)](https://doi.org/10.1257/aer.20221576).

## Tests

- Run the full backend matrix, including PySpark: `make test`
- Use the faster run without PySpark: `make ftest`
