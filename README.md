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

By default binscatter chooses bins via the DPI (Direct Plug-In) selector; when you want more bins, override via `num_bins`.
```python
binscatter(df, "gdpPercap", "lifeExp", num_bins=120)
```

<img src="https://raw.githubusercontent.com/matthiaskaeding/binscatter/images/images/readme/gapminder_gdp_lifeexp_fixed.png" alt="Binscatter: GDP per capita vs Life Expectancy (120 bins)" width="640" />

### Confidence Intervals

Binscatter supports computing confidence intervals for the bin-level estimates using two methods:

```python
# Pointwise confidence intervals (heteroskedasticity-robust)
binscatter(df, "gdpPercap", "lifeExp", ci="pointwise")

# Robust bias-corrected (RBC) confidence intervals
binscatter(df, "gdpPercap", "lifeExp", ci="rbc", ci_level=0.95)
```

When `return_type="native"`, the output dataframe includes `ci_lower`, `ci_upper`, and `ci_std_error` columns:

```python
result_df = binscatter(df, "gdpPercap", "lifeExp", ci="pointwise", return_type="native")
# result_df has columns: bin, gdpPercap, lifeExp, ci_lower, ci_upper, ci_std_error
```

Both CI methods work seamlessly with controls and across all supported backends.

This package implements binscatter plots following:

- Cattaneo, Matias D.; Crump, Richard K.; Farrell, Max H.; Feng, Yingjie (2024), “On Binscatter,” *American Economic Review*, 114(5), 1488–1514. [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

## Tests

- Run the full backend matrix, including PySpark: `just test`
- Use the faster run without PySpark: `just ftest`
