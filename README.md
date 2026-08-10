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

See the [rendered demo notebook](https://github.com/matthiaskaeding/binscatter/blob/notebooks/demo.ipynb) for more examples. This package implements binscatter following [Cattaneo et al. (2024)](https://doi.org/10.1257/aer.20221576).

## High-dimensional categorical controls

`binscatter` handles high-dimensional categorical controls efficiently with the
method of alternating projections (MAP), without building a large one-hot matrix.

Example timings for 5,000 pandas rows and 10 bins:

| categories | `binscatter` | explicit one-hot | speedup |
|---:|---:|---:|---:|
| 10 | ~0.009s | ~0.017s | ~2× |
| 25 | ~0.008s | ~0.064s | ~8× |
| 50 | ~0.008s | ~0.28s | ~36× |

Reference environment: MacBook Air (Apple M1, 8 CPU cores, 16 GB RAM), macOS
15.6.1 arm64, Python 3.11.9, and pandas 3.0.5. The values are medians of three
calls from `make benchmark-fe-fast`, run locally on 10 August 2026; they are
comparative measurements, not performance guarantees for other machines.

```bash
make benchmark-fe-fast  # short suite
make benchmark-fe       # up to 1M rows and 50K levels
```

## Tests

- Check a representative sample while working: `make test-fast` (~18s)
- Run the whole suite without PySpark: `make test`
- Run the full backend matrix, including PySpark: `make test-spark`
