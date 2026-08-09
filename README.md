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

## Fixed effects without the dummy explosion

Fixed effects are as convenient as any other control: pass the columns and
`binscatter` absorbs them automatically. There is no dummy preparation, reference
category, or separate estimator to manage.

```python
binscatter(
    panel,
    "experience",
    "wage",
    controls=["age", "firm_id", "year"],
    categorical=["firm_id", "year"],  # needed for integer-coded identifiers
)
```

String and categorical columns are detected automatically; `categorical=` tells
`binscatter` that an integer identifier is a factor rather than a numeric slope.
One, two, or several crossed fixed effects work with the same API, including
automatic bin selection, confidence intervals, polynomial overlays, native output,
and every supported dataframe backend.

Under the hood, the fixed effects are projected out from group-level sufficient
statistics. The estimator is the same as an explicit dummy regression, but the
library never builds the wide dummy matrix. A one-way model scales with observations
and observed levels; a multi-way model stores observed fixed-effect intersections,
not the full Cartesian product. This is especially useful for ordinary panel data,
where many observations share a firm-year or similar cell.

An illustrative pandas benchmark on an Apple Silicon laptop (Python 3.11.9,
pandas 3.0.5, median of three runs, 5,000 rows and 10 bins) gives:

| fixed-effect levels | absorbed | explicit one-hot | speedup |
|---:|---:|---:|---:|
| 10 | ~0.007s | ~0.015s | ~2× |
| 25 | ~0.007s | ~0.06s | ~9× |
| 50 | ~0.007s | ~0.27s | ~39× |

At higher cardinality the comparison becomes less interesting because the explicit
dummy route quickly becomes impractical: in the same short suite, absorption took
about 0.02s for 100,000 rows with 5,000 fixed-effect levels, and 0.04s for 20,000 rows
with 1,000 × 20 crossed levels (12,636 observed intersections). Timings depend on
hardware and dataframe versions, so reproduce the short suite with:

```bash
make benchmark-fe-fast
# README-ready table:
uv run scripts/benchmark_fixed_effects.py --quick --markdown
```

`make benchmark-fe` runs the larger scaling suite up to one million rows and 50,000
levels.

## Tests

- Check a representative sample while working: `make test-fast` (~18s)
- Run the whole suite without PySpark: `make test`
- Run the full backend matrix, including PySpark: `make test-spark`
