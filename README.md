# Dataframe agnostic binscatter plots

This package implements binscatter plots following:

> Cattaneo, Crump, Farrell and Feng (2024)  
> "On Binscatter"  
> American Economic Review, 114(5), pp. 1488-1514  
> [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

- Uses `narwhals` as dataframe layer.
  - Currently supports: pandas, Polars, DuckDB, Dask, and PySpark
  - All other Narwhals backends fall back to a generic quantile handler if a native path is unavailable
- Lightweight - little dependencies
- Uses `plotly` as graphics backend - because: (1) it's great (2) it uses `narwhals` as well, minimizing dependencies
- Pythonic alternative to the excellent **binsreg** package

## Installation

```bash
pip install binscatter
```

---

## Example

We made this noisy scatterplot:

![Noisy scatterplot](https://raw.githubusercontent.com/matthiaskaeding/binscatter/images/images/readme/scatter.png)

This is how we make a nice binscatter plot, controlling for a set of features:

```python
from binscatter import binscatter

binscatter(
    df,
    "mtr90_lag3",
    "lnpat",
    controls=[
        "top_corp_lag3",
        "real_gdp_pc",
        "population_density",
        "rd_credit_lag3",
        "statenum",
        "year",
    ],
    num_bins=35,
)
```

![Binscatter with controls (35 bins)](https://raw.githubusercontent.com/matthiaskaeding/binscatter/images/images/readme/binscatter_controls.png)

The data originates from:

Akcigit, Ufuk; Grigsby, John; Nicholas, Tom; Stantcheva, Stefanie, 2021, "Replication Data for: 'Taxation and Innovation in the 20th Century'", https://doi.org/10.7910/DVN/SR410I, Harvard Dataverse, V1

## Tests

- Run the full backend matrix, including PySpark: `just test`
- Use the faster run without PySpark: `just test-fast`
