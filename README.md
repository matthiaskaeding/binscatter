# Dataframe agnostic binscatter plots

This package implements binscatter plots following:

> Cattaneo, Crump, Farrell and Feng (2024)  
> "On Binscatter"  
> American Economic Review, 114(5), pp. 1488-1514  
> [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

- Uses `narwhals` as dataframe layer `binscatter`.
  - Currently supports: pandas, Polars, DuckDB, Dask, and PySpark
  - All other Narwhals backends fall back to a generic quantile handler if a native path is unavailable
- Lightweight - little dependencies
- Uses `plotly` as graphics backend - because: (1) its great (2) it uses `narwhals` as well, minimizing dependencies
- Pythonic alternative to the excellent **binsreg** package

---

## Example

![combined](https://github.com/matthiaskaeding/binscatter/blob/images/images/readme/combined.png?raw=true)

## Tests

- Run the full backend matrix, including PySpark: `just test`
- Use the faster run without PySpark: `just test-fast`
