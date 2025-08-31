# Dataframe agnostic binscatter plots

This package implements binscatter plots following:

> Cattaneo, Crump, Farrell and Feng (2024)  
> "On Binscatter"  
> American Economic Review, 114(5), pp. 1488-1514  
> [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

- Uses `narwhals` as dataframe layer `binscatter` can support: cuDF, Modin, pandas, Polars, DuckDB, Ibis, PySpark, SQLFrame
- Lightweight - little dependencies
- Uses `plotly` as graphics backend - because: (1) its great (2) it uses `narwhals` as well, minimizing dependencies
- Pythonic alternative to the excellent **binsreg** package

---

## Example

![combined](https://github.com/matthiaskaeding/binscatter/blob/images/images/readme/combined.png?raw=true)

