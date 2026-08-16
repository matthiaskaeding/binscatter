# Dataframe agnostic binscatter plots

**TL;DR:** Fast binscatter plots with automatic bin selection for all kinds of dataframes

- Handles pandas, Polars, DuckDB, Dask, and PySpark via `narwhals`
- Supports covariate adjustment (including high-dimensional categorical controls) and confidence intervals
- Interactive plots via `plotly`
- Implementation follows [Cattaneo et al. (2024)](https://doi.org/10.1257/aer.20221576).

## What are binscatter plots?

Binscatter plots group the x-axis into bins and plot the average y value for each bin, giving a cleaner view of the relationship between two variables.
They estimate the conditional mean rather than displaying all the underlying observations, and can also adjust for covariates and provide statistical inference.

## Installation

```bash
pip install binscatter
```

## Example

```python
import pandas as pd
from binscatter import binscatter

df = pd.read_csv("data/nhanes_age_bp.csv")
binscatter(df, "age", "systolic_bp")
```

<img src="https://raw.githubusercontent.com/matthiaskaeding/binscatter/images/images/readme/nhanes_age_bp.png" alt="Binscatter of age and average systolic blood pressure in the NHANES example data" width="640" />

`binscatter` automatically chooses the number of bins with the DPI selector. For
covariate adjustment, confidence intervals, alternative bin selectors, polynomial
overlays, and dataframe-backend examples, see the
[rendered demo notebook](https://github.com/matthiaskaeding/binscatter/blob/notebooks/demo.ipynb).


## Tests

- Check a representative sample while working: `make test-fast` (~18s)
- Run the whole suite without PySpark: `make test`
- Run the full backend matrix, including PySpark: `make test-spark`
