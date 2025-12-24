# Dataframe agnostic binscatter plots

**TL;DR:** Fast binscatter plots for all kinds of dataframes.

- Built on the `narwhals` dataframe abstraction, so pandas, Polars, DuckDB, Dask, and PySpark inputs all work out of the box.
  - All other Narwhals backends fall back to a generic quantile handler if a native path is unavailable
- Lightweight - little dependencies
- Just works: by default picks the number of bins automatically via the rule-of-thumb selector from Cattaneo et al. (2024) - no manual tuning
- Efficiently avoids materializing large intermediate datasets
- Optional polynomial regression overlay computed directly from the raw data (and any controls) for quick visual comparison
- Uses `plotly` as graphics backend - because: (1) it's great (2) it uses `narwhals` as well, minimizing dependencies
- Pythonic alternative to the excellent **binsreg** package

## What are binscatter plots? 

Binscatter plots group the x-axis into bins and plot average outcomes for each bin, giving a cleaner view of the relationship between two variables—possibly controlling for confounders. They show an estimate of the conditional mean, rather than all the underlying data as in a classical scatter plot.

## Installation

```bash
pip install binscatter
```

---

## Example

A binscatter plot showing patenting activity against the 3-year net of tax rate controlling for several state-level covariates.

<img src="images/readme/binscatter_controls.png" alt="Scatter and binscatter" width="640" />

See code below:

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
    # num_bins="rule-of-thumb",  # optional: let the selector choose the bin count
    # return_type="native",  # optional: get the aggregated dataframe instead of a Plotly figure
    # poly_line=2,  # optional: overlay a degree-2 polynomial fit using the raw data plus controls
).update_layout(  # binscatter returns a Plotly figure, so you can tweak labels, colors, etc.
    xaxis_title="Log net of tax rate := log(1 - tax rate)",
    yaxis_title="Log number of patents",
)
```
This is how a classical scatter of the same data looks like, clearly showing a lot of noise:

<img src="images/readme/scatter.png" alt="Scatter" width="640" />


This package implements binscatter plots following:

- Cattaneo, Matias D.; Crump, Richard K.; Farrell, Max H.; Feng, Yingjie (2024), “On Binscatter,” *American Economic Review*, 114(5), 1488–1514. [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

Data for the example originates from:

- Akcigit, Ufuk; Grigsby, John; Nicholas, Tom; Stantcheva, Stefanie (2021), “Replication Data for: ‘Taxation and Innovation in the 20th Century’,” *Harvard Dataverse*, V1. [DOI: 10.7910/DVN/SR410I](https://doi.org/10.7910/DVN/SR410I)

## Tests

- Run the full backend matrix, including PySpark: `just test`
- Use the faster run without PySpark: `just ftest`
