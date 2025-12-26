# Dataframe agnostic binscatter plots

**TL;DR:** Fast binscatter plots for all kinds of dataframes.

- Built on the `narwhals` dataframe abstraction, so pandas, Polars, DuckDB, Dask, and PySpark inputs all work out of the box.
- Uses `plotly` as graphics backend - because: (1) it's great (2) it uses `narwhals` as well, minimizing dependencies
- Lightweight - little dependencies
- Just works: by default picks the number of bins automatically via the rule-of-thumb selector from Cattaneo et al. (2024) - no manual tuning

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
).update_layout(  # binscatter returns a Plotly figure, so you can tweak labels, colors, etc.
    xaxis_title="Log net of tax rate := log(1 - tax rate)",
    yaxis_title="Log number of patents",
)
```


This package implements binscatter plots following:

- Cattaneo, Matias D.; Crump, Richard K.; Farrell, Max H.; Feng, Yingjie (2024), “On Binscatter,” *American Economic Review*, 114(5), 1488–1514. [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

Data for the example originates from:

- Akcigit, Ufuk; Grigsby, John; Nicholas, Tom; Stantcheva, Stefanie (2021), “Replication Data for: ‘Taxation and Innovation in the 20th Century’,” *Harvard Dataverse*, V1. [DOI: 10.7910/DVN/SR410I](https://doi.org/10.7910/DVN/SR410I)

## Tests

- Run the full backend matrix, including PySpark: `just test`
- Use the faster run without PySpark: `just ftest`
