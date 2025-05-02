# Modern binscatter plots

This package implements binscatter plots following:

> Cattaneo, Crump, Farrell and Feng (2024)  
> "On Binscatter"  
> American Economic Review, 114(5), pp. 1488-1514  
> [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

- Uses `polars` for scalability  
- Uses `plotnine` as graphics backend — allowing composable plots 
  - Can return data for use in other packages 
- Pythonic alternative to the excellent **binsreg** package

---

## Example

We made this noisy scatterplot:

![Noisy scatterplot](https://github.com/matthiaskaeding/binscatter/blob/images/artifacts/images/scatter.png?raw=true)

This is how we make a nice binscatter plot, controlling for a set of features:

```python
from binscatter import binscatter
p_binscatter_controls = binscatter(
    df,
    "mtr90_lag3",
    "lnpat",
    [
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

![Binscatter](https://github.com/matthiaskaeding/binscatter/blob/images/artifacts/images/binscatter_controls.png?raw=true)

Data used for example:

Akcigit, Ufuk; Grigsby, John; Nicholas, Tom; Stantcheva, Stefanie, 2021, "Replication Data for: 'Taxation and Innovation in the 20th Century'", https://doi.org/10.7910/DVN/SR410I, Harvard Dataverse, V1
