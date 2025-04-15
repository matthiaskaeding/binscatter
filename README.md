# Modern binscatter plots

This package implements binscatter plots following:

> Cattaneo, Crump, Farrell and Feng (2024)  
> "On Binscatter"  
> American Economic Review, 114(5), pp. 1488-1514  
> [DOI: 10.1257/aer.20221576](https://doi.org/10.1257/aer.20221576)

- Uses `polars` for scalability  
- Uses `plotnine` as graphics backend — allowing composable plots  

---

## Example

We have created this noisy scatterplot:

![Noisy scatterplot](https://github.com/matthiaskaeding/binscatter/blob/images/readme/scatter.png?raw=true)

This is how we would make a binscatter plot

```python
from binscatter import binscatter
binscatter(
    df,
    x="x",
    y="y",
)
```

![Noisy scatterplot](https://github.com/matthiaskaeding/binscatter/blob/images/readme/binscatter.png?raw=true)
