# Binscatter Performance Notes

## Stage timings

The `scripts/debug_binscatter.py` harness (250k rows, 8 controls, 50 bins) surfaces the following per-stage timings (seconds):

| Stage | pandas | PySpark |
| --- | --- | --- |
| `clean_df` | 0.048 | 0.45 |
| `maybe_add_regression_features` | 0.091 | 3.02 |
| `quantile` prep | ~0.010 | ~0.50 |
| `partial_out_controls` | 0.047 | 2.75 |
| Plotting | 0.11 | 0.12 |

Key takeaways: pandas runs finish in under 0.15 s per stage, while PySpark spends ~3 s each in regression-feature expansion and control partialling (plus ~0.5 s in cleanup/quantiles).

## Hotspot analysis

- `maybe_add_regression_features` (PySpark ~3 s): each categorical control triggers its own `df.select(col).unique().collect()` round trip to fetch distinct levels before adding dummy columns. With three categorical controls this means re-scanning the full dataset three times and materialising the values on the driver.
- `partial_out_controls` (PySpark ~2.75 s): after binning, we collect all feature sums and cross-moments via `_ensure_feature_moments`, which synthesises a large `select` expression and forces another shuffle-heavy aggregation. The subsequent solve happens on the driver, so the Spark job is dedicated to computing these aggregates.
- `clean_df` + quantile phases (~0.5 s combined): Spark still scans the data twice (once to materialise the minimal columns, once to compute quantiles) though the absolute time is modest relative to the two hotspots above.
