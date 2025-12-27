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
