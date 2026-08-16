# Example data

`state_data_processed.parquet` is derived from:

> Akcigit, Ufuk; Grigsby, John; Nicholas, Tom; Stantcheva, Stefanie (2021),
> “Replication Data for: ‘Taxation and Innovation in the 20th Century’,” Harvard
> Dataverse. <https://doi.org/10.7910/DVN/SR410I>

The source dataset is dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The Parquet remains
under those terms and is not covered by the repository's MIT license.

Starting from `REPLICATION_PACKET/Data/state_data.dta`, its preparation:

1. Keeps observations from 1939 onward.
2. Log-transforms `population_density` and `real_gdp_pc`.
3. Converts `mtr90_lag3` and `top_corp_lag3` from percentages to
   `log(1 - rate / 100)`.

The reproducible preparation is implemented in
`scripts/replicate_binscatter/prep_data.py`.
