# Example data

## `nhanes_age_bp.csv`

This two-column teaching dataset is derived from the `NHANES` dataframe in version
2.1.0 of the R package
[`NHANES`](https://cran.r-project.org/package=NHANES). The package combines the
2009–2010 and 2011–2012 National Health and Nutrition Examination Survey cycles and
resamples the survey-weighted source data to 10,000 rows for educational use.

The committed file selects `Age` and `BPSysAve`, drops rows missing either value,
and renames the columns to `age` and `systolic_bp`. `BPSysAve` is the combined
systolic blood pressure reading in mm Hg. The resulting file contains 8,551 rows.
Run `uv run scripts/prepare_nhanes_example.py` to reproduce it from the
pinned `NHANES` 2.1.0 data object in CRAN's package mirror. The preparation
dependency declared by that script is isolated from the package's runtime
dependencies.

The underlying data were produced by the Centers for Disease Control and
Prevention, National Center for Health Statistics. Federal-agency data are generally
in the public domain and may be reproduced without permission; please cite the CDC
and NCHS. Use of NCHS public-use files is limited to statistical reporting and
analysis, and users must not attempt to identify survey participants. This extract
contains no participant identifiers.

The R package documentation describes this resampled dataframe as an educational
dataset, not a research database. Research analyses should instead use the original
CDC files with their survey weights and design variables.

Sources:

- [NHANES package documentation](https://search.r-project.org/CRAN/refmans/NHANES/html/NHANES.html)
- [CDC NHANES data and citation guidance](https://wwwn.cdc.gov/nchs/nhanes/NhanesCitation.aspx)
- [NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html)
