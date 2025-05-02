# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "plotnine",
#     "polars",
#     "pyarrow",
#     "pyfixest",
# ]
# ///

# This roughly replicates figure 2 in
# Cattaneo, Matias D., Richard K. Crump, Max H. Farrell, and Yingjie Feng. 2024. "On Binscatter." American Economic Review 114 (5): 1488–1514.
# using the data available in the replication package
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SR410I
# This is the original replication package,
# Akcigit, Ufuk; Grigsby, John; Nicholas, Tom; Stantcheva, Stefanie, 2021, "Replication Data for: 'Taxation and Innovation in the 20th Century'", https://doi.org/10.7910/DVN/SR410I, Harvard Dataverse, V1
#  so the data will not be filtered
# Exactly as in Catteneo et al.
# %%
import polars as pl
import plotnine as pn
from pathlib import Path
from pandas import read_stata as pd_read_stata
import pyfixest as pf
import sys

sys.path.append(".")
from binscatter import binscatter


def read_stata(*args, **kwargs):
    return pl.from_pandas(pd_read_stata(*args, **kwargs))


proj_dir = Path(__file__).parent.parent.resolve()
print("project dir =", proj_dir)
data_dir = proj_dir / "artifacts"
data_dir.mkdir(exist_ok=True, parents=True)
img_dir = data_dir / "images"
img_dir.mkdir(exist_ok=True, parents=True)
# %%
fl = data_dir / "dataverse_files/REPLICATION_PACKET/Data/state_data.dta"
df = read_stata(fl).filter(pl.col("year") >= 1939)
df = df.with_columns(
    pl.col("population_density", "real_gdp_pc").log(),
    *[
        (1 - pl.col(x) / 100).log().alias(x)
        for x in ["mtr90_lag3", "top_corp", "top_corp_lag3"]
    ],
)
df.describe()
# %%
# Check
pf.feols(
    "lnpat ~ mtr90_lag3 + top_corp_lag3 + real_gdp_pc + population_density + rd_credit_lag3 | statenum + year",
    df,
).summary()

# %%
p_scatter = pn.ggplot(df) + pn.aes("mtr90_lag3", "lnpat") + pn.geom_point()
p_scatter.save(img_dir / "scatter.png")
# %%
p_binscatter = binscatter(
    df,
    "mtr90_lag3",
    "lnpat",
    num_bins=20,
)
p_binscatter.save(img_dir / "binscatter.png")
# %%
df = df.with_columns(pl.col("statenum", "year").cast(pl.String))
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
p_binscatter_controls.save(img_dir / "binscatter_controls.png")
# %%
