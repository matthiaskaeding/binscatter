# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "polars",
#     "pyarrow",
#     "pyfixest",
#     "plotly",
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
import importlib
import sys
from pathlib import Path

import plotly.express as px
import polars as pl
from pandas import read_stata as pd_read_stata
import pyfixest as pf


def read_stata(*args, **kwargs):
    return pl.from_pandas(pd_read_stata(*args, **kwargs))


proj_dir = Path(__file__).parent.parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

binscatter = importlib.import_module("binscatter").binscatter

print("project dir =", proj_dir)
data_dir = proj_dir / "artifacts"
data_dir.mkdir(exist_ok=True, parents=True)
assets_dir = proj_dir / "images" / "readme"
assets_dir.mkdir(exist_ok=True, parents=True)
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
p_scatter = px.scatter(
    df.select("mtr90_lag3", "lnpat").to_pandas(),
    x="mtr90_lag3",
    y="lnpat",
)
p_scatter.write_image(assets_dir / "scatter.png", width=800, height=600)
# %%
p_binscatter = binscatter(
    df,
    "mtr90_lag3",
    "lnpat",
    num_bins=20,
)
p_binscatter.write_image(assets_dir / "binscatter.png", width=800, height=600)
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
p_binscatter_controls.write_image(
    assets_dir / "binscatter_controls.png", width=800, height=600
)
# %%
