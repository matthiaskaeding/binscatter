# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "polars",
#     "kaleido",
#     "pyarrow",
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
import sys
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402

import plotly.express as px  # noqa: E402
import polars as pl  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.binscatter.core import binscatter  # noqa: E402

proj_dir = ROOT
log_file = proj_dir / "artifacts" / "binscatter.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("binscatter.core").setLevel(logging.DEBUG)
# Ensure pandas/numpy logger noise stays low while keeping binscatter debug lines
logging.getLogger("numpy").setLevel(logging.WARNING)
for n in ("choreographer", "kaleido", "numba", "matplotlib", "asyncio", "browser_proc"):
    logging.getLogger(n).setLevel(logging.ERROR)


print("project dir =", proj_dir)
data_dir = proj_dir / "artifacts"
data_dir.mkdir(exist_ok=True, parents=True)
assets_dir = proj_dir / "images" / "readme"
assets_dir.mkdir(exist_ok=True, parents=True)
# %%
parquet_path = data_dir / "state_data_processed.parquet"
if not parquet_path.exists():
    raise FileNotFoundError(
        f"Missing {parquet_path}. Run scripts/replicate_binscatter/prep_data.py first."
    )
df = pl.read_parquet(parquet_path)
df.describe()
# %%
axis_labels = {
    "x": "Log net of tax rate := log(1 - tax rate)",
    "y": "Log number of patents",
}
p_scatter = px.scatter(
    df.select("mtr90_lag3", "lnpat"),
    x="mtr90_lag3",
    y="lnpat",
    template="simple_white",
    color_discrete_sequence=["black"],
    labels={"x": "mtr90_lag3", "y": "lnpat"},
)
# Hide the single-trace legend for cleaner screenshots.
p_scatter.update_layout(
    showlegend=False,
    xaxis_title=axis_labels["x"],
    yaxis_title=axis_labels["y"],
)
p_scatter.write_image(assets_dir / "scatter.png", width=640, height=480)
# %%
p_binscatter = binscatter(
    df,
    "mtr90_lag3",
    "lnpat",
    num_bins=20,
)
p_binscatter.update_layout(
    xaxis_title=axis_labels["x"],
    yaxis_title=axis_labels["y"],
)
p_binscatter.write_image(assets_dir / "binscatter_bare.png", width=640, height=480)
# %%
df = df.with_columns(pl.col("statenum", "year").cast(pl.String))
# Capture all binscatter logs into artifacts/binscatter.log
log_file = data_dir / "binscatter.log"

print("Generating binscatter with controls...")
p_binscatter_controls = binscatter(
    df.to_pandas(),
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
    num_bins="rule-of-thumb",
)
logging.shutdown()

p_binscatter_controls.update_layout(
    xaxis_title=axis_labels["x"],
    yaxis_title=axis_labels["y"],
)
p_binscatter_controls.write_image(
    assets_dir / "binscatter_controls.png", width=640, height=480
)
# %%
combined = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("(a) Scatter", " (b) Binscatter with controls"),
    horizontal_spacing=0.08,
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)
for trace in p_scatter.data:
    combined.add_trace(trace, row=1, col=1)
for trace in p_binscatter_controls.data:
    combined.add_trace(trace, row=1, col=2)
combined.update_xaxes(title_text=axis_labels["x"], row=1, col=1)
combined.update_yaxes(title_text=axis_labels["y"], row=1, col=1)
combined.update_xaxes(title_text=axis_labels["x"], row=1, col=2)
combined.update_yaxes(title_text=axis_labels["y"], row=1, col=2)
combined.update_layout(
    template="simple_white",
    showlegend=False,
)
for annotation in combined.layout.annotations:
    annotation.update(font=dict(family="Georgia", size=16))
combined.write_image(assets_dir / "scatter_vs_binscatter.png", width=960, height=480)
combined.show()
