# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "polars",
#     "pyarrow",
# ]
# ///

from __future__ import annotations

from pathlib import Path

from typing import cast

import polars as pl
from pandas import read_stata as pd_read_stata


def read_stata(*args, **kwargs) -> pl.DataFrame:
    return cast(pl.DataFrame, pl.from_pandas(pd_read_stata(*args, **kwargs)))


def main() -> None:
    proj_dir = Path(__file__).parent.parent.parent.resolve()
    artifacts = proj_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    stata_path = artifacts / "dataverse_files/REPLICATION_PACKET/Data/state_data.dta"
    parquet_path = artifacts / "state_data_processed.parquet"

    df = read_stata(stata_path).filter(pl.col("year") >= 1939)
    df = df.with_columns(
        pl.col("population_density", "real_gdp_pc").log(),
        *[
            (1 - pl.col(x) / 100).log().alias(x)
            for x in ["mtr90_lag3", "top_corp", "top_corp_lag3"]
        ],
    )
    df.write_parquet(parquet_path)
    print(f"Wrote processed data to {parquet_path}")  # noqa: T201


if __name__ == "__main__":
    main()
