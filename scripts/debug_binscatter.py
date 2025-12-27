# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "binscatter",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "pyspark",
# ]
#
# [tool.uv.sources]
# binscatter = { path = "..", editable = true }
# ///
"""Run binscatter with a configurable backend and a feature-rich synthetic dataset."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from binscatter import binscatter

BackendType = Literal["pandas", "pyspark"]

CONTROL_COLUMNS = [
    "control_linear",
    "control_nonlinear",
    "control_interaction",
    "control_category",
    "control_region",
    "control_segment",
    "control_year",
    "control_binary",
]


def make_large_dataframe(num_rows: int = 250_000) -> pd.DataFrame:
    """Create a synthetic dataframe brimming with numeric and categorical controls."""
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=0.0, scale=1.0, size=num_rows)
    control_linear = rng.normal(loc=0.0, scale=1.5, size=num_rows)
    control_nonlinear = np.sin(x) + rng.normal(scale=0.2, size=num_rows)
    categories = np.array(["a", "b", "c", "d"])
    control_category = rng.choice(categories, size=num_rows, replace=True)
    control_region = rng.choice(["north", "south", "east", "west"], size=num_rows)
    control_segment = rng.choice(["consumer", "enterprise", "public"], size=num_rows)
    control_interaction = x * control_linear
    control_year = rng.integers(2000, 2020, size=num_rows)
    control_binary = (rng.random(size=num_rows) > 0.5).astype(int)

    noise = rng.normal(scale=0.5, size=num_rows)
    y = (
        0.75 * x
        + 0.25 * control_linear
        + 0.1 * control_nonlinear
        + 0.2 * (control_segment == "enterprise").astype(float)
        + 0.15 * (control_region == "north").astype(float)
        + 0.05 * control_binary
        + 0.01 * (control_year - 2000)
        + noise
    )

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "control_linear": control_linear,
            "control_nonlinear": control_nonlinear,
            "control_interaction": control_interaction,
            "control_category": control_category,
            "control_region": control_region,
            "control_segment": control_segment,
            "control_year": control_year,
            "control_binary": control_binary,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a large dataset and run binscatter with a chosen backend."
    )
    parser.add_argument(
        "--type",
        choices=["pandas", "pyspark"],
        default="pandas",
        help="Select which dataframe backend to hand to binscatter.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=250_000,
        help="Number of rows to generate for the synthetic dataset.",
    )
    return parser.parse_args()


def _prepare_backend_dataframe(
    df: pd.DataFrame, backend: BackendType
) -> tuple[object, object | None]:
    if backend == "pandas":
        logging.info("Using pandas backend for binscatter input...")
        return df, None

    logging.info("Using PySpark backend for binscatter input...")
    try:
        from pyspark.sql import SparkSession
    except ImportError as err:  # pragma: no cover - runtime guard
        msg = "PySpark is required when --type pyspark is selected."
        raise RuntimeError(msg) from err

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("binscatter-debug-script")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "ERROR"))
    spark_df = spark.createDataFrame(df)
    return spark_df, spark


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("binscatter.core").setLevel(logging.DEBUG)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)

    df = make_large_dataframe(num_rows=args.rows)
    logging.info(
        "Generated dataframe with shape %s containing columns: %s",
        df.shape,
        list(df.columns),
    )

    backend_df, spark_session = _prepare_backend_dataframe(df, args.type)
    try:
        fig = binscatter(
            backend_df,
            x="x",
            y="y",
            controls=CONTROL_COLUMNS,
            num_bins=50,
            poly_line=3,
        )
    finally:
        if spark_session is not None:
            logging.info("Stopping Spark session...")
            spark_session.stop()
            logging.info("Done!")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    output_path = artifacts_dir / f"debug_binscatter_{args.type}.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    logging.info("Wrote binscatter output with controls to %s", output_path.resolve())


if __name__ == "__main__":
    main()
