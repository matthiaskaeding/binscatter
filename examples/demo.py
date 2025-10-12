"""
Binscatter Demo Script

Demonstrates binscatter functionality across different dataframe backends:
- Polars
- DuckDB
- PySpark (if available)
"""

import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_data(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create sample data for demonstration."""
    np.random.seed(seed)
    x = np.linspace(0, 10, n)
    y = x + np.random.normal(0, 1, n)
    return x, y


def demo_polars():
    """Demonstrate binscatter with Polars DataFrame."""
    try:
        import polars as pl
        from binscatter import binscatter

        logger.info("Running Polars demo...")
        x, y = create_sample_data()
        df = pl.DataFrame({"x": x, "y": y})

        result = binscatter(df, "x", "y", num_bins=10, return_type="native")
        print("Polars result:")
        print(result.head())
        print()

    except ImportError as e:
        logger.warning(f"Polars demo skipped: {e}")
    except Exception as e:
        logger.error(f"Polars demo failed: {e}")


def demo_duckdb():
    """Demonstrate binscatter with DuckDB relation."""
    try:
        import duckdb
        import pandas as pd
        from binscatter import binscatter

        logger.info("Running DuckDB demo...")
        x, y = create_sample_data()

        tmp_df = pd.DataFrame({"x": x, "y": y})
        con = duckdb.connect(":memory:")
        rel = con.from_df(tmp_df)

        result = binscatter(rel, "x", "y", num_bins=10, return_type="native")
        print("DuckDB result:")
        print(result)
        print()

    except ImportError as e:
        logger.warning(f"DuckDB demo skipped: {e}")
    except Exception as e:
        logger.error(f"DuckDB demo failed: {e}")


def demo_pyspark():
    """Demonstrate binscatter with PySpark DataFrame."""
    try:
        from pyspark.sql import SparkSession
        from binscatter import binscatter

        logger.info("Running PySpark demo...")

        # Initialize Spark with minimal configuration
        spark = (
            SparkSession.builder.appName("BinscatterDemo")
            .master("local[*]")
            .config("spark.sql.adaptive.enabled", "false")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        x, y = create_sample_data()

        # Convert to Spark DataFrame
        import pandas as pd

        pdf = pd.DataFrame({"x": x, "y": y})
        df_spark = spark.createDataFrame(pdf)

        result = binscatter(df_spark, "x", "y", num_bins=10, return_type="native")
        print("PySpark result:")
        result.show()
        print()
        spark.stop()

    except ImportError as e:
        logger.warning(f"PySpark demo skipped: {e}")
    except Exception as e:
        logger.error(f"PySpark demo failed: {e}")


def demo_plotly():
    """Demonstrate plotly output."""
    try:
        import polars as pl
        from binscatter import binscatter

        logger.info("Running Plotly demo...")
        x, y = create_sample_data()
        df = pl.DataFrame({"x": x, "y": y})

        fig = binscatter(df, "x", "y", num_bins=10, return_type="plotly")
        print("Plotly figure created successfully!")

        # Optionally save or show
        # fig.write_html("binscatter_demo.html")
        fig.show()  # Opens in browser if available

    except ImportError as e:
        logger.warning(f"Plotly demo skipped: {e}")
    except Exception as e:
        logger.error(f"Plotly demo failed: {e}")


def main():
    """Run all available demos."""
    logger.info("Starting binscatter demos...")

    demos = [
        ("Polars", demo_polars),
        ("DuckDB", demo_duckdb),
        ("PySpark", demo_pyspark),
        ("Plotly", demo_plotly),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error in {name} demo: {e}")

    logger.info("Demo completed!")


if __name__ == "__main__":
    main()
