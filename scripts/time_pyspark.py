import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from pyspark.sql import SparkSession
from binscatter import binscatter
import time
from math import ceil

N = 100_000_000
CHUNK_SIZE = 1000_000
N_chunks = ceil(N / CHUNK_SIZE)
spark = SparkSession.builder.getOrCreate()

with tempfile.TemporaryDirectory() as dirname:
    dir = Path(dirname)
    print("Writing test files to chunks")
    for i in range(N_chunks):
        if i % 50 == 0:
            print(f"Chunk {i} / {N_chunks} ")
        x = np.random.normal(0, 1, CHUNK_SIZE)
        y = np.random.normal(0, 1, CHUNK_SIZE)
        df = pl.DataFrame({"x": x, "y": y})
        file = dir / f"df{i}.parquet"
        df.write_parquet(file)

    df = spark.read.parquet(dirname)
    print(df)

    print("Making binscatter - timing starts here")
    t0 = time.perf_counter()
    p = binscatter(df, "x", "y")
    dt = time.perf_counter() - t0
    print(f"binscatter took {dt:.3f} s")
    p.write_html("binscatter.html", auto_open=True, include_plotlyjs="cdn")
