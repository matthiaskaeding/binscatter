# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyreadr>=0.5.3",
# ]
# ///
"""Build the small NHANES dataset used by the README example.

The pinned source is the educational ``NHANES`` dataframe from version 2.1.0 of the
R package of the same name. Its reader is a script-only dependency and is not added
to the binscatter package.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

import pyreadr

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "nhanes_age_bp.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/cran/NHANES/"
    "0d8e4c9f9bd0fc2e4737cb60165d29de8a3eb5aa/data/NHANES.rda"
)
SOURCE_SHA256 = "a553613f1affafdfb3f903f1924643a4b4c59ed7ca8a95ca828f8d118391ee3b"
EXPECTED_ROWS = 8_551


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read()

    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        msg = f"Unexpected source checksum: {digest}"
        raise RuntimeError(msg)

    with tempfile.NamedTemporaryFile(suffix=".rda") as rdata:
        rdata.write(source)
        rdata.flush()
        nhanes = pyreadr.read_r(rdata.name)["NHANES"]

    example = (
        nhanes.loc[:, ["Age", "BPSysAve"]]
        .dropna()
        .rename(columns={"Age": "age", "BPSysAve": "systolic_bp"})
        .astype(int)
    )
    if len(example) != EXPECTED_ROWS:
        msg = f"Expected {EXPECTED_ROWS:,} complete rows, found {len(example):,}"
        raise RuntimeError(msg)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(OUTPUT, index=False, lineterminator="\n")

    print(f"Wrote {len(example):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
