# test_add_fallback.py
import numpy as np
import pandas as pd
import narwhals as nw
import narwhals.selectors as ncs

from binscatter.core import prep, _add_fallback, _remove_bad_values, _make_probs


def test_add_fallback_adds_bins_and_is_monotonic():
    # Build a simple, well-behaved dataset (unique quantiles)
    rng = np.random.default_rng(0)
    df_in = pd.DataFrame(
        {
            "x": np.arange(1, 101, dtype=float),  # 1..100
            "y": rng.normal(loc=0.0, scale=1.0, size=100),
        }
    )

    # Use prep to get a real Profile (and implicitly validate inputs)
    # We don't need the returned df_with_bins here, only the profile.
    # This seems bad but okay.
    _, profile = prep(df_in, x_name="x", y_name="y", controls=(), num_bins=4)

    # Recreate the filtered lazy frame exactly like prep does
    dfn = nw.from_native(df_in)
    dfl = dfn.lazy()
    df_xy = dfl.select("x", "y")
    cols_numeric = tuple(df_xy.select(ncs.numeric()).columns)
    cols_cat = tuple(df_xy.select(ncs.categorical()).columns)
    df_filtered = _remove_bad_values(df_xy, cols_numeric, cols_cat)

    # Same probabilities prep/configure_quantile_handler would use
    probs = _make_probs(profile.num_bins)

    # Run the fallback path we want to test
    lf_out = _add_fallback(df_filtered, profile, probs)
    out = lf_out.collect()  # narwhals DataFrame

    # 1) Bin column is present and row count preserved
    assert profile.bin_name in out.columns
    assert out.shape[0] == df_in.shape[0]

    out = out.sort("x")
    assert out.item(0, profile.bin_name) == 0
    assert out.item(out.shape[0] - 1, profile.bin_name) == 3

    assert "x" in out.columns, f"x not found - columns: {out.columns}"
    # 2) Bins should be non-decreasing when x is sorted (join_asof forward fill)
    xs = out.get_column("x").to_list()
    bins = out.get_column(profile.bin_name).to_list()
    assert xs == sorted(xs), "Expected output to be sorted by x"
    assert all(curr >= prev for prev, curr in zip(bins, bins[1:]))

    assert set(bins) == {0, 1, 2, 3}

    bin_counts = (
        out.group_by(profile.bin_name)
        .agg(nw.len().alias("count"))
        .sort(profile.bin_name)
        .get_column("count")
        .to_list()
    )
    prev = bin_counts[0]
    for i in range(1, 4):
        curr = bin_counts[i]
        assert abs(curr - prev) < 1, (
            f"Groups must be roughly equal. Sizes = {bin_counts}"
        )
