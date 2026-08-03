"""Property-based tests for :func:`binscatter` using Hypothesis (#53).

These complement the example-based tests by asserting invariants that must hold
for *any* input, rather than for hand-picked frames. The properties fall into
three groups:

* **Structural** -- bin counts, ordering, and range containment.
* **Invariance** -- results that must not change under row permutation or under
  rescaling of a control.
* **Equivariance** -- results that must transform predictably when ``x`` or
  ``y`` is put through an affine map.
* **Missing data** -- rows carrying a null or non-finite value in any consumed
  column are dropped, so they cannot influence the result.
* **Categorical controls** -- string controls depend only on the partition they
  induce, not on the labels themselves.

The ``x`` strategy deliberately generates well-separated values (a cumulative
sum of positive gaps). ``binscatter`` legitimately reduces the bin count when
quantile edges collide, so guaranteeing separation keeps these tests focused on
the invariants above instead of re-testing the deduplication path, which
``tests/test_binscatter.py`` already covers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from binscatter.core import binscatter

# Binscatter is not cheap, so keep the example count modest and disable the
# per-example deadline (the first call in a run pays import/JIT-style costs).
PROPERTY_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_Y_BOUND = 1e6

finite_y = st.floats(
    min_value=-_Y_BOUND,
    max_value=_Y_BOUND,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def frames(draw, with_control: bool = False):
    """Draw a frame with strictly increasing, well-separated ``x``."""
    num_bins = draw(st.integers(min_value=2, max_value=8))
    n = draw(st.integers(min_value=num_bins * 5, max_value=num_bins * 15))

    start = draw(st.floats(min_value=-1e3, max_value=1e3))
    gaps = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=100.0),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    x = np.concatenate([[start], start + np.cumsum(gaps)])

    y = np.asarray(draw(st.lists(finite_y, min_size=n, max_size=n)), dtype=float)

    data = {"x": x, "y": y}
    if with_control:
        data["z"] = np.asarray(
            draw(st.lists(finite_y, min_size=n, max_size=n)), dtype=float
        )
    return pd.DataFrame(data), num_bins


#: Values that ``_remove_bad_values`` must strip out. ``None`` becomes ``NaN``
#: once pandas coerces the column to float, which is the point -- both routes
#: into "missing" should behave identically.
BAD_VALUES = [float("nan"), float("inf"), float("-inf"), None]

LABELS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


@st.composite
def bad_rows(draw, columns: list[str]):
    """Draw rows that each carry at least one null / non-finite value."""
    count = draw(st.integers(min_value=1, max_value=4))
    rows = []
    for _ in range(count):
        row = {col: draw(finite_y) for col in columns}
        # Corrupt at least one column; possibly more.
        corrupted = draw(
            st.lists(st.sampled_from(columns), min_size=1, max_size=len(columns))
        )
        for col in corrupted:
            row[col] = draw(st.sampled_from(BAD_VALUES))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


@st.composite
def categorical_frames(draw):
    """Draw a frame with a string control column."""
    num_bins = draw(st.integers(min_value=2, max_value=5))
    num_levels = draw(st.integers(min_value=2, max_value=4))
    # Keep plenty of rows per (bin, level) cell so the design stays full rank.
    n = draw(st.integers(min_value=num_bins * 20, max_value=num_bins * 40))

    start = draw(st.floats(min_value=-1e3, max_value=1e3))
    gaps = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=100.0),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    x = np.concatenate([[start], start + np.cumsum(gaps)])
    y = np.asarray(draw(st.lists(finite_y, min_size=n, max_size=n)), dtype=float)

    levels = LABELS[:num_levels]
    g = np.asarray(
        draw(st.lists(st.sampled_from(levels), min_size=n, max_size=n)), dtype=object
    )
    # Every level must appear, otherwise relabelling is not a bijection on the
    # observed data and the comparison below is not meaningful.
    assume(len(set(g.tolist())) == num_levels)

    return pd.DataFrame({"x": x, "y": y, "g": g}), num_bins, levels


def _run(df: pd.DataFrame, num_bins: int, controls=None) -> pd.DataFrame:
    result = binscatter(
        df,
        "x",
        "y",
        num_bins=num_bins,
        controls=controls,
        return_type="native",
    )
    return result.sort_values("bin").reset_index(drop=True)


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------


@pytest.mark.quick
@PROPERTY_SETTINGS
@given(frames())
def test_produces_requested_bins_with_contiguous_labels(case):
    df, num_bins = case
    result = _run(df, num_bins)

    assert result.shape[0] == num_bins
    np.testing.assert_array_equal(
        np.sort(result["bin"].to_numpy()), np.arange(num_bins)
    )


@PROPERTY_SETTINGS
@given(frames())
def test_bin_x_means_are_strictly_increasing(case):
    df, num_bins = case
    result = _run(df, num_bins)

    x_means = result["x"].to_numpy()
    assert np.all(np.diff(x_means) > 0)


@PROPERTY_SETTINGS
@given(frames())
def test_bin_means_stay_within_the_data_range(case):
    df, num_bins = case
    result = _run(df, num_bins)

    for col in ("x", "y"):
        lo, hi = df[col].min(), df[col].max()
        span = max(hi - lo, 1.0)
        values = result[col].to_numpy()
        assert np.all(values >= lo - 1e-9 * span)
        assert np.all(values <= hi + 1e-9 * span)


# --------------------------------------------------------------------------
# Invariance properties
# --------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(frames(), st.integers(min_value=0, max_value=2**32 - 1))
def test_row_order_does_not_affect_result(case, seed):
    df, num_bins = case
    expected = _run(df, num_bins)

    permutation = np.random.default_rng(seed).permutation(df.shape[0])
    shuffled = df.iloc[permutation].reset_index(drop=True)
    actual = _run(shuffled, num_bins)

    np.testing.assert_allclose(
        actual["x"].to_numpy(), expected["x"].to_numpy(), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-9, atol=1e-9
    )


@PROPERTY_SETTINGS
@given(
    frames(with_control=True),
    st.floats(min_value=-1e3, max_value=1e3),
    st.floats(min_value=0.01, max_value=100.0),
)
def test_control_is_invariant_to_its_own_affine_rescaling(case, shift, scale):
    """Shifting or scaling a control leaves the partialled-out fit unchanged.

    The control enters only through ``beta + mean(W) @ gamma``; an affine
    change to ``W`` moves ``beta``, ``gamma`` and ``mean(W)`` so that the
    fitted values cancel out exactly.
    """
    df, num_bins = case
    expected = _run(df, num_bins, controls=["z"])

    rescaled = df.assign(z=df["z"] * scale + shift)
    actual = _run(rescaled, num_bins, controls=["z"])

    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-7, atol=1e-7
    )


# --------------------------------------------------------------------------
# Equivariance properties
# --------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    frames(),
    st.floats(min_value=-100.0, max_value=100.0),
    st.floats(min_value=-1e3, max_value=1e3),
)
def test_affine_map_on_y_carries_through_to_bin_means(case, scale, shift):
    df, num_bins = case
    assume(abs(scale) > 1e-6)

    expected = _run(df, num_bins)
    actual = _run(df.assign(y=df["y"] * scale + shift), num_bins)

    np.testing.assert_allclose(
        actual["y"].to_numpy(),
        expected["y"].to_numpy() * scale + shift,
        rtol=1e-7,
        atol=1e-7,
    )


@PROPERTY_SETTINGS
@given(
    frames(),
    st.floats(min_value=0.01, max_value=100.0),
    st.floats(min_value=-1e3, max_value=1e3),
)
def test_increasing_affine_map_on_x_moves_x_but_not_y(case, scale, shift):
    df, num_bins = case

    expected = _run(df, num_bins)
    actual = _run(df.assign(x=df["x"] * scale + shift), num_bins)

    # A positive scale preserves the ordering of x, so the bins hold exactly the
    # same rows: x means follow the map and y means are untouched.
    np.testing.assert_allclose(
        actual["x"].to_numpy(),
        expected["x"].to_numpy() * scale + shift,
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-9, atol=1e-9
    )


def _sits_on_a_bin_edge(x: np.ndarray, num_bins: int) -> bool:
    """Whether any observation lands exactly on an interior quantile edge."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, num_bins + 1))[1:-1]
    if edges.size == 0:
        return False
    spread = float(np.ptp(x)) or 1.0
    return bool(np.any(np.abs(x[:, None] - edges[None, :]) <= 1e-12 * spread))


@PROPERTY_SETTINGS
@given(frames())
def test_reversing_x_reverses_the_bin_ordering(case):
    """Negating ``x`` reverses the bin ordering, so bin ``k`` becomes ``J-1-k``.

    This holds only when no observation sits exactly on an interior quantile
    edge. Bins are left-closed (``[lo, hi)``, i.e. ``right=False``), and
    reflection turns that into right-closed, so a point sitting exactly on an
    edge switches sides. That is a deliberate convention rather than a bug, so
    those inputs are excluded instead of being asserted over.
    """
    df, num_bins = case
    assume(not _sits_on_a_bin_edge(df["x"].to_numpy(), num_bins))

    expected = _run(df, num_bins)
    actual = _run(df.assign(x=-df["x"]), num_bins)

    np.testing.assert_allclose(
        actual["x"].to_numpy(),
        -expected["x"].to_numpy()[::-1],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual["y"].to_numpy(),
        expected["y"].to_numpy()[::-1],
        rtol=1e-9,
        atol=1e-9,
    )


# --------------------------------------------------------------------------
# Validation properties
# --------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(frames(), st.integers(min_value=-100, max_value=1))
def test_non_positive_bin_counts_are_rejected(case, num_bins):
    df, _ = case
    with pytest.raises(ValueError):
        binscatter(df, "x", "y", num_bins=num_bins, return_type="native")


# --------------------------------------------------------------------------
# Missing / non-finite data
# --------------------------------------------------------------------------


@pytest.mark.quick
@PROPERTY_SETTINGS
@given(data=st.data())
def test_rows_with_missing_or_non_finite_values_are_ignored(data):
    """Appending unusable rows must not change the result.

    ``_remove_bad_values`` drops them before any quantile is computed, so the
    surviving frame is exactly the clean one.
    """
    df, num_bins = data.draw(frames(with_control=True))
    columns = ["x", "y", "z"]
    junk = data.draw(bad_rows(columns))

    expected = _run(df, num_bins, controls=["z"])
    # Cast before concatenating: an all-null junk column would otherwise stay
    # object-dtyped and trip pandas' empty/all-NA concat deprecation.
    polluted = pd.concat([df[columns], junk.astype(float)], ignore_index=True)
    actual = _run(polluted, num_bins, controls=["z"])

    np.testing.assert_allclose(
        actual["x"].to_numpy(), expected["x"].to_numpy(), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-7, atol=1e-7
    )


@PROPERTY_SETTINGS
@given(categorical_frames(), st.integers(min_value=0, max_value=2**32 - 1))
def test_missing_categorical_values_drop_their_rows(case, seed):
    """A null category drops the row, exactly as deleting it up front would."""
    df, num_bins, _ = case
    rng = np.random.default_rng(seed)
    victims = rng.choice(df.shape[0], size=min(3, df.shape[0]), replace=False)

    blanked = df.copy()
    blanked.loc[victims, "g"] = None
    removed = df.drop(index=victims).reset_index(drop=True)

    # Dropping rows can collapse quantile edges; only compare when both runs
    # produced the same bin count.
    actual = _run(blanked, num_bins, controls=["g"])
    expected = _run(removed, num_bins, controls=["g"])
    assume(actual.shape[0] == expected.shape[0])

    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-7, atol=1e-7
    )


# --------------------------------------------------------------------------
# Categorical (string) controls
# --------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(categorical_frames(), st.integers(min_value=0, max_value=2**32 - 1))
def test_renaming_categories_does_not_change_the_fit(case, seed):
    """Only the partition matters, not the labels.

    Relabelling can change which level sorts first, and therefore which dummy is
    dropped as the reference category -- but fitted values are invariant to that
    choice, so the bin means must not move.
    """
    df, num_bins, levels = case
    rng = np.random.default_rng(seed)

    # Map onto a fresh set of labels in a shuffled order, so the alphabetically
    # first level (the reference category) generally changes.
    replacements = list(LABELS[-len(levels) :])
    rng.shuffle(replacements)
    mapping = dict(zip(levels, replacements))

    expected = _run(df, num_bins, controls=["g"])
    actual = _run(df.assign(g=df["g"].map(mapping)), num_bins, controls=["g"])

    np.testing.assert_allclose(
        actual["x"].to_numpy(), expected["x"].to_numpy(), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-7, atol=1e-7
    )


@PROPERTY_SETTINGS
@given(categorical_frames())
def test_constant_categorical_control_is_a_no_op(case):
    """A single-level control carries no information, so it changes nothing."""
    df, num_bins, _ = case

    expected = _run(df.drop(columns="g"), num_bins)
    actual = _run(df.assign(g="only-one-level"), num_bins, controls=["g"])

    np.testing.assert_allclose(
        actual["y"].to_numpy(), expected["y"].to_numpy(), rtol=1e-7, atol=1e-7
    )
