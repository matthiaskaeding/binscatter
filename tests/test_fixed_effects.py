"""Tests for absorbing a categorical control via the Mundlak / within transform.

Absorption is a silent rewrite of an existing code path: the model is unchanged and
nothing raises if it is subtly wrong, the plotted values just quietly move. So the
gate is equivalence against the one-hot path, checked on every backend, backed by an
independent statsmodels oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import binscatter.fixed_effects as fe_mod
from binscatter import binscatter, core
from binscatter.fixed_effects import (
    demean_centered,
    demean_within,
    group_codes,
    within_correct,
)
from tests.conftest import DF_BACKENDS, convert_to_backend

DF_TYPE_PARAMS = [
    pytest.param(df_type) for df_type in [b for b in DF_BACKENDS if b != "pyspark"]
]
if "pyspark" in DF_BACKENDS:
    DF_TYPE_PARAMS.append(pytest.param("pyspark", marks=pytest.mark.pyspark))

# Distributed engines use approximate quantiles, so bins (and therefore fitted
# values) can differ slightly from the exact backends.
LOOSE_BACKENDS = {"dask", "pyspark"}


def tolerances(df_type, reshaped_input=False):
    """Tolerances for comparing two runs.

    ``reshaped_input`` marks comparisons where the two runs see *different* frames
    (shuffled rows, dropped rows). On dask and PySpark the quantile edges are
    approximate, so the bin boundaries themselves move and only a loose comparison
    is meaningful -- matching the convention used elsewhere in the suite.
    """
    if df_type not in LOOSE_BACKENDS:
        return 1e-9, 1e-9
    return (0.1, 0.15) if reshaped_input else (1e-6, 1e-6)


def make_panel(n=2000, n_groups=25, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    region = rng.integers(0, 3, n)
    x = rng.normal(size=n)
    age = rng.normal(size=n)
    y = (
        0.8 * x
        + 1.5 * age
        + rng.normal(scale=2.0, size=n_groups)[g]
        + rng.normal(scale=1.0, size=3)[region]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "age": age,
            "grp": [f"g{i:03d}" for i in g],
            "region": [f"r{i}" for i in region],
        }
    )


def run_native(df, **kwargs):
    out = binscatter(df, "x", "y", return_type="native", **kwargs)
    pdf = out if isinstance(out, pd.DataFrame) else _to_pandas(out)
    return pdf.sort_values("bin")["y"].to_numpy()


def _to_pandas(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    for attr in ("to_pandas", "df", "compute"):
        if hasattr(obj, attr):
            return getattr(obj, attr)()
    raise TypeError(f"cannot convert {type(obj)}")


def without_absorption(monkeypatch):
    """Force the legacy one-hot path so the two can be compared."""
    monkeypatch.setattr(core, "select_absorbed_factors", lambda *a, **k: ())


@pytest.fixture
def force_absorption(monkeypatch):
    """Absorb even tiny categoricals.

    Production only absorbs at ``ABSORB_MIN_LEVELS`` or more, but the one-hot path
    it is compared against costs O(G^2) aggregations, so the equivalence tests would
    be unusably slow at a realistic threshold. Lowering it here exercises exactly the
    same code on small data.
    """
    monkeypatch.setattr(fe_mod, "ABSORB_MIN_LEVELS", 2)


# --------------------------------------------------------------------------
# 1. Equivalence: the gate
# --------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
@pytest.mark.parametrize(
    "controls",
    [["grp"], ["age", "grp"], ["age", "grp", "region"]],
    ids=["cat_only", "cat_plus_numeric", "two_cats_plus_numeric"],
)
def test_absorbed_matches_one_hot(df_type, controls, monkeypatch, force_absorption):
    df = convert_to_backend(make_panel(), df_type)
    absorbed = run_native(df, controls=controls, num_bins=6)
    without_absorption(monkeypatch)
    one_hot = run_native(df, controls=controls, num_bins=6)

    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_x_means_unchanged_by_absorption(df_type, monkeypatch, force_absorption):
    """Absorption touches y only; the per-bin x means must not move at all."""
    df = convert_to_backend(make_panel(), df_type)
    absorbed = binscatter(
        df, "x", "y", controls=["grp"], num_bins=6, return_type="native"
    )
    without_absorption(monkeypatch)
    one_hot = binscatter(
        df, "x", "y", controls=["grp"], num_bins=6, return_type="native"
    )

    a = _to_pandas(absorbed).sort_values("bin")["x"].to_numpy()
    b = _to_pandas(one_hot).sort_values("bin")["x"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.quick
def test_absorbed_matches_statsmodels_oracle(force_absorption):
    """Independent check against explicit bin + fixed-effect dummies."""
    sm = pytest.importorskip("statsmodels.api")
    df = make_panel(n=1500, n_groups=12, seed=3)
    num_bins = 6

    fitted = run_native(df, controls=["age", "grp"], num_bins=num_bins)

    # Rebuild the same design by hand and fit it densely.
    edges = np.quantile(df["x"], np.linspace(0, 1, num_bins + 1))
    bin_idx = np.clip(
        np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1
    )
    bin_dummies = pd.get_dummies(pd.Series(bin_idx), prefix="bin", drop_first=False)
    grp_dummies = pd.get_dummies(df["grp"], prefix="grp", drop_first=True)
    age = df["age"].to_numpy()[:, None]
    design = np.column_stack(
        [bin_dummies.to_numpy(), age, grp_dummies.to_numpy()]
    ).astype(float)
    theta = sm.OLS(df["y"].to_numpy(), design).fit().params
    beta = theta[:num_bins]
    gamma = theta[num_bins:]
    means = np.concatenate([[age.mean()], grp_dummies.mean().to_numpy()])
    expected = beta + means @ gamma

    np.testing.assert_allclose(fitted, expected, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# 2. Ordering and label handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_row_order_does_not_change_result(df_type, force_absorption):
    """group_by output order is not stable across backends; labels must be mapped."""
    df = make_panel(n=1200, n_groups=15, seed=7)
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    a = run_native(convert_to_backend(df, df_type), controls=["age", "grp"], num_bins=5)
    b = run_native(
        convert_to_backend(shuffled, df_type), controls=["age", "grp"], num_bins=5
    )
    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(lambda i: f"g{i}", id="string"),
        pytest.param(lambda i: i, id="int"),
        pytest.param(lambda i: f"grüße-{i}/ü", id="unicode"),
        pytest.param(lambda i: "L" * 200 + str(i), id="very_long"),
    ],
)
def test_group_label_types(labels, force_absorption):
    """Group labels are dict keys, so exotic values must round-trip."""
    df = make_panel(n=800, n_groups=6, seed=5)
    codes = df["grp"].str.replace("g", "").astype(int)
    df["grp"] = [labels(int(c)) for c in codes]
    kwargs = {"controls": ["age", "grp"], "num_bins": 4}
    if not isinstance(labels(0), str):
        kwargs["categorical"] = ["grp"]
    out = run_native(df, **kwargs)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# 3. Degenerate structure
# --------------------------------------------------------------------------


def test_singleton_groups(force_absorption):
    """Levels with a single observation demean to exactly zero."""
    df = make_panel(n=400, n_groups=5, seed=9)
    # Give 20 rows their own private group.
    df.loc[:19, "grp"] = [f"solo{i}" for i in range(20)]
    out = run_native(df, controls=["age", "grp"], num_bins=4)
    assert np.all(np.isfinite(out))


def test_level_confined_to_single_bin(force_absorption):
    """A level appearing in only one bin is collinear with that bin."""
    df = make_panel(n=600, n_groups=6, seed=13)
    lowest = df["x"] < df["x"].quantile(0.15)
    df.loc[lowest, "grp"] = "only_low"
    out = run_native(df, controls=["grp"], num_bins=4)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_constant_categorical_is_not_absorbed(df_type, force_absorption):
    """A one-level categorical carries no information and must be a no-op."""
    df = make_panel(n=600, seed=17)
    df["const"] = "same"
    with_const = run_native(
        convert_to_backend(df, df_type), controls=["age", "const"], num_bins=5
    )
    without = run_native(convert_to_backend(df, df_type), controls=["age"], num_bins=5)
    np.testing.assert_allclose(with_const, without, rtol=1e-9, atol=1e-9)


def test_disconnected_bin_group_structure(force_absorption):
    """Bins and groups forming disconnected blocks stay finite and identified."""
    rng = np.random.default_rng(21)
    n = 800
    x = rng.normal(size=n)
    half = x > np.median(x)
    grp = np.where(
        half,
        "hi_" + (rng.integers(0, 3, n)).astype(str),
        "lo_" + (rng.integers(0, 3, n)).astype(str),
    )
    df = pd.DataFrame({"x": x, "y": rng.normal(size=n) + half * 3.0, "grp": grp})
    out = run_native(df, controls=["grp"], num_bins=4)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# 4. Level recovery
# --------------------------------------------------------------------------


def test_fitted_values_stay_on_y_scale(force_absorption):
    """Without the mean fixed effect added back, fitted values lose their level."""
    df = make_panel(n=1500, n_groups=20, seed=23)
    df["y"] = df["y"] + 500.0  # large offset carried entirely by the fixed effects
    out = run_native(df, controls=["grp"], num_bins=5)
    assert 400.0 < out.mean() < 600.0


def test_group_shift_moves_level_not_bin_effects(force_absorption):
    """The defining fixed-effect property: a within-group shift is absorbed."""
    df = make_panel(n=1500, n_groups=20, seed=29)
    base = run_native(df, controls=["age", "grp"], num_bins=5)

    shifted = df.copy()
    target = shifted["grp"] == shifted["grp"].iloc[0]
    shifted.loc[target, "y"] += 100.0

    out = run_native(shifted, controls=["age", "grp"], num_bins=5)
    # Bin effects are unchanged once the common level shift is removed.
    np.testing.assert_allclose(
        out - out.mean(), base - base.mean(), rtol=1e-6, atol=1e-6
    )


# --------------------------------------------------------------------------
# 5. Interactions with the other consumers of regression_features
# --------------------------------------------------------------------------


def test_rot_selector_matches_one_hot(monkeypatch, force_absorption):
    """The rule-of-thumb selector is moment-based, so absorption is exact for it."""
    df = make_panel(n=2000, n_groups=15, seed=31)
    absorbed = binscatter(
        df, "x", "y", controls=["age", "grp"], num_bins="rot", return_type="native"
    )
    without_absorption(monkeypatch)
    one_hot = binscatter(
        df, "x", "y", controls=["age", "grp"], num_bins="rot", return_type="native"
    )
    assert len(absorbed) == len(one_hot)
    np.testing.assert_allclose(
        absorbed.sort_values("bin")["y"].to_numpy(),
        one_hot.sort_values("bin")["y"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_dpi_selector_matches_one_hot(monkeypatch, force_absorption):
    """DPI is invariant to representing the same categorical by dummies or absorption."""
    df = make_panel(n=2000, n_groups=15, seed=31)
    absorbed = binscatter(
        df, "x", "y", controls=["age", "grp"], num_bins="dpi", return_type="native"
    )
    without_absorption(monkeypatch)
    one_hot = binscatter(
        df, "x", "y", controls=["age", "grp"], num_bins="dpi", return_type="native"
    )
    assert len(absorbed) == len(one_hot)
    np.testing.assert_allclose(
        absorbed.sort_values("bin")["y"].to_numpy(),
        one_hot.sort_values("bin")["y"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_dpi_variance_matches_centered_dummy_oracle():
    """The absorbed sandwich equals explicit centered fixed-effect dummies."""
    df = make_panel(n=1800, n_groups=12, seed=67)
    num_bins = 9
    edges = np.quantile(df["x"], np.linspace(0, 1, num_bins + 1))
    bin_idx = np.clip(
        np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1
    )
    bin_counts = np.bincount(bin_idx, minlength=num_bins)
    y = df["y"].to_numpy()
    age = df["age"].to_numpy()

    dummies = pd.get_dummies(df["grp"], drop_first=True).to_numpy(dtype=float)
    explicit_controls = np.column_stack([age, dummies])
    explicit_controls -= explicit_controls.mean(axis=0)
    expected = core._compute_dpi_variance_constant(
        y, explicit_controls, bin_idx, bin_counts
    )

    codes, counts = group_codes(df["grp"].to_numpy())
    absorbed_y = demean_centered(y, codes, counts)
    centered_age = age - age.mean()
    absorbed_controls = demean_centered(centered_age, codes, counts)[:, None]
    actual = core._compute_dpi_variance_constant(
        absorbed_y,
        absorbed_controls,
        bin_idx,
        bin_counts,
        codes,
        counts,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_dpi_one_hot_is_invariant_to_reference_label(monkeypatch):
    """Relabeling categories cannot change DPI by changing the omitted dummy."""
    without_absorption(monkeypatch)
    df = make_panel(n=1800, n_groups=15, seed=71)
    relabeled = df.copy()
    labels = sorted(df["grp"].unique())
    relabeled["grp"] = relabeled["grp"].map(
        dict(zip(labels, reversed(labels), strict=True))
    )

    original = run_native(df, controls=["age", "grp"], num_bins="dpi")
    changed_reference = run_native(relabeled, controls=["age", "grp"], num_bins="dpi")
    np.testing.assert_allclose(original, changed_reference, rtol=1e-10, atol=1e-10)


def test_absorption_threshold_preserves_existing_behaviour(monkeypatch):
    """Below the threshold nothing is absorbed, so results cannot move."""
    df = make_panel(n=1500, n_groups=20, seed=59)
    assert (
        fe_mod.select_absorbed_factors(
            core.clean_df(df, ("grp",), "x", "y")[0], ("grp",)
        )
        == ()
    )

    default = run_native(df, controls=["age", "grp"], num_bins="dpi")
    without_absorption(monkeypatch)
    legacy = run_native(df, controls=["age", "grp"], num_bins="dpi")
    np.testing.assert_allclose(default, legacy, rtol=1e-12, atol=1e-12)


def test_poly_line_with_absorbed_fixed_effect(monkeypatch, force_absorption):
    df = make_panel(n=1500, n_groups=12, seed=37)
    fig = binscatter(df, "x", "y", controls=["age", "grp"], num_bins=5, poly_line=2)
    absorbed_line = fig.data[1].y

    without_absorption(monkeypatch)
    fig2 = binscatter(df, "x", "y", controls=["age", "grp"], num_bins=5, poly_line=2)
    np.testing.assert_allclose(absorbed_line, fig2.data[1].y, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# 6. Nulls
# --------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_nulls_in_fixed_effect_column_are_dropped(df_type, force_absorption):
    df = make_panel(n=900, n_groups=8, seed=41)
    df.loc[:49, "grp"] = None
    clean = df.dropna(subset=["grp"])

    with_nulls = run_native(
        convert_to_backend(df, df_type), controls=["grp"], num_bins=4
    )
    dropped = run_native(
        convert_to_backend(clean, df_type), controls=["grp"], num_bins=4
    )
    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(with_nulls, dropped, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# 7. The point of the change
# --------------------------------------------------------------------------


def test_high_cardinality_completes():
    """5k levels would build ~12M aggregations on the one-hot path."""
    rng = np.random.default_rng(43)
    n, n_groups = 50_000, 5_000
    g = rng.integers(0, n_groups, n)
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": 0.7 * x + rng.normal(scale=2.0, size=n_groups)[g] + rng.normal(size=n),
            "firm_id": g,
        }
    )
    # Assert the routing decision directly rather than by timing it: if this ever
    # falls back to one-hot the test would still pass, just take minutes.
    lazy, _, _, categorical = core.clean_df(df, ("firm_id",), "x", "y", ("firm_id",))
    assert fe_mod.select_absorbed_factors(lazy, categorical) == ("firm_id",)

    out = run_native(df, controls=["firm_id"], categorical=["firm_id"], num_bins=10)
    assert out.shape == (10,)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# 8. Unit tests for the algebra itself
# --------------------------------------------------------------------------


def test_within_correct_matches_explicit_demeaning():
    rng = np.random.default_rng(47)
    n, g_count = 500, 20
    groups = rng.integers(0, g_count, n)
    A = rng.normal(size=(n, 3))
    B = rng.normal(size=(n, 2))

    codes, counts = group_codes(groups)
    S_A = np.zeros((g_count, 3))
    S_B = np.zeros((g_count, 2))
    for j in range(3):
        np.add.at(S_A[:, j], codes, A[:, j])
    for j in range(2):
        np.add.at(S_B[:, j], codes, B[:, j])

    got = within_correct(A.T @ B, S_A, S_B, 1.0 / counts)

    A_d = np.column_stack([demean_within(A[:, j], codes, counts) for j in range(3)])
    B_d = np.column_stack([demean_within(B[:, j], codes, counts) for j in range(2)])
    np.testing.assert_allclose(got, A_d.T @ B_d, rtol=1e-10, atol=1e-10)


def test_demean_within_zeroes_group_means():
    rng = np.random.default_rng(53)
    groups = rng.integers(0, 7, 300)
    values = rng.normal(size=300)
    codes, counts = group_codes(groups)
    out = demean_within(values, codes, counts)
    sums = np.bincount(codes, weights=out, minlength=counts.size)
    np.testing.assert_allclose(sums, 0.0, atol=1e-10)


def test_demean_centered_preserves_mean_and_equalizes_group_means():
    values = np.array([1.0, 3.0, 7.0, 9.0, 11.0])
    codes = np.array([0, 0, 1, 1, 1])
    counts = np.bincount(codes)

    result = demean_centered(values, codes, counts)

    expected_mean = values.mean()
    assert result.mean() == pytest.approx(expected_mean)
    for group in range(counts.size):
        assert result[codes == group].mean() == pytest.approx(expected_mean)
