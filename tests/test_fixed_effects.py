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
    FEProjector,
    demean_centered,
    demean_within,
    factor_codes,
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
    """Force the one-hot path so the two can be compared.

    Every categorical control is absorbed in production, so the one-hot route
    survives only as the oracle these equivalence tests measure against.
    """
    monkeypatch.setattr(core, "select_absorbed", lambda *a, **k: ())


# --------------------------------------------------------------------------
# 1. Equivalence: the gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
@pytest.mark.parametrize(
    "controls",
    [["grp"], ["age", "grp"], ["age", "grp", "region"]],
    ids=["cat_only", "cat_plus_numeric", "two_cats_plus_numeric"],
)
def test_absorbed_matches_one_hot(df_type, controls, monkeypatch):
    df = convert_to_backend(make_panel(), df_type)
    absorbed = run_native(df, controls=controls, num_bins=6)
    without_absorption(monkeypatch)
    one_hot = run_native(df, controls=controls, num_bins=6)

    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_x_means_unchanged_by_absorption(df_type, monkeypatch):
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


def test_absorbed_matches_statsmodels_oracle():
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
def test_row_order_does_not_change_result(df_type):
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
def test_group_label_types(labels):
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


def test_singleton_groups():
    """Levels with a single observation demean to exactly zero."""
    df = make_panel(n=400, n_groups=5, seed=9)
    # Give 20 rows their own private group.
    df.loc[:19, "grp"] = [f"solo{i}" for i in range(20)]
    out = run_native(df, controls=["age", "grp"], num_bins=4)
    assert np.all(np.isfinite(out))


def test_level_confined_to_single_bin():
    """A level appearing in only one bin is collinear with that bin."""
    df = make_panel(n=600, n_groups=6, seed=13)
    lowest = df["x"] < df["x"].quantile(0.15)
    df.loc[lowest, "grp"] = "only_low"
    out = run_native(df, controls=["grp"], num_bins=4)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_constant_categorical_is_not_absorbed(df_type):
    """A one-level categorical carries no information and must be a no-op."""
    df = make_panel(n=600, seed=17)
    df["const"] = "same"
    with_const = run_native(
        convert_to_backend(df, df_type), controls=["age", "const"], num_bins=5
    )
    without = run_native(convert_to_backend(df, df_type), controls=["age"], num_bins=5)
    np.testing.assert_allclose(with_const, without, rtol=1e-9, atol=1e-9)


def test_disconnected_bin_group_structure():
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


def test_fitted_values_stay_on_y_scale():
    """Without the mean fixed effect added back, fitted values lose their level."""
    df = make_panel(n=1500, n_groups=20, seed=23)
    df["y"] = df["y"] + 500.0  # large offset carried entirely by the fixed effects
    out = run_native(df, controls=["grp"], num_bins=5)
    assert 400.0 < out.mean() < 600.0


def test_group_shift_moves_level_not_bin_effects():
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


def test_rot_selector_matches_one_hot(monkeypatch):
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


def test_dpi_selector_matches_one_hot(monkeypatch):
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

    row_codes = factor_codes([df["grp"].to_numpy()])
    projector = FEProjector.from_row_codes(row_codes, ("grp",))
    absorbed_y = demean_centered(y, projector, row_codes)
    centered_age = age - age.mean()
    absorbed_controls = demean_centered(centered_age, projector, row_codes)[:, None]
    actual = core._compute_dpi_variance_constant(
        absorbed_y,
        absorbed_controls,
        bin_idx,
        bin_counts,
        projector,
        row_codes,
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


def test_absorption_matches_one_hot_at_low_cardinality(monkeypatch):
    """Absorbing a small categorical must not move results off the one-hot path."""
    df = make_panel(n=1500, n_groups=20, seed=59)
    assert fe_mod.select_absorbed(
        core.clean_df(df, ("grp",), "x", "y")[0], ("grp",)
    ) == ("grp",)

    default = run_native(df, controls=["age", "grp"], num_bins="dpi")
    without_absorption(monkeypatch)
    one_hot = run_native(df, controls=["age", "grp"], num_bins="dpi")
    np.testing.assert_allclose(default, one_hot, rtol=1e-9, atol=1e-9)


def test_poly_line_with_absorbed_fixed_effect(monkeypatch):
    df = make_panel(n=1500, n_groups=12, seed=37)
    fig = binscatter(df, "x", "y", controls=["age", "grp"], num_bins=5, poly_line=2)
    absorbed_line = fig.data[1].y

    without_absorption(monkeypatch)
    fig2 = binscatter(df, "x", "y", controls=["age", "grp"], num_bins=5, poly_line=2)
    np.testing.assert_allclose(absorbed_line, fig2.data[1].y, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# 6. Nulls
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_nulls_in_fixed_effect_column_are_dropped(df_type):
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
    assert fe_mod.select_absorbed(lazy, categorical) == ("firm_id",)

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

    projector = FEProjector.from_row_codes(codes[:, None], ("grp",))
    got = within_correct(A.T @ B, S_A, projector.solve(S_B))

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
    row_codes = codes[:, None]
    projector = FEProjector.from_row_codes(row_codes, ("grp",))

    result = demean_centered(values, projector, row_codes)

    expected_mean = values.mean()
    assert result.mean() == pytest.approx(expected_mean)
    for group in range(2):
        assert result[codes == group].mean() == pytest.approx(expected_mean)


# --------------------------------------------------------------------------
# 9. Several fixed effects at once
# --------------------------------------------------------------------------


def make_multiway_panel(n=3000, n_firms=30, n_years=8, n_regions=4, seed=0):
    """A panel with three crossed factors of quite different cardinality."""
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, n)
    year = rng.integers(0, n_years, n)
    region = rng.integers(0, n_regions, n)
    x = rng.normal(size=n)
    age = rng.normal(size=n)
    y = (
        0.8 * x
        + 1.5 * age
        + rng.normal(scale=2.0, size=n_firms)[firm]
        + rng.normal(scale=1.0, size=n_years)[year]
        + rng.normal(scale=0.7, size=n_regions)[region]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "age": age,
            "firm": [f"f{i:03d}" for i in firm],
            "year": [f"y{i}" for i in year],
            "region": [f"r{i}" for i in region],
        }
    )


def dense_projector(row_codes, names=("a", "b")):
    """Build a projector plus the dense dummy matrix ``D`` it stands in for."""
    projector = FEProjector.from_row_codes(row_codes, names)
    blocks = [
        np.eye(int(projector.counts[f].size))[row_codes[:, f]]
        for f in range(projector.num_factors)
    ]
    return projector, np.column_stack(blocks)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
@pytest.mark.parametrize(
    "controls",
    [
        ["firm", "year"],
        ["age", "firm", "year"],
        ["age", "firm", "year", "region"],
    ],
    ids=["two_way", "two_way_plus_numeric", "three_way_plus_numeric"],
)
def test_multiway_matches_one_hot(df_type, controls, monkeypatch):
    """The gate: absorbing several factors must equal encoding them all."""
    df = convert_to_backend(make_multiway_panel(), df_type)
    absorbed = run_native(df, controls=controls, num_bins=6)
    without_absorption(monkeypatch)
    one_hot = run_native(df, controls=controls, num_bins=6)

    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "factors", [("firm", "year"), ("firm", "year", "region")], ids=["two", "three"]
)
def test_multiway_matches_statsmodels_oracle(factors):
    """Independent check against explicit bin + fixed-effect dummies."""
    sm = pytest.importorskip("statsmodels.api")
    df = make_multiway_panel(n=2500, seed=3)
    num_bins = 6

    fitted = run_native(df, controls=["age", *factors], num_bins=num_bins)

    edges = np.quantile(df["x"], np.linspace(0, 1, num_bins + 1))
    bin_idx = np.clip(
        np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1
    )
    bin_dummies = pd.get_dummies(pd.Series(bin_idx), prefix="bin", drop_first=False)
    fe_dummies = [
        pd.get_dummies(df[name], prefix=name, drop_first=True) for name in factors
    ]
    age = df["age"].to_numpy()[:, None]
    design = np.column_stack(
        [bin_dummies.to_numpy(), age, *(d.to_numpy() for d in fe_dummies)]
    ).astype(float)
    theta = sm.OLS(df["y"].to_numpy(), design).fit().params
    beta = theta[:num_bins]
    gamma = theta[num_bins:]
    means = np.concatenate([[age.mean()], *(d.mean().to_numpy() for d in fe_dummies)])
    expected = beta + means @ gamma

    np.testing.assert_allclose(fitted, expected, rtol=1e-8, atol=1e-8)


def test_projector_solves_the_stacked_normal_equations():
    """``G alpha = rhs`` to machine precision, for several right-hand sides."""
    rng = np.random.default_rng(17)
    n = 4000
    row_codes = np.column_stack(
        [rng.integers(0, 40, n), rng.integers(0, 12, n), rng.integers(0, 5, n)]
    )
    projector = FEProjector.from_row_codes(row_codes, ("a", "b", "c"))
    values = rng.normal(size=(n, 4))
    rhs = np.concatenate(
        [
            np.column_stack(
                [
                    np.bincount(
                        row_codes[:, f],
                        weights=values[:, j],
                        minlength=projector.counts[f].size,
                    )
                    for j in range(values.shape[1])
                ]
            )
            for f in range(3)
        ]
    )

    alpha = projector.solve(rhs)
    residual = np.linalg.norm(projector.matvec(alpha) - rhs) / np.linalg.norm(rhs)
    assert residual < fe_mod.FE_TOL


def test_projector_cross_product_matches_explicit_projection():
    """``A'B - (D'A)' alpha_B`` equals ``A' M_D B`` with ``D`` materialized."""
    rng = np.random.default_rng(19)
    n = 1200
    row_codes = np.column_stack([rng.integers(0, 15, n), rng.integers(0, 6, n)])
    projector, D = dense_projector(row_codes)

    A = rng.normal(size=(n, 3))
    B = rng.normal(size=(n, 2))
    S_A = D.T @ A
    S_B = D.T @ B

    got = within_correct(A.T @ B, S_A, projector.solve(S_B))

    residual_maker = np.eye(n) - D @ np.linalg.pinv(D.T @ D) @ D.T
    expected = A.T @ residual_maker @ B
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_projector_one_way_is_the_closed_form():
    """A single factor must not iterate at all -- ``G`` is already diagonal."""
    rng = np.random.default_rng(23)
    n = 800
    row_codes = rng.integers(0, 12, n)[:, None]
    projector = FEProjector.from_row_codes(row_codes, ("grp",))
    rhs = rng.normal(size=(projector.total_levels, 3))

    np.testing.assert_array_equal(
        projector.solve(rhs), rhs / projector.counts[0][:, None]
    )


def test_conjugate_gradient_fallback_reaches_the_same_solution():
    """The CG rescue path solves the same system the sweeps do."""
    rng = np.random.default_rng(29)
    n = 2000
    row_codes = np.column_stack([rng.integers(0, 25, n), rng.integers(0, 9, n)])
    projector = FEProjector.from_row_codes(row_codes, ("a", "b"))
    rhs = projector.matvec(rng.normal(size=(projector.total_levels, 2)))

    gauss_seidel = projector.solve(rhs)
    conjugate = projector._solve_cg(rhs, np.zeros_like(rhs))

    # alpha itself is not unique -- G is singular -- but D alpha is, so compare the
    # projection rather than the coefficients.
    np.testing.assert_allclose(
        projector.row_effects(gauss_seidel, row_codes),
        projector.row_effects(conjugate, row_codes),
        rtol=1e-7,
        atol=1e-7,
    )


def test_non_convergence_raises_naming_the_columns(monkeypatch):
    """A starved iteration budget must fail loudly rather than return a guess."""
    monkeypatch.setattr(fe_mod, "FE_MAXITER", 1)
    rng = np.random.default_rng(31)
    n = 3000
    row_codes = np.column_stack([rng.integers(0, 60, n), rng.integers(0, 40, n)])
    projector = FEProjector.from_row_codes(row_codes, ("firm", "worker"))
    rhs = projector.matvec(rng.normal(size=(projector.total_levels, 1)))

    with pytest.raises(ValueError, match="firm"):
        projector.solve(rhs)


def test_disconnected_components_still_give_unique_fitted_values():
    """Separable factors leave alpha unidentified but the projection well defined."""
    rng = np.random.default_rng(37)
    n = 1600
    # Two islands: the first half of the firms only ever appears in the early years.
    island = rng.integers(0, 2, n)
    firm = island * 8 + rng.integers(0, 8, n)
    year = island * 5 + rng.integers(0, 5, n)
    row_codes = np.column_stack([firm, year])
    projector, D = dense_projector(row_codes, ("firm", "year"))

    values = rng.normal(size=(n, 2))
    rhs = D.T @ values
    alpha = projector.solve(rhs)

    expected = D @ np.linalg.lstsq(D.T @ D, rhs, rcond=None)[0]
    np.testing.assert_allclose(
        projector.row_effects(alpha, row_codes), expected, rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_multiway_row_order_does_not_change_result(df_type, monkeypatch):
    """Level labels are dict keys across two aggregations; both must map by label."""
    df = make_multiway_panel(n=2000, seed=41)
    shuffled = df.sample(frac=1.0, random_state=13).reset_index(drop=True)

    base = run_native(
        convert_to_backend(df, df_type), controls=["age", "firm", "year"], num_bins=6
    )
    moved = run_native(
        convert_to_backend(shuffled, df_type),
        controls=["age", "firm", "year"],
        num_bins=6,
    )

    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(base, moved, rtol=rtol, atol=atol)


@pytest.mark.parametrize("selector", ["rot", "dpi"])
def test_multiway_selectors_match_one_hot(selector, monkeypatch):
    """Bin selection must not depend on whether the factors were absorbed."""
    df = make_multiway_panel(n=2500, seed=43)
    absorbed = run_native(df, controls=["age", "firm", "year"], num_bins=selector)
    without_absorption(monkeypatch)
    one_hot = run_native(df, controls=["age", "firm", "year"], num_bins=selector)

    assert len(absorbed) == len(one_hot), f"{selector} chose a different bin count"
    np.testing.assert_allclose(absorbed, one_hot, rtol=1e-8, atol=1e-8)


def test_dpi_variance_two_way_matches_centered_dummy_oracle():
    """The dense multi-way sandwich equals explicit centered dummies for both factors."""
    df = make_multiway_panel(n=2000, n_firms=12, n_years=5, seed=67)
    num_bins = 8
    edges = np.quantile(df["x"], np.linspace(0, 1, num_bins + 1))
    bin_idx = np.clip(
        np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1
    )
    bin_counts = np.bincount(bin_idx, minlength=num_bins)
    y = df["y"].to_numpy()
    age = df["age"].to_numpy()

    dummies = [
        pd.get_dummies(df[name], drop_first=True).to_numpy(dtype=float)
        for name in ("firm", "year")
    ]
    explicit_controls = np.column_stack([age, *dummies])
    explicit_controls -= explicit_controls.mean(axis=0)
    expected = core._compute_dpi_variance_constant(
        y, explicit_controls, bin_idx, bin_counts
    )

    row_codes = factor_codes([df["firm"].to_numpy(), df["year"].to_numpy()])
    projector = FEProjector.from_row_codes(row_codes, ("firm", "year"))
    absorbed_y = demean_centered(y, projector, row_codes)
    absorbed_controls = demean_centered(age - age.mean(), projector, row_codes)[:, None]
    actual = core._compute_dpi_variance_constant(
        absorbed_y, absorbed_controls, bin_idx, bin_counts, projector, row_codes
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_poly_line_with_several_absorbed_fixed_effects(monkeypatch):
    """The overlay restores its level from the mean of all absorbed factors."""
    df = make_multiway_panel(n=2000, seed=47)
    absorbed = binscatter(
        df, "x", "y", controls=["age", "firm", "year"], num_bins=6, poly_line=2
    )
    without_absorption(monkeypatch)
    one_hot = binscatter(
        df, "x", "y", controls=["age", "firm", "year"], num_bins=6, poly_line=2
    )
    np.testing.assert_allclose(
        absorbed.data[1].y, one_hot.data[1].y, rtol=1e-8, atol=1e-8
    )


def test_joint_cell_compression_shrinks_the_driver_side():
    """A panel with repeated FE tuples must collapse well below the row count."""
    rng = np.random.default_rng(53)
    n, n_firms, n_years = 40_000, 200, 10
    firm = rng.integers(0, n_firms, n)
    year = rng.integers(0, n_years, n)
    row_codes = np.column_stack([firm, year])
    projector = FEProjector.from_row_codes(row_codes, ("firm", "year"))

    # Every firm-year pair is observed many times over, so the driver holds a few
    # thousand combinations rather than 40k rows.
    assert projector.num_cells <= n_firms * n_years
    assert projector.num_cells < n / 10


def test_high_cardinality_two_way_completes():
    """5k firms crossed with 200 years is hopeless for the one-hot path."""
    rng = np.random.default_rng(59)
    n, n_firms, n_years = 60_000, 5_000, 200
    firm = rng.integers(0, n_firms, n)
    year = rng.integers(0, n_years, n)
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": (
                0.7 * x
                + rng.normal(scale=2.0, size=n_firms)[firm]
                + rng.normal(scale=1.0, size=n_years)[year]
                + rng.normal(size=n)
            ),
            "firm_id": firm,
            "year": year,
        }
    )
    lazy, _, _, categorical = core.clean_df(
        df, ("firm_id", "year"), "x", "y", ("firm_id", "year")
    )
    assert fe_mod.select_absorbed(lazy, categorical) == ("firm_id", "year")

    out = run_native(
        df,
        controls=["firm_id", "year"],
        categorical=["firm_id", "year"],
        num_bins=10,
    )
    assert out.shape == (10,)
    assert np.all(np.isfinite(out))
