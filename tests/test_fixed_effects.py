"""Tests for absorbing a categorical control via the Mundlak / within transform.

Absorption is a silent rewrite of an existing code path: the model is unchanged and
nothing raises if it is subtly wrong, the plotted values just quietly move. So the
gate is equivalence against the one-hot path, checked on every backend, backed by an
independent statsmodels oracle.
"""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from binsreg import binsregselect

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


def dense_reference(
    df: pd.DataFrame,
    num_bins: int,
    *,
    numeric: tuple[str, ...] = (),
    categorical: tuple[str, ...] = (),
) -> np.ndarray:
    """Independent centred-design OLS oracle for adjusted dots."""
    edges = np.quantile(df["x"], np.linspace(0.0, 1.0, num_bins + 1))
    bin_idx = np.clip(
        np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1
    )
    bins = pd.get_dummies(bin_idx, drop_first=False, dtype=float).to_numpy()
    blocks = [df[list(numeric)].to_numpy(dtype=float)] if numeric else []
    blocks.extend(
        pd.get_dummies(df[name], drop_first=True, dtype=float).to_numpy()
        for name in categorical
    )
    if blocks:
        controls = np.column_stack(blocks)
        controls -= controls.mean(axis=0)
        design = np.column_stack([bins, controls])
    else:
        design = bins
    theta = np.linalg.lstsq(design, df["y"].to_numpy(dtype=float), rcond=None)[0]
    return theta[:num_bins]


def _statsmodels_reference(df: pd.DataFrame, num_bins: int) -> np.ndarray:
    """Adjusted dots from an explicit dense statsmodels design."""
    sm = pytest.importorskip("statsmodels.api")
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
    means = np.concatenate([[age.mean()], grp_dummies.mean().to_numpy()])
    return theta[:num_bins] + means @ theta[num_bins:]


def _disconnected_panel() -> tuple[pd.DataFrame, np.ndarray]:
    """One-way FE panel whose low and high bins have disjoint group sets."""
    rng = np.random.default_rng(21)
    n = 800
    x = rng.normal(size=n)
    high = x > np.median(x)
    grp = np.where(
        high,
        "hi_" + (rng.integers(0, 3, n)).astype(str),
        "lo_" + (rng.integers(0, 3, n)).astype(str),
    )
    frame = pd.DataFrame({"x": x, "y": rng.normal(size=n) + high * 3.0, "grp": grp})
    return frame, high


def _dpi_oracle_case():
    """Inputs shared by independent absorbed-DPI specification tests."""
    df = make_panel(n=1800, n_groups=12, seed=67)
    num_bins = 9
    edges = np.quantile(df["x"], np.linspace(0, 1, num_bins + 1))
    bin_idx = np.clip(
        np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1
    )
    bin_counts = np.bincount(bin_idx, minlength=num_bins)
    dummies = pd.get_dummies(df["grp"], drop_first=True).to_numpy(dtype=float)
    explicit_controls = np.column_stack([df["age"].to_numpy(), dummies])
    explicit_controls -= explicit_controls.mean(axis=0)
    return df, bin_idx, bin_counts, explicit_controls


def _high_cardinality_panel() -> pd.DataFrame:
    """Panel large enough that production must absorb its categorical control."""
    rng = np.random.default_rng(43)
    n, n_groups = 50_000, 5_000
    g = rng.integers(0, n_groups, n)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "y": 0.7 * x + rng.normal(scale=2.0, size=n_groups)[g] + rng.normal(size=n),
            "firm_id": g,
        }
    )


def _to_pandas(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    for attr in ("to_pandas", "toPandas", "df", "compute"):
        if hasattr(obj, attr):
            return getattr(obj, attr)()
    raise TypeError(f"cannot convert {type(obj)}")


def assert_complete_case_rows(
    frame: nw.LazyFrame, expected: pd.DataFrame, columns: tuple[str, ...]
) -> None:
    """Assert exactly which rows reach the estimator, without numeric tolerance."""
    observed_native = frame.select(*columns).collect().to_native()
    observed = _to_pandas(observed_native)[list(columns)]
    observed = observed.sort_values(columns[0]).reset_index(drop=True)
    expected = expected[list(columns)].sort_values(columns[0]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_dtype=False,
        check_exact=True,
    )


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
    if df_type == "pyspark":
        df = df.repartition(3)
    absorbed = run_native(df, controls=controls, num_bins=6)
    without_absorption(monkeypatch)
    one_hot = run_native(df, controls=controls, num_bins=6)

    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_integer_coded_factor_matches_one_hot_on_every_backend(
    df_type, monkeypatch, force_absorption
):
    """Explicit integer categoricals take the same FE model on every dataframe."""
    frame = make_panel(n=1200, n_groups=15, seed=11)
    frame["grp_id"] = pd.factorize(frame["grp"], sort=True)[0]
    native = convert_to_backend(frame, df_type)
    if df_type == "pyspark":
        native = native.repartition(3)
    kwargs = {
        "controls": ["age", "grp_id"],
        "categorical": ["grp_id"],
        "num_bins": 5,
    }
    absorbed = run_native(native, **kwargs)
    without_absorption(monkeypatch)
    one_hot = run_native(native, **kwargs)
    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_one_way_native_output_preserves_input_backend(df_type, force_absorption):
    """A controlled native result stays in the caller's dataframe ecosystem."""
    native = convert_to_backend(make_panel(n=1200, n_groups=15, seed=12), df_type)
    if df_type == "pyspark":
        native = native.repartition(3)
    result = binscatter(
        native,
        "x",
        "y",
        controls=["age", "grp"],
        num_bins=5,
        return_type="native",
    )
    assert isinstance(result, type(native))


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
    df = make_panel(n=1500, n_groups=12, seed=3)
    num_bins = 6
    fitted = run_native(df, controls=["age", "grp"], num_bins=num_bins)
    expected = _statsmodels_reference(df, num_bins)
    np.testing.assert_allclose(fitted, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("control_name", ["__fe_resp_0", "__fe_count"])
def test_temporary_aliases_do_not_shadow_control_columns(
    control_name, force_absorption
):
    """Valid user column names must not collide with aggregation temporaries."""
    df = make_panel(n=1500, n_groups=12, seed=3)
    expected = run_native(df, controls=["age", "grp"], num_bins=6)
    renamed = df.rename(columns={"grp": control_name})
    actual = run_native(renamed, controls=["age", control_name], num_bins=6)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


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
    expected = dense_reference(df, 4, numeric=("age",), categorical=("grp",))
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------
# 3. Degenerate structure
# --------------------------------------------------------------------------


def test_singleton_groups(force_absorption):
    """Levels with a single observation demean to exactly zero."""
    df = make_panel(n=400, n_groups=5, seed=9)
    # Give 20 rows their own private group.
    df.loc[:19, "grp"] = [f"solo{i}" for i in range(20)]
    out = run_native(df, controls=["age", "grp"], num_bins=4)
    expected = dense_reference(df, 4, numeric=("age",), categorical=("grp",))
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-8)


def test_level_confined_to_single_bin(force_absorption):
    """A level appearing in only one bin is collinear with that bin."""
    df = make_panel(n=600, n_groups=6, seed=13)
    lowest = df["x"] < df["x"].quantile(0.15)
    df.loc[lowest, "grp"] = "only_low"
    out = run_native(df, controls=["grp"], num_bins=4)
    expected = dense_reference(df, 4, categorical=("grp",))
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-8)


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


def test_disconnected_bin_group_structure_is_rejected(force_absorption):
    """Adjusted dots are not estimable across disconnected bin/group components."""
    df, _ = _disconnected_panel()
    with pytest.raises(ValueError, match="not identif|collinear with .*bin indicators"):
        run_native(df, controls=["grp"], num_bins=4)


def test_single_group_bridge_matches_dense_ols(force_absorption):
    """One crossing group identifies a one-way FE curve, even at the boundary."""
    df, high = _disconnected_panel()
    bridged = df.copy()
    bridged.loc[bridged.index[high][0], "grp"] = "lo_0"
    actual = run_native(bridged, controls=["grp"], num_bins=4)
    expected = dense_reference(bridged, 4, categorical=("grp",))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_one_way_rejects_numeric_control_collinear_with_bins(force_absorption):
    """A one-way FE does not rescue a control that reproduces the bin block."""
    frame, high = _disconnected_panel()
    frame.loc[frame.index[high][0], "grp"] = "lo_0"
    edges = frame["x"].quantile([idx / 4 for idx in range(5)]).to_numpy()
    frame["bin_control"] = np.clip(
        np.searchsorted(edges, frame["x"], side="right") - 1, 0, 3
    ).astype(float)
    with pytest.raises(ValueError, match="not identif|collinear with .*bin indicators"):
        run_native(frame, controls=["bin_control", "grp"], num_bins=4)


# --------------------------------------------------------------------------
# 4. Level recovery
# --------------------------------------------------------------------------


def test_response_translation_moves_every_dot_exactly(force_absorption):
    """A response location shift must survive absorption without changing contrasts."""
    df = make_panel(n=1500, n_groups=20, seed=23)
    baseline = run_native(df, controls=["grp"], num_bins=5)
    shifted = run_native(df.assign(y=df["y"] + 500.0), controls=["grp"], num_bins=5)
    np.testing.assert_allclose(shifted, baseline + 500.0, rtol=1e-10, atol=1e-10)


def test_group_shift_moves_level_not_bin_effects(force_absorption):
    """The defining fixed-effect property: a within-group shift is absorbed."""
    df = make_panel(n=1500, n_groups=20, seed=29)
    base = run_native(df, controls=["age", "grp"], num_bins=5)

    shifted = df.copy()
    target = shifted["grp"] == shifted["grp"].iloc[0]
    shifted.loc[target, "y"] += 100.0

    out = run_native(shifted, controls=["age", "grp"], num_bins=5)
    expected_shift = 100.0 * float(target.mean())
    np.testing.assert_allclose(out, base + expected_shift, rtol=1e-8, atol=1e-8)


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
    """The absorbed sandwich equals explicit centred fixed-effect dummies."""
    df, bin_idx, bin_counts, explicit_controls = _dpi_oracle_case()
    y = df["y"].to_numpy()
    age = df["age"].to_numpy()
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


def test_absorbed_rot_pilot_matches_authors_package(force_absorption):
    """The absorbed ROT pilot agrees with the centred explicit-dummy oracle."""
    df, _, _, explicit_controls = _dpi_oracle_case()
    pilot = len(
        binscatter(
            df,
            "x",
            "y",
            controls=["age", "grp"],
            num_bins="rot",
            return_type="native",
        )
    )
    authors_rot = int(
        binsregselect(df["y"], df["x"], w=explicit_controls).nbinsrot_regul
    )
    assert abs(pilot - authors_rot) <= 1


def test_absorbed_dpi_matches_authors_package_with_same_pilot(force_absorption):
    """Pinning the ROT pilot isolates the authors package's DPI calculation."""
    df, _, _, explicit_controls = _dpi_oracle_case()
    pilot = len(
        binscatter(
            df,
            "x",
            "y",
            controls=["age", "grp"],
            num_bins="rot",
            return_type="native",
        )
    )
    ours = len(
        binscatter(
            df,
            "x",
            "y",
            controls=["age", "grp"],
            num_bins="dpi",
            return_type="native",
        )
    )
    theirs = int(
        binsregselect(df["y"], df["x"], w=explicit_controls, nbinsrot=pilot).nbinsdpi
    )
    assert ours == theirs


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


def test_below_threshold_factor_is_not_absorbed():
    """A low-cardinality factor stays on the established one-hot route."""
    df = make_panel(n=1500, n_groups=20, seed=59)
    assert (
        fe_mod.select_absorbed_factors(
            core.clean_df(df, ("grp",), "x", "y")[0], ("grp",)
        )
        == ()
    )


def test_below_threshold_result_matches_forced_one_hot(monkeypatch):
    """Routing below the threshold cannot change the estimator."""
    df = make_panel(n=1500, n_groups=20, seed=59)
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


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
@pytest.mark.parametrize(
    "invalid_column",
    ["x", "y", "age", "grp"],
    ids=["x_nan", "y_infinity", "control_infinity", "fixed_effect_null"],
)
def test_invalid_rows_are_dropped_before_one_way_absorption(
    df_type, invalid_column, monkeypatch, force_absorption
):
    """Exactly the complete cases, not merely similar dots, reach absorption."""
    df = make_panel(n=900, n_groups=8, seed=41)
    invalid_value = None if invalid_column == "grp" else np.inf
    if invalid_column == "x":
        invalid_value = np.nan
    df.loc[:19, invalid_column] = invalid_value
    clean = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["x", "y", "age", "grp"])
        .reset_index(drop=True)
    )

    seen: list[int] = []
    original = core.partial_out_controls

    def checked_partial_out(frame, profile, *args, **kwargs):
        assert_complete_case_rows(frame, clean, ("x", "y", "age", "grp"))
        seen.append(len(clean))
        return original(frame, profile, *args, **kwargs)

    monkeypatch.setattr(core, "partial_out_controls", checked_partial_out)
    run_native(convert_to_backend(df, df_type), controls=["age", "grp"], num_bins=4)
    assert seen == [len(clean)]


# --------------------------------------------------------------------------
# 7. The point of the change
# --------------------------------------------------------------------------


def test_high_cardinality_factor_uses_absorption():
    """Five thousand levels must route away from quadratic one-hot aggregation."""
    df = _high_cardinality_panel()
    lazy, _, _, categorical = core.clean_df(df, ("firm_id",), "x", "y", ("firm_id",))
    assert fe_mod.select_absorbed_factors(lazy, categorical) == ("firm_id",)


def test_high_cardinality_absorption_completes():
    """The absorbed route completes and reports one finite dot per requested bin."""
    df = _high_cardinality_panel()
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
