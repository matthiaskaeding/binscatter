import logging
import math
import operator
import time
import uuid
import warnings
from collections.abc import Iterable
from functools import reduce, wraps
from typing import (
    Any,
    Literal,
    NamedTuple,
    cast,
    overload,
)

import narwhals as nw
import narwhals.selectors as ncs
import numpy as np
import plotly.express as px
from narwhals import Implementation
from narwhals.typing import IntoDataFrame
from plotly import graph_objects as go

from binscatter.dummy_builders import configure_build_dummies
from binscatter.fixed_effects import (
    FEMoments,
    compute_fe_moments,
    demean_centered,
    group_codes,
    recover_levels,
    select_absorbed,
    within_correct,
)
from binscatter.inference import IntervalResult, compute_intervals
from binscatter.quantiles import (
    configure_add_bins,
    configure_compute_quantiles,
)

logger = logging.getLogger(__name__)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        logger.info("[%s] start", name)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            logger.info("[%s] done in %.3fs", name, duration)

    return wrapper


@overload
def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    *,
    controls: Iterable[str] | str | None = None,
    categorical: Iterable[str] | str | None = None,
    num_bins: int | Literal["rule-of-thumb", "rot", "dpi"] = "dpi",
    return_type: Literal["plotly"] = "plotly",
    poly_line: int | None = None,
    ci: Literal["none", "pointwise", "rbc"] = "none",
    ci_level: float = 0.95,
    **kwargs,
) -> go.Figure: ...


@overload
def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    *,
    controls: Iterable[str] | str | None = None,
    categorical: Iterable[str] | str | None = None,
    num_bins: int | Literal["rule-of-thumb", "rot", "dpi"] = "dpi",
    return_type: Literal["native"] = "native",
    poly_line: int | None = None,
    ci: Literal["none", "pointwise", "rbc"] = "none",
    ci_level: float = 0.95,
    **kwargs,
) -> object: ...


def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    *,
    controls: Iterable[str] | str | None = None,
    categorical: Iterable[str] | str | None = None,
    num_bins: int | Literal["rule-of-thumb", "rot", "dpi"] = "dpi",
    return_type: Literal["plotly", "native"] = "plotly",
    poly_line: int | None = None,
    ci: Literal["none", "pointwise", "rbc"] = "none",
    ci_level: float = 0.95,
    **kwargs,
) -> object:
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df: Any dataframe supported by narwhals.
        x: Name of the x column.
        y: Name of the y column.
        controls: Optional control columns to partial out (either a string or iterable of strings).
        categorical: Controls to treat as categorical regardless of dtype, for integer-coded
            identifiers such as ``firm_id`` that would otherwise enter as a single linear term.
            Must be a subset of ``controls``. This is an override, not a declaration: string
            columns are still detected automatically.
        num_bins: Number of quantile bins to form, or ``"dpi"`` (default), ``"rule-of-thumb"`` (``"rot"``) for automatic selection.
        return_type: If ``plotly`` (default) return a Plotly figure; if ``native`` return a dataframe matching the input backend.
        poly_line: Optional integer degree (1, 2, or 3) to fit a polynomial in ``x`` using the raw data (plus controls) and overlay it on the Plotly figure.
        ci: Confidence intervals to report alongside the dots. ``"pointwise"`` gives the
            heteroskedasticity-robust interval for the binscatter estimator itself, centred
            on each dot. ``"rbc"`` gives the robust bias-corrected interval for the
            underlying regression function, built from the next-order fit; it is a valid
            interval for the true conditional mean and is deliberately not centred on the
            dot. Default ``"none"``.
        ci_level: Confidence level used when ``ci`` is not ``"none"``. Default ``0.95``.
        kwargs: Extra keyword args forwarded to ``plotly.express.scatter`` when plotting.

    Returns:
        A Plotly figure or native dataframe, depending on ``return_type``.
    """
    if return_type not in ("plotly", "native"):
        msg = f"Invalid return_type: {return_type}"
        raise ValueError(msg)
    manual_bins: int = 0
    if isinstance(num_bins, str):
        if num_bins in ("rule-of-thumb", "rot"):
            auto_bins = "rot"
        elif num_bins == "dpi":
            auto_bins = "dpi"
        else:
            msg = f"Unknown num_bins string option: {num_bins}"
            raise ValueError(msg)
    else:
        try:
            manual_bins = int(num_bins)
        except (TypeError, ValueError) as err:
            raise TypeError(
                "num_bins must be an integer when provided explicitly"
            ) from err
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        auto_bins = None

    if not isinstance(x, str):
        raise TypeError("x_name must be a string")
    if not isinstance(y, str):
        raise TypeError("y_name must be a string")
    if poly_line is not None:
        if not isinstance(poly_line, int):
            raise TypeError("poly_line must be an integer in {1, 2, 3}")
        if poly_line not in (1, 2, 3):
            raise ValueError("poly_line must be one of {1, 2, 3}")
    if ci not in ("none", "pointwise", "rbc"):
        msg = f"Invalid ci option: {ci}. Expected one of 'none', 'pointwise', 'rbc'."
        raise ValueError(msg)
    if ci != "none":
        if isinstance(ci_level, bool) or not isinstance(ci_level, (int, float)):
            raise TypeError("ci_level must be a number strictly between 0 and 1")
        if not 0.0 < float(ci_level) < 1.0:
            raise ValueError("ci_level must lie strictly between 0 and 1")

    controls = _clean_controls(controls)
    logger.debug(
        "[binscatter] cleaned controls count=%d values=%s", len(controls), controls
    )
    if x in controls:
        raise ValueError("x cannot be in controls")
    if y in controls:
        raise ValueError("y cannot be in controls")
    if len(set(controls)) < len(controls):
        raise ValueError("controls contains duplicate entries")

    categorical = _clean_controls(categorical)
    if len(set(categorical)) < len(categorical):
        raise ValueError("categorical contains duplicate entries")
    not_controls = [c for c in categorical if c not in controls]
    if not_controls:
        msg = (
            f"categorical entries must also appear in controls: {not_controls}. "
            "categorical only overrides how a control is classified; it does not add controls."
        )
        raise ValueError(msg)

    distinct_suffix = str(uuid.uuid4()).replace(
        "-", "_"
    )  # "-" in col names can cause issues
    bin_name = f"bins____{distinct_suffix}"

    df, is_lazy_input, numeric_columns, categorical_columns = clean_df(
        df, controls, x, y, categorical
    )
    logger.debug(
        "[binscatter] clean_df backend=%s lazy=%s numeric_controls=%d categorical_controls=%d",
        df.implementation,
        is_lazy_input,
        len(numeric_columns),
        len(categorical_columns),
    )

    df_with_regression_features, regression_features, absorbed_fe = (
        add_regression_features(
            df,
            numeric_controls=numeric_columns,
            categorical_controls=categorical_columns,
        )
    )
    logger.debug("[binscatter] regression features total=%d", len(regression_features))
    if poly_line is not None:
        (
            df_with_regression_features,
            polynomial_features,
        ) = add_polynomial_features(
            df_with_regression_features,
            x_name=x,
            degree=poly_line,
            distinct_suffix=distinct_suffix,
        )
        logger.debug(
            "[binscatter] polynomial features total=%d", len(polynomial_features)
        )
    else:
        polynomial_features = ()
        logger.debug("[binscatter] polynomial features skipped")

    if auto_bins == "rot":
        computed_num_bins = _select_rule_of_thumb_bins(
            df_with_regression_features, x, y, regression_features, absorbed_fe
        )
        logger.info(
            "Selected %d bins using rule-of-thumb (ROT) selector", computed_num_bins
        )
    elif auto_bins == "dpi":
        computed_num_bins = _select_dpi_bins(
            df_with_regression_features, x, y, regression_features, absorbed_fe
        )
        logger.info(
            "Selected %d bins using direct plug-in (DPI) selector", computed_num_bins
        )
    else:
        computed_num_bins = manual_bins

    # Compute quantiles and deduplicate
    compute_quantiles = configure_compute_quantiles(
        computed_num_bins, df_with_regression_features.implementation
    )
    quantiles = compute_quantiles(df_with_regression_features, x)
    unique_quantiles = tuple(dict.fromkeys(quantiles))
    logger.debug(
        "[binscatter] quantiles total=%d unique=%d",
        len(quantiles),
        len(unique_quantiles),
    )

    # Compute feasible number of bins
    n_unique = len(unique_quantiles)
    final_num_bins = 1 if n_unique <= 1 else max(2, n_unique - 1)
    logger.debug(
        "[binscatter] final_num_bins=%d unique_quantiles=%d", final_num_bins, n_unique
    )

    if final_num_bins < 2:
        raise ValueError(
            "Could not produce at least 2 bins; check the distribution of the x column."
        )

    if final_num_bins < computed_num_bins and not auto_bins:
        logger.warning(
            "Requested %d bins but only %d unique quantile boundaries found. "
            "Using %d bins instead.",
            computed_num_bins,
            len(unique_quantiles),
            final_num_bins,
        )

    profile = Profile(
        num_bins=final_num_bins,
        x_name=x,
        y_name=y,
        bin_name=bin_name,
        regression_features=regression_features,
        polynomial_features=polynomial_features,
        distinct_suffix=distinct_suffix,
        is_lazy_input=is_lazy_input,
        implementation=df.implementation,
        x_bounds=(unique_quantiles[0], unique_quantiles[-1]),
        fe_name=absorbed_fe,
        bin_edges=unique_quantiles,
    )
    add_bins = configure_add_bins(profile)
    df_prepped = add_bins(df_with_regression_features, unique_quantiles)

    moment_cache: dict[str, float] | None = None
    if controls or poly_line is not None:
        moment_cache = {}

    # Check that we have enough rows for the bins (reuse cache if available)
    if moment_cache is None:
        total_rows = int(df_prepped.select(nw.len()).collect().item(0, 0))
    else:
        _ensure_moments(
            df_prepped, moment_cache, {_moment_alias("total_count"): nw.len()}
        )
        total_rows = int(moment_cache[_moment_alias("total_count")])
    if total_rows < final_num_bins:
        if auto_bins:
            msg = (
                f"Rule-of-thumb selected {final_num_bins} bins but only {total_rows} valid rows remain. "
                "Specify num_bins manually or check your data for invalid values."
            )
        else:
            msg = (
                f"Requested {final_num_bins} bins but only {total_rows} valid rows remain. "
                "Number of bins cannot exceed the number of valid rows."
            )
        raise ValueError(msg)

    if not controls:
        df_plotting = compute_bin_means(df_prepped, profile)
    else:
        df_plotting, _ = partial_out_controls(
            df_prepped, profile, moment_cache=moment_cache
        )

    if ci != "none":
        if auto_bins:
            warnings.warn(
                "Confidence intervals are only valid when the number of bins is well "
                f"above the IMSE-optimal choice, but num_bins={final_num_bins} was "
                f"picked by the {auto_bins.upper()} selector to minimise IMSE. Pass a "
                "larger num_bins explicitly for intervals you can rely on.",
                stacklevel=2,
            )
        intervals = compute_intervals(df_prepped, profile, ci, float(ci_level))
        df_plotting = _attach_intervals(df_plotting, profile, intervals)

    polynomial_line: PolynomialFit | None = None
    if poly_line is not None and return_type == "plotly":
        cache = moment_cache or {}
        polynomial_line = _fit_polynomial_line(df_prepped, profile, poly_line, cache)

    match return_type:
        case "plotly":
            return make_plot_plotly(
                df_plotting,
                profile,
                kwargs_binscatter=kwargs,
                polynomial_line=polynomial_line,
            )
        case "native":
            return make_native_dataframe(df_plotting, profile)


class Profile(NamedTuple):
    """Main profile which holds bunch of data derived from dataframe."""

    x_name: str
    y_name: str
    num_bins: int
    bin_name: str
    distinct_suffix: str
    is_lazy_input: bool
    implementation: Implementation
    regression_features: tuple[str, ...]
    polynomial_features: tuple[str, ...]
    x_bounds: tuple[float, float]
    fe_name: str | None = None
    #: The ``num_bins + 1`` quantile boundaries the bins were cut on. Only the
    #: inference path needs them; everything else works off the bin labels.
    bin_edges: tuple[float, ...] = ()

    @property
    def x_col(self) -> nw.Expr:
        return nw.col(self.x_name)

    @property
    def y_col(self) -> nw.Expr:
        return nw.col(self.y_name)


def _moment_alias(kind: str, *parts: str) -> str:
    if kind == "sumprod" and len(parts) == 2:
        parts = tuple(sorted(parts))
    suffix = "__".join(parts)
    return f"__moment_{kind}" + (f"__{suffix}" if suffix else "")


def _ensure_moments(
    df: nw.LazyFrame, cache: dict[str, float], expr_map: dict[str, nw.Expr]
) -> None:
    """Populate the cache with any missing scalar moments defined in expr_map."""
    missing = {alias: expr for alias, expr in expr_map.items() if alias not in cache}
    if not missing:
        return
    collected = df.select(
        *(expr.alias(alias) for alias, expr in missing.items())
    ).collect()
    for alias in missing:
        value = collected.item(0, alias)
        if value is None:
            value = 0.0
        cache[alias] = float(value)


def _ensure_feature_moments(
    df: nw.LazyFrame,
    feature_names: tuple[str, ...],
    y_expr: nw.Expr,
    cache: dict[str, float],
) -> None:
    """Gather feature-level sums, y cross-products, and cross-moments into the cache."""
    if not feature_names:
        return
    exprs: dict[str, nw.Expr] = {}
    for col in feature_names:
        exprs[_moment_alias("sum", col)] = nw.col(col).sum()
        exprs[_moment_alias("sum_y", col)] = (nw.col(col) * y_expr).sum()
    for i, col_i in enumerate(feature_names):
        for j, col_j in enumerate(feature_names[i:], start=i):
            exprs[_moment_alias("sumprod", col_i, col_j)] = (
                nw.col(col_i) * nw.col(col_j)
            ).sum()
    _ensure_moments(df, cache, exprs)


def _build_feature_normal_equations(
    feature_names: tuple[str, ...], cache: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    if not feature_names:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    size = len(feature_names)
    xtx = np.zeros((size, size), dtype=float)
    xty = np.zeros(size, dtype=float)
    for i, col_i in enumerate(feature_names):
        xty[i] = cache[_moment_alias("sum_y", col_i)]
        xtx[i, i] = cache[_moment_alias("sumprod", col_i, col_i)]
        for j in range(i + 1, size):
            col_j = feature_names[j]
            value = cache[_moment_alias("sumprod", col_i, col_j)]
            xtx[i, j] = value
            xtx[j, i] = value
    return xtx, xty


class PolynomialFit(NamedTuple):
    """Container for polynomial overlay metadata."""

    degree: int
    coefficients: np.ndarray
    x: np.ndarray
    y: np.ndarray


@timed
def partial_out_controls(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    moment_cache: dict[str, float] | None = None,
) -> tuple[nw.LazyFrame, dict[str, np.ndarray]]:
    """Compute binscatter means after partialling out controls following Cattaneo et al. (2024)."""

    if moment_cache is None:
        moment_cache = {}

    agg_exprs = [
        nw.len().alias("__count"),
        profile.x_col.mean().alias(profile.x_name),
        profile.y_col.sum().alias("__sum_y"),
        *[nw.col(c).sum().alias(c) for c in profile.regression_features],
    ]

    per_bin = (
        df_prepped.group_by(profile.bin_name).agg(*agg_exprs).sort(profile.bin_name)
    ).collect()

    counts = per_bin.get_column("__count").to_numpy()
    if counts.size < profile.num_bins:
        msg = "Quantiles are not unique. Decrease number of bins."
        raise ValueError(msg)
    sum_y = per_bin.get_column("__sum_y").to_numpy()
    if profile.regression_features:
        bin_control_sums = np.column_stack(
            [
                per_bin.get_column(alias).to_numpy()
                for alias in profile.regression_features
            ]
        )
    else:
        bin_control_sums = np.zeros((profile.num_bins, 0))

    totals_cache = {_moment_alias("total_count"): nw.len()}
    _ensure_moments(df_prepped, moment_cache, totals_cache)
    _ensure_feature_moments(
        df_prepped,
        profile.regression_features,
        profile.y_col,
        moment_cache,
    )

    total_count = moment_cache[_moment_alias("total_count")]
    if profile.regression_features:
        total_ctrl_sums = np.array(
            [
                moment_cache[_moment_alias("sum", alias)]
                for alias in profile.regression_features
            ],
            dtype=float,
        )
        ww, wy = _build_feature_normal_equations(
            profile.regression_features, moment_cache
        )
    else:
        total_ctrl_sums = np.array([], dtype=float)
        wy = np.array([], dtype=float)
        ww = np.zeros((0, 0), dtype=float)

    num_bins = profile.num_bins
    k = len(profile.regression_features)
    size = num_bins + k

    bin_block = np.diag(counts.astype(float, copy=False))
    bin_y = sum_y.astype(float, copy=False)

    fe_moments: FEMoments | None = None
    if profile.fe_name is not None:
        fe_moments = compute_fe_moments(
            df_prepped,
            fe_name=profile.fe_name,
            bin_name=profile.bin_name,
            bin_labels=per_bin.get_column(profile.bin_name).to_list(),
            feature_names=profile.regression_features,
            response_exprs={"y": profile.y_col},
        )
        inv = fe_moments.inv_counts
        s_bin = fe_moments.bin_counts
        s_feat = fe_moments.feature_sums
        s_y = fe_moments.response_sums["y"]

        bin_block = within_correct(bin_block, s_bin, s_bin, inv)
        bin_y = within_correct(bin_y, s_bin, s_y, inv)
        if k:
            bin_control_sums = within_correct(bin_control_sums, s_bin, s_feat, inv)
            ww = within_correct(ww, s_feat, s_feat, inv)
            wy = within_correct(wy, s_feat, s_y, inv)

    XTX = np.zeros((size, size), dtype=float)
    XTy = np.zeros(size, dtype=float)

    XTX[:num_bins, :num_bins] = bin_block
    if k:
        XTX[:num_bins, num_bins:] = bin_control_sums
        XTX[num_bins:, :num_bins] = bin_control_sums.T
        XTX[num_bins:, num_bins:] = ww
        XTy[num_bins:] = wy

    XTy[:num_bins] = bin_y

    if fe_moments is None:
        theta = _solve_normal_equations(XTX, XTy)
    else:
        # The within system is rank-deficient by exactly one (demeaned bin dummies
        # sum to zero row-wise), and a near-singular matrix often makes
        # np.linalg.solve return garbage rather than raise, so go straight to the
        # minimum-norm solution.
        theta = np.linalg.lstsq(XTX, XTy, rcond=None)[0]

    beta = theta[:num_bins]
    gamma = theta[num_bins:]
    mean_controls = total_ctrl_sums / total_count if k else np.array([])
    fitted = beta + (mean_controls @ gamma if k else 0.0)
    if fe_moments is not None:
        fitted = fitted + recover_levels(fe_moments, beta, gamma, "y")

    y_vals = nw.new_series(
        name=profile.y_name, values=fitted, backend=per_bin.implementation
    )

    df_plotting = (
        per_bin.select(profile.bin_name, profile.x_name).with_columns(y_vals).lazy()
    )

    return df_plotting, {"beta": beta, "gamma": gamma}


def _attach_intervals(
    df_plotting: nw.LazyFrame, profile: Profile, intervals: IntervalResult
) -> nw.LazyFrame:
    """Add the interval columns to the plotting frame, aligned by bin order.

    ``compute_bin_means`` does not sort, so the frame is sorted here before the
    interval arrays (which are in bin order) are attached.
    """
    collected = df_plotting.sort(profile.bin_name).collect()
    backend = collected.implementation
    return collected.with_columns(
        nw.new_series(name="ci_lower", values=intervals.lower, backend=backend),
        nw.new_series(name="ci_upper", values=intervals.upper, backend=backend),
        nw.new_series(name="ci_std_error", values=intervals.std_error, backend=backend),
    ).lazy()


@timed
def _fit_polynomial_line(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    degree: int,
    cache: dict[str, float],
) -> PolynomialFit:
    if not profile.polynomial_features or len(profile.polynomial_features) < degree:
        raise ValueError(
            "Polynomial features not initialized; ensure poly_line was set."
        )
    if profile.x_bounds is None:
        raise ValueError("x_bounds not set in profile.")
    poly_cols = profile.polynomial_features[:degree]
    feature_names = poly_cols + profile.regression_features
    base_exprs: dict[str, nw.Expr] = {
        _moment_alias("total_count"): nw.len(),
        _moment_alias("sum_y_total"): profile.y_col.sum(),
    }
    _ensure_moments(df_prepped, cache, base_exprs)
    _ensure_feature_moments(df_prepped, feature_names, profile.y_col, cache)

    total_count = cache[_moment_alias("total_count")]
    if total_count <= 0:
        raise ValueError("Polynomial overlay requires at least one observation.")
    feature_xtx, feature_xty = _build_feature_normal_equations(feature_names, cache)
    size = len(feature_names) + 1
    xtx = np.zeros((size, size), dtype=float)
    xty = np.zeros(size, dtype=float)
    xtx[0, 0] = total_count
    xty[0] = cache[_moment_alias("sum_y_total")]
    xtx[1:, 1:] = feature_xtx
    xty[1:] = feature_xty
    if feature_names:
        column_sums = np.array(
            [cache[_moment_alias("sum", col)] for col in feature_names],
            dtype=float,
        )
        xtx[0, 1:] = column_sums
        xtx[1:, 0] = column_sums

    fe_level = 0.0
    if profile.fe_name is not None and feature_names:
        # Absorbing removes the intercept, so the design is the features alone.
        # The overlay level is restored from the mean fixed effect, exactly as in
        # partial_out_controls.
        moments = compute_fe_moments(
            df_prepped,
            fe_name=profile.fe_name,
            feature_names=feature_names,
            response_exprs={"y": profile.y_col},
        )
        inv = moments.inv_counts
        s_feat = moments.feature_sums
        s_y = moments.response_sums["y"]
        within_xtx = within_correct(xtx[1:, 1:], s_feat, s_feat, inv)
        within_xty = within_correct(xty[1:], s_feat, s_y, inv)
        feature_coefs = _solve_normal_equations(within_xtx, within_xty)
        alpha = inv * (s_y - s_feat @ feature_coefs)
        fe_level = float(moments.counts @ alpha / moments.total_count)
        coefficients = np.concatenate([[0.0], feature_coefs])
    else:
        coefficients = _solve_normal_equations(xtx, xty)
    coefficients = coefficients.copy()
    coefficients[0] += fe_level
    x_min, x_max = profile.x_bounds
    x_grid = _build_prediction_grid(x_min, x_max)
    if profile.regression_features:
        control_means = np.array(
            [
                cache[_moment_alias("sum", ctrl)] / total_count
                for ctrl in profile.regression_features
            ],
            dtype=float,
        )
    else:
        control_means = np.array([], dtype=float)
    y_pred = _evaluate_polynomial_predictions(
        coefficients, degree, control_means, x_grid
    )
    return PolynomialFit(
        degree=degree,
        coefficients=coefficients,
        x=x_grid,
        y=y_pred,
    )


def _build_prediction_grid(
    x_min: float, x_max: float, grid_size: int = 200
) -> np.ndarray:
    if not (math.isfinite(x_min) and math.isfinite(x_max)):
        raise ValueError("Polynomial overlay requires finite x bounds.")
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if math.isclose(x_min, x_max):
        return np.array([x_min], dtype=float)
    return np.linspace(x_min, x_max, num=grid_size)


def _evaluate_polynomial_predictions(
    coefficients: np.ndarray, degree: int, control_means: np.ndarray, x_grid: np.ndarray
) -> np.ndarray:
    intercept = float(coefficients[0])
    poly_coeffs = coefficients[1 : degree + 1]
    control_coeffs = coefficients[degree + 1 :]
    baseline = intercept
    if control_coeffs.size and control_means.size:
        baseline += float(control_coeffs @ control_means)
    if not x_grid.size:
        return np.array([], dtype=float)
    poly_terms = np.vstack([x_grid**power for power in range(1, degree + 1)])
    y_poly = poly_terms.T @ poly_coeffs
    return baseline + y_poly


@timed
def make_plot_plotly(
    df_plotting: nw.LazyFrame,
    profile: Profile,
    kwargs_binscatter: dict[str, Any],
    polynomial_line: PolynomialFit | None = None,
) -> go.Figure:
    """Make plot from prepared dataframe.

    Args:
      df_prepped (nw.LazyFrame): Prepared dataframe. Has three columns: bin, x, y with names in profile"""
    collected = df_plotting.collect()
    data = collected.select(profile.x_name, profile.y_name)
    if data.shape[0] < profile.num_bins:
        raise ValueError("Quantiles are not unique. Decrease number of bins.")

    x = data.get_column(profile.x_name).to_list()
    if len(set(x)) < profile.num_bins:
        msg = f"Unique number of bins is {len(set(x))} fewer than {profile.num_bins} as desired. Decrease parameter num_bins."
        raise ValueError(msg)
    y = data.get_column(profile.y_name).to_list()

    has_ci = {"ci_lower", "ci_upper"}.issubset(set(collected.columns))
    if has_ci:
        ci_lower = collected.get_column("ci_lower").to_list()
        ci_upper = collected.get_column("ci_upper").to_list()

    raw_x_min, raw_x_max = profile.x_bounds
    pad_ratio = 0.06872
    x_span = raw_x_max - raw_x_min
    pad_x = x_span * pad_ratio if x_span > 0 else 1.0
    padded_range_x = (raw_x_min - pad_x, raw_x_max + pad_x)

    # Calculate y-axis range based on scatter points only
    # Plotly auto-range expands ~6.9% beyond the data span, so match that feel
    y_min, y_max = min(y), max(y)
    if has_ci:
        # Interval bars are data, not decoration: clipping them would understate
        # the uncertainty the caller asked to see.
        y_min = min(y_min, *ci_lower)
        y_max = max(y_max, *ci_upper)
    y_span = y_max - y_min
    pad_ratio_y = 0.06872
    pad_y = y_span * pad_ratio_y if y_span > 0 else 1.0
    padded_range_y = (y_min - pad_y, y_max + pad_y)

    scatter_args = {
        "x": x,
        "y": y,
        "range_x": padded_range_x,
        "range_y": padded_range_y,
        "labels": {
            "x": profile.x_name,
            "y": profile.y_name,
        },
        "template": "simple_white",
        "color_discrete_sequence": ["black"],
    }
    for k, v in kwargs_binscatter.items():
        if k in ("x", "y", "range_x", "range_y"):
            msg = f"px.scatter will ignore keyword argument '{k}'"
            warnings.warn(msg)
            continue
        scatter_args[k] = v

    figure = px.scatter(**scatter_args)
    if "size" not in kwargs_binscatter:
        figure.update_traces(marker={"size": 8})
    else:
        warnings.warn(
            "binscatter plot respects provided 'size' keyword; default marker size of 8 skipped."
        )
    if has_ci:
        # Drawn as segments rather than plotly error bars: a robust bias-corrected
        # interval is not centred on its dot, and error bars can only express an
        # offset from the marker.
        bar_x: list[float | None] = []
        bar_y: list[float | None] = []
        for xi, lo, hi in zip(x, ci_lower, ci_upper):
            bar_x.extend((xi, xi, None))
            bar_y.extend((lo, hi, None))
        figure.add_trace(
            go.Scatter(
                x=bar_x,
                y=bar_y,
                mode="lines",
                name="Confidence interval",
                showlegend=False,
                hoverinfo="skip",
                line={"color": "black", "width": 1},
            )
        )
    if polynomial_line is not None:
        figure.add_trace(
            go.Scatter(
                x=polynomial_line.x.tolist(),
                y=polynomial_line.y.tolist(),
                mode="lines",
                name=f"Polynomial fit (deg {polynomial_line.degree})",
                showlegend=False,
                line={"color": "rgba(31, 119, 180, 0.95)", "width": 2},
            )
        )

    return figure


def _remove_bad_values(
    df: nw.LazyFrame, cols_numeric: Iterable[str], cols_cat: Iterable[str]
) -> nw.LazyFrame:
    """Removes nulls and non-finite values for the provided columns."""

    bad_conditions = []

    df = df.drop_nulls(subset=[*cols_numeric, *cols_cat])

    for c in cols_numeric:
        col = nw.col(c)
        bad_conditions.append(~col.is_finite() | col.is_nan())

    if not bad_conditions:
        return df

    final_bad_condition = reduce(operator.or_, bad_conditions)

    return df.filter(~final_bad_condition)


def split_columns(
    frame: nw.LazyFrame | nw.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return tuples of numeric and categorical column names for a narwhals frame."""

    def _safe_columns(selection: Any) -> tuple[str, ...]:
        if selection is None:
            return ()
        if hasattr(selection, "collect_schema"):
            try:
                schema = selection.collect_schema()
            except Exception:  # pragma: no cover - backend quirk
                schema = None
            else:
                if schema is not None:
                    if hasattr(schema, "names") and callable(schema.names):
                        names = schema.names()
                        if names:
                            return tuple(names)
                    if isinstance(schema, dict):
                        return tuple(schema.keys())
        columns: tuple[str, ...] = ()
        if hasattr(selection, "columns"):
            try:
                columns = tuple(selection.columns)
            except Exception:  # pragma: no cover - backend quirk
                columns = ()
        if columns:
            return columns
        return ()

    numeric_cols = _safe_columns(frame.select(ncs.numeric()))
    frame_columns = _safe_columns(frame)
    categorical_cols = tuple(col for col in frame_columns if col not in numeric_cols)

    return numeric_cols, categorical_cols


def _clean_controls(controls: Iterable[str] | str | None) -> tuple[str, ...]:
    if not controls:
        return ()
    if isinstance(controls, str):
        return (controls,)

    try:
        controls = tuple(controls)
    except Exception as e:
        raise ValueError(
            "Failed to cast controls to tuple: check that controls is iterable of strings"
        ) from e
    if not all(isinstance(c, str) for c in controls):
        raise TypeError("controls must contain only strings")

    return controls


@timed
def clean_df(
    df_in: IntoDataFrame,
    controls: tuple[str, ...],
    x: str,
    y: str,
    categorical: tuple[str, ...] = (),
) -> tuple[nw.LazyFrame, bool, tuple[str, ...], tuple[str, ...]]:
    """Normalize the input dataframe and split controls by type.

    Returns a lazy narwhals frame containing only the requested columns, whether
    the original input was lazy, and tuples of numeric / categorical controls.

    ``categorical`` names controls that must be treated as categorical even when
    their dtype is numeric.
    """
    # Try cheap column access for non-lazy inputs
    cols: list[str] | None = None
    if hasattr(df_in, "collect_schema"):
        pass  # lazy frame, defer to narwhals
    elif hasattr(df_in, "columns"):
        cols = list(cast(Any, df_in).columns)

    dfn: nw.DataFrame | nw.LazyFrame = nw.from_native(df_in)

    if type(dfn) is nw.DataFrame:
        is_lazy_input = False
        if cols is None:
            cols = list(dfn.columns)
    elif type(dfn) is nw.LazyFrame:
        is_lazy_input = True
        if cols is None:
            cols = dfn.collect_schema().names()
    else:
        msg = f"Unexpected narwhals type {(type(dfn))}"
        raise ValueError(msg)

    if cols is None or len(cols) <= 1:
        msg = "Input dataframe must have at least 2 columns"
        raise TypeError(msg)
    for c in [x, y, *controls]:
        if c not in cols:
            msg = f"{c} not in input dataframe"
            raise ValueError(msg)

    logger.debug("Type after calling to native: %s", type(dfn.to_native()))
    dfl: nw.LazyFrame = dfn.lazy()

    df = dfl.select(x, y, *controls)
    cols_numeric, cols_cat = split_columns(df)
    union_cols = set(cols_numeric) | set(cols_cat)
    missing_controls = [c for c in controls if c not in union_cols]
    if missing_controls:
        msg = f"Unknown control columns (neither numeric nor categorical): {missing_controls}"
        raise TypeError(msg)
    if x not in cols_numeric:
        msg = f"x column '{x}' must be numeric"
        raise TypeError(msg)
    if y not in cols_numeric:
        msg = f"y column '{y}' must be numeric"
        raise TypeError(msg)

    # Filtering must key off the *actual* dtypes: a numeric column reclassified as
    # categorical still needs its non-finite values removed, or NaN silently becomes
    # its own category.
    df_filtered = _remove_bad_values(df, cols_numeric, cols_cat)

    if categorical:
        schema = df.collect_schema()
        float_cats = [c for c in categorical if schema[c].is_float()]
        if float_cats:
            msg = (
                f"categorical columns must not be floating point: {float_cats}. "
                "Grouping on floats is unreliable; cast to an integer or string column first."
            )
            raise TypeError(msg)
        cols_numeric = tuple(c for c in cols_numeric if c not in categorical)
        cols_cat = tuple(cols_cat) + tuple(c for c in categorical if c not in cols_cat)

    numeric_controls = tuple(c for c in controls if c in cols_numeric)
    categorical_controls = tuple(c for c in controls if c in cols_cat)

    return df_filtered.lazy(), is_lazy_input, numeric_controls, categorical_controls


@timed
def _select_rule_of_thumb_bins(
    df: nw.LazyFrame,
    x: str,
    y: str,
    regression_features: tuple[str, ...],
    fe_name: str | None = None,
) -> int:
    """Implement the SA-4.1 rule-of-thumb selector (currently for p=0, v=0)."""
    data_cols: tuple[str, ...] = (x, *regression_features)
    stats = _collect_rule_of_thumb_stats(df, data_cols, y)
    n_obs = stats.item(0, "__n")
    if n_obs is None or n_obs <= 1:
        raise ValueError(
            "Rule-of-thumb selector needs at least two observations in the design."
        )
    n_obs_f = float(n_obs)
    design_size = len(data_cols) + 1  # add intercept
    xtx = np.zeros((design_size, design_size), dtype=float)
    xty = np.zeros(design_size, dtype=float)
    xty_sq = np.zeros(design_size, dtype=float)
    column_sums = np.zeros(design_size, dtype=float)

    xtx[0, 0] = n_obs_f
    column_sums[0] = n_obs_f
    xty[0] = stats.item(0, "__sum_y")
    xty_sq[0] = stats.item(0, "__sum_y2")

    # Fill entries that involve actual data columns.
    for idx, col_name in enumerate(data_cols):
        alias_sum = f"__sum_{idx}"
        alias_xty = f"__xty_{idx}"
        alias_xty2 = f"__xty2_{idx}"
        col_sum = stats.item(0, alias_sum)
        column_sums[idx + 1] = col_sum
        xtx[0, idx + 1] = col_sum
        xtx[idx + 1, 0] = col_sum
        xty[idx + 1] = stats.item(0, alias_xty)
        xty_sq[idx + 1] = stats.item(0, alias_xty2)
        for jdx in range(idx, len(data_cols)):
            alias_xtx = f"__xtx_{idx}_{jdx}"
            value = stats.item(0, alias_xtx)
            xtx[idx + 1, jdx + 1] = value
            xtx[jdx + 1, idx + 1] = value

    if fe_name is None:
        beta_y = _solve_normal_equations(xtx, xty)
        beta_y_sq = _solve_normal_equations(xtx, xty_sq)
        # With an intercept in the design, OLS fitted values sum to the response
        # total, so this equals the raw sum of y-squared.
        sum_y_sq_resid = float(column_sums @ beta_y_sq)
        design_xtx = xtx
        slope_index = 1
    else:
        # Absorbing the fixed effect kills the intercept, so the design is the data
        # columns alone, within-transformed via their group sums.
        moments = compute_fe_moments(
            df,
            fe_name=fe_name,
            feature_names=data_cols,
            response_exprs={"y": nw.col(y), "y_sq": nw.col(y) * nw.col(y)},
        )
        inv = moments.inv_counts
        s_feat = moments.feature_sums
        s_y = moments.response_sums["y"]
        design_xtx = within_correct(xtx[1:, 1:], s_feat, s_feat, inv)
        design_xty = within_correct(xty[1:], s_feat, s_y, inv)
        beta_y = _solve_normal_equations(design_xtx, design_xty)
        # Within-transformed total sum of squares for y.
        sum_y_sq_resid = float(xty_sq[0] - s_y @ (inv * s_y))
        slope_index = 0

    # Sample moments of x for the Gaussian reference density.
    x_sum = column_sums[1]
    x_sq_sum = xtx[1, 1]
    mean_x = x_sum / n_obs_f
    var_x = (x_sq_sum / n_obs_f) - mean_x**2
    if var_x <= 0:
        raise ValueError(
            "Rule-of-thumb selector requires the x column to have positive variance."
        )
    std_x = math.sqrt(var_x)

    # Per Cattaneo et al. (2024) SA-4.1, for p=0, s=0, v=0:
    # B_hat = bcons * (1/n) * Σᵢ [μ'(xᵢ)]² / f_X(xᵢ)²
    # where bcons = 1/(2*(p+1)+1) / ((p+1)!)² / C(2(p+1), p+1)²
    # For p=0: bcons = 1/3 / 1 / 4 = 1/12
    # For linear fit, μ'(x) = slope (constant), so:
    # B_hat = (1/12) * slope² * (1/n) * Σᵢ 1/f_X(xᵢ)²
    sum_inv_density_sq = _gaussian_inverse_density_squared_sum(df, x, mean_x, std_x)

    slope = beta_y[slope_index]
    # bcons for p=0, s=0, v=0: 1/(2*1+1) / 1!² / C(2,1)² = 1/3 / 1 / 4 = 1/12
    imse_bcons = 1.0 / 12.0
    bias_constant = imse_bcons * (slope**2) * (sum_inv_density_sq / n_obs_f)
    if bias_constant <= 0 or not math.isfinite(bias_constant):
        raise ValueError(
            "Rule-of-thumb selector estimated a non-positive bias constant; "
            "consider specifying num_bins explicitly."
        )

    # V_hat = (1/n) * Σᵢ σ²(xᵢ) where σ²(x) = E[y²|x] - E[y|x]²
    # We estimate this as the average residual variance from the polynomial fit
    quad_form = float(beta_y.T @ design_xtx @ beta_y)
    variance_constant = (sum_y_sq_resid - quad_form) / n_obs_f
    variance_constant = max(variance_constant, 0.0)
    if variance_constant <= 0 or not math.isfinite(variance_constant):
        raise ValueError(
            "Rule-of-thumb selector estimated a non-positive variance constant; "
            "consider specifying num_bins explicitly."
        )

    # J_IMSE = (2B/V)^(1/3) * n^(1/3)
    prefactor = (2.0 * bias_constant) / variance_constant
    j_float = prefactor ** (1.0 / 3.0) * n_obs_f ** (1.0 / 3.0)
    # Cap at ~10 observations per bin to avoid noisy estimates
    max_bins = max(2, int(n_obs) // 10)
    computed_bins = max(2, round(j_float))
    return min(max_bins, computed_bins)


@timed
def _select_dpi_bins(
    df: nw.LazyFrame,
    x: str,
    y: str,
    regression_features: tuple[str, ...],
    fe_name: str | None = None,
) -> int:
    """Implement the SA-4.2 direct plug-in selector (for p=0, s=0, v=0)."""
    pilot_bins = _select_rule_of_thumb_bins(df, x, y, regression_features, fe_name)

    required_cols = (x, y) + regression_features
    if fe_name is not None:
        required_cols = required_cols + (fe_name,)
    collected = df.select(*(nw.col(col) for col in required_cols)).collect()
    native = collected.to_pandas()

    x_values = native[x].to_numpy(dtype=float, copy=True)
    y_values = native[y].to_numpy(dtype=float, copy=True)

    if regression_features:
        controls = native[list(regression_features)].to_numpy(dtype=float, copy=True)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
    else:
        controls = None

    raw_groups = native[fe_name].to_numpy() if fe_name is not None else None

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if controls is not None:
        mask &= np.all(np.isfinite(controls), axis=1)

    if not np.all(mask):
        x_values = x_values[mask]
        y_values = y_values[mask]
        if controls is not None:
            controls = controls[mask]
        if raw_groups is not None:
            raw_groups = raw_groups[mask]

    n_obs = x_values.size
    if n_obs < 5:
        return pilot_bins

    # The plotted curve evaluates controls at their sample means. Centering them
    # makes the bin coefficients represent that same curve and makes their
    # covariance invariant to dummy reference coding.
    if controls is not None:
        controls = controls - controls.mean(axis=0)

    fe_codes: np.ndarray | None = None
    fe_counts: np.ndarray | None = None
    if raw_groups is not None:
        # Project out centered group effects rather than the full group-indicator
        # space. This preserves the intercept, so the DPI bin coefficients stay on
        # the sample-average outcome level instead of becoming rank-deficient.
        # x stays raw because bins are formed on the original scale.
        fe_codes, fe_counts = group_codes(raw_groups)
        y_values = demean_centered(y_values, fe_codes, fe_counts)
        if controls is not None:
            controls = np.column_stack(
                [
                    demean_centered(controls[:, j], fe_codes, fe_counts)
                    for j in range(controls.shape[1])
                ]
            )

    constants = _estimate_dpi_imse_constants(
        x_values,
        y_values,
        controls,
        pilot_bins,
        fe_codes,
        fe_counts,
    )
    if constants is None:
        return pilot_bins

    imse_b, imse_v = constants
    if imse_b <= 0 or not math.isfinite(imse_b):
        return pilot_bins
    if imse_v <= 0 or not math.isfinite(imse_v):
        return pilot_bins

    prefactor = (2.0 * imse_b) / imse_v
    if prefactor <= 0 or not math.isfinite(prefactor):
        return pilot_bins

    computed_bins = max(2, math.ceil(prefactor ** (1.0 / 3.0)))
    return computed_bins


def _estimate_dpi_imse_constants(
    x_values: np.ndarray,
    y_values: np.ndarray,
    controls: np.ndarray | None,
    pilot_bins: int,
    fe_codes: np.ndarray | None = None,
    fe_counts: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Compute the IMSE bias/variance constants for the DPI selector.

    When ``fe_codes`` is given the caller has already removed centered fixed effects
    from ``y_values`` and ``controls``. The spline design and bin dummies receive the
    same projection here so the whole system stays consistent with
    Frisch-Waugh-Lovell while retaining the intercept.
    """
    n_obs = x_values.size
    if n_obs == 0:
        return None

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    span = x_max - x_min
    if not math.isfinite(span) or span <= 0:
        return None
    x_norm = (x_values - x_min) / span

    edges = _quantile_edges(x_norm, pilot_bins)
    if edges.size < 3:
        return None

    bin_idx = np.searchsorted(edges, x_norm, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, edges.size - 2)
    bin_counts = np.bincount(bin_idx, minlength=edges.size - 1)
    if np.any(bin_counts == 0):
        return None

    interior_knots = edges[1:-1]
    spline_design = _linear_spline_design(x_norm, interior_knots)
    fit_design = spline_design
    if fe_codes is not None and fe_counts is not None:
        # Remove centered group effects from the basis. In particular, the intercept
        # remains one rather than collapsing to zero.
        fit_design = np.column_stack(
            [
                demean_centered(spline_design[:, j], fe_codes, fe_counts)
                for j in range(spline_design.shape[1])
            ]
        )
    if controls is not None and controls.size:
        design_matrix = np.column_stack((fit_design, controls))
    else:
        design_matrix = fit_design

    try:
        beta, *_ = np.linalg.lstsq(design_matrix, y_values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    spline_coeffs = beta[: spline_design.shape[1]]
    derivatives = _linear_spline_derivative(x_norm, interior_knots, spline_coeffs)

    bias_cons = _compute_dpi_bias_constant(
        x_norm,
        derivatives,
        edges,
        bin_idx,
        bin_counts,
    )
    if bias_cons is None:
        return None

    variance_cons = _compute_dpi_variance_constant(
        y_values,
        controls,
        bin_idx,
        bin_counts,
        fe_codes,
        fe_counts,
    )
    if variance_cons is None:
        return None

    num_bins = bin_counts.size
    imse_b = bias_cons * (num_bins**2)
    imse_v = variance_cons / num_bins
    return imse_b, imse_v


def _quantile_edges(x_norm: np.ndarray, num_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(x_norm, quantiles, method="linear")
    unique_edges = np.unique(edges)
    return unique_edges


def _linear_spline_design(x_norm: np.ndarray, interior_knots: np.ndarray) -> np.ndarray:
    design = np.empty((x_norm.size, 2 + interior_knots.size), dtype=float)
    design[:, 0] = 1.0
    design[:, 1] = x_norm
    for idx, knot in enumerate(interior_knots):
        design[:, idx + 2] = np.clip(x_norm - knot, 0.0, None)
    return design


def _linear_spline_derivative(
    x_norm: np.ndarray,
    interior_knots: np.ndarray,
    spline_coeffs: np.ndarray,
) -> np.ndarray:
    derivatives = np.full(x_norm.shape, float(spline_coeffs[1]))
    gamma = spline_coeffs[2:]
    if gamma.size:
        for knot, coeff in zip(interior_knots, gamma):
            derivatives += coeff * (x_norm >= knot)
    return derivatives


def _compute_dpi_bias_constant(
    x_norm: np.ndarray,
    derivatives: np.ndarray,
    edges: np.ndarray,
    bin_idx: np.ndarray,
    bin_counts: np.ndarray,
) -> float | None:
    widths = np.diff(edges)
    widths = widths[bin_idx]
    valid = widths > 0
    if not np.any(valid):
        return None
    left_edges = edges[bin_idx]
    bias_weights = (x_norm - left_edges) - 0.5 * widths
    bias_raw = derivatives * bias_weights
    bin_sums = np.zeros(bin_counts.size, dtype=float)
    np.add.at(bin_sums, bin_idx[valid], bias_raw[valid])
    bin_means = np.divide(
        bin_sums,
        bin_counts,
        out=np.zeros_like(bin_sums),
        where=bin_counts > 0,
    )
    bias_residuals = bias_raw[valid] - bin_means[bin_idx[valid]]
    return float(np.mean(bias_residuals**2))


def _compute_dpi_variance_constant(
    y_values: np.ndarray,
    controls: np.ndarray | None,
    bin_idx: np.ndarray,
    bin_counts: np.ndarray,
    fe_codes: np.ndarray | None = None,
    fe_counts: np.ndarray | None = None,
) -> float | None:
    num_bins = bin_counts.size
    q = 0 if controls is None else controls.shape[1]
    size = num_bins + q

    # Under centered absorption the bin dummies become
    # e_b - shares[g] + global_shares. Everything below is expressible from the
    # (group, bin) cell aggregates, so the n x J projected design is never formed.
    shares: np.ndarray | None = None
    global_shares: np.ndarray | None = None
    if fe_codes is not None and fe_counts is not None:
        cell_counts = np.zeros((fe_counts.size, num_bins), dtype=float)
        np.add.at(cell_counts, (fe_codes, bin_idx), 1.0)
        shares = cell_counts / fe_counts[:, None]
        global_shares = bin_counts / y_values.size

    XtX = np.zeros((size, size), dtype=float)
    XtY = np.zeros(size, dtype=float)
    bin_block = np.diag(bin_counts)
    bin_y = np.bincount(bin_idx, weights=y_values, minlength=num_bins)
    if shares is not None:
        assert fe_counts is not None and global_shares is not None
        bin_block = (
            bin_block
            - shares.T @ (fe_counts[:, None] * shares)
            + y_values.size * np.outer(global_shares, global_shares)
        )
    XtX[:num_bins, :num_bins] = bin_block
    XtY[:num_bins] = bin_y

    if q:
        assert controls is not None  # type narrowing for checker
        sum_controls = np.zeros((num_bins, q), dtype=float)
        for idx in range(q):
            np.add.at(sum_controls[:, idx], bin_idx, controls[:, idx])
        XtX[:num_bins, num_bins:] = sum_controls
        XtX[num_bins:, :num_bins] = sum_controls.T
        XtX[num_bins:, num_bins:] = controls.T @ controls
        XtY[num_bins:] = controls.T @ y_values

    try:
        params = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
    beta_bins = params[:num_bins]
    gamma = params[num_bins:]

    fitted = beta_bins[bin_idx]
    if shares is not None:
        assert fe_codes is not None and global_shares is not None
        fitted = (
            fitted - (shares @ beta_bins)[fe_codes] + float(global_shares @ beta_bins)
        )
    if q:
        assert controls is not None  # type narrowing for checker
        fitted = fitted + controls @ gamma
    residuals = y_values - fitted
    u_sq = residuals**2

    XtUX = np.zeros_like(XtX)
    bin_meat = np.diag(np.bincount(bin_idx, weights=u_sq, minlength=num_bins))
    if shares is not None:
        assert (
            fe_codes is not None and fe_counts is not None and global_shares is not None
        )
        cell_u = np.zeros((fe_counts.size, num_bins), dtype=float)
        np.add.at(cell_u, (fe_codes, bin_idx), u_sq)
        group_u = cell_u.sum(axis=1)
        residualized_u_sums = (
            np.bincount(bin_idx, weights=u_sq, minlength=num_bins) - shares.T @ group_u
        )
        bin_meat = (
            bin_meat
            - cell_u.T @ shares
            - shares.T @ cell_u
            + shares.T @ (group_u[:, None] * shares)
            + np.outer(residualized_u_sums, global_shares)
            + np.outer(global_shares, residualized_u_sums)
            + float(u_sq.sum()) * np.outer(global_shares, global_shares)
        )
    XtUX[:num_bins, :num_bins] = bin_meat
    if q:
        assert controls is not None  # type narrowing for checker
        cross = np.zeros((num_bins, q), dtype=float)
        for idx in range(q):
            np.add.at(cross[:, idx], bin_idx, controls[:, idx] * u_sq)
        if shares is not None:
            assert (
                fe_codes is not None
                and fe_counts is not None
                and global_shares is not None
            )
            group_cross = np.zeros((fe_counts.size, q), dtype=float)
            for idx in range(q):
                np.add.at(group_cross[:, idx], fe_codes, controls[:, idx] * u_sq)
            total_cross = (controls * u_sq[:, None]).sum(axis=0)
            cross = (
                cross - shares.T @ group_cross + np.outer(global_shares, total_cross)
            )
        XtUX[:num_bins, num_bins:] = cross
        XtUX[num_bins:, :num_bins] = cross.T
        XtUX[num_bins:, num_bins:] = controls.T @ (controls * u_sq[:, None])

    # Absorbing does not make the fixed effect free: it still consumes G-1
    # parameters (exactly what drop_first dummies would have cost), and leaving them
    # out of the count inflates the degrees of freedom and shifts the selected bins.
    n_params = size + (fe_counts.size - 1 if fe_counts is not None else 0)
    dof = max(y_values.size - n_params, 1)
    XtUX *= y_values.size / dof

    XtX_inv = np.linalg.pinv(XtX)
    cov = XtX_inv @ XtUX @ XtX_inv
    cov_bins = cov[:num_bins, :num_bins]
    cov_diag = np.clip(np.diag(cov_bins), 0.0, None)
    return float(np.sum(bin_counts * cov_diag) / y_values.size)


def _collect_rule_of_thumb_stats(
    df: nw.LazyFrame, data_cols: tuple[str, ...], y: str
) -> nw.DataFrame:
    """Gather the global cross-moments needed by the rule-of-thumb selector."""
    y_expr = nw.col(y)
    y_sq_expr = y_expr * y_expr
    exprs: list[nw.Expr] = [
        nw.len().alias("__n"),
        y_expr.sum().alias("__sum_y"),
        y_sq_expr.sum().alias("__sum_y2"),
    ]
    for idx, col in enumerate(data_cols):
        col_expr = nw.col(col)
        exprs.append(col_expr.sum().alias(f"__sum_{idx}"))
        exprs.append((col_expr * y_expr).sum().alias(f"__xty_{idx}"))
        exprs.append((col_expr * y_sq_expr).sum().alias(f"__xty2_{idx}"))
        for jdx in range(idx, len(data_cols)):
            exprs.append(
                (col_expr * nw.col(data_cols[jdx])).sum().alias(f"__xtx_{idx}_{jdx}")
            )
    return df.select(*exprs).collect()


def _solve_normal_equations(xtx: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the normal equations, falling back to a pseudo-inverse when needed."""
    try:
        return np.linalg.solve(xtx, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ rhs


def _gaussian_inverse_density_squared_sum(
    df: nw.LazyFrame, x: str, mean_x: float, std_x: float, z_cutoff: float = 1.96
) -> float:
    """Compute sum_i 1 / f_G(x_i)^2 for the Gaussian reference f_G.

    Per Cattaneo et al. (2024) SA-4.1, the bias constant for p=0, v=0 uses
    f_X(x)^(2p+2-2v) = f_X(x)^2 in the denominator.

    We trim the density from below by capping z-scores at z_cutoff (default 1.96,
    i.e. 97.5th percentile) to prevent extreme outliers from causing the inverse
    to explode.
    """
    if std_x <= 0:
        raise ValueError("Standard deviation must be positive.")

    # Maximum allowed z^2 before capping
    z_sq_max = z_cutoff**2

    z_sq = ((nw.col(x) - mean_x) / std_x) ** 2
    # Cap z^2 at the cutoff to prevent density from getting too small
    z_sq_capped = nw.when(z_sq > z_sq_max).then(z_sq_max).otherwise(z_sq)
    # For 1/f(x)^2, we need exp(z^2) (not exp(0.5*z^2))
    # f(x) = 1/(sqrt(2pi)*std) * exp(-0.5*z^2)
    # 1/f(x)^2 = 2*pi*std^2 * exp(z^2)
    exp_expr = z_sq_capped.exp()
    sum_exp = (
        df.select(exp_expr.sum().alias("__sum_exp")).collect().item(0, "__sum_exp")
    )
    return float(sum_exp) * 2.0 * math.pi * std_x * std_x


@timed
def compute_bin_means(df: nw.LazyFrame, profile: Profile) -> nw.LazyFrame:
    df_plotting: nw.LazyFrame = (
        df.group_by(profile.bin_name)
        .agg(profile.x_col.mean(), profile.y_col.mean())
        .with_columns(nw.col(profile.bin_name).cast(nw.Int32))
    ).lazy()

    return df_plotting


@timed
def add_regression_features(
    df: nw.LazyFrame,
    numeric_controls: tuple[str, ...],
    categorical_controls: tuple[str, ...],
) -> tuple[nw.LazyFrame, tuple[str, ...], str | None]:
    """Inject numeric controls and one-hot categorical controls when requested.

    The highest-cardinality categorical is absorbed rather than encoded, so it is
    excluded from the dummies and returned as the third element. Remaining
    categoricals are one-hot encoded as before and ride along inside the control
    block, where the within correction handles them for free.
    """
    if not numeric_controls and not categorical_controls:
        return df, (), None
    if numeric_controls and not categorical_controls:
        return df, numeric_controls, None

    absorbed = select_absorbed(df, categorical_controls)
    remaining = tuple(c for c in categorical_controls if c != absorbed)
    if not remaining:
        return df, numeric_controls, absorbed

    build_dummies = configure_build_dummies(df.implementation)
    df_with_dummies, dummy_cols = build_dummies(df, remaining)
    if not dummy_cols:
        logger.debug("No dummy expressions created, all categorical controls constant")
        return df_with_dummies, numeric_controls, absorbed

    return df_with_dummies, numeric_controls + dummy_cols, absorbed


@timed
def add_polynomial_features(
    df: nw.LazyFrame,
    *,
    x_name: str,
    degree: int,
    distinct_suffix: str,
) -> tuple[nw.LazyFrame, tuple[str, ...]]:
    """Append polynomial columns in x up to the requested degree."""
    exprs: list[nw.Expr] = []
    names: list[str] = []
    for power in range(1, degree + 1):
        alias = f"__poly_{power}_{distinct_suffix}"
        exprs.append((nw.col(x_name) ** power).alias(alias))
        names.append(alias)
    if not exprs:
        return df, ()
    return df.with_columns(*exprs), tuple(names)


@timed
def make_native_dataframe(df_plotting: nw.LazyFrame, profile: Profile) -> IntoDataFrame:
    """Convert the plotting frame into the native backend expected by the caller."""
    df_out_nw = df_plotting.rename({profile.bin_name: "bin"}).sort("bin")
    logger.debug(
        "Type of df_out_nw: %s, implementation: %s",
        type(df_out_nw),
        df_out_nw.implementation,
    )

    if profile.implementation in (
        Implementation.PYSPARK,
        Implementation.DUCKDB,
        Implementation.DASK,
    ):
        return df_out_nw.to_native()
    else:
        return df_out_nw.collect().to_native()
