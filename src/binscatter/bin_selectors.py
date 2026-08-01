"""Helper utilities for automatic bin selection (ROT and DPI)."""

from __future__ import annotations

import math
from typing import Tuple

import narwhals as nw
import numpy as np


def select_rule_of_thumb_bins(
    df: nw.LazyFrame, x: str, y: str, regression_features: tuple[str, ...]
) -> int:
    """Rule-of-thumb selector following Cattaneo et al. (2024) SA-4.1."""
    data_cols: Tuple[str, ...] = (x, *regression_features)
    stats = _collect_rule_of_thumb_stats(df, data_cols, y)
    n_obs = stats.item(0, "__n")
    if n_obs is None or n_obs <= 1:
        raise ValueError(
            "Rule-of-thumb selector needs at least two observations in the design."
        )
    n_obs_f = float(n_obs)
    design_size = len(data_cols) + 1
    xtx = np.zeros((design_size, design_size), dtype=float)
    xty = np.zeros(design_size, dtype=float)
    xty_sq = np.zeros(design_size, dtype=float)
    column_sums = np.zeros(design_size, dtype=float)

    xtx[0, 0] = n_obs_f
    column_sums[0] = n_obs_f
    xty[0] = stats.item(0, "__sum_y")
    xty_sq[0] = stats.item(0, "__sum_y2")

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

    beta_y = _solve_normal_equations(xtx, xty)
    beta_y_sq = _solve_normal_equations(xtx, xty_sq)

    x_sum = column_sums[1]
    x_sq_sum = xtx[1, 1]
    mean_x = x_sum / n_obs_f
    var_x = (x_sq_sum / n_obs_f) - mean_x**2
    if var_x <= 0:
        raise ValueError(
            "Rule-of-thumb selector requires the x column to have positive variance."
        )
    std_x = math.sqrt(var_x)

    sum_inv_density_sq = _gaussian_inverse_density_squared_sum(df, x, mean_x, std_x)
    slope = beta_y[1]
    imse_bcons = 1.0 / 12.0
    bias_constant = imse_bcons * (slope**2) * (sum_inv_density_sq / n_obs_f)
    if bias_constant <= 0 or not math.isfinite(bias_constant):
        raise ValueError("Rule-of-thumb selector produced invalid bias constant.")

    sum_pred_y_sq = float(column_sums @ beta_y_sq)
    quad_form = float(beta_y.T @ xtx @ beta_y)
    variance_constant = (sum_pred_y_sq - quad_form) / n_obs_f
    variance_constant = max(variance_constant, 0.0)
    if variance_constant <= 0 or not math.isfinite(variance_constant):
        raise ValueError("Rule-of-thumb selector produced invalid variance constant.")

    prefactor = (2.0 * bias_constant) / variance_constant
    j_float = prefactor ** (1.0 / 3.0) * n_obs_f ** (1.0 / 3.0)
    max_bins = max(2, int(n_obs) // 10)
    computed_bins = max(2, int(round(j_float)))
    return min(max_bins, computed_bins)


def select_dpi_bins(
    df: nw.LazyFrame, x: str, y: str, regression_features: tuple[str, ...]
) -> int:
    """Direct plug-in selector (SA-4.2)"""
    pilot_bins = select_rule_of_thumb_bins(df, x, y, regression_features)

    required_cols = (x, y) + regression_features
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

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if controls is not None:
        mask &= np.all(np.isfinite(controls), axis=1)

    if not np.all(mask):
        x_values = x_values[mask]
        y_values = y_values[mask]
        if controls is not None:
            controls = controls[mask]

    n_obs = x_values.size
    if n_obs < 5:
        return pilot_bins

    constants = estimate_dpi_imse_constants(
        x_values,
        y_values,
        controls,
        pilot_bins,
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


def estimate_dpi_imse_constants(
    x_values: np.ndarray,
    y_values: np.ndarray,
    controls: np.ndarray | None,
    pilot_bins: int,
) -> tuple[float, float] | None:
    """Compute the IMSE bias/variance constants for the DPI selector."""
    n_obs = x_values.size
    if n_obs == 0:
        return None

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    span = x_max - x_min
    if not np.isfinite(span) or span <= 0:
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
    if controls is not None and controls.size:
        design_matrix = np.column_stack((spline_design, controls))
    else:
        design_matrix = spline_design

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
    )
    if variance_cons is None:
        return None

    num_bins = bin_counts.size
    imse_b = bias_cons * (num_bins**2)
    imse_v = variance_cons / num_bins
    return imse_b, imse_v


def _quantile_edges(x_norm: np.ndarray, num_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    try:
        edges = np.quantile(x_norm, quantiles, method="linear")
    except TypeError:  # numpy<1.22 compatibility
        edges = np.quantile(x_norm, quantiles, interpolation="linear")  # type: ignore[call-overload]
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
) -> float | None:
    num_bins = bin_counts.size
    q = 0 if controls is None else controls.shape[1]
    size = num_bins + q
    XtX = np.zeros((size, size), dtype=float)
    XtX[:num_bins, :num_bins] = np.diag(bin_counts)
    XtY = np.zeros(size, dtype=float)
    XtY[:num_bins] = np.bincount(bin_idx, weights=y_values, minlength=num_bins)

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
    if q:
        assert controls is not None  # type narrowing for checker
        fitted = fitted + controls @ gamma
    residuals = y_values - fitted
    u_sq = residuals**2

    XtUX = np.zeros_like(XtX)
    XtUX[:num_bins, :num_bins] = np.diag(
        np.bincount(bin_idx, weights=u_sq, minlength=num_bins)
    )
    if q:
        assert controls is not None  # type narrowing for checker
        cross = np.zeros((num_bins, q), dtype=float)
        for idx in range(q):
            np.add.at(cross[:, idx], bin_idx, controls[:, idx] * u_sq)
        XtUX[:num_bins, num_bins:] = cross
        XtUX[num_bins:, :num_bins] = cross.T
        XtUX[num_bins:, num_bins:] = controls.T @ (controls * u_sq[:, None])

    dof = max(y_values.size - size, 1)
    XtUX *= y_values.size / dof

    XtX_inv = np.linalg.pinv(XtX)
    cov = XtX_inv @ XtUX @ XtX_inv
    cov_bins = cov[:num_bins, :num_bins]
    cov_diag = np.clip(np.diag(cov_bins), 0.0, None)
    return float(np.sum(bin_counts * cov_diag) / y_values.size)


def _collect_rule_of_thumb_stats(
    df: nw.LazyFrame, data_cols: tuple[str, ...], y: str
) -> nw.DataFrame:
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
    try:
        return np.linalg.solve(xtx, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ rhs


def _gaussian_inverse_density_squared_sum(
    df: nw.LazyFrame, x: str, mean_x: float, std_x: float, z_cutoff: float = 1.96
) -> float:
    if std_x <= 0:
        raise ValueError("Standard deviation must be positive.")

    z_sq_max = z_cutoff**2
    z_sq = ((nw.col(x) - mean_x) / std_x) ** 2
    z_sq_capped = nw.when(z_sq > z_sq_max).then(z_sq_max).otherwise(z_sq)
    exp_expr = z_sq_capped.exp()
    sum_exp = (
        df.select(exp_expr.sum().alias("__sum_exp")).collect().item(0, "__sum_exp")
    )
    return float(sum_exp) * 2.0 * math.pi * std_x * std_x
