"""Helper utilities for the direct plug-in (DPI) bin selector."""

from __future__ import annotations

import numpy as np


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
        edges = np.quantile(x_norm, quantiles, interpolation="linear")
    unique_edges = np.unique(edges)
    return unique_edges


def _linear_spline_design(
    x_norm: np.ndarray, interior_knots: np.ndarray
) -> np.ndarray:
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
        fitted = fitted + controls @ gamma
    residuals = y_values - fitted
    u_sq = residuals**2

    XtUX = np.zeros_like(XtX)
    XtUX[:num_bins, :num_bins] = np.diag(
        np.bincount(bin_idx, weights=u_sq, minlength=num_bins)
    )
    if q:
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
