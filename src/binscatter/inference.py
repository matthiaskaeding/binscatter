from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional

import numpy as np


@dataclass
class BinAggregates:
    counts: np.ndarray
    mean_x: np.ndarray
    sum_x: np.ndarray
    sum_x2: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray
    sum_xy: np.ndarray
    control_sums: np.ndarray
    control_x_sums: np.ndarray


@dataclass
class RegressionResult:
    beta: np.ndarray
    gamma: np.ndarray
    mean_controls: np.ndarray
    xtx: np.ndarray
    xty: np.ndarray
    rss: float
    dof_resid: int
    total_count: int
    total_y2: float
    bin_stats: BinAggregates
    control_cross: np.ndarray
    wy: np.ndarray


@dataclass
class IntervalResult:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    std_error: np.ndarray
    method: str


def _standard_normal_quantile(level: float) -> float:
    """Return the two-sided z critical value for a given confidence level."""
    if not (0.0 < level < 1.0):
        raise ValueError("ci_level must be strictly between 0 and 1.")
    alpha = 1.0 - level
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _homoskedastic_vcov(xtx: np.ndarray, rss: float, dof_resid: int) -> Optional[np.ndarray]:
    """Compute sigma^2 (X'X)^-1 using a pseudo inverse if needed."""
    if dof_resid <= 0:
        return None
    sigma2 = max(rss / dof_resid, 0.0)
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    return sigma2 * xtx_inv


def compute_pointwise_ci(result: RegressionResult, ci_level: float) -> IntervalResult:
    """Construct bin-level confidence intervals for the fixed-J parameter Xi."""
    vcov = _homoskedastic_vcov(result.xtx, result.rss, result.dof_resid)
    num_bins = result.beta.size
    size = num_bins + result.gamma.size
    centers = np.empty(num_bins, dtype=float)
    lowers = np.empty_like(centers)
    uppers = np.empty_like(centers)
    std_errors = np.empty_like(centers)
    z = _standard_normal_quantile(ci_level)

    if vcov is None:
        centers.fill(np.nan)
        lowers.fill(np.nan)
        uppers.fill(np.nan)
        std_errors.fill(np.nan)
        return IntervalResult(centers, lowers, uppers, std_errors, method="pointwise")

    control_adjustment = (
        float(result.mean_controls @ result.gamma) if result.gamma.size else 0.0
    )
    for j in range(num_bins):
        g = np.zeros(size, dtype=float)
        g[j] = 1.0
        if result.gamma.size:
            g[num_bins:] = result.mean_controls
        centers[j] = result.beta[j] + control_adjustment
        variance = float(g @ vcov @ g)
        std_errors[j] = float(np.sqrt(max(variance, 0.0)))
        lowers[j] = centers[j] - z * std_errors[j]
        uppers[j] = centers[j] + z * std_errors[j]

    return IntervalResult(centers, lowers, uppers, std_errors, method="pointwise")


def _build_linear_spline_system(
    result: RegressionResult, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the constrained piecewise linear system used for RBC intervals."""
    bin_stats = result.bin_stats
    num_bins = bin_stats.counts.size
    num_controls = result.gamma.size
    num_basis = num_bins + 1
    size = num_basis + num_controls
    xtx = np.zeros((size, size), dtype=float)
    xty = np.zeros(size, dtype=float)

    for j in range(num_bins):
        a = bin_edges[j]
        b = bin_edges[j + 1]
        h = float(b - a)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("Bin edges must be finite and strictly increasing for RBC inference.")
        s0 = float(bin_stats.counts[j])
        sx = float(bin_stats.sum_x[j])
        sx2 = float(bin_stats.sum_x2[j])
        sy = float(bin_stats.sum_y[j])
        sxy = float(bin_stats.sum_xy[j])
        left = j
        right = j + 1
        left_sq = (s0 * b * b - 2.0 * b * sx + sx2) / (h * h)
        right_sq = (sx2 - 2.0 * a * sx + s0 * a * a) / (h * h)
        cross = ((a + b) * sx - sx2 - s0 * a * b) / (h * h)

        xtx[left, left] += left_sq
        xtx[right, right] += right_sq
        xtx[left, right] += cross
        xtx[right, left] += cross

        left_y = (b * sy - sxy) / h
        right_y = (sxy - a * sy) / h
        xty[left] += left_y
        xty[right] += right_y

        if num_controls:
            ctrl_sum = bin_stats.control_sums[j]
            ctrl_x_sum = bin_stats.control_x_sums[j]
            left_ctrl = (b * ctrl_sum - ctrl_x_sum) / h
            right_ctrl = (ctrl_x_sum - a * ctrl_sum) / h
            xtx[left, num_basis:] += left_ctrl
            xtx[num_basis:, left] += left_ctrl
            xtx[right, num_basis:] += right_ctrl
            xtx[num_basis:, right] += right_ctrl

    if num_controls:
        xtx[num_basis:, num_basis:] = result.control_cross
        xty[num_basis:] = result.wy

    return xtx, xty


def compute_rbc_ci(
    result: RegressionResult, bin_edges: np.ndarray, ci_level: float
) -> IntervalResult:
    """Construct robust-bias-corrected confidence intervals using a continuous linear spline."""
    xtx, xty = _build_linear_spline_system(result, bin_edges)
    try:
        theta = np.linalg.solve(xtx, xty)
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        theta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
        xtx_inv = np.linalg.pinv(xtx)

    rss = max(result.total_y2 - float(theta @ xty), 0.0)
    rank = np.linalg.matrix_rank(xtx)
    dof = max(result.total_count - rank, 1)
    sigma2 = max(rss / dof, 0.0)
    vcov = sigma2 * xtx_inv

    num_bins = result.bin_stats.counts.size
    num_basis = num_bins + 1
    centers = np.empty(num_bins, dtype=float)
    lowers = np.empty_like(centers)
    uppers = np.empty_like(centers)
    std_errors = np.empty_like(centers)
    z = _standard_normal_quantile(ci_level)

    mean_controls = result.mean_controls
    k = mean_controls.size
    for j in range(num_bins):
        a = bin_edges[j]
        b = bin_edges[j + 1]
        h = float(b - a)
        if h <= 0:
            raise ValueError("Bin edges must be strictly increasing for RBC inference.")
        x_mean = float(result.bin_stats.mean_x[j])
        t = 0.0 if b == a else (x_mean - a) / h
        t = min(max(t, 0.0), 1.0)
        g = np.zeros(num_basis + k, dtype=float)
        g[j] = 1.0 - t
        g[j + 1] = t
        if k:
            g[num_basis:] = mean_controls
        centers[j] = float(g @ theta)
        variance = float(g @ vcov @ g)
        std_errors[j] = float(np.sqrt(max(variance, 0.0)))
        lowers[j] = centers[j] - z * std_errors[j]
        uppers[j] = centers[j] + z * std_errors[j]

    return IntervalResult(centers, lowers, uppers, std_errors, method="rbc")
