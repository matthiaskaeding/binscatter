"""Absorb a high-cardinality categorical control via the Mundlak / within transform.

One-hot encoding a categorical control produces G-1 dummy columns, which makes
``_ensure_feature_moments`` build ``k(k+1)/2`` sum-product aggregations and
``partial_out_controls`` solve a dense ``(J+k)x(J+k)`` system. At a few thousand
levels that dies while building the query plan.

Instead, absorb the control. For any two matrices ``A``, ``B`` and the group-dummy
matrix ``D``, the within-transformed cross product is

    (M_D A)'(M_D B) = A'B - S_A' N^-1 S_B

where ``S_A = D'A`` are group sums and ``N = diag(group counts)``. So only group
counts, group sums, and the bin x group crosstab are needed -- two aggregations,
independent of the number of levels, and no per-row residuals are materialized.

This is the Mundlak transform applied at the moment level. It is exact for a single
categorical, because ``D'D`` is diagonal; with two absorbed factors ``D'D`` picks up
incidence blocks and there is no closed form (the two-way fixed effects problem).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

logger = logging.getLogger(__name__)

_FE_COUNT = "__fe_count"


@dataclass
class FEMoments:
    """Group-level sufficient statistics for the absorbed categorical.

    Attributes:
        counts: ``(G,)`` observations per group.
        inv_counts: ``(G,)`` reciprocal of ``counts``, i.e. the diagonal of ``N^-1``.
        bin_counts: ``(G, J)`` bin-by-group crosstab, ``S_B``.
        feature_sums: ``(G, q)`` group sums of the regression features, ``S_W``.
        response_sums: group sums keyed by response name, e.g. ``S_y``.
        total_count: total number of observations.
    """

    counts: np.ndarray
    inv_counts: np.ndarray
    bin_counts: np.ndarray
    feature_sums: np.ndarray
    response_sums: dict[str, np.ndarray]
    total_count: float


def within_correct(
    raw: np.ndarray,
    group_sums_a: np.ndarray,
    group_sums_b: np.ndarray,
    inv_counts: np.ndarray,
) -> np.ndarray:
    """Return ``raw - S_A' N^-1 S_B``, the within-transformed cross product.

    ``group_sums_b`` may be ``(G, b)`` for a matrix block or ``(G,)`` for a vector,
    matching the shape of ``raw``.
    """
    weighted = inv_counts[:, None] * group_sums_a
    return raw - weighted.T @ group_sums_b


def group_codes(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(codes, counts)`` for ``groups``, where codes index into counts."""
    _, codes = np.unique(groups, return_inverse=True)
    return codes, np.bincount(codes).astype(float)


def demean_within(
    values: np.ndarray, codes: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Subtract group means from ``values``.

    Used only where the data has already been materialized (the DPI selector); the
    lazy paths use :func:`within_correct` on the moments instead.
    """
    sums = np.bincount(codes, weights=values, minlength=counts.size)
    return values - (sums / counts)[codes]


#: Below this many levels the one-hot path is kept, so existing results do not move.
#:
#: Absorbing is exact for the bin estimates, but the DPI selector's sandwich variance
#: is not invariant to the reparameterization: the one-hot design is full rank
#: (drop_first pins the level) while the within system is rank-deficient by one, so
#: the individual bin coefficients -- and hence their variances -- are identified
#: only up to a shift. Crossing the threshold can therefore shift the selected bin
#: count, which is why small categoricals stay where they are.
#:
#: 50 is where the one-hot path starts costing real time without yet being
#: unusable. Measured on pandas at n=20,000, num_bins=10 (`make benchmark-fe`):
#:
#:     levels   one-hot   absorbed
#:         50     0.74s      0.04s
#:        100     3.52s      0.02s
#:        200    31.32s      0.03s
#:        400   ~6 min       0.02s
#:
#: It scales roughly cubically in levels, so the exact cutoff matters less than
#: being on the right side of the knee.
ABSORB_MIN_LEVELS = 50


def select_absorbed(
    df: nw.LazyFrame, categorical_controls: tuple[str, ...]
) -> str | None:
    """Pick the categorical control to absorb: the one with the most levels.

    Returns ``None`` when there is nothing to absorb. Costs a single pass computing
    ``n_unique`` for every categorical control at once.

    A constant categorical is never absorbed: it carries no information, and the
    one-hot path already drops it, so absorbing it would only introduce a degenerate
    single-group demeaning.
    """
    if not categorical_controls:
        return None

    counts = df.select(
        *(nw.col(c).n_unique().alias(c) for c in categorical_controls)
    ).collect()
    cardinalities = {c: int(counts.item(0, c)) for c in categorical_controls}
    absorbed = max(categorical_controls, key=lambda c: cardinalities[c])
    if cardinalities[absorbed] < ABSORB_MIN_LEVELS:
        logger.debug(
            "[fixed_effects] max cardinality %d < %d, keeping the one-hot path",
            cardinalities[absorbed],
            ABSORB_MIN_LEVELS,
        )
        return None
    logger.debug(
        "[fixed_effects] cardinalities=%s absorbing=%s", cardinalities, absorbed
    )
    return absorbed


def compute_fe_moments(
    df: nw.LazyFrame,
    *,
    fe_name: str,
    feature_names: tuple[str, ...],
    response_exprs: Mapping[str, nw.Expr],
    bin_name: str | None = None,
    bin_labels: Sequence[Any] | None = None,
) -> FEMoments:
    """Collect the group-level sufficient statistics needed to absorb ``fe_name``.

    One aggregation over the fixed effect, plus a second over (bin, fixed effect)
    when ``bin_name``/``bin_labels`` are given. The bin selectors run before bins
    exist, so they omit the crosstab and get a ``(G, 0)`` ``bin_counts``.

    ``bin_labels`` fixes the column order of the crosstab so it lines up with the
    caller's per-bin frame -- group and bin ordering out of ``group_by`` is not
    guaranteed to be stable across backends, so both are mapped explicitly by label
    rather than by position.
    """
    response_names = list(response_exprs)
    aliases = {name: f"__fe_resp_{i}" for i, name in enumerate(response_names)}

    # Materialize derived responses before grouping. Dask rejects non-elementary
    # aggregations such as (col * col).sum() inside group_by, so the expression has
    # to become a column first.
    df = df.with_columns(
        *(response_exprs[name].alias(aliases[name]) for name in response_names)
    )

    group_aggs = [nw.len().alias(_FE_COUNT)]
    group_aggs += [
        nw.col(aliases[name]).sum().alias(aliases[name]) for name in response_names
    ]
    group_aggs += [nw.col(c).sum().alias(c) for c in feature_names]

    grouped = df.group_by(fe_name).agg(*group_aggs).collect()

    labels = grouped.get_column(fe_name).to_list()
    num_groups = len(labels)
    if num_groups == 0:
        raise ValueError(f"No groups found for fixed effect column '{fe_name}'.")
    group_index = {label: i for i, label in enumerate(labels)}

    counts = np.asarray(grouped.get_column(_FE_COUNT).to_list(), dtype=float)
    response_sums = {
        name: np.asarray(grouped.get_column(aliases[name]).to_list(), dtype=float)
        for name in response_names
    }
    if feature_names:
        feature_sums = np.column_stack(
            [
                np.asarray(grouped.get_column(c).to_list(), dtype=float)
                for c in feature_names
            ]
        )
    else:
        feature_sums = np.zeros((num_groups, 0), dtype=float)

    if bin_name is None or bin_labels is None:
        bin_counts = np.zeros((num_groups, 0), dtype=float)
        logger.debug(
            "[fixed_effects] fe=%s groups=%d (no crosstab)", fe_name, num_groups
        )
    else:
        bin_index = {label: i for i, label in enumerate(bin_labels)}
        crosstab = (
            df.group_by(bin_name, fe_name).agg(nw.len().alias(_FE_COUNT)).collect()
        )
        bin_counts = np.zeros((num_groups, len(bin_labels)), dtype=float)
        for label, bin_label, count in zip(
            crosstab.get_column(fe_name).to_list(),
            crosstab.get_column(bin_name).to_list(),
            crosstab.get_column(_FE_COUNT).to_list(),
        ):
            bin_counts[group_index[label], bin_index[bin_label]] = float(count)

        logger.debug(
            "[fixed_effects] fe=%s groups=%d crosstab_cells=%d",
            fe_name,
            num_groups,
            crosstab.shape[0],
        )

    return FEMoments(
        counts=counts,
        inv_counts=1.0 / counts,
        bin_counts=bin_counts,
        feature_sums=feature_sums,
        response_sums=response_sums,
        total_count=float(counts.sum()),
    )


def recover_levels(
    moments: FEMoments,
    beta: np.ndarray,
    gamma: np.ndarray,
    response: str,
) -> float:
    """Return the count-weighted mean fixed effect.

    Absorbing removes the level from ``beta``: the within system is rank-deficient by
    exactly one, so ``beta`` is identified only up to a shift. Adding this back makes
    ``beta_j + mean_alpha`` invariant to that shift and puts fitted values back on the
    original ``y`` scale.
    """
    alpha = moments.inv_counts * (
        moments.response_sums[response]
        - moments.bin_counts @ beta
        - (moments.feature_sums @ gamma if gamma.size else 0.0)
    )
    return float(moments.counts @ alpha / moments.total_count)
