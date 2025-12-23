import logging
import math
import operator
import uuid
import warnings
from functools import reduce
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Tuple,
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

logger = logging.getLogger(__name__)


@overload
def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    *,
    controls: Iterable[str] | str | None = None,
    num_bins: int | Literal["rule-of-thumb"] = "rule-of-thumb",
    return_type: Literal["plotly"] = "plotly",
    **kwargs,
) -> go.Figure: ...


@overload
def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    *,
    controls: Iterable[str] | str | None = None,
    num_bins: int | Literal["rule-of-thumb"] = "rule-of-thumb",
    return_type: Literal["native"] = "native",
    **kwargs,
) -> object: ...


def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    *,
    controls: Iterable[str] | str | None = None,
    num_bins: int | Literal["rule-of-thumb"] = "rule-of-thumb",
    return_type: Literal["plotly", "native"] = "plotly",
    **kwargs,
) -> object:
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df: Any dataframe supported by narwhals.
        x: Name of the x column.
        y: Name of the y column.
        controls: Optional control columns to partial out (either a string or iterable of strings).
        num_bins: Number of quantile bins to form, or ``"rule-of-thumb"`` for the automatic selector.
        return_type: If ``plotly`` (default) return a Plotly figure; if ``native`` return a dataframe matching the input backend.
        kwargs: Extra keyword args forwarded to ``plotly.express.scatter`` when plotting.

    Returns:
        A Plotly figure or native dataframe, depending on ``return_type``.
    """
    if return_type not in ("plotly", "native"):
        msg = f"Invalid return_type: {return_type}"
        raise ValueError(msg)
    if isinstance(num_bins, str):
        auto_bins = num_bins == "rule-of-thumb"
        if not auto_bins:
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
        auto_bins = False

    if not isinstance(x, str):
        raise TypeError("x_name must be a string")
    if not isinstance(y, str):
        raise TypeError("y_name must be a string")

    controls = _clean_controls(controls)
    if x in controls:
        raise ValueError("x cannot be in controls")
    if y in controls:
        raise ValueError("y cannot be in controls")
    if len(set(controls)) < len(controls):
        raise ValueError("controls contains duplicate entries")

    df, is_lazy_input, numeric_columns, categorical_columns = clean_df(
        df, controls, x, y
    )
    df_with_regression_features, regression_features = maybe_add_regression_features(
        df,
        numeric_controls=numeric_columns,
        categorical_controls=categorical_columns,
    )

    distinct_suffix = str(uuid.uuid4()).replace(
        "-", "_"
    )  # "-" in col names can cause issues
    bin_name = f"bins____{distinct_suffix}"
    if auto_bins:
        computed_num_bins = _select_rule_of_thumb_bins(
            df_with_regression_features, x, y, regression_features
        )
    else:
        computed_num_bins = manual_bins

    profile = Profile(
        num_bins=computed_num_bins,
        x_name=x,
        y_name=y,
        bin_name=bin_name,
        regression_features=regression_features,
        distinct_suffix=distinct_suffix,
        is_lazy_input=is_lazy_input,
        implementation=df.implementation,
    )
    add_bins: Callable = configure_quantile_handler(profile)
    df_prepped = add_bins(df_with_regression_features)

    if not controls:
        # The easy one, just compute the means by bin
        df_plotting = compute_bin_means(df_prepped, profile)
    else:
        # Compute regression coefficients
        # and partial out the effect of the controls
        # (see section 2.2 in Cattaneo, Crump, Farrell and Feng (2024))
        df_plotting, _ = partial_out_controls(df_prepped, profile)

    match return_type:
        case "plotly":
            return make_plot_plotly(df_plotting, profile, kwargs_binscatter=kwargs)
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
    regression_features: Tuple[str, ...]

    @property
    def x_col(self) -> nw.Expr:
        return nw.col(self.x_name)

    @property
    def y_col(self) -> nw.Expr:
        return nw.col(self.y_name)


def partial_out_controls(
    df_prepped: nw.LazyFrame, profile: Profile
) -> tuple[nw.LazyFrame, dict[str, np.ndarray]]:
    """Compute binscatter means after partialling out controls following Cattaneo et al. (2024)."""

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
    else:  # This should never happen
        bin_control_sums = np.zeros((profile.num_bins, 0))

    total_exprs = [nw.len().alias("__total_count")]
    total_exprs.extend(
        nw.col(alias).sum().alias(f"__total_ctrl_{idx}")
        for idx, alias in enumerate(profile.regression_features)
    )
    total_exprs.extend(
        (nw.col(alias) * profile.y_col).sum().alias(f"__wy_{idx}")
        for idx, alias in enumerate(profile.regression_features)
    )
    for i, alias_i in enumerate(profile.regression_features):
        for j, alias_j in enumerate(profile.regression_features[i:], start=i):
            total_exprs.append(
                (nw.col(alias_i) * nw.col(alias_j)).sum().alias(f"__ww_{i}_{j}")
            )

    totals = df_prepped.select(*total_exprs).collect()
    total_count = totals.item(0, "__total_count")
    if profile.regression_features:
        total_ctrl_sums = np.array(
            [
                totals.item(0, f"__total_ctrl_{idx}")
                for idx in range(len(profile.regression_features))
            ]
        )
        wy = np.array(
            [
                totals.item(0, f"__wy_{idx}")
                for idx in range(len(profile.regression_features))
            ]
        )
        ww = np.zeros(
            (len(profile.regression_features), len(profile.regression_features))
        )
        for i in range(len(profile.regression_features)):
            for j in range(i, len(profile.regression_features)):
                alias = f"__ww_{i}_{j}"
                value = totals.item(0, alias)
                ww[i, j] = value
                ww[j, i] = value
    else:
        total_ctrl_sums = np.array([])
        wy = np.array([])
        ww = np.zeros((0, 0))

    # Assemble normal equations
    num_bins = profile.num_bins
    k = len(profile.regression_features)
    size = num_bins + k
    XTX = np.zeros((size, size))
    XTy = np.zeros(size)

    XTX[:num_bins, :num_bins] = np.diag(counts)
    if k:
        XTX[:num_bins, num_bins:] = bin_control_sums
        XTX[num_bins:, :num_bins] = bin_control_sums.T
        XTX[num_bins:, num_bins:] = ww
        XTy[num_bins:] = wy

    XTy[:num_bins] = sum_y

    try:
        theta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        theta, *_ = np.linalg.lstsq(XTX, XTy, rcond=None)

    beta = theta[:num_bins]
    gamma = theta[num_bins:]
    mean_controls = total_ctrl_sums / total_count if k else np.array([])
    fitted = beta + (mean_controls @ gamma if k else 0.0)

    y_vals = nw.new_series(
        name=profile.y_name, values=fitted, backend=per_bin.implementation
    )

    df_plotting = (
        per_bin.select(profile.bin_name, profile.x_name).with_columns(y_vals).lazy()
    )

    return df_plotting, {"beta": beta, "gamma": gamma}


def make_plot_plotly(
    df_plotting: nw.LazyFrame, profile: Profile, kwargs_binscatter: dict[str, Any]
) -> go.Figure:
    """Make plot from prepared dataframe.

    Args:
      df_prepped (nw.LazyFrame): Prepared dataframe. Has three columns: bin, x, y with names in profile"""
    data = df_plotting.select(profile.x_name, profile.y_name).collect()
    if data.shape[0] < profile.num_bins:
        raise ValueError("Quantiles are not unique. Decrease number of bins.")

    x_min = data.get_column(profile.x_name).min()
    x_max = data.get_column(profile.x_name).max()

    x = data.get_column(profile.x_name).to_list()
    if len(set(x)) < profile.num_bins:
        msg = f"Unique number of bins is {len(set(x))} fewer than {profile.num_bins} as desired. Decrease parameter num_bins."
        raise ValueError(msg)
    y = data.get_column(profile.y_name).to_list()

    pad = (x_max - x_min) * 0.04
    padded_range_x = (x_min - pad, x_max + pad)
    scatter_args = {
        "x": x,
        "y": y,
        "range_x": padded_range_x,
        "title": "Binscatter",
        "labels": {
            "x": profile.x_name,
            "y": profile.y_name,
        },
        "template": "simple_white",
    }
    for k in kwargs_binscatter:
        if k in ("x", "y", "range_x"):
            msg = f"px.scatter will ignore keyword argument '{k}'"
            warnings.warn(msg)
            continue
        scatter_args[k] = kwargs_binscatter[k]

    return px.scatter(**scatter_args)


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
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return tuples of numeric and categorical column names for a narwhals frame."""

    def _safe_columns(selection: Any) -> Tuple[str, ...]:
        if selection is None:
            return tuple()
        columns: Tuple[str, ...] = tuple()
        if hasattr(selection, "columns"):
            try:
                columns = tuple(selection.columns)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - backend quirk
                columns = tuple()
        if columns:
            return columns
        if hasattr(selection, "collect_schema"):
            try:
                schema = selection.collect_schema()
            except Exception:  # pragma: no cover - backend quirk
                schema = None
            else:
                if schema is not None:
                    if hasattr(schema, "names") and callable(schema.names):
                        return tuple(schema.names())
                    if isinstance(schema, dict):
                        return tuple(schema.keys())
        return tuple()

    numeric_cols = _safe_columns(frame.select(ncs.numeric()))
    frame_columns = tuple(frame.columns)
    categorical_cols = tuple(col for col in frame_columns if col not in numeric_cols)

    return numeric_cols, categorical_cols


# Quantiles


# Defined here for testability
def _add_fallback(
    df: nw.LazyFrame, profile: Profile, probs: List[float]
) -> nw.LazyFrame:
    try:
        qs = df.select(
            [
                profile.x_col.quantile(p, interpolation="linear").alias(f"q{p}")
                for p in probs
            ]
        ).collect()
    except TypeError:
        expr = cast(Any, profile.x_col)
        qs = df.select([expr.quantile(p).alias(f"q{p}") for p in probs]).collect()
    except Exception as e:
        logger.error(
            "Tried making quantiles with and without interpolation method for df of type: %s",
            type(df),
        )
        raise e
    qs_long = (
        qs.unpivot(variable_name="prob", value_name="quantile")
        .sort("quantile")
        .with_row_index(profile.bin_name)
    )

    quantile_bins = qs_long.select("quantile", profile.bin_name).lazy()

    # Sorting is not always necessary - but for safety we sort
    return (
        df.sort(profile.x_name)
        .join_asof(
            quantile_bins,
            left_on=profile.x_name,
            right_on="quantile",
            strategy="forward",
        )
        .drop("quantile")
    )


def _make_probs(num_bins) -> List[float]:
    return [i / num_bins for i in range(1, num_bins + 1)]


def configure_quantile_handler(profile: Profile) -> Callable:
    """Return a backend-specific function that assigns quantile bins."""
    probs = _make_probs(profile.num_bins)

    def add_fallback(df: nw.LazyFrame):
        return _add_fallback(df, profile, probs)

    def add_to_dask(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            from pandas import cut
        except ImportError:
            raise ImportError("Dask support requires dask and pandas to be installed.")

        df_native = df.to_native()
        logger.debug("Type of df_native (should be dask): %s", type(df_native))
        quantiles = df_native[profile.x_name].quantile(probs[:-1]).compute()
        bins = (float("-inf"), *quantiles, float("inf"))
        df_native[profile.bin_name] = df_native[profile.x_name].map_partitions(
            cut,
            bins=bins,
            labels=range(len(probs)),
            include_lowest=False,
            right=False,
        )

        return nw.from_native(df_native).lazy()

    def add_to_pandas(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            from pandas import cut
        except ImportError:
            raise ImportError("Pandas support requires pandas to be installed.")
        df_native = df.to_native()
        x = df_native[profile.x_name]
        quantiles = x.quantile(probs[:-1])

        bins = (float("-Inf"), *quantiles, float("Inf"))
        logger.debug("bins: %s", bins)
        logger.debug("quantiles: %s", quantiles)

        buckets = cut(
            df_native[profile.x_name],
            bins=bins,
            labels=range(len(probs)),
            include_lowest=False,
            right=False,
        )
        df_native[profile.bin_name] = buckets

        return nw.from_native(df_native).lazy()

    def add_to_polars(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars support requires Polars to be installed.")
        # Because cut and qcut are not stable we use when-then
        df_native = df.to_native()
        x_col = pl.col(profile.x_name)

        qs = df_native.select(
            [x_col.quantile(p, interpolation="linear").alias(f"q{p}") for p in probs]
        ).collect()
        expr = pl
        n = qs.width
        for i in range(n):
            thr = qs.item(0, i)
            cond = x_col.le(thr) if i == n - 1 else x_col.lt(thr)
            expr = expr.when(cond).then(pl.lit(i))
        expr = expr.alias(profile.bin_name)
        df_native_with_bin = df_native.with_columns(expr)

        return nw.from_native(df_native_with_bin).lazy()

    def add_to_duckdb(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            import duckdb

            rel = df.to_native()
            assert isinstance(rel, duckdb.DuckDBPyRelation), f"{type(rel)=}"

        except Exception as e:
            raise RuntimeError(
                "Failed to use df.to_native(); DuckDB may not be installed."
            ) from e

        order_expr = f"{profile.x_name} ASC"
        rel_with_bins = rel.project(
            f"*, ntile({len(probs)}) OVER (ORDER BY {order_expr}) - 1 AS {profile.bin_name}"
        )
        assert isinstance(rel_with_bins, duckdb.DuckDBPyRelation), (
            f"{type(rel_with_bins)=}"
        )

        return nw.from_native(rel_with_bins).lazy()

    def add_to_pyspark(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            from pyspark.ml.feature import Bucketizer
            from pyspark.sql.functions import col
        except ImportError as e:
            raise ImportError(
                f"PySpark support requires pyspark to be installed. Original error: {e}"
            ) from e
        sdf = df.to_native()
        qs = sdf.approxQuantile(profile.x_name, (0.0, *probs), relativeError=0.01)
        if logger.isEnabledFor(logging.DEBUG):
            sample = sdf.sample(False, 0.02, seed=1).select(profile.x_name).toPandas()
            pd_qs = sample[profile.x_name].quantile((0.0, *probs)).to_list()
            logger.debug(
                "Pyspark vs pandas (sample) quantiles: %s", list(zip(qs, pd_qs))
            )
        if len(set(qs)) < len(qs):
            raise ValueError("Quantiles not unique. Decrease number of bins.")

        bucketizer = Bucketizer(
            splits=qs,
            inputCol=profile.x_name,
            outputCol=profile.bin_name,
            handleInvalid="keep",
        )

        sdf_binned = bucketizer.transform(sdf).withColumn(
            profile.bin_name, col(profile.bin_name).cast("int")
        )

        return nw.from_native(sdf_binned).lazy()

    logger.debug("Configuring quantile handler for %s", profile.implementation)
    if profile.implementation == Implementation.PANDAS:
        return add_to_pandas
    elif profile.implementation == Implementation.POLARS:
        return add_to_polars
    elif profile.implementation == Implementation.PYSPARK:
        return add_to_pyspark
    elif profile.implementation == Implementation.DUCKDB:
        return add_to_duckdb
    elif profile.implementation == Implementation.DASK:
        return add_to_dask
    else:
        return add_fallback


def _compute_quantiles(
    df: nw.DataFrame, colname: str, probs: Iterable[float], bin_name: str
) -> nw.LazyFrame:
    """Get multiple quantiles in one operation"""
    col = nw.col(colname)
    if df.implementation != nw.Implementation.PYSPARK:
        qs = df.select(
            [col.quantile(p, interpolation="linear").alias(f"q{p}") for p in probs]
        )
    else:
        # Pyspark - ugly hack
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise ImportError("PySpark support requires pyspark to be installed.")
        spark = SparkSession.builder.getOrCreate()

        quantiles: list[float] = (
            df.select(colname).to_native().approxQuantile(colname, probs, 0.03)
        )
        q_data = {}
        for p, q in zip(probs, quantiles):
            k = f"q{p}"
            q_data[k] = [q]
        qs_spark = spark.createDataFrame(q_data)
        qs = nw.from_native(qs_spark)

    return (
        qs.unpivot(variable_name="prob", value_name="quantile")
        .sort("quantile")
        .with_row_index(bin_name, order_by="quantile")
        .lazy()
    )


def _clean_controls(controls: Iterable[str] | str | None) -> Tuple[str, ...]:
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


def clean_df(
    df_in: IntoDataFrame, controls: Tuple[str, ...], x: str, y: str
) -> Tuple[nw.LazyFrame, bool, Tuple[str, ...], Tuple[str, ...]]:
    """Normalize the input dataframe and split controls by type.

    Returns a lazy narwhals frame containing only the requested columns, whether
    the original input was lazy, and tuples of numeric / categorical controls.
    """
    cols = getattr(df_in, "columns", None)
    if cols is None or len(cols) <= 1:
        msg = "Input dataframe must have 'columns' attribute and at least 2 cols"
        raise TypeError(msg)
    for c in [x, y, *controls]:
        if c not in cols:
            msg = f"{c} not in input dataframe"
            raise ValueError(msg)

    dfn: nw.DataFrame | nw.LazyFrame = nw.from_native(df_in)

    if type(dfn) is nw.DataFrame:
        is_lazy_input = False
    elif type(dfn) is nw.LazyFrame:
        is_lazy_input = True
    else:
        msg = f"Unexpected narwhals type {(type(dfn))}"
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

    df_filtered = _remove_bad_values(df, cols_numeric, cols_cat)

    numeric_controls = tuple(c for c in controls if c in cols_numeric)
    categorical_controls = tuple(c for c in controls if c in cols_cat)

    return df_filtered.lazy(), is_lazy_input, numeric_controls, categorical_controls


def add_bins(df, profile):
    quantile_handler = configure_quantile_handler(profile)
    try:
        df_with_bins = quantile_handler(df)
    except ValueError as err:
        err_text = str(err)
        if (
            "Quantiles are not unique" in err_text
            or "Bin edges must be unique" in err_text
        ):
            raise ValueError(
                "Quantiles are not unique. Decrease number of bins."
            ) from err
        raise

    return df_with_bins


def _select_rule_of_thumb_bins(
    df: nw.LazyFrame, x: str, y: str, regression_features: Tuple[str, ...]
) -> int:
    """Implement the SA-4.1 rule-of-thumb selector (currently for p=0, v=0)."""
    data_cols: Tuple[str, ...] = (x, *regression_features)
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

    beta_y = _solve_normal_equations(xtx, xty)
    beta_y_sq = _solve_normal_equations(xtx, xty_sq)

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

    sum_inv_density = _gaussian_inverse_density_sum(df, x, mean_x, std_x)

    slope = beta_y[1]
    shifted_legendre_weight = 1.0 / 3.0  # ∫_0^1 B_1(z)^2 dz
    bias_constant = (slope**2 / n_obs_f) * sum_inv_density * shifted_legendre_weight
    if bias_constant <= 0 or not math.isfinite(bias_constant):
        raise ValueError(
            "Rule-of-thumb selector estimated a non-positive bias constant; "
            "consider specifying num_bins explicitly."
        )

    sum_pred_y_sq = float(column_sums @ beta_y_sq)
    quad_form = float(beta_y.T @ xtx @ beta_y)
    avg_sigma_sq = (sum_pred_y_sq - quad_form) / n_obs_f
    avg_sigma_sq = max(avg_sigma_sq, 0.0)
    if avg_sigma_sq <= 0 or not math.isfinite(avg_sigma_sq):
        raise ValueError(
            "Rule-of-thumb selector estimated a non-positive variance constant; "
            "consider specifying num_bins explicitly."
        )

    prefactor = (2.0 * bias_constant) / avg_sigma_sq
    j_float = prefactor ** (1.0 / 3.0) * n_obs_f ** (1.0 / 3.0)
    max_bins = max(2, int(n_obs) - 1)
    computed_bins = max(2, int(round(j_float)))
    return min(max_bins, computed_bins)


def _collect_rule_of_thumb_stats(
    df: nw.LazyFrame, data_cols: Tuple[str, ...], y: str
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


def _gaussian_inverse_density_sum(
    df: nw.LazyFrame, x: str, mean_x: float, std_x: float
) -> float:
    """Compute sum_i 1 / f_G(x_i) for the Gaussian reference f_G."""
    if std_x <= 0:
        raise ValueError("Standard deviation must be positive.")
    z_sq = ((nw.col(x) - mean_x) / std_x) ** 2
    exp_expr = (0.5 * z_sq).exp()
    sum_exp = (
        df.select(exp_expr.sum().alias("__sum_exp")).collect().item(0, "__sum_exp")
    )
    return float(sum_exp) * math.sqrt(2.0 * math.pi) * std_x


def compute_bin_means(df: nw.LazyFrame, profile: Profile) -> nw.LazyFrame:
    df_plotting: nw.LazyFrame = (
        df.group_by(profile.bin_name)
        .agg(profile.x_col.mean(), profile.y_col.mean())
        .with_columns(nw.col(profile.bin_name).cast(nw.Int32))
    ).lazy()

    return df_plotting


def maybe_add_regression_features(
    df: nw.LazyFrame,
    numeric_controls: Tuple[str, ...],
    categorical_controls: Tuple[str, ...],
) -> Tuple[nw.LazyFrame, Tuple[str, ...]]:
    """Inject numeric controls and one-hot categorical controls when requested."""
    if not numeric_controls and not categorical_controls:
        return df, ()
    if numeric_controls and not categorical_controls:
        return df, numeric_controls

    dummy_exprs: list[nw.Expr] = []
    dummy_cols: list[str] = []
    for c in categorical_controls:
        distinct_values: List[Any] = (
            df.select(c).unique().collect().get_column(c).sort().to_list()
        )
        if len(distinct_values) <= 1:
            continue
        for i, v in enumerate(distinct_values[1:]):
            alias = f"__ctrl_{v}_{i}"
            expr = (nw.col(c) == v).cast(nw.Float64).alias(alias)
            dummy_exprs.append(expr)
            dummy_cols.append(alias)

    if not dummy_exprs:
        logger.debug("No dummy expressions created, all categorical controls constant")
        return df, tuple(numeric_controls)

    return df.with_columns(*dummy_exprs), numeric_controls + tuple(dummy_cols)


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
