from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Tuple
import numpy as np
import narwhals as nw
from narwhals.typing import IntoDataFrame
from binscatter.df_utils import (
    _remove_bad_values,
    _quantiles_from_sorted,
    _get_quantile_bins,
)
import plotly.express as px
import narwhals.selectors as ncs


def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    controls: Iterable[str] = [],
    num_bins=20,
    return_type: Literal["plotly", "native"] = "plotly",
):
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df (Union[polars.DataFrame, pandas.DataFrame]): Input dataframe
        x (str): Name of x column
        y (str): Name y column
        controls (Iterable[str]): names of control variables (not used yet)
        num_bins (int, optional): Number of bins to use. Defaults to 20
        return_type (str): Return type. Default a ggplot, otherwise "polars" for polars dataframe or "pandas" for pandas dataframe

    Returns:
        plotnine.ggplot: A ggplot object containing the binned scatter plot with x and y axis labels
    """
    if return_type not in ("plotly", "native"):
        raise ValueError("return_type must be either 'plotly', or 'native'")

    # Prepare dataframe: sort, remove non numerics and add bins
    df_prepped, config = prep(df, x, y, controls, num_bins)

    # Currently there are 2 cases:
    # (1) no controls: the easy one, just compute the means by bin
    # (2) controls: here we need to compute regression coefficients
    # and partial out the effect of the controls
    # (see section 2.2 in Cattaneo, Crump, Farrell and Feng (2024))
    if not controls:
        df_plotting = df_prepped.group_by(config.bin_name).agg(
            config.x_col.mean(), config.y_col.mean()
        )
    else:
        x_controls = make_controls_mat(df_prepped, controls)
        x_bins = make_b(df_prepped, config)
        df_plotting = partial_out_controls(x_bins, x_controls, df_prepped, config)

    match return_type:
        case "plotly":
            return make_plot_plotly(df_plotting, config)
        case "native":
            return df_plotting.to_native()


@dataclass
class Config:
    x_name: str
    y_name: str
    controls: Sequence[str]
    num_bins: int
    N: int
    x_col: nw.Expr
    y_col: nw.Expr
    bin_name: str
    x_min: float
    x_max: float


def prep(
    df: IntoDataFrame,
    x: str | None,
    y: str | None,
    controls: Iterable[str] = [],
    num_bins: int = 20,
) -> Tuple[nw.DataFrame, Config]:
    """Prepares the input data and builds configuration.

    Args:
        df: Input dataframe, either polars or pandas. Must have at least 2 columns.
            First column is treated as y variable, second as x variable.
        num_bins: Number of bins to use for binscatter. Must be less than number of rows.

    Returns:
        tuple: (polars.DataFrame, Config)
            - Sorted input dataframe converted to polars
            - Config object with metadata about the data

    Raises:
        AssertionError: If input validation fails
    """
    assert num_bins > 1
    assert isinstance(x, str)
    assert isinstance(y, str)

    df = nw.from_native(df)
    N = df.shape[0]
    assert df.shape[1] >= 2
    assert N > 0
    assert num_bins < N
    assert num_bins > 1
    assert df.shape[1] > 1
    cols = df.columns
    assert x in cols
    assert y in cols
    missing = [c for c in controls if c not in cols]
    assert not missing, f"Missing controls in df: {missing}"

    # Find name for bins
    bin_name = "bins____"
    for _ in range(100):
        if bin_name in cols:
            continue
        bin_name = bin_name + "_"
    if bin_name in cols:
        msg = f"'{bin_name}' and 99 versions with less underscores are columns in df"
        raise ValueError(msg)

    df_prepped = (
        df.select(x, y, *controls)
        .pipe(_remove_bad_values)
        .sort(x)
        .pipe(add_quantile_bins, x, bin_name, num_bins)
    )

    config = Config(
        num_bins=num_bins,
        x_name=x,
        y_name=y,
        x_col=nw.col(x),
        y_col=nw.col(y),
        N=df_prepped.shape[0],
        bin_name=bin_name,
        controls=controls,
        x_min=df_prepped.get_column(x).item(0),
        x_max=df_prepped.get_column(x).item(-1),
    )

    return df_prepped, config


def make_controls_mat(df: nw.DataFrame, controls: Iterable[str]) -> np.ndarray:
    """Creates a matrix of control variables by combining numeric and categorical features.

    Args:
        df (pl.DataFrame): Input dataframe containing control variables
        controls (Iterable[str]): Names of control variables to include

    Returns:
        numpy.ndarray: Matrix of control variables with numeric columns and dummy-encoded categorical columns
    """
    dfc = df.select(controls)
    df_num = dfc.select(ncs.numeric())
    df_cat = dfc.select(~ncs.categorical())
    if df_cat.shape[1] == 0:
        return df_num.to_numpy()

    sub_dfs = []
    for col in df_cat.columns:
        tmp = df_cat.get_column(col).to_dummies(drop_first=True).to_numpy()
        sub_dfs.append(tmp)
    x_cat = np.concat(sub_dfs, axis=1)
    if df_num.shape[1] == 0:
        return x_cat
    x_num = df_num.to_numpy()

    return np.concat((x_num, x_cat), axis=1)


def add_quantile_bins(df: nw.DataFrame, x_name: str, bin_name: str, num_bins: int):
    """Gets quantile bins

    Args:
        df (Union[polars.DataFrame, pandas.DataFrame]): Input dataframe with x and y variables
        config (Config): Configuration object with metadata about the data

    Returns:
        polars.DataFrame: Input dataframe with additional 'bin' column containing quantile bin assignments
    """

    probs = [i / num_bins for i in range(1, num_bins + 1)]
    x_quantiles: List[int] = _quantiles_from_sorted(df.get_column(x_name), probs)

    n_unique_quantiles = len(set(x_quantiles))
    n_duplicates = len(x_quantiles) - n_unique_quantiles
    if len(x_quantiles) != n_unique_quantiles:
        msg = f"""{n_duplicates} duplicate quantiles in variable "{x_name}", choose lower number of bins"""
        raise ValueError(msg)

    quantile_bins = _get_quantile_bins(df, x_name, x_quantiles)

    return df.with_columns(quantile_bins.alias(bin_name))


def make_b(df_prepped: nw.DataFrame, config: Config):
    """Makes the design matrix corresponding to the bins"""

    B = df_prepped.get_column(config.bin_name).to_dummies(
        drop_first=False, separator="_"
    )
    assert B.shape[1] == config.num_bins, (
        f"B must have {config.num_bins} columns but has {B.width}, this indicates too many bins"
    )
    # Reorder so that column i corresponds always to bin i
    cols = [f"{config.bin_name}_{i}" for i in range(config.num_bins)]

    return B.select(cols).to_numpy()


def partial_out_controls(
    x_bins: np.array, x_controls: np.ndarray, df_prepped: nw.DataFrame, config: Config
) -> nw.DataFrame:
    """Concatenate bin dummies and control variables horizontally."""
    x_conc = np.concatenate((x_bins, x_controls), axis=1)
    assert x_conc.shape[0] == config.N

    y = df_prepped.get_column(config.y_name).to_numpy()
    theta = np.linalg.lstsq(x_conc, y)[0]
    x_controls_means = np.mean(x_controls, axis=0)
    # Extract coefficients
    beta = theta[: config.num_bins]
    gamma = theta[config.num_bins :]
    # Evaluate
    y_vals = nw.new_series(
        name=config.y_name,
        values=beta + np.dot(x_controls_means, gamma),
        backend=df_prepped.implementation,
    )

    # Build output data frame - should be analog to the one build without controls
    df_plotting = (
        df_prepped.group_by(config.bin_name)
        .agg(
            config.x_col.mean(),
        )
        .sort(config.bin_name)
        .with_columns(y_vals)
    )

    assert df_plotting.shape[0] == config.num_bins
    assert df_plotting.shape[1] == 3

    return df_plotting


def make_plot_plotly(df: nw.DataFrame, config: Config):
    x = df.get_column(config.x_name).to_list()
    y = df.get_column(config.y_name).to_list()
    range_x = (config.x_min, config.x_max)

    fig = px.scatter(x=x, y=y, range_x=range_x).update_layout(
        title="Binscatter",
        xaxis_title=config.x_name,
        yaxis_title=config.y_name,
    )

    return fig
