import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
from binscatter import binscatter
from plotly.subplots import make_subplots

images_folder = Path(__file__).parents[1] / "images" / "readme"
images_folder.mkdir(exist_ok=True, parents=True)
assert images_folder.exists(), f"{images_folder} does not exist"
np.random.seed(0)


def make_binscatter_data(
    n_points: int = 3000,
    x_range: tuple[float, float] = (-12.0, 12.0),
    trend: str = "quadratic",  # {"linear","quadratic","cubic","sine","piecewise","none"}
    trend_params: dict | None = None,
    heteroskedastic: bool = True,
    noise_base: float = 2.0,
    noise_x_slope: float = 0.25,  # how much noise grows with |x|
    n_groups: int = 0,  # 0 means no groups; else creates 1..n_groups (ints)
    group_effect_strength: float = 8.0,  # vertical separation between groups
    group_slope_jitter: float = 0.12,  # per-group slope tweak
    outlier_frac: float = 0.015,  # fraction of points to boost as outliers
    outlier_scale: float = 5.0,  # outlier noise multiplier
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate visually interesting scatter data for binscatter demos.

    Returns a DataFrame with columns:
      - x, y
      - group (int, optional if n_groups>0)
      - is_outlier (bool)

    Parameters let you dial in: nonlinear trend type, heteroskedasticity,
    subgroup structure, and rare outliers to make the plot less uniform.
    """
    rng = np.random.default_rng(seed)

    # X support
    x_low, x_high = x_range
    x = rng.uniform(x_low, x_high, n_points)

    # Build underlying trend f(x)
    trend_params = trend_params or {}
    if trend == "linear":
        b0 = trend_params.get("intercept", 0.0)
        b1 = trend_params.get("slope", 0.6)
        f = b0 + b1 * x
    elif trend == "quadratic":
        b0 = trend_params.get("intercept", 1.0)
        b1 = trend_params.get("slope1", 0.3)
        b2 = trend_params.get("slope2", -0.06)
        f = b0 + b1 * x + b2 * x**2
    elif trend == "cubic":
        b0 = trend_params.get("intercept", 0.0)
        b1 = trend_params.get("slope1", 0.2)
        b2 = trend_params.get("slope2", -0.03)
        b3 = trend_params.get("slope3", 0.002)
        f = b0 + b1 * x + b2 * x**2 + b3 * x**3
    elif trend == "sine":
        amp = trend_params.get("amp", 8.0)
        freq = trend_params.get("freq", 0.35)
        phase = trend_params.get("phase", 0.0)
        base = trend_params.get("base", 0.0)
        f = base + amp * np.sin(freq * x + phase)
    elif trend == "piecewise":
        # two-piece with a kink; good for binscatter illustrations
        kink = trend_params.get("kink", 0.0)
        left_slope = trend_params.get("left_slope", 0.2)
        right_slope = trend_params.get("right_slope", -0.4)
        base = trend_params.get("base", 1.0)
        f = np.where(
            x <= kink, base + left_slope * (x - kink), base + right_slope * (x - kink)
        )
    elif trend == "none":
        f = np.zeros_like(x)
    else:
        raise ValueError(f"Unknown trend: {trend}")

    # Optional group structure
    if n_groups and n_groups > 0:
        groups = rng.integers(1, n_groups + 1, n_points)
        # vertical offsets + slight slope jitter per group
        offsets = (groups - 1) * group_effect_strength
        slope_shifts = (groups - 1) * group_slope_jitter
        f_grouped = f + offsets + slope_shifts * x
    else:
        groups = None
        f_grouped = f

    # Noise: heteroskedastic by default so bins stand out
    if heteroskedastic:
        sigma = noise_base + noise_x_slope * np.abs(x)
    else:
        sigma = np.full_like(x, noise_base)
    eps = rng.normal(0.0, sigma)

    y = f_grouped + eps

    # Sprinkle some outliers
    n_out = int(outlier_frac * n_points)
    is_outlier = np.zeros(n_points, dtype=bool)
    if n_out > 0:
        idx = rng.choice(n_points, size=n_out, replace=False)
        is_outlier[idx] = True
        # explode noise for outliers
        y[idx] += rng.normal(0.0, outlier_scale * sigma[idx])

    df = pd.DataFrame({"x": x, "y": y})
    if groups is not None:
        df["group"] = groups
    df["is_outlier"] = is_outlier
    return df


# %%
df = make_binscatter_data(100_00)
p_scatter = px.scatter(df, "x", "y")
p_binscatter = binscatter(df, "x", "y")
combined = make_subplots(rows=1, cols=2, subplot_titles=("Scatter", "Binscatter"))

# Add traces from fig1
for trace in p_scatter.data:
    combined.add_trace(trace, row=1, col=1)

# Add traces from fig2
for trace in p_binscatter.data:
    combined.add_trace(trace, row=1, col=2)

combined.write_image(
    images_folder / "combined.png",
    width=1200,
    height=600,  # pixels
    scale=2,
)
