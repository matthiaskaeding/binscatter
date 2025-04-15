# Make some pictures for README.md
# %%
from binscatter import binscatter
from plotnine import ggplot, aes, geom_point
import numpy as np
from pathlib import Path
import pandas as pd

images_folder = Path(__file__).parents[1] / "images" / "readme"
images_folder.mkdir(exist_ok=True, parents=True)
assert images_folder.exists(), f"{images_folder} does not exist"
np.random.seed(0)


def generate_scatter_data(n_points=2000):
    x_values = np.random.uniform(low=-11.0, high=-1.0, size=n_points)

    intercept = 1.0
    slope1 = 0.1
    slope2 = -0.19
    noise_std_dev = 5.7

    noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=n_points)
    y_values = intercept + slope1 * x_values + slope2 * (x_values**2) + noise

    df = pd.DataFrame({"x": x_values, "y": y_values})
    return df


# %%
df = generate_scatter_data(100_000)
p_scatter = ggplot(df) + aes("x", "y") + geom_point()
p_scatter
# %%
p_binscatter = binscatter(df, "x", "y")
p_binscatter
# %%
p_scatter.save(images_folder / "scatter.png")
p_binscatter.save(images_folder / "binscatter.png")

# %%
