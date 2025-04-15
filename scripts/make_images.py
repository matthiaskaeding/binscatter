# Make some pictures for README.md
from binscatter import binscatter
from plotnine import ggplot, aes, geom_point
import numpy as np
from pathlib import Path
import pandas as pd

images_folder = Path(__file__).parents[1] /  "images"
images_folder.mkdir(exist_ok=True)
assert images_folder.exists(), f"{images_folder} does not exist"


np.random.seed(0)


def generate_dataset(size=1000, seed=0):
    np.random.seed(seed)
    x = np.random.normal(loc=50, scale=10, size=size)
    y = 2 * x + 10 * np.sin(0.1 * x) + np.random.normal(loc=0, scale=20, size=size)
    category = np.random.choice(["A", "B", "C"], size=size)
    df = pd.DataFrame({"x": x, "y": y, "category": category})
    return df


def generate_log_dataset(size=1000, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(low=1, high=100, size=size)
    y = 5 * np.log(x) + np.random.normal(loc=0, scale=3, size=size)

    df = pd.DataFrame({"x": x, "y": y})
    return df
def generate_log_dense_dataset(size=100000, seed=0):
    np.random.seed(seed)
    x = np.random.lognormal(mean=2, sigma=1, size=size)
    x = np.clip(x, a_min=1e-3, a_max=None)  # avoid extreme small values
    log_x = np.log(x)
    scale = np.maximum(0.1, 0.5 + 0.2 * log_x)  # ensure positive scale
    noise = np.random.normal(loc=0, scale=scale, size=size)
    y = 0.5 * log_x + noise
    df = pd.DataFrame({'x': np.log(x), 'y': np.log(y)})
    return df

df = generate_log_dense_dataset(100_000)

p0 = ggplot(df) + aes("x", "y") + geom_point()

p0.save(images_folder / "scatter.png")
p_bin = binscatter(df, "x", "y")
p_bin.save(images_folder / "binscatter.png")
