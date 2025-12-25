# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "optuna",
#     "pandas",
#     "pyarrow",
#     "scikit-learn",
# ]
# ///

"""Optimize an ElasticNet regressor on the diabetes dataset via Optuna."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TRIALS_PATH = ARTIFACTS / "optuna_elasticnet_trials.parquet"
SUMMARY_PATH = ARTIFACTS / "optuna_elasticnet_summary.json"


def _load_dataset() -> tuple[np.ndarray, np.ndarray]:
    data = load_diabetes()
    return data.data, data.target


def _run_study(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    *,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=5000, random_state=seed)),
        ]
    )

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        pipeline.set_params(model__alpha=alpha, model__l1_ratio=l1_ratio)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds) ** 0.5
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def _record_trials(study: optuna.Study) -> None:
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {
            "trial": trial.number,
            "rmse": float(trial.value),
            "duration_seconds": trial.duration.total_seconds() if trial.duration else 0.0,
        }
        row.update(trial.params)
        rows.append(row)
    if not rows:
        raise RuntimeError("No completed trials to record.")
    pd.DataFrame(rows).to_parquet(TRIALS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize ElasticNet on the diabetes dataset via Optuna",
    )
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    X, y = _load_dataset()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    study = _run_study(
        X_train,
        X_valid,
        y_train,
        y_valid,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    _record_trials(study)

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trials_record_path": str(TRIALS_PATH),
            },
            indent=2,
        )
    )
    print(f"Wrote {TRIALS_PATH}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")


if __name__ == "__main__":
    main()
