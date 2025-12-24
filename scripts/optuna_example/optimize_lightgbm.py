# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lightgbm",
#     "optuna",
#     "polars",
#     "scikit-learn",
# ]
# ///

"""Optimize a LightGBM regressor on the binscatter dataset using Optuna."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import optuna
import polars as pl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "artifacts" / "state_data_processed.parquet"
TRIALS_PATH = ROOT / "artifacts" / "optuna_lightgbm_trials.parquet"
SUMMARY_PATH = ROOT / "artifacts" / "optuna_lightgbm_summary.json"
FEATURES = [
    "mtr90_lag3",
    "top_corp_lag3",
    "real_gdp_pc",
    "population_density",
    "rd_credit_lag3",
]
TARGET = "lnpat"


def _load_dataset() -> tuple[pl.DataFrame, pl.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. Run scripts/replicate_binscatter/prep_data.py first."
        )
    df = pl.read_parquet(DATA_PATH).select([*FEATURES, TARGET]).drop_nulls()
    return df.select(FEATURES), df.select(TARGET)


def _run_study(
    X_train,
    X_valid,
    y_train,
    y_valid,
    *,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "verbosity": -1,
            "force_col_wise": True,
        }
        booster = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        trial.set_user_attr("best_iteration", int(booster.best_iteration or 200))
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def _record_trials(study: optuna.Study) -> None:
    records: list[dict[str, float | int]] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        record: dict[str, float | int] = {
            "trial": trial.number,
            "rmse": float(trial.value),
            "duration_seconds": trial.duration.total_seconds()
            if trial.duration
            else 0.0,
        }
        for key, value in trial.params.items():
            record[key] = value
        for key, value in trial.user_attrs.items():
            record[key] = value
        records.append(record)
    if not records:
        raise RuntimeError("No completed trials found.")
    pl.DataFrame(records).write_parquet(TRIALS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize a LightGBM regressor via Optuna on the binscatter dataset."
    )
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = _load_dataset()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X.to_pandas(),
        y.to_pandas().squeeze(),
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
