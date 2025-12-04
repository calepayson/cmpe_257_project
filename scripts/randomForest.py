#!/usr/bin/env python
"""
RandomForest trainer.
Trains on normalized data, evaluates, and saves model + metrics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "normalized"
OUTPUT_MODELS = BASE_DIR / "models"
OUTPUT_ARTIFACTS = BASE_DIR / "artifacts"


def prepare_dirs() -> None:
    for folder in (OUTPUT_MODELS, OUTPUT_ARTIFACTS):
        folder.mkdir(parents=True, exist_ok=True)


def load_fold_data(window_size: int, fold_idx: int):
    """Load train/test/eval data for a given window and fold."""
    window_dir = DATA_DIR / f"window_{window_size}"
    fold_dir = window_dir / f"fold_{fold_idx}"

    X_train = pd.read_csv(fold_dir / "X_train.csv")
    y_train = pd.read_csv(fold_dir / "y_train.csv").squeeze()
    X_test = pd.read_csv(fold_dir / "X_test.csv")
    y_test = pd.read_csv(fold_dir / "y_test.csv").squeeze()
    X_eval = pd.read_csv(window_dir / "X_eval.csv")
    y_eval = pd.read_csv(window_dir / "y_eval.csv").squeeze()

    return X_train, y_train, X_test, y_test, X_eval, y_eval


def evaluate_model(model, X, y) -> Dict[str, float]:
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    return {"rmse": float(rmse), "r2": float(r2)}


def parse_hyperparameters(json_str: str) -> Dict:
    try:
        return json.loads(json_str) if json_str else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON parameters: {json_str}") from e


def save_metrics(metrics: Dict, model_name: str) -> None:
    path = OUTPUT_ARTIFACTS / f"metrics_{model_name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def train_random_forest(window: int, fold: int, params: Dict) -> None:
    prepare_dirs()
    X_train, y_train, X_test, y_test, X_eval, y_eval = load_fold_data(window, fold)

    logger.info("Starting RandomForest training for window=%d, fold=%d", window, fold)
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    test_scores = evaluate_model(rf, X_test, y_test)
    eval_scores = evaluate_model(rf, X_eval, y_eval)

    logger.info(
        "Test => RMSE: %.4f | R2: %.4f | Eval => RMSE: %.4f | R2: %.4f",
        test_scores["rmse"],
        test_scores["r2"],
        eval_scores["rmse"],
        eval_scores["r2"],
    )

    model_name = f"rf_window{window}_fold{fold}"
    joblib.dump(rf, OUTPUT_MODELS / f"{model_name}.pkl")

    metrics = {
        "window": window,
        "fold": fold,
        "model": "rf",
        "params": params,
        "test_rmse": test_scores["rmse"],
        "test_r2": test_scores["r2"],
        "eval_rmse": eval_scores["rmse"],
        "eval_r2": eval_scores["r2"],
    }
    save_metrics(metrics, model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RandomForest training (alternative style).")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--params",
        default="{}",
        help='RandomForest hyperparameters as JSON string, e.g. \'{"n_estimators":100}\'',
    )
    args = parser.parse_args()

    hyperparams = parse_hyperparameters(args.params)
    train_random_forest(args.window, args.fold, hyperparams)
