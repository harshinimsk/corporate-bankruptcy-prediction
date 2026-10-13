"""Train and save the bankruptcy prediction models.

Expected input: data/taiwanese_bankruptcy.csv, downloaded from the UCI
Taiwanese Bankruptcy Prediction dataset. The target column should be
"Bankrupt?" (1 = bankrupt, 0 = not bankrupt).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import DATA_DIR, MODEL_DIR, find_target_column, read_csv_safely, IDENTIFIER_CANDIDATES

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = DATA_DIR / "taiwanese_bankruptcy.csv"


def safe_metric(metric_fn, y_true, y_score):
    try:
        return float(metric_fn(y_true, y_score))
    except (ValueError, TypeError):
        return None


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Download the Taiwanese Bankruptcy Prediction CSV from UCI and save it as "
            "data/taiwanese_bankruptcy.csv. See README.md."
        )

    df = read_csv_safely(DATA_FILE)
    target_col = find_target_column(df)
    if target_col is None:
        raise ValueError(
            "Could not find the target column. Expected a column named 'Bankrupt?'. "
            f"Columns found: {list(df.columns[:8])} ..."
        )

    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    valid = y_raw.isin([0, 1])
    df = df.loc[valid].copy()
    y = y_raw.loc[valid].astype(int)
    excluded = {target_col}
    excluded.update(c for c in IDENTIFIER_CANDIDATES if c in df.columns)
    # The target is binary; only numeric columns are used as model features.
    X = df.drop(columns=list(excluded), errors="ignore").copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop columns that contain no usable numeric information.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] < 2:
        raise ValueError("Too few numeric feature columns were found in the dataset.")
    # Remove constant columns, which add no predictive information.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols)
    if y.nunique() != 2:
        raise ValueError("The target must contain both classes: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ]),
    }

    metrics = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": safe_metric(roc_auc_score, y_test, probabilities),
            "pr_auc": safe_metric(average_precision_score, y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
            "test_rows": int(len(y_test)),
            "threshold": 0.5,
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["random_forest"], MODEL_DIR / "random_forest.pkl")
    joblib.dump(fitted["logistic_regression"], MODEL_DIR / "logistic_regression.pkl")
    with (MODEL_DIR / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2, ensure_ascii=False)
    with (MODEL_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (MODEL_DIR / "training_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_file": DATA_FILE.name,
            "target_column": target_col,
            "rows_used": int(len(df)),
            "feature_count": int(X.shape[1]),
            "bankrupt_rows": int((y == 1).sum()),
            "non_bankrupt_rows": int((y == 0).sum()),
            "dropped_constant_columns": constant_cols,
            "random_state": 42,
            "test_size": 0.2,
        }, f, indent=2)

    print("Training complete.")
    print(f"Rows used: {len(df):,} | Features used: {X.shape[1]}")
    print("Saved models and metrics in the models/ folder.")
    for name, result in metrics.items():
        print(f"\n{name}")
        for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
            value = result[key]
            print(f"  {key}: {'N/A' if value is None else f'{value:.3f}'}")


if __name__ == "__main__":
    main()
