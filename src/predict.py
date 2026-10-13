"""CSV validation and prediction helpers used by the Streamlit app."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_utils import MODEL_DIR, clean_feature_frame, IDENTIFIER_CANDIDATES


def load_model_and_columns(model_name="random_forest"):
    model_path = MODEL_DIR / f"{model_name}.pkl"
    columns_path = MODEL_DIR / "feature_columns.json"
    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError("Model files are missing. Run `python -m src.train` first.")
    model = joblib.load(model_path)
    with columns_path.open("r", encoding="utf-8") as f:
        columns = json.load(f)
    return model, columns


def predict_dataframe(df: pd.DataFrame, model_name="random_forest"):
    model, feature_columns = load_model_and_columns(model_name)
    X, missing = clean_feature_frame(df, feature_columns)
    if missing:
        raise ValueError(
            "The uploaded CSV is missing required model feature columns. "
            f"Missing {len(missing)} column(s): " + ", ".join(missing[:12]) +
            (" ..." if len(missing) > 12 else "")
        )
    if X.empty:
        raise ValueError("The uploaded CSV has no rows to predict.")
    missing_cells = int(X.isna().sum().sum())
    if missing_cells:
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Found {missing_cells} blank or non-numeric feature value(s) in: "
            + ", ".join(bad_cols[:10])
            + ". Please correct these values and upload the CSV again."
        )
    probabilities = model.predict_proba(X)[:, 1]
    results = df.copy().reset_index(drop=True)
    results["Bankruptcy_Probability"] = probabilities
    results["Risk_Level"] = pd.cut(
        probabilities, bins=[-0.000001, 0.20, 0.50, 1.000001],
        labels=["Low", "Medium", "High"], include_lowest=True
    ).astype(str)
    results["Financial_Health_Score"] = np.rint((1 - probabilities) * 100).astype(int)
    results["Early_Warning"] = np.where(
        probabilities >= 0.50, "High risk: review financial indicators promptly",
        np.where(probabilities >= 0.20, "Watchlist: monitor financial indicators", "No immediate model alert")
    )
    # Keep identifiers as supplied by the user; they are not model features.
    if not any(c in results.columns for c in IDENTIFIER_CANDIDATES):
        results.insert(0, "Company_ID", [f"Company {i+1}" for i in range(len(results))])
    return results, X, model
