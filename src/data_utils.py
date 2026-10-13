"""Shared data-loading and validation helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
TARGET_CANDIDATES = ["Bankrupt?", "Bankrupt", "bankrupt", "target", "Target", "class", "Class"]
IDENTIFIER_CANDIDATES = ["Company_ID", "Company ID", "company_id", "Company", "Company_Name", "Company Name", "company_name", "ID", "id"]


def find_target_column(df: pd.DataFrame) -> str | None:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    # Handle whitespace around the usual UCI target name.
    for col in df.columns:
        if str(col).strip().lower() in {"bankrupt?", "bankrupt", "target", "class"}:
            return col
    return None


def read_csv_safely(file_or_path) -> pd.DataFrame:
    """Read a CSV with common encodings and trim accidental column whitespace."""
    try:
        df = pd.read_csv(file_or_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_feature_frame(df: pd.DataFrame, expected_columns: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    """Select expected feature columns and coerce them to numeric values."""
    expected = list(expected_columns)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        return pd.DataFrame(), missing
    features = df[expected].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    # Infinity is treated as missing so the user sees a useful validation message.
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, []


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
