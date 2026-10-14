"""Create a template CSV for checking the upload format after model training.

This template contains column names only and is NOT a real financial dataset.
Fill it with valid numeric company financial indicators before uploading.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
columns_file = ROOT / "models" / "feature_columns.json"
if not columns_file.exists():
    raise SystemExit("Run `python -m src.train` first so feature_columns.json is created.")
columns = json.loads(columns_file.read_text(encoding="utf-8"))
template = pd.DataFrame(columns=["Company_ID", *columns])
out = ROOT / "data" / "company_upload_template.csv"
template.to_csv(out, index=False)
print(f"Created empty upload template: {out}")
print("Fill one row per company with numeric values before uploading it to the app.")
