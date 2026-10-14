# Corporate Bankruptcy Prediction & Early Warning System

A student-level machine-learning project that predicts a company's bankruptcy risk from financial indicators in an uploaded CSV. The Streamlit dashboard reports a model probability, a simple financial-health presentation score, risk bands, global feature importance, and an optional SHAP explanation.

> **Important:** This is an academic prototype, not a credit-rating tool or professional financial advice. A prediction is not a guarantee. The commonly used Taiwanese Bankruptcy Prediction dataset reflects historical Taiwanese companies and should not be assumed to represent every country or current market.

## 1. Project structure

```text
Corporate-Bankruptcy-EWS/
├── app.py
├── requirements.txt
├── README.md
├── make_sample_upload.py
├── data/
│   └── (put the downloaded training CSV here)
├── models/
│   └── (trained model files are created here)
├── notebooks/
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── train.py
    ├── predict.py
    ├── explain.py
    └── health_score.py
```

## 2. Set it up in VS Code

1. Install Python 3.10 or 3.11 and VS Code.
2. Extract this ZIP and open the **Corporate-Bankruptcy-EWS** folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment (recommended):

   **Windows PowerShell**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 3. Get the training dataset

Download the **Taiwanese Bankruptcy Prediction** dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction

Save the downloaded CSV as:

`data/taiwanese_bankruptcy.csv`

The file must contain the target column **`Bankrupt?`**, where `1` means bankrupt and `0` means not bankrupt. If the downloaded file uses a different filename, rename it to the name above. If you receive an Excel file, export/save it as CSV first.

## 4. Train the models (one-time setup)

Run from the project root, the folder containing `app.py`:

```bash
python -m src.train
```

The training script prepares numeric financial features, splits data into training and test sets using stratification, trains Logistic Regression and Random Forest models, and saves model files and evaluation metrics in `models/`.

It prints precision, recall, F1, ROC-AUC, and PR-AUC. Because bankruptcy datasets are usually imbalanced, do not judge the model only by accuracy. The 0.5 decision threshold and the risk bands are simple project defaults, not professionally calibrated thresholds.

## 5. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open that URL in your browser.

## 6. Upload a company CSV

- The app uses **CSV upload only**; there is no manual financial-data entry form.
- Each row should represent one company.
- Required financial feature column names must match the columns used during training exactly.
- Include all required model feature columns. Optional company identifier/name columns such as `Company_ID` or `Company_Name` are displayed for convenience but are not used by the model.
- All required financial feature values must be numeric and non-empty.
- Do not include the target column `Bankrupt?` in a new-company upload. The app predicts that outcome.

After training, create a header-only template by running:

```bash
python make_sample_upload.py
```

This creates `data/company_upload_template.csv`. It contains the expected headers but **no real financial values**. Fill it with legitimate numeric values before uploading. Do not use random numbers and interpret the resulting predictions as meaningful company assessments.

## 7. Dashboard pages

1. **Financial Risk Dashboard** — risk distribution, average predicted risk, health-score summary, and downloadable predictions.
2. **Company Risk Assessment** — select a row and inspect its bankruptcy probability, risk band, and score.
3. **Explainable AI** — global Random Forest feature importance and an optional local SHAP explanation.
4. **Early Warning & Recommendations** — high/medium-risk rows and a general financial-review checklist.

## 8. Models and concepts used

- Logistic Regression: a baseline classification model.
- Random Forest: an ensemble of decision trees; used as the default model.
- Class weighting: helps the models account for the imbalanced target without generating synthetic rows.
- Stratified train/test split: preserves the approximate class ratio in both sets.
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix.
- SHAP and feature importance: help explain how the fitted model uses features; these explanations describe model behavior, not causation.

## 9. Troubleshooting

- **Model files missing:** run `python -m src.train` before `streamlit run app.py`.
- **Dataset not found:** check the exact path `data/taiwanese_bankruptcy.csv`.
- **Target column not found:** confirm the training CSV has a column named `Bankrupt?` (extra surrounding spaces are tolerated).
- **Missing columns during upload:** upload a CSV built from the same feature set used to train the saved model. Do not mix files from different datasets or versions.
- **Blank/non-numeric cells:** correct the CSV and upload it again. This app intentionally asks users to correct missing values instead of silently inventing financial data.
- **SHAP issue:** install/update dependencies with `pip install -r requirements.txt`. If SHAP fails on a specific Python version, use the other dashboard pages; predictions and global feature importance do not depend on SHAP.

## 10. Limitations and responsible use

The output is a model estimate, not proof that a company will or will not go bankrupt. The training data has a specific historical and geographic context, and a model can perform differently on companies from other regions or periods. The health score is simply `100 × (1 − predicted bankruptcy probability)`, rounded to an integer; it is not a validated financial-health index. Always combine model output with current financial statements and qualified human review.
