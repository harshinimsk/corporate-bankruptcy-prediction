from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_utils import MODEL_DIR, read_csv_safely
from src.predict import load_model_and_columns, predict_dataframe
from src.explain import global_feature_importance, shap_local_explanation

ROOT_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Corporate Bankruptcy EWS", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: #f4f7fb; border: 1px solid #e2e8f0; padding: 14px; border-radius: 10px;}
    .small-note {color: #64748b; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("Corporate Bankruptcy Prediction & Early Warning System")
st.caption("A student machine-learning project for exploring financial risk from uploaded CSV data.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", [
        "Financial Risk Dashboard", "Company Risk Assessment", "Explainable AI", "Early Warning & Recommendations"
    ])
    st.divider()
    st.caption("Workflow: train the model once, then upload a CSV containing financial features.")

metrics_path = MODEL_DIR / "model_metrics.json"
columns_path = MODEL_DIR / "feature_columns.json"
model_ready = (MODEL_DIR / "random_forest.pkl").exists() and columns_path.exists()

if not model_ready:
    st.warning("The trained model is not available yet.")
    st.markdown("Run these commands in the VS Code terminal from the project folder:")
    st.code("pip install -r requirements.txt\npython -m src.train\nstreamlit run app.py", language="bash")
    st.info("Before training, save the UCI dataset CSV as `data/taiwanese_bankruptcy.csv`. See README.md for details.")
    st.stop()

with columns_path.open("r", encoding="utf-8") as f:
    feature_columns = json.load(f)

if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "input_df" not in st.session_state:
    st.session_state.input_df = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "random_forest"

# CSV-only workflow: there is no manual company-entry form.
st.sidebar.subheader("Upload company data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Each row should represent one company.")
model_choice = st.sidebar.selectbox("Prediction model", ["Random Forest", "Logistic Regression"], index=0)
model_name = "random_forest" if model_choice == "Random Forest" else "logistic_regression"

if uploaded_file is not None:
    try:
        input_df = read_csv_safely(uploaded_file)
        results_df, feature_df, _ = predict_dataframe(input_df, model_name=model_name)
        st.session_state.results_df = results_df
        st.session_state.input_df = input_df
        st.session_state.feature_df = feature_df
        st.session_state.model_name = model_name
        st.sidebar.success(f"Predictions ready for {len(results_df)} row(s).")
    except Exception as exc:
        st.session_state.results_df = None
        st.session_state.input_df = None
        st.sidebar.error(str(exc))

results = st.session_state.results_df

if page == "Financial Risk Dashboard":
    st.header("Financial Risk Dashboard")
    if results is None:
        st.info("Upload a company CSV using the sidebar to see predictions and dashboard summaries.")
        st.markdown("**CSV requirements**")
        st.write(f"- Include the **{len(feature_columns)} financial feature columns** used during training.")
        st.write("- Column names must match the training dataset exactly.")
        st.write("- Each row should represent one company; optional company ID/name columns are allowed.")
        st.write("- Remove blank or non-numeric values from the required financial feature columns.")
    else:
        avg_risk = results["Bankruptcy_Probability"].mean() * 100
        high_risk = int((results["Risk_Level"] == "High").sum())
        medium_risk = int((results["Risk_Level"] == "Medium").sum())
        low_risk = int((results["Risk_Level"] == "Low").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Companies assessed", f"{len(results):,}")
        c2.metric("Average predicted risk", f"{avg_risk:.1f}%")
        c3.metric("High-risk companies", high_risk)
        c4.metric("Average health score", f"{results['Financial_Health_Score'].mean():.0f}/100")
        st.subheader("Risk distribution")
        distribution = results["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        st.bar_chart(distribution)
        st.subheader("Company-level results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download prediction results (CSV)", results.to_csv(index=False).encode("utf-8-sig"),
                           file_name="bankruptcy_predictions.csv", mime="text/csv")
        st.caption("Risk bands are project thresholds: Low < 20%, Medium 20–49.9%, High ≥ 50%.")

elif page == "Company Risk Assessment":
    st.header("Company Risk Assessment")
    if results is None:
        st.info("Upload a CSV in the sidebar to assess one or more companies.")
    else:
        display_cols = [c for c in ["Company_ID", "Company ID", "Company_Name", "Company Name", "company_name", "Bankruptcy_Probability", "Risk_Level", "Financial_Health_Score", "Early_Warning"] if c in results.columns]
        st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
        options = list(range(len(results)))
        selected = st.selectbox("Select a company row", options, format_func=lambda i: str(
            next((results.iloc[i][c] for c in ["Company_ID", "Company ID", "Company_Name", "Company Name", "company_name"] if c in results.columns), f"Company {i+1}")
        ))
        row = results.iloc[selected]
        a, b, c = st.columns(3)
        a.metric("Bankruptcy probability", f"{row['Bankruptcy_Probability']:.1%}")
        b.metric("Risk level", str(row["Risk_Level"]))
        c.metric("Financial health score", f"{int(row['Financial_Health_Score'])}/100")
        st.progress(float(row["Bankruptcy_Probability"]), text="Predicted bankruptcy probability")
        st.info(str(row["Early_Warning"]))
        st.caption("The score is derived from the model probability for this project. It is not a formal credit rating or a financial guarantee.")

elif page == "Explainable AI":
    st.header("Explainable AI")
    st.write("This page shows which financial features are influential overall and, when SHAP is available, which features influence an individual prediction.")
    try:
        model, cols = load_model_and_columns("random_forest")
        importance = global_feature_importance(model, cols, limit=15)
        if not importance.empty:
            st.subheader("Top features in the Random Forest model")
            st.bar_chart(importance.set_index("Feature")["Importance"])
            st.dataframe(importance, use_container_width=True, hide_index=True)
        else:
            st.info("Global feature importance is available for the Random Forest model.")
        if results is not None:
            row_index = st.selectbox("Company row to explain", list(range(len(results))), format_func=lambda i: f"Company row {i+1}")
            if st.button("Generate SHAP explanation"):
                with st.spinner("Calculating SHAP values..."):
                    _, X, fitted_model = predict_dataframe(st.session_state.input_df, st.session_state.model_name)
                    # Random Forest TreeExplainer is used for this local explanation.
                    rf_model, _ = load_model_and_columns("random_forest")
                    explanation = shap_local_explanation(rf_model, X, row_index=row_index, limit=12)
                st.subheader("Features influencing this row")
                st.dataframe(explanation, use_container_width=True, hide_index=True)
                st.caption("Positive SHAP values push the model output toward the bankruptcy class; negative values push it away. This describes model behavior, not causation.")
    except Exception as exc:
        st.warning(f"Explanation could not be generated: {exc}")

elif page == "Early Warning & Recommendations":
    st.header("Early Warning & Recommendations")
    if results is None:
        st.info("Upload a CSV in the sidebar to generate risk alerts.")
    else:
        high = results[results["Risk_Level"] == "High"]
        medium = results[results["Risk_Level"] == "Medium"]
        low = results[results["Risk_Level"] == "Low"]
        c1, c2, c3 = st.columns(3)
        c1.metric("High-risk alerts", len(high))
        c2.metric("Watchlist", len(medium))
        c3.metric("Low-risk rows", len(low))
        st.subheader("Rows requiring attention")
        attention = results[results["Risk_Level"].isin(["High", "Medium"])].sort_values("Bankruptcy_Probability", ascending=False)
        if attention.empty:
            st.success("No rows crossed the project's medium- or high-risk thresholds.")
        else:
            st.dataframe(attention, use_container_width=True, hide_index=True)
        st.subheader("General review checklist")
        st.markdown("""
- **High risk:** review cash flow, debt obligations, profitability, liquidity, and recent financial statements with a qualified finance professional.
- **Medium risk:** monitor changes in the same indicators and compare with earlier reporting periods.
- **Low risk:** continue routine monitoring; a low model score does not mean that risk is absent.

These are general review prompts, not automated financial advice. The model cannot identify a cause with certainty or guarantee a future outcome.
""")
        st.download_button("Download early-warning list", attention.to_csv(index=False).encode("utf-8-sig"),
                           file_name="early_warning_list.csv", mime="text/csv")

st.divider()
st.caption("Academic prototype only. Predictions depend on the training dataset and model; do not use this app as the sole basis for real financial decisions.")
