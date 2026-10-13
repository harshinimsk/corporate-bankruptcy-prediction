"""Global and per-company explanation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def global_feature_importance(model, feature_names, limit=15) -> pd.DataFrame:
    """Get global feature importance from the fitted Random Forest pipeline."""
    estimator = model.named_steps.get("model", model)
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    values = estimator.feature_importances_
    return pd.DataFrame({"Feature": list(feature_names), "Importance": values}).sort_values(
        "Importance", ascending=False
    ).head(limit)


def shap_local_explanation(model, X: pd.DataFrame, row_index: int = 0, limit=10):
    """Return SHAP values for the positive (bankruptcy) class, if SHAP is installed."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Run `pip install shap`.") from exc

    estimator = model.named_steps.get("model", model)
    imputer = model.named_steps.get("imputer")
    X_ready = X.copy()
    if imputer is not None:
        X_ready = pd.DataFrame(imputer.transform(X_ready), columns=X.columns, index=X.index)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_ready.iloc[[row_index]])
    # SHAP versions return either a list by class or an ndarray with a class axis.
    if isinstance(values, list):
        positive_values = np.asarray(values[1])[0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            positive_values = values[0, :, 1] if values.shape[-1] == 2 else values[0, :, 0]
        elif values.ndim == 2:
            positive_values = values[0]
        else:
            raise ValueError("Unexpected SHAP output format.")
    result = pd.DataFrame({
        "Feature": X.columns,
        "SHAP_Value": positive_values,
        "Company_Value": X_ready.iloc[0].to_numpy(),
    })
    result["Direction"] = np.where(
        result["SHAP_Value"] > 0, "Pushes prediction toward bankruptcy", "Pushes prediction away from bankruptcy"
    )
    return result.reindex(result.SHAP_Value.abs().sort_values(ascending=False).index).head(limit)
