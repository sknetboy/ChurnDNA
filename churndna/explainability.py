from __future__ import annotations

import pandas as pd
import shap


def shap_by_segment(model, X: pd.DataFrame, segment_col: str = "adoption_score") -> dict[str, pd.DataFrame]:
    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)

    low = X[segment_col] <= X[segment_col].quantile(0.33)
    high = X[segment_col] >= X[segment_col].quantile(0.66)
    segments = {
        "low_adoption": pd.DataFrame(shap_values.values[low], columns=X.columns),
        "high_adoption": pd.DataFrame(shap_values.values[high], columns=X.columns),
    }
    return {k: v.abs().mean().sort_values(ascending=False).to_frame("mean_abs_shap") for k, v in segments.items()}
