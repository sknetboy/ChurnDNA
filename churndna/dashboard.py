from __future__ import annotations

import joblib
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="ChurnDNA", layout="wide")
st.title("ChurnDNA · Explorador de riesgo de abandono")

artifact = joblib.load("artifacts/churndna_model.joblib")
model = artifact["model"]
feature_columns = artifact["feature_columns"]
metrics = artifact["metrics"]

st.subheader("Métricas de entrenamiento")
col1, col2, col3 = st.columns(3)
col1.metric("Baseline AUC", f"{metrics['baseline_auc']:.3f}")
col2.metric("XGBoost AUC", f"{metrics['xgb_auc']:.3f}")
col3.metric("Precision@10%", f"{metrics['precision_at_10']:.3f}")

st.subheader("Perfil de usuario")
input_row = {}
for col in feature_columns:
    input_row[col] = st.number_input(col, value=0.0)

if st.button("Predecir churn"):
    X = pd.DataFrame([input_row], columns=feature_columns)
    proba = float(model.predict_proba(X)[0, 1])
    st.success(f"Probabilidad de churn: {proba:.2%}")

    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)
    st.subheader("SHAP waterfall")
    fig = shap.plots.waterfall(shap_values[0, :, 1], max_display=10, show=False)
    st.pyplot(fig.figure)
