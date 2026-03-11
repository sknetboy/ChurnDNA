from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ChurnDNA API", version="0.1.0")
artifact = joblib.load("artifacts/churndna_model.joblib")
model = artifact["model"]
feature_columns = artifact["feature_columns"]


class PredictRequest(BaseModel):
    user_id: str
    features: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    row = {c: float(payload.features.get(c, 0.0)) for c in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)
    proba = float(model.predict_proba(X)[0, 1])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "calibrated_classifiers_"):
        importances = model.calibrated_classifiers_[0].estimator.feature_importances_
    else:
        importances = model.estimator.feature_importances_

    top_idx = importances.argsort()[::-1][:3]
    reasons = [feature_columns[i] for i in top_idx]
    return {"user_id": payload.user_id, "churn_probability": proba, "top_3_reasons": reasons}
