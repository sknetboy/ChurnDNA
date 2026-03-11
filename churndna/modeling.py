from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class TrainArtifacts:
    baseline_auc: float
    xgb_auc: float
    precision_at_10: float
    model_path: str


def precision_at_top_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.1) -> float:
    threshold_idx = int(len(y_proba) * (1 - k))
    cutoff = np.partition(y_proba, threshold_idx)[threshold_idx]
    y_pred = (y_proba >= cutoff).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def train_models(feature_df: pd.DataFrame, model_path: str = "artifacts/churndna_model.joblib") -> TrainArtifacts:
    train_df = feature_df.drop(columns=["user_id"]).copy()
    X = train_df.drop(columns=["churn_30d"])
    y = train_df["churn_30d"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    baseline = LogisticRegression(max_iter=500)
    baseline.fit(X_train_bal, y_train_bal)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "auc",
            "random_state": 42,
        }
        clf = XGBClassifier(**params)
        clf.fit(X_train_bal, y_train_bal)
        return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best = XGBClassifier(**study.best_params, eval_metric="auc", random_state=42)
    calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=3)
    calibrated.fit(X_train_bal, y_train_bal)

    y_proba = calibrated.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, y_proba)
    p_at_10 = precision_at_top_k(y_test.to_numpy(), y_proba, k=0.1)

    payload = {
        "model": calibrated,
        "feature_columns": list(X.columns),
        "metrics": {
            "baseline_auc": baseline_auc,
            "xgb_auc": xgb_auc,
            "precision_at_10": p_at_10,
        },
    }
    joblib.dump(payload, model_path)
    return TrainArtifacts(baseline_auc=baseline_auc, xgb_auc=xgb_auc, precision_at_10=p_at_10, model_path=model_path)
