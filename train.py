from __future__ import annotations

from churndna.data_generation import GenerationConfig, build_churn_label, generate_synthetic_events
from churndna.features import build_feature_table
from churndna.modeling import train_models


def main() -> None:
    config = GenerationConfig(n_users=50_000, months_history=18)
    events = generate_synthetic_events(config)
    labels = build_churn_label(events, horizon_days=30)
    feature_df = build_feature_table(events, labels)
    artifacts = train_models(feature_df)

    print("=== ChurnDNA Entrenamiento ===")
    print(f"Modelo guardado en: {artifacts.model_path}")
    print(f"Baseline AUC: {artifacts.baseline_auc:.3f}")
    print(f"XGBoost AUC: {artifacts.xgb_auc:.3f}")
    print(f"Precision@10%: {artifacts.precision_at_10:.3f}")


if __name__ == "__main__":
    main()
