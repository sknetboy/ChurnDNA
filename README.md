# ChurnDNA

Sistema avanzado de predicción de abandono basado en señales de comportamiento digital.

## Componentes implementados

- **Generación sintética (Faker + SDV)**: 50k usuarios con 18 meses de historial y eventos `login`, `search`, `click`, `page_view`, `purchase`, `support_ticket`.
- **Pipeline de features**:
  - RFE (Recency-Frequency-Engagement) adaptado a comportamiento digital.
  - Embeddings Word2Vec a partir de secuencias de eventos por usuario.
  - Detección de fatiga digital en ventanas 7d/14d/30d.
  - Score de adopción de features.
- **Modelado**:
  - Baseline Logistic Regression.
  - Modelo principal XGBoost con tuning de hiperparámetros mediante Optuna.
  - Balanceo con SMOTE.
  - Calibración con `CalibratedClassifierCV`.
- **Explicabilidad**:
  - SHAP values por segmento (`low_adoption` / `high_adoption`).
- **Serving y visualización**:
  - API FastAPI `POST /predict`.
  - Dashboard Streamlit con explorador de riesgo y waterfall SHAP.

## Instalación

```bash
pip install -e .
```

## Entrenamiento

```bash
python train.py
```

## API

```bash
uvicorn churndna.api:app --reload --port 8000
```

`POST /predict` body:

```json
{
  "user_id": "U000001",
  "features": {
    "recency_days": 4,
    "frequency": 35
  }
}
```

## Dashboard

```bash
streamlit run churndna/dashboard.py
```

## Objetivo de métricas

Objetivo de negocio:
- AUC-ROC > 0.85
- Precision@10% > 0.70

> Nota: al usar datos sintéticos, las métricas pueden variar por seed y configuración de generación.
