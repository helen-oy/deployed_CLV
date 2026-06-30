# Customer Lifetime Value and Retention Intelligence API

A production-ready customer retention intelligence service for CLV forecasting, churn risk assessment, RFM segmentation, and campaign ROI analysis.

## Features

- FastAPI-based business retention API
- LightGBM-powered CLV and churn prediction
- RFM analysis and customer segmentation
- Personalized recommendations and retention actions
- Campaign ROI and value-at-risk economics
- Artifact-only deployment support for FastAPI Cloud
- Modular pipeline code for training, serving, and evaluation
- Unit tests and deployment helpers

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an editable install:

```bash
pip install -e .
```

## Usage

Run the API locally:

```bash
fastapi dev src/api.py
```

Or with Uvicorn:

```bash
uvicorn src.api:app --reload
```

Run the training pipeline:

```python
from src.clv_pipeline import CLVPipeline

pipeline = CLVPipeline()
pipeline.run()
```

## Live Dashboard

View the deployed customer retention dashboard:

```text
https://deployedclv-pa6zwcq9vmfyyow3cagnfx.streamlit.app/
```

## Docker API

Run the API locally with Docker:

```bash
docker compose up --build -d
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## Business Retention API

The API exposes customer decisioning endpoints for executive CLV, churn risk, RFM health, and retention economics:

- `GET /business/summary`: portfolio CLV, churn risk, value-at-risk, and priority retention KPIs.
- `GET /business/decision-rules`: churn bands, CLV bands, priority-score logic, value-at-risk formula, and campaign ROI definitions.
- `GET /segments/summary`: high-level segment distribution and health summary.
- `GET /segments/rfm-summary`: current-state RFM segment health, including recency, frequency, monetary value, customer count, and customer share.
- `GET /customers/{customer_id}/profile`: customer-level RFM, CLV, churn risk, value-at-risk, priority score, and recommended actions.
- `GET /customers/top/{n}`: top priority customers with CLV, churn risk, value-at-risk, priority score, and recommendations.
- `POST /predict/clv`: saved LightGBM predictions for known customers.
- `POST /business/campaign-roi`: campaign economics using margin rate, offer cost, expected uplift, and either selected customers or top priority customers.

## Deployment Notes

For FastAPI Cloud, the API runs in artifact-only mode. It reads packaged CSV and JSON serving artifacts from `src/model_artifacts/`, with `models/` kept as a local fallback, and it does not import the training pipeline or LightGBM at startup. This avoids native runtime dependencies such as OpenMP `libgomp` that are needed only for live model inference and training.

The `POST /predict/clv` endpoint returns saved LightGBM predictions for customers present in `models/lightgbm_predictions.csv`. Live inference for new customers is disabled in the cloud API; regenerate the artifacts locally before redeploying when you need to refresh predictions.

LightGBM is trained with a snapshot-based target design:

- Features are built from customer behavior in the 90 days before `calibration_end`.
- `LGBMRegressor` predicts `clv_90d`, the total spend in the next 90 days.
- `LGBMClassifier` predicts `churn_90d`, no purchase in the next 90 days.
- LightGBM is evaluated with a time-based validation snapshot rather than a random split: training uses an older snapshot, validation uses `calibration_end`.
- Feature importance is exported to `models/lightgbm_feature_importance.csv`.
- Metrics are exported to `models/lightgbm_metrics.json`; if prior metrics exist, the old-vs-new comparison is written to `models/lightgbm_model_comparison.json`.
- The current feature set uses clearer behavioral variables such as `spend_last_90d`, `spend_last_30d`, order counts, basket size, product diversity, tenure, and trend ratios.
- Each new LightGBM challenger is logged to local MLflow under `mlruns/` and saved under `models/lightgbm_registry/runs/<model_version>/`.
- After local training, export refreshed serving artifacts before redeploying.

The cloud API accepts `model_type: "lightgbm"` only. The `baseline` and `both` options require the full local modeling environment and are intentionally not available in artifact-only deployment.

Example request:

```json
{
  "model_type": "lightgbm",
  "customers": [
    {
      "customer_id": "12347",
      "recency": 30,
      "frequency": 50,
      "monetary": 4000
    }
  ]
}
```

## Project Structure

- `src/`: source code modules, including the FastAPI app and training pipeline
- `src/model_artifacts/`: packaged serving artifacts for deployment
- `tests/`: unit tests
- `data/`: input data and schemas
- `models/`: saved model artifacts and exports
- `docs/`: documentation
- `notebooks/`: Jupyter notebooks for exploration
- `scripts/`: utility and runner scripts
