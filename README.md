# CLV Prediction Project

A production-ready Customer Lifetime Value prediction system with churn propensity estimation and personalized recommendations.

## Features

- SQL database data ingestion
- Data preprocessing and feature engineering
- RFM analysis and customer segmentation
- CLV prediction using BG/NBD and Gamma-Gamma models
- Churn risk assessment
- Personalized recommendation engine
- Modular, testable code with OOP design
- Unit tests and CI/CD pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Usage

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

Run the API locally:

```bash
docker compose up --build -d
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## Business Decision API

The API exposes non-contractual customer decisioning endpoints for executive
CLV, inactivity risk, RFM health, and retention economics:

- `GET /business/summary`: portfolio CLV, churn risk, value at risk, and
  priority retention KPIs.
- `GET /business/decision-rules`: churn bands, CLV bands, priority-score logic,
  value-at-risk formula, and campaign ROI definitions.
- `GET /segments/rfm-summary`: current-state RFM segment health, including
  recency, frequency, monetary value, customer count, and customer share.
- `GET /customers/{customer_id}/profile`: customer-level RFM, CLV, churn risk,
  value at risk, priority score, and recommended actions.
- `GET /customers/top/{n}`: top priority customers with CLV, churn risk,
  value at risk, priority score, and recommendations.
- `POST /business/campaign-roi`: campaign economics using margin rate, offer
  cost, expected uplift, and either selected customers or top priority customers.

For FastAPI Cloud, the API runs in artifact-only mode. It reads packaged CSV
and JSON serving artifacts from `src/model_artifacts/`, with `models/` kept as
a local fallback, and does not import the training pipeline or LightGBM at
startup. This avoids native runtime dependencies such as OpenMP `libgomp` that
are needed only for live model inference/training.

The `POST /predict/clv` endpoint returns saved LightGBM predictions for
customers present in `models/lightgbm_predictions.csv`. Live inference for new
customers is disabled in the cloud API; regenerate the artifacts locally when
you need to refresh predictions.
LightGBM is trained with a snapshot-based target design:

- Features are built from customer behavior in the 90 days before `calibration_end`.
- `LGBMRegressor` predicts `clv_90d`, the total spend in the next 90 days.
- `LGBMClassifier` predicts `churn_90d`, no purchase in the next 90 days.
- LightGBM is evaluated with a time-based validation snapshot rather than a
  random split: training uses an older snapshot, validation uses `calibration_end`.
- Feature importance is exported to `models/lightgbm_feature_importance.csv`.
- Metrics are exported to `models/lightgbm_metrics.json`; if previous metrics
  exist, old-vs-new comparison is written to `models/lightgbm_model_comparison.json`.
- The current v2 feature set removes duplicate spend aliases such as
  `total_spend`, `monetary_value`, and `monetary_log`, keeping clearer features
  like `spend_last_90d`, `spend_last_30d`, order counts, basket size, product
  diversity, tenure, and trend ratios.
- Each new LightGBM challenger is logged to local MLflow under `mlruns/` and
  saved under `models/lightgbm_registry/runs/<model_version>/`.
- After local training, export refreshed serving artifacts before redeploying.

The cloud API accepts `model_type: "lightgbm"` only. The `baseline` and `both`
options require the full local modeling environment and are intentionally not
available in artifact-only deployment.

Example:

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

- `src/`: Source code modules
- `tests/`: Unit tests
- `data/`: Data files and schemas
- `models/`: Saved model artifacts
- `docs/`: Documentation
- `notebooks/`: Jupyter notebooks for exploration
- `scripts/`: Utility scripts
