"""
FastAPI application for CLV prediction and recommendation service.
"""

import ast
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value prediction and recommendation service",
    version="1.0.0"
)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RFM_SEGMENT_SUMMARY_PATH = MODELS_DIR / "current_state_segment_summary.csv"
LIGHTGBM_PREDICTIONS_PATH = MODELS_DIR / "lightgbm_predictions.csv"
CUSTOMER_RECOMMENDATIONS_PATH = MODELS_DIR / "customer_recommendations.csv"
CUSTOMER_SEGMENTS_PATH = MODELS_DIR / "customer_segments.csv"
LIGHTGBM_FEATURE_IMPORTANCE_PATH = MODELS_DIR / "lightgbm_feature_importance.csv"
LIGHTGBM_METRICS_PATH = MODELS_DIR / "lightgbm_metrics.json"

LOW_CHURN_MAX = 0.20
HIGH_CHURN_MIN = 0.50
DEFAULT_MARGIN_RATE = 0.40
DEFAULT_OFFER_COST = 5.00
DEFAULT_EXPECTED_UPLIFT = 0.10
FEATURE_WINDOW_DAYS = 90
TARGET_WINDOW_DAYS = 90

RECOMMENDATION_MATRIX = {
    ('Champions', 'Low'): [
        "VIP loyalty rewards program",
        "Exclusive early access to new products",
        "Personalized thank you communications",
        "Premium customer support"
    ],
    ('Champions', 'Moderate'): [
        "Upsell premium products",
        "Cross-sell complementary items",
        "Referral program incentives",
        "Exclusive member events"
    ],
    ('Champions', 'High'): [
        "Immediate retention campaign",
        "Personal concierge service",
        "Custom product bundles",
        "Loyalty program reactivation"
    ],
    ('Loyal', 'Low'): [
        "Loyalty program rewards",
        "Personalized product recommendations",
        "Birthday/anniversary specials",
        "Exclusive member discounts"
    ],
    ('Loyal', 'Moderate'): [
        "Upsell opportunities",
        "Cross-sell recommendations",
        "Engagement campaigns",
        "Loyalty tier upgrades"
    ],
    ('Loyal', 'High'): [
        "Retention incentives",
        "Personal outreach",
        "Custom retention packages",
        "Loyalty program reactivation"
    ],
    ('Active', 'Low'): [
        "Re-engagement campaigns",
        "Personalized offers",
        "Product discovery recommendations",
        "Loyalty program enrollment"
    ],
    ('Active', 'Moderate'): [
        "Upsell campaigns",
        "Cross-sell opportunities",
        "Engagement incentives",
        "Personalized communications"
    ],
    ('Active', 'High'): [
        "Win-back campaigns",
        "Discounted retention offers",
        "Personal outreach",
        "Re-engagement incentives"
    ],
    ('Occasional', 'Low'): [
        "Re-engagement emails",
        "Special promotional offers",
        "Product recommendations",
        "Simplified purchase process"
    ],
    ('Occasional', 'Moderate'): [
        "Promotional campaigns",
        "Discount incentives",
        "Cross-sell opportunities",
        "Engagement building"
    ],
    ('Occasional', 'High'): [
        "Reactivation campaigns",
        "Deep discount offers",
        "Personal outreach",
        "Simplified re-engagement"
    ],
    ('Low', 'Low'): [
        "Basic re-engagement",
        "Generic promotional offers",
        "Product awareness campaigns",
        "Low-cost incentives"
    ],
    ('Low', 'Moderate'): [
        "Targeted promotional offers",
        "Discount campaigns",
        "Product recommendations",
        "Re-engagement incentives"
    ],
    ('Low', 'High'): [
        "Aggressive reactivation",
        "Deep discount promotions",
        "Personal outreach",
        "High-touch re-engagement"
    ],
}


class CustomerData(BaseModel):
    """Customer features for CLV prediction."""
    customer_id: str
    recency: int
    frequency: int
    monetary: float
    customer_age: Optional[float] = None
    monetary_value: Optional[float] = None


class PredictionRequest(BaseModel):
    """Request for CLV predictions."""
    customers: List[CustomerData]
    model_type: Literal['lightgbm', 'baseline', 'both'] = 'lightgbm'


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    customer_id: str
    clv_segment: str
    churn_risk: str


class CampaignEconomicsRequest(BaseModel):
    """Inputs for campaign-level retention economics."""
    customer_ids: Optional[List[str]] = None
    top_n: Optional[int] = Field(default=100, ge=1, le=10000)
    margin_rate: float = Field(default=DEFAULT_MARGIN_RATE, ge=0, le=1)
    offer_cost_per_customer: float = Field(default=DEFAULT_OFFER_COST, ge=0)
    expected_uplift: float = Field(default=DEFAULT_EXPECTED_UPLIFT, ge=0, le=1)


@app.on_event("startup")
async def startup_event():
    """Validate artifact availability at startup without importing LightGBM."""
    required_artifacts = [
        RFM_SEGMENT_SUMMARY_PATH,
        LIGHTGBM_PREDICTIONS_PATH,
        CUSTOMER_RECOMMENDATIONS_PATH,
        CUSTOMER_SEGMENTS_PATH,
        LIGHTGBM_FEATURE_IMPORTANCE_PATH,
        LIGHTGBM_METRICS_PATH,
    ]
    missing = [str(path.relative_to(BASE_DIR)) for path in required_artifacts if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required deployment artifacts: {missing}")
    logger.info("Artifact-only API startup complete")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(float(value))


def _churn_risk_band(churn_probability: float) -> str:
    if churn_probability <= LOW_CHURN_MAX:
        return "Low"
    if churn_probability <= HIGH_CHURN_MIN:
        return "Moderate"
    return "High"


def _value_at_risk(clv: float, churn_probability: float) -> float:
    return max(_safe_float(clv), 0.0) * max(_safe_float(churn_probability), 0.0)


def _campaign_economics(value_at_risk: float, customer_count: int,
                        margin_rate: float, offer_cost_per_customer: float,
                        expected_uplift: float) -> Dict[str, float]:
    expected_retained_revenue = value_at_risk * expected_uplift
    expected_gross_margin = expected_retained_revenue * margin_rate
    campaign_cost = customer_count * offer_cost_per_customer
    expected_net_value = expected_gross_margin - campaign_cost
    roi = None if campaign_cost == 0 else expected_net_value / campaign_cost
    return {
        "expected_retained_revenue": float(expected_retained_revenue),
        "expected_gross_margin": float(expected_gross_margin),
        "campaign_cost": float(campaign_cost),
        "expected_net_value": float(expected_net_value),
        "roi": None if roi is None else float(roi),
    }


def _parse_recommendations(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    return parsed if isinstance(parsed, list) else [str(parsed)]


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.replace({pd.NA: None}).where(pd.notna(df), None).to_dict("records")


def _lightgbm_predictions_df() -> pd.DataFrame:
    if not LIGHTGBM_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="LightGBM predictions are not available"
        )
    df = pd.read_csv(LIGHTGBM_PREDICTIONS_PATH)
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["value_at_risk_90d"] = (
        df["lightgbm_clv_90d"] * df["lightgbm_churn_90d_probability"]
    )
    df["churn_risk_band"] = df["lightgbm_churn_90d_probability"].apply(_churn_risk_band)
    return df


def _recommendations_df() -> pd.DataFrame:
    if not CUSTOMER_RECOMMENDATIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Customer recommendations are not available"
        )
    df = pd.read_csv(CUSTOMER_RECOMMENDATIONS_PATH)
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["value_at_risk"] = df["CLV"] * df["churn_probability"]
    df["Recommendations"] = df["Recommendations"].apply(_parse_recommendations)
    return df


def _rfm_summary_df() -> pd.DataFrame:
    if RFM_SEGMENT_SUMMARY_PATH.exists():
        df = pd.read_csv(RFM_SEGMENT_SUMMARY_PATH)
    else:
        raise HTTPException(
            status_code=500,
            detail="RFM segment summary is not available"
        )
    total_customers = df["Customer Count"].sum()
    df["Customer Share"] = df["Customer Count"] / total_customers
    return df


def _customer_segments_df() -> pd.DataFrame:
    if not CUSTOMER_SEGMENTS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Customer RFM segments are not available"
        )
    df = pd.read_csv(CUSTOMER_SEGMENTS_PATH)
    df["CustomerID"] = df["CustomerID"].astype(str)
    return df


def _recommendations_for_segment(clv_segment: str, churn_risk: str) -> List[str]:
    return RECOMMENDATION_MATRIX.get(
        (clv_segment, churn_risk),
        [
            "General re-engagement campaign",
            "Product awareness communications",
            "Basic promotional offers",
        ],
    )


def _segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(['CLV_Segment', 'Churn_Risk']).agg({
        'CustomerID': 'count',
        'CLV': ['mean', 'median', 'sum'],
        'churn_probability': 'mean'
    }).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.rename(columns={
        'CustomerID_count': 'customer_count',
        'CLV_mean': 'avg_clv',
        'CLV_median': 'median_clv',
        'CLV_sum': 'total_clv',
        'churn_probability_mean': 'avg_churn_risk'
    })


def _lightgbm_metrics() -> Dict[str, Any]:
    if not LIGHTGBM_METRICS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="LightGBM metrics are not available"
        )
    with LIGHTGBM_METRICS_PATH.open() as f:
        return json.load(f)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CLV Prediction API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/business/decision-rules")
async def get_decision_rules():
    """Explain the decision rules used for non-contractual customer management."""
    return {
        "business_context": (
            "Non-contractual churn is inferred as purchase inactivity rather "
            "than a known cancellation event."
        ),
        "target_window_days": TARGET_WINDOW_DAYS,
        "feature_window_days": FEATURE_WINDOW_DAYS,
        "churn_risk_bands": {
            "Low": f"churn_probability <= {LOW_CHURN_MAX}",
            "Moderate": f"{LOW_CHURN_MAX} < churn_probability <= {HIGH_CHURN_MIN}",
            "High": f"churn_probability > {HIGH_CHURN_MIN}",
        },
        "clv_banding": {
            "method": "Customer CLV quintiles",
            "bands": ["Low", "Occasional", "Active", "Loyal", "Champions"],
        },
        "priority_score": {
            "formula": "(CLV percentile rank + churn probability percentile rank) / 2",
            "interpretation": (
                "Higher scores identify customers with stronger value potential "
                "and greater retention urgency."
            ),
        },
        "value_at_risk": {
            "formula": "predicted_clv * churn_probability",
            "interpretation": (
                "Expected future value exposed to inactivity risk over the prediction window."
            ),
        },
        "campaign_roi": {
            "expected_retained_revenue": "value_at_risk * expected_uplift",
            "expected_gross_margin": "expected_retained_revenue * margin_rate",
            "expected_net_value": "expected_gross_margin - campaign_cost",
            "roi": "expected_net_value / campaign_cost",
        },
    }


@app.get("/business/summary")
async def get_business_summary():
    """Return executive KPIs for CLV, churn exposure, RFM health, and retention focus."""
    try:
        predictions = _lightgbm_predictions_df()
        rfm_summary = _rfm_summary_df()
        recommendations = _recommendations_df()

        largest_segment = rfm_summary.loc[rfm_summary["Customer Count"].idxmax()]
        highest_value_segment = rfm_summary.loc[rfm_summary["Monetary"].idxmax()]
        highest_frequency_segment = rfm_summary.loc[rfm_summary["Frequency"].idxmax()]
        high_risk = predictions[
            predictions["lightgbm_churn_90d_probability"] > HIGH_CHURN_MIN
        ]
        top_priority = recommendations.head(100)

        return {
            "total_customers": _safe_int(rfm_summary["Customer Count"].sum()),
            "prediction_customer_count": len(predictions),
            "total_predicted_clv_90d": _safe_float(
                predictions["lightgbm_clv_90d"].sum()
            ),
            "average_predicted_clv_90d": _safe_float(
                predictions["lightgbm_clv_90d"].mean()
            ),
            "average_churn_90d_probability": _safe_float(
                predictions["lightgbm_churn_90d_probability"].mean()
            ),
            "high_churn_customer_count": len(high_risk),
            "high_churn_customer_share": _safe_float(len(high_risk) / len(predictions)),
            "total_value_at_risk_90d": _safe_float(predictions["value_at_risk_90d"].sum()),
            "largest_segment": {
                "segment": largest_segment["customer_segment"],
                "customer_count": _safe_int(largest_segment["Customer Count"]),
            },
            "highest_value_segment": {
                "segment": highest_value_segment["customer_segment"],
                "average_monetary_value": _safe_float(highest_value_segment["Monetary"]),
            },
            "highest_frequency_segment": {
                "segment": highest_frequency_segment["customer_segment"],
                "average_frequency": _safe_float(highest_frequency_segment["Frequency"]),
            },
            "priority_retention": {
                "customer_count": len(top_priority),
                "total_clv": _safe_float(top_priority["CLV"].sum()),
                "average_clv": _safe_float(top_priority["CLV"].mean()),
                "total_value_at_risk": _safe_float(top_priority["value_at_risk"].sum()),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Business summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/clv")
async def predict_clv(request: PredictionRequest):
    """
    Return saved CLV/churn predictions for customers in the artifact snapshot.

    Live LightGBM inference is intentionally disabled in the FastAPI Cloud
    artifact-only deployment because the managed runtime does not provide the
    native OpenMP library required by LightGBM.
    """
    try:
        if request.model_type in ('baseline', 'both'):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Baseline/live model inference is not available in the "
                    "artifact-only cloud deployment."
                )
            )

        saved_predictions = _lightgbm_predictions_df().set_index("CustomerID")
        predictions = []
        missing_customer_ids = []
        for customer in request.customers:
            customer_id = str(customer.customer_id)
            if customer_id not in saved_predictions.index:
                missing_customer_ids.append(customer_id)
                continue

            row = saved_predictions.loc[customer_id]
            churn_probability = _safe_float(row['lightgbm_churn_90d_probability'])
            clv_90d = _safe_float(row['lightgbm_clv_90d'])
            predictions.append({
                "customer_id": customer_id,
                "lightgbm": {
                    "predicted_clv_90d": clv_90d,
                    "churn_90d_probability": churn_probability,
                    "churn_risk_band": _churn_risk_band(churn_probability),
                    "value_at_risk_90d": _value_at_risk(clv_90d, churn_probability),
                    "source": "saved_artifact_snapshot",
                }
            })

        if missing_customer_ids:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": (
                        "Saved prediction artifacts do not contain every requested "
                        "customer. Live inference for unseen customers is disabled "
                        "in this artifact-only deployment."
                    ),
                    "missing_customer_ids": missing_customer_ids,
                },
            )

        return {"predictions": predictions}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CLV prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a customer."""
    try:
        return {
            "customer_id": request.customer_id,
            "clv_segment": request.clv_segment,
            "churn_risk": request.churn_risk,
            "recommendations": _recommendations_for_segment(
                request.clv_segment,
                request.churn_risk,
            )
        }

    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments/summary")
async def get_segment_summary():
    """Get summary of customer segments."""
    try:
        recommendations = _recommendations_df()
        summary = _segment_summary(recommendations)
        return summary.reset_index().to_dict('records')

    except Exception as e:
        logger.error(f"Segment summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments/rfm-summary")
async def get_rfm_segment_summary():
    """Get current-state RFM segment health for non-contractual customers."""
    try:
        summary = _rfm_summary_df()
        summary = summary.rename(columns={
            "customer_segment": "segment",
            "Customer Count": "customer_count",
            "Customer Share": "customer_share",
            "Recency": "average_recency_days",
            "Frequency": "average_frequency",
            "Monetary": "average_monetary_value",
        })
        return _records(summary)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RFM segment summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/top/{n}")
async def get_top_customers(n: int = 10):
    """Get top N priority customers."""
    try:
        recommendations = _recommendations_df().head(n).copy()
        recommendations["churn_risk_band"] = recommendations["churn_probability"].apply(
            _churn_risk_band
        )
        fields = [
            "CustomerID",
            "CLV",
            "churn_probability",
            "churn_risk_band",
            "CLV_Segment",
            "Churn_Risk",
            "value_at_risk",
            "Priority_Score",
            "Recommendations",
        ]
        return _records(recommendations[fields])

    except Exception as e:
        logger.error(f"Top customers query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/{customer_id}/profile")
async def get_customer_profile(customer_id: str):
    """Get a full customer profile for CLV, inactivity risk, RFM, and action planning."""
    try:
        customer_id = str(customer_id)
        recommendations = _recommendations_df()
        predictions = _lightgbm_predictions_df()
        customer_segments = _customer_segments_df()

        recommendation_match = recommendations[
            recommendations["CustomerID"] == customer_id
        ]
        prediction_match = predictions[predictions["CustomerID"] == customer_id]
        segment_match = customer_segments[customer_segments["CustomerID"] == customer_id]

        if (
            recommendation_match.empty and
            prediction_match.empty and
            segment_match.empty
        ):
            raise HTTPException(status_code=404, detail="Customer not found")

        profile: Dict[str, Any] = {"customer_id": customer_id}

        if not segment_match.empty:
            rfm = segment_match.iloc[0]
            profile["rfm"] = {
                "recency_days": _safe_float(rfm.get("Recency")),
                "frequency": _safe_float(rfm.get("Frequency")),
                "monetary_value": _safe_float(rfm.get("Monetary")),
                "cluster": None if pd.isna(rfm.get("Cluster")) else rfm.get("Cluster"),
            }

        if not prediction_match.empty:
            prediction = prediction_match.iloc[0]
            profile["lightgbm"] = {
                "predicted_clv_90d": _safe_float(prediction["lightgbm_clv_90d"]),
                "churn_90d_probability": _safe_float(
                    prediction["lightgbm_churn_90d_probability"]
                ),
                "churn_risk_band": prediction["churn_risk_band"],
                "value_at_risk_90d": _safe_float(prediction["value_at_risk_90d"]),
            }

        if not recommendation_match.empty:
            recommendation = recommendation_match.iloc[0]
            profile["decisioning"] = {
                "clv_segment": recommendation["CLV_Segment"],
                "churn_risk": recommendation["Churn_Risk"],
                "priority_score": _safe_float(recommendation["Priority_Score"]),
                "value_at_risk": _safe_float(recommendation["value_at_risk"]),
                "recommendations": recommendation["Recommendations"],
            }

        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Customer profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business/campaign-roi")
async def estimate_campaign_roi(request: CampaignEconomicsRequest):
    """
    Estimate retention campaign economics from CLV, churn risk, and expected uplift.

    If customer_ids is omitted, the endpoint evaluates the top N priority customers.
    """
    try:
        recommendations = _recommendations_df().copy()
        if request.customer_ids:
            requested_ids = {str(customer_id) for customer_id in request.customer_ids}
            recommendations = recommendations[
                recommendations["CustomerID"].isin(requested_ids)
            ]
        else:
            recommendations = recommendations.head(request.top_n)

        if recommendations.empty:
            raise HTTPException(
                status_code=404,
                detail="No customers found for campaign economics"
            )

        total_value_at_risk = _safe_float(recommendations["value_at_risk"].sum())
        economics = _campaign_economics(
            total_value_at_risk,
            len(recommendations),
            request.margin_rate,
            request.offer_cost_per_customer,
            request.expected_uplift,
        )

        by_segment = (
            recommendations.groupby(["CLV_Segment", "Churn_Risk"])
            .agg(
                customer_count=("CustomerID", "count"),
                total_clv=("CLV", "sum"),
                average_churn_probability=("churn_probability", "mean"),
                total_value_at_risk=("value_at_risk", "sum"),
            )
            .reset_index()
        )
        by_segment["expected_retained_revenue"] = (
            by_segment["total_value_at_risk"] * request.expected_uplift
        )
        by_segment["campaign_cost"] = (
            by_segment["customer_count"] * request.offer_cost_per_customer
        )
        by_segment["expected_net_value"] = (
            by_segment["expected_retained_revenue"] * request.margin_rate -
            by_segment["campaign_cost"]
        )
        by_segment["roi"] = by_segment.apply(
            lambda row: (
                None
                if row["campaign_cost"] == 0
                else row["expected_net_value"] / row["campaign_cost"]
            ),
            axis=1,
        )

        return {
            "inputs": {
                "customer_count": len(recommendations),
                "margin_rate": request.margin_rate,
                "offer_cost_per_customer": request.offer_cost_per_customer,
                "expected_uplift": request.expected_uplift,
            },
            "portfolio": {
                "total_clv": _safe_float(recommendations["CLV"].sum()),
                "average_churn_probability": _safe_float(
                    recommendations["churn_probability"].mean()
                ),
                "total_value_at_risk": total_value_at_risk,
                **economics,
            },
            "by_segment": _records(by_segment),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Campaign ROI failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/lightgbm/feature-importance")
async def get_lightgbm_feature_importance():
    """Get LightGBM feature importance for CLV and churn models."""
    try:
        if not LIGHTGBM_FEATURE_IMPORTANCE_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="LightGBM feature importance artifact is not available"
            )
        return pd.read_csv(LIGHTGBM_FEATURE_IMPORTANCE_PATH).to_dict('records')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature importance query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/lightgbm/metrics")
async def get_lightgbm_metrics():
    """Get current LightGBM model metrics."""
    metadata = _lightgbm_metrics()

    return {
        "model_version": metadata.get("model_version"),
        "training_snapshot_date": metadata.get("training_snapshot_date"),
        "validation_snapshot_date": metadata.get("validation_snapshot_date"),
        "feature_window_days": metadata.get('feature_window_days', FEATURE_WINDOW_DAYS),
        "target_window_days": metadata.get('target_window_days', TARGET_WINDOW_DAYS),
        "metrics": metadata.get("metrics", {}),
    }
