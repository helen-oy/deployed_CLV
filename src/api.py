"""
FastAPI application for CLV prediction and recommendation service.
"""

import ast
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
import logging
from pathlib import Path

from .clv_pipeline import CLVPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value prediction and recommendation service",
    version="1.0.0"
)

# Global pipeline instance (loaded on startup)
pipeline = None
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RFM_SEGMENT_SUMMARY_PATH = MODELS_DIR / "current_state_segment_summary.csv"
LIGHTGBM_PREDICTIONS_PATH = MODELS_DIR / "lightgbm_predictions.csv"
CUSTOMER_RECOMMENDATIONS_PATH = MODELS_DIR / "customer_recommendations.csv"
CUSTOMER_SEGMENTS_PATH = MODELS_DIR / "customer_segments.csv"

LOW_CHURN_MAX = 0.20
HIGH_CHURN_MIN = 0.50
DEFAULT_MARGIN_RATE = 0.40
DEFAULT_OFFER_COST = 5.00
DEFAULT_EXPECTED_UPLIFT = 0.10


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
    """Load the CLV pipeline on startup."""
    global pipeline
    try:
        config_path = Path(__file__).parent.parent / 'config.yaml'
        pipeline = CLVPipeline(str(config_path))
        logger.info("CLV pipeline initialized")

        run_pipeline_on_startup = os.getenv(
            "CLV_API_RUN_PIPELINE_ON_STARTUP",
            "false"
        ).lower() in {"1", "true", "yes"}
        if run_pipeline_on_startup:
            # Run the pipeline to generate recommendations
            pipeline.run()
            logger.info("CLV pipeline executed successfully")
        else:
            champion_dir = pipeline._lightgbm_champion_dir()
            if champion_dir.exists():
                pipeline.advanced_model.load(str(champion_dir))
                logger.info("Loaded champion LightGBM model from %s", champion_dir)
            logger.info(
                "Skipping pipeline.run() because CLV_API_RUN_PIPELINE_ON_STARTUP is false"
            )
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise


def _ensure_pipeline() -> CLVPipeline:
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")
    return pipeline


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
    active_pipeline = _ensure_pipeline()
    if active_pipeline.lightgbm_predictions is not None:
        df = active_pipeline.lightgbm_predictions.copy()
    elif LIGHTGBM_PREDICTIONS_PATH.exists():
        df = pd.read_csv(LIGHTGBM_PREDICTIONS_PATH)
    else:
        raise HTTPException(
            status_code=500,
            detail="LightGBM predictions are not available"
        )
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["value_at_risk_90d"] = (
        df["lightgbm_clv_90d"] * df["lightgbm_churn_90d_probability"]
    )
    df["churn_risk_band"] = df["lightgbm_churn_90d_probability"].apply(_churn_risk_band)
    return df


def _recommendations_df() -> pd.DataFrame:
    active_pipeline = _ensure_pipeline()
    if active_pipeline.recommendations is not None:
        df = active_pipeline.recommendations.copy()
    elif CUSTOMER_RECOMMENDATIONS_PATH.exists():
        df = pd.read_csv(CUSTOMER_RECOMMENDATIONS_PATH)
    else:
        raise HTTPException(
            status_code=500,
            detail="Customer recommendations are not available"
        )
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
    active_pipeline = _ensure_pipeline()
    if active_pipeline.customer_segments is not None:
        df = active_pipeline.customer_segments.reset_index().copy()
    elif CUSTOMER_SEGMENTS_PATH.exists():
        df = pd.read_csv(CUSTOMER_SEGMENTS_PATH)
    else:
        raise HTTPException(
            status_code=500,
            detail="Customer RFM segments are not available"
        )
    df["CustomerID"] = df["CustomerID"].astype(str)
    return df


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
        "target_window_days": _ensure_pipeline().config.get("target_window_days", 90),
        "feature_window_days": _ensure_pipeline().config.get("feature_window_days", 90),
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
    Predict CLV for given customers.

    For known customers, the API uses the fitted model features created during
    startup. For new customers, provide customer_age (T) and monetary_value
    along with frequency and recency.
    """
    active_pipeline = _ensure_pipeline()

    try:
        feature_rows = []
        lightgbm_feature_rows = []
        trained_customer_lookup = {}
        if active_pipeline.summary_data is not None:
            trained_customer_lookup = {
                str(customer_id): customer_id
                for customer_id in active_pipeline.summary_data.index
            }
        lightgbm_customer_lookup = {}
        if active_pipeline.lightgbm_features is not None:
            lightgbm_customer_lookup = {
                str(customer_id): customer_id
                for customer_id in active_pipeline.lightgbm_features.index
            }

        for customer in request.customers:
            trained_customer_id = trained_customer_lookup.get(customer.customer_id)
            if trained_customer_id is not None:
                customer_features = active_pipeline.summary_data.loc[trained_customer_id]
            else:
                if customer.customer_age is None and request.model_type in ('baseline', 'both'):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "customer_age is required for customers that are not "
                            "already present in the trained pipeline data"
                        )
                    )

                customer_features = pd.Series({
                    'frequency': customer.frequency,
                    'recency': customer.recency,
                    'T': customer.customer_age if customer.customer_age is not None else 0,
                    'monetary_value': (
                        customer.monetary_value
                        if customer.monetary_value is not None
                        else customer.monetary
                    )
                })

            feature_rows.append({
                'CustomerID': customer.customer_id,
                'frequency': float(customer_features['frequency']),
                'recency': float(customer_features['recency']),
                'T': float(customer_features['T']),
                'monetary_value': float(customer_features['monetary_value'])
            })

            lightgbm_customer_id = lightgbm_customer_lookup.get(customer.customer_id)
            if lightgbm_customer_id is not None:
                lightgbm_features = active_pipeline.lightgbm_features.loc[lightgbm_customer_id]
            else:
                lightgbm_features = pd.Series({
                    'frequency': customer.frequency,
                    'recency': customer.recency,
                    'avg_basket_size_90d': 0,
                    'orders_last_7d': 0,
                    'orders_last_30d': customer.frequency,
                    'orders_last_90d': customer.frequency,
                    'spend_last_30d': customer.monetary,
                    'spend_last_90d': customer.monetary,
                    'unique_products_90d': 0,
                    'customer_tenure_days': (
                        customer.customer_age
                        if customer.customer_age is not None
                        else 0
                    ),
                    'spend_trend_30d_vs_90d': 1 if customer.monetary > 0 else 0,
                    'orders_trend_30d_vs_90d': 1 if customer.frequency > 0 else 0,
                    'frequency_recency_ratio': customer.frequency / (customer.recency + 1),
                })

            lightgbm_feature_rows.append({
                'CustomerID': customer.customer_id,
                'frequency': float(lightgbm_features['frequency']),
                'recency': float(lightgbm_features['recency']),
                'avg_basket_size_90d': float(lightgbm_features['avg_basket_size_90d']),
                'orders_last_7d': float(lightgbm_features['orders_last_7d']),
                'orders_last_30d': float(lightgbm_features['orders_last_30d']),
                'orders_last_90d': float(lightgbm_features['orders_last_90d']),
                'spend_last_30d': float(lightgbm_features['spend_last_30d']),
                'spend_last_90d': float(lightgbm_features['spend_last_90d']),
                'unique_products_90d': float(lightgbm_features['unique_products_90d']),
                'customer_tenure_days': float(lightgbm_features['customer_tenure_days']),
                'spend_trend_30d_vs_90d': float(lightgbm_features['spend_trend_30d_vs_90d']),
                'orders_trend_30d_vs_90d': float(lightgbm_features['orders_trend_30d_vs_90d']),
                'frequency_recency_ratio': float(lightgbm_features['frequency_recency_ratio']),
            })

        feature_df = pd.DataFrame(feature_rows).set_index('CustomerID')
        lightgbm_feature_df = pd.DataFrame(lightgbm_feature_rows).set_index('CustomerID')
        if request.model_type in ('lightgbm', 'both') and not active_pipeline.advanced_model.is_fitted:
            raise HTTPException(
                status_code=503,
                detail="LightGBM model is not available. Use model_type='baseline'."
            )

        predictions = []
        baseline_predictions = None
        lightgbm_predictions = None

        if request.model_type in ('baseline', 'both'):
            baseline_predictions = (
                active_pipeline.predict_customer_features(feature_df)
                .set_index('CustomerID')
            )

        if request.model_type in ('lightgbm', 'both'):
            lightgbm_predictions = (
                active_pipeline.predict_customer_features_lightgbm(lightgbm_feature_df)
                .set_index('CustomerID')
            )

        for customer_id in feature_df.index:
            prediction = {'customer_id': str(customer_id)}

            if lightgbm_predictions is not None:
                lightgbm_row = lightgbm_predictions.loc[customer_id]
                churn_probability = _safe_float(lightgbm_row['lightgbm_churn_90d_probability'])
                clv_90d = _safe_float(lightgbm_row['lightgbm_clv_90d'])
                prediction['lightgbm'] = {
                    'predicted_clv_90d': clv_90d,
                    'churn_90d_probability': churn_probability,
                    'churn_risk_band': _churn_risk_band(churn_probability),
                    'value_at_risk_90d': _value_at_risk(clv_90d, churn_probability),
                }

            if baseline_predictions is not None:
                baseline_row = baseline_predictions.loc[customer_id]
                churn_probability = _safe_float(baseline_row['churn_probability'])
                clv = _safe_float(baseline_row['CLV'])
                prediction['baseline'] = {
                    'predicted_purchases': _safe_float(baseline_row['predicted_purchases']),
                    'predicted_monetary': _safe_float(baseline_row['predicted_monetary']),
                    'predicted_clv': clv,
                    'churn_probability': churn_probability,
                    'churn_risk_band': _churn_risk_band(churn_probability),
                    'value_at_risk': _value_at_risk(clv, churn_probability),
                }

            predictions.append(prediction)

        return {"predictions": predictions}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CLV prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a customer."""
    if pipeline is None or pipeline.recommender is None:
        raise HTTPException(status_code=500, detail="Recommender not loaded")

    try:
        # Get recommendations based on segment
        recommendations = pipeline.recommender._get_recommendations_for_customer(
            pd.Series({
                'CLV_Segment': request.clv_segment,
                'Churn_Risk': request.churn_risk
            })
        )

        return {
            "customer_id": request.customer_id,
            "clv_segment": request.clv_segment,
            "churn_risk": request.churn_risk,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments/summary")
async def get_segment_summary():
    """Get summary of customer segments."""
    try:
        active_pipeline = _ensure_pipeline()
        recommendations = _recommendations_df()
        summary = active_pipeline.recommender.get_segment_summary(recommendations)
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
    active_pipeline = _ensure_pipeline()
    if not active_pipeline.advanced_model.is_fitted:
        raise HTTPException(status_code=500, detail="LightGBM model not available")

    try:
        return active_pipeline.advanced_model.get_feature_importance().to_dict('records')
    except Exception as e:
        logger.error(f"Feature importance query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/lightgbm/metrics")
async def get_lightgbm_metrics():
    """Get current LightGBM model metrics."""
    active_pipeline = _ensure_pipeline()
    if not active_pipeline.advanced_model.metrics:
        raise HTTPException(status_code=500, detail="LightGBM metrics not available")

    return {
        "feature_window_days": active_pipeline.config.get('feature_window_days', 90),
        "target_window_days": active_pipeline.config.get('target_window_days', 90),
        "metrics": active_pipeline.advanced_model.metrics,
    }
