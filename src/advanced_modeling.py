"""
Advanced supervised CLV and churn modeling with LightGBM.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class AdvancedCLVModel:
    """
    Supervised LightGBM models for holdout CLV and churn prediction.

    The model uses customer-level calibration features and trains against
    observed holdout-period outcomes.
    """

    feature_columns = [
        'frequency',
        'recency',
        'avg_basket_size_90d',
        'orders_last_7d',
        'orders_last_30d',
        'orders_last_90d',
        'spend_last_30d',
        'spend_last_90d',
        'unique_products_90d',
        'customer_tenure_days',
        'spend_trend_30d_vs_90d',
        'orders_trend_30d_vs_90d',
        'frequency_recency_ratio',
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.clv_model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
        self.churn_model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
        self.is_fitted = False
        self.has_churn_classes = False
        self.metrics = {}
        self.metadata = {}

    def prepare_features(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """Prepare LightGBM features from customer-level feature data."""
        features = customer_features.copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        for column in self.feature_columns:
            if column not in features.columns:
                features[column] = 0
        return features[self.feature_columns]

    def build_snapshot_features(self, transactions: pd.DataFrame,
                                snapshot_date: str,
                                feature_window_days: int = 90) -> pd.DataFrame:
        """
        Build customer features from the past rolling feature window.

        Features use transactions after snapshot_date - feature_window_days and
        up to and including snapshot_date.
        """
        snapshot = pd.to_datetime(snapshot_date)
        feature_start = snapshot - pd.Timedelta(days=feature_window_days)
        self._validate_snapshot_windows(snapshot, feature_start, feature_window_days)

        history = transactions[transactions['InvoiceDate'] <= snapshot].copy()
        history['TotalSales'] = history['Quantity'] * history['UnitPrice']

        past = history[
            (history['InvoiceDate'] > feature_start) &
            (history['InvoiceDate'] <= snapshot)
        ].copy()

        features = past.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'InvoiceDate': 'max',
            'TotalSales': ['sum', 'mean'],
            'StockCode': 'nunique',
        })
        features.columns = [
            'frequency',
            'last_purchase_date',
            'spend_last_90d',
            'avg_order_value_90d',
            'unique_products_90d',
        ]
        features['recency'] = (snapshot - features['last_purchase_date']).dt.days
        features['frequency_recency_ratio'] = features['frequency'] / (features['recency'] + 1)

        first_purchase = history.groupby('CustomerID')['InvoiceDate'].min()
        features['customer_tenure_days'] = (
            snapshot - first_purchase.reindex(features.index)
        ).dt.days.fillna(0)

        features['orders_last_7d'] = self._orders_in_window(past, snapshot, 7).reindex(features.index).fillna(0)
        features['orders_last_30d'] = self._orders_in_window(past, snapshot, 30).reindex(features.index).fillna(0)
        features['orders_last_90d'] = features['frequency']
        features['spend_last_30d'] = self._spend_in_window(past, snapshot, 30).reindex(features.index).fillna(0)
        features['avg_basket_size_90d'] = self._avg_basket_size(past).reindex(features.index).fillna(0)
        features['spend_trend_30d_vs_90d'] = (
            features['spend_last_30d'] / features['spend_last_90d'].replace(0, np.nan)
        ).fillna(0)
        features['orders_trend_30d_vs_90d'] = (
            features['orders_last_30d'] / features['orders_last_90d'].replace(0, np.nan)
        ).fillna(0)

        features = features.drop(columns=['last_purchase_date'])

        return self.prepare_features(features)

    def _validate_snapshot_windows(self, snapshot: pd.Timestamp,
                                   feature_start: pd.Timestamp,
                                   feature_window_days: int) -> None:
        """Guard against accidental feature/target window leakage."""
        if feature_start >= snapshot:
            raise ValueError("Feature window start must be before snapshot date")
        if feature_window_days <= 0:
            raise ValueError("feature_window_days must be positive")

    def _orders_in_window(self, transactions: pd.DataFrame,
                          snapshot: pd.Timestamp,
                          days: int) -> pd.Series:
        """Count unique orders in a lookback window."""
        window_start = snapshot - pd.Timedelta(days=days)
        window = transactions[
            (transactions['InvoiceDate'] > window_start) &
            (transactions['InvoiceDate'] <= snapshot)
        ]
        return window.groupby('CustomerID')['InvoiceNo'].nunique()

    def _spend_in_window(self, transactions: pd.DataFrame,
                         snapshot: pd.Timestamp,
                         days: int) -> pd.Series:
        """Sum spend in a lookback window."""
        window_start = snapshot - pd.Timedelta(days=days)
        window = transactions[
            (transactions['InvoiceDate'] > window_start) &
            (transactions['InvoiceDate'] <= snapshot)
        ]
        return window.groupby('CustomerID')['TotalSales'].sum()

    def _avg_basket_size(self, transactions: pd.DataFrame) -> pd.Series:
        """Calculate average item quantity per order in the feature window."""
        order_quantities = transactions.groupby(['CustomerID', 'InvoiceNo'])['Quantity'].sum()
        return order_quantities.groupby('CustomerID').mean()

    def prepare_targets(self, customer_ids: pd.Index,
                        transactions: pd.DataFrame,
                        snapshot_date: str,
                        target_window_days: int = 90) -> pd.DataFrame:
        """
        Create snapshot-based future targets.

        churn_90d is 1 when the customer makes no purchase in the next target
        window. clv_90d is the total spend in that same future window.
        """
        snapshot = pd.to_datetime(snapshot_date)
        target_end = snapshot + pd.Timedelta(days=target_window_days)
        if target_end <= snapshot:
            raise ValueError("target_window_days must be positive")

        future = transactions[
            (transactions['InvoiceDate'] > snapshot) &
            (transactions['InvoiceDate'] <= target_end)
        ].copy()
        future['TotalSales'] = future['Quantity'] * future['UnitPrice']

        outcomes = future.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'TotalSales': 'sum',
        }).rename(columns={
            'InvoiceNo': 'future_purchases',
            'TotalSales': 'clv_90d',
        })

        targets = pd.DataFrame(index=customer_ids).join(outcomes, how='left').fillna(0)
        targets['churn_90d'] = (targets['future_purchases'] == 0).astype(int)
        return targets

    def fit(self, transactions: pd.DataFrame,
            snapshot_date: str,
            feature_window_days: int = 90,
            target_window_days: int = 90,
            validation_snapshot_date: Optional[str] = None) -> None:
        """Fit LightGBM CLV regression and churn classification models."""
        logger.info("Fitting LightGBM CLV and churn models")

        features = self.build_snapshot_features(
            transactions,
            snapshot_date,
            feature_window_days
        )
        targets = self.prepare_targets(
            features.index,
            transactions,
            snapshot_date,
            target_window_days
        )

        self.clv_model.fit(features, targets['clv_90d'])

        churn_target = targets['churn_90d']
        self.has_churn_classes = churn_target.nunique() > 1
        if self.has_churn_classes:
            self.churn_model.fit(features, churn_target)
        else:
            logger.warning(
                "Skipping LightGBM churn classifier because the holdout target "
                "contains only one class."
            )

        self.is_fitted = True

        if validation_snapshot_date:
            validation_features = self.build_snapshot_features(
                transactions,
                validation_snapshot_date,
                feature_window_days
            )
            validation_targets = self.prepare_targets(
                validation_features.index,
                transactions,
                validation_snapshot_date,
                target_window_days
            )
            self.metrics = self.evaluate(validation_features, validation_targets)
        else:
            self.metrics = self.evaluate(features, targets)

        logger.info("LightGBM models fitted successfully")

    def evaluate(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate LightGBM models on the labeled snapshot data."""
        clv_pred = np.maximum(self.clv_model.predict(features), 0)
        clv_true = targets['clv_90d']
        metrics = {
            'clv_mae': float(mean_absolute_error(clv_true, clv_pred)),
            'clv_rmse': float(np.sqrt(mean_squared_error(clv_true, clv_pred))),
            'clv_r2': float(r2_score(clv_true, clv_pred)),
        }

        if self.has_churn_classes:
            churn_true = targets['churn_90d']
            churn_prob = self.churn_model.predict_proba(features)[:, 1]
            churn_pred = (churn_prob >= 0.5).astype(int)
            metrics.update({
                'churn_roc_auc': float(roc_auc_score(churn_true, churn_prob)),
                'churn_pr_auc': float(average_precision_score(churn_true, churn_prob)),
                'churn_precision': float(precision_score(churn_true, churn_pred, zero_division=0)),
                'churn_recall': float(recall_score(churn_true, churn_pred, zero_division=0)),
                'churn_f1': float(f1_score(churn_true, churn_pred, zero_division=0)),
            })

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Return gain-based and split-count feature importance."""
        if not self.is_fitted:
            raise ValueError("LightGBM models must be fitted before feature importance is available")

        importance = pd.DataFrame({'feature': self.feature_columns})
        importance['clv_importance_gain'] = self.clv_model.booster_.feature_importance(
            importance_type='gain'
        )
        importance['clv_importance_split'] = self.clv_model.booster_.feature_importance(
            importance_type='split'
        )

        if self.has_churn_classes:
            importance['churn_importance_gain'] = self.churn_model.booster_.feature_importance(
                importance_type='gain'
            )
            importance['churn_importance_split'] = self.churn_model.booster_.feature_importance(
                importance_type='split'
            )
        else:
            importance['churn_importance_gain'] = 0
            importance['churn_importance_split'] = 0

        importance['combined_importance_gain'] = (
            importance['clv_importance_gain'] + importance['churn_importance_gain']
        )
        return importance.sort_values('combined_importance_gain', ascending=False)

    def predict(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """Predict supervised CLV and churn probability for customer features."""
        if not self.is_fitted:
            raise ValueError("LightGBM models must be fitted before prediction")

        features = self.prepare_features(customer_features)
        clv = np.maximum(self.clv_model.predict(features), 0)

        if self.has_churn_classes:
            churn_probability = self.churn_model.predict_proba(features)[:, 1]
        else:
            churn_probability = np.zeros(len(features))

        return pd.DataFrame({
            'CustomerID': features.index,
            'lightgbm_clv_90d': clv,
            'lightgbm_churn_90d_probability': churn_probability,
        })

    def save(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save fitted LightGBM model artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clv_model, output_path / 'lightgbm_clv_model.joblib')
        if self.has_churn_classes:
            joblib.dump(self.churn_model, output_path / 'lightgbm_churn_model.joblib')
        if metadata is not None:
            self.metadata = metadata
            with open(output_path / 'metadata.json', 'w') as f:
                import json
                json.dump(metadata, f, indent=2)

    def load(self, model_dir: str) -> None:
        """Load fitted LightGBM model artifacts."""
        model_path = Path(model_dir)
        metadata_path = model_path / 'metadata.json'

        self.clv_model = joblib.load(model_path / 'lightgbm_clv_model.joblib')
        churn_path = model_path / 'lightgbm_churn_model.joblib'
        self.has_churn_classes = churn_path.exists()
        if self.has_churn_classes:
            self.churn_model = joblib.load(churn_path)

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            self.metadata = metadata
            self.metrics = metadata.get('metrics', {})
            if metadata.get('feature_columns'):
                self.feature_columns = metadata['feature_columns']

        self.is_fitted = True
