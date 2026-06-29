"""
Main CLV prediction pipeline.
Orchestrates the entire CLV prediction and recommendation process.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import logging
import yaml
import json
import shutil
from datetime import datetime
from dotenv import load_dotenv
import os
from urllib.parse import quote

from .data_ingestion import DataIngestion
from .data_preprocessing import DataPreprocessor
from .segmentation import CustomerSegmenter
from .clv_modeling import CLVModel
from .advanced_modeling import AdvancedCLVModel
from .recommendations import RecommendationEngine

logger = logging.getLogger(__name__)


class CLVPipeline:
    """
    Main pipeline for Customer Lifetime Value prediction and recommendations.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the CLV pipeline.

        Args:
            config_path: Path to configuration YAML file
        """
        # Load environment variables from .env file
        load_dotenv()

        self.config = self._load_config(config_path)
        self.data_ingestion = DataIngestion(self.config.get('data_ingestion', {}))
        self.preprocessor = DataPreprocessor()
        self.segmenter = CustomerSegmenter(random_state=self.config.get('random_state', 42))
        self.clv_model = CLVModel(penalizer_coef=self.config.get('penalizer_coef', 0.0))
        self.advanced_model = AdvancedCLVModel(random_state=self.config.get('random_state', 42))
        self.recommender = RecommendationEngine()

        # Storage for intermediate results
        self.raw_data = None
        self.all_clean_data = None
        self.clean_data = None
        self.holdout_data = None
        self.rfm_data = None
        self.clv_predictions = None
        self.churn_probabilities = None
        self.lightgbm_predictions = None
        self.lightgbm_features = None
        self.lightgbm_loaded_from_champion = False
        self.lightgbm_challenger_metadata = None
        self.customer_segments = None
        self.recommendations = None
        self.summary_data = None

        logger.info("CLV Pipeline initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file or use defaults.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        default_config = {
            'data_source': 'csv',  # 'csv' or 'sql'
            'csv_path': 'data/customer_segmentation.csv',
            'sql_connection': None,
            'sql_query': None,
            'cutoff_date': '2011-06-30',
            'calibration_end': '2011-06-30',
            'holdout_end': '2011-12-31',
            'penalizer_coef': 0.0,
            'time_periods': 90,
            'discount_rate': 0.01,
            'random_state': 42,
            'use_lightgbm': True,
            'load_champion_lightgbm': True,
            'force_train_lightgbm_challenger': False,
            'feature_window_days': 90,
            'target_window_days': 90,
            'mlflow_tracking_uri': 'file:./mlruns',
            'mlflow_experiment_name': 'clv-lightgbm',
            'model_registry_dir': 'models/lightgbm_registry',
            'output_dir': 'models/',
            'plots_dir': 'models/plots/'
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            default_config.update(user_config)

        default_config = self._resolve_env_vars(default_config)
        return default_config

    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve environment variables for configuration values."""
        if config.get('sql_connection'):
            config['sql_connection'] = os.path.expandvars(config['sql_connection'])
        return config

    def run(self) -> Dict[str, Any]:
        """
        Run the complete CLV prediction pipeline.

        Returns:
            Dictionary with results and metrics
        """
        logger.info("Starting CLV prediction pipeline")

        try:
            # Step 1: Data Ingestion
            self.raw_data = self._ingest_data()

            # Step 2: Data Preprocessing
            self.clean_data = self._preprocess_data()

            # Step 3: Feature Engineering (RFM)
            self.rfm_data = self._create_rfm_features()

            # Step 4: Customer Segmentation
            self.customer_segments = self._perform_segmentation()

            # Step 5: CLV Modeling
            self._fit_clv_models()

            # Step 5b: Advanced supervised modeling
            self._fit_advanced_models()

            # Step 6: CLV and Churn Prediction
            self.clv_predictions, self.churn_probabilities = self._predict_clv_and_churn()
            self.lightgbm_predictions = self._predict_lightgbm_clv_and_churn()

            # Step 7: Generate Recommendations
            self.recommendations = self._generate_recommendations()

            # Step 8: Save Results
            self._save_results()

            # Step 9: Generate Summary Report
            summary = self._generate_summary_report()

            logger.info("CLV prediction pipeline completed successfully")
            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _ingest_data(self) -> pd.DataFrame:
        """Step 1: Data ingestion"""
        logger.info("Step 1: Data ingestion")

        if self.config['data_source'] == 'sql':
            if not self.config.get('sql_connection') or not self.config.get('sql_query'):
                raise ValueError("SQL connection string and query must be provided")
            self.data_ingestion.connect_to_sql(self.config['sql_connection'])
            data = self.data_ingestion.load_from_sql(self.config['sql_query'])
        else:
            data = self.data_ingestion.load_from_csv(self.config['csv_path'])

        # Validate required columns
        required_cols = ['CustomerID', 'InvoiceNo', 'StockCode', 'Description',
                        'Quantity', 'UnitPrice', 'InvoiceDate', 'Country']
        if not self.data_ingestion.validate_data(data, required_cols):
            raise ValueError("Data validation failed")

        logger.info(f"Ingested {len(data)} records")
        return data

    def _preprocess_data(self) -> pd.DataFrame:
        """Step 2: Data preprocessing"""
        logger.info("Step 2: Data preprocessing")

        clean_data = self.preprocessor.clean_data(self.raw_data)
        self.all_clean_data = clean_data

        # Split into calibration and holdout if needed
        if self.config.get('use_holdout', True):
            cal_data, holdout_data = self.preprocessor.split_calibration_holdout(
                clean_data,
                self.config['calibration_end'],
                self.config['holdout_end']
            )
            self.holdout_data = holdout_data
            return cal_data

        return clean_data

    def _create_rfm_features(self) -> pd.DataFrame:
        """Step 3: Feature engineering"""
        logger.info("Step 3: Feature engineering")

        rfm_data = self.preprocessor.create_rfm_features(
            self.clean_data,
            self.config['cutoff_date']
        )

        logger.info(f"Created RFM features for {len(rfm_data)} customers")
        return rfm_data

    def _perform_segmentation(self) -> pd.DataFrame:
        """Step 4: Customer segmentation"""
        logger.info("Step 4: Customer segmentation")

        # Prepare scaled data for clustering
        X_scaled, _ = self.preprocessor.prepare_model_data(self.rfm_data)

        # Find optimal clusters
        optimal_k = self.segmenter.find_optimal_clusters(X_scaled)

        # Perform clustering
        cluster_assignments = self.segmenter.perform_clustering(X_scaled, optimal_k)

        # Get cluster characteristics
        cluster_stats = self.segmenter.get_cluster_characteristics(self.rfm_data, cluster_assignments)

        # Combine with RFM data
        segments = self.rfm_data.join(cluster_assignments.set_index('CustomerID'))

        logger.info(f"Created {optimal_k} customer segments")
        return segments

    def _fit_clv_models(self):
        """Step 5: Fit CLV models"""
        logger.info("Step 5: Fit CLV models")

        # Prepare summary data for lifetimes library
        self.summary_data = self.clv_model.prepare_summary_data(
            self.clean_data,
            observation_end=self.config['cutoff_date']
        )

        # Fit BG/NBD model
        self.clv_model.fit_bgf_model(self.summary_data)

        # Fit Gamma-Gamma model
        self.clv_model.fit_gamma_gamma_model(self.summary_data)

    def _fit_advanced_models(self):
        """Step 5b: Fit supervised LightGBM models."""
        if not self.config.get('use_lightgbm', True):
            logger.info("Skipping LightGBM models because use_lightgbm is false")
            return

        if self.summary_data is None:
            raise ValueError("Summary data must be prepared before fitting LightGBM models")

        if self.all_clean_data is None or self.all_clean_data.empty:
            logger.warning("Skipping LightGBM models because no cleaned transaction data is available")
            return

        validation_snapshot = pd.to_datetime(self.config['calibration_end'])
        training_snapshot = validation_snapshot - pd.Timedelta(
            days=self.config.get('target_window_days', 90)
        )

        self.lightgbm_features = self.advanced_model.build_snapshot_features(
            self.all_clean_data,
            snapshot_date=validation_snapshot,
            feature_window_days=self.config.get('feature_window_days', 90)
        )

        champion_dir = self._lightgbm_champion_dir()
        should_load_champion = (
            self.config.get('load_champion_lightgbm', True) and
            not self.config.get('force_train_lightgbm_challenger', False)
        )
        if should_load_champion and champion_dir.exists():
            self.advanced_model.load(str(champion_dir))
            self.lightgbm_loaded_from_champion = True
            logger.info("Loaded champion LightGBM model from %s", champion_dir)
            return

        self.advanced_model.fit(
            self.all_clean_data,
            snapshot_date=training_snapshot,
            feature_window_days=self.config.get('feature_window_days', 90),
            target_window_days=self.config.get('target_window_days', 90),
            validation_snapshot_date=validation_snapshot
        )
        self.lightgbm_challenger_metadata = self._build_lightgbm_metadata()

    def _predict_clv_and_churn(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Step 6: CLV and churn prediction"""
        logger.info("Step 6: CLV and churn prediction")

        if self.summary_data is None:
            self.summary_data = self.clv_model.prepare_summary_data(
                self.clean_data,
                observation_end=self.config['cutoff_date']
            )

        # Predict CLV
        clv_predictions = self.clv_model.predict_clv(
            self.summary_data,
            time_periods=self.config['time_periods'],
            discount_rate=self.config['discount_rate']
        )

        # Predict churn probability
        churn_probabilities = self.clv_model.predict_churn_probability(self.summary_data)

        logger.info("CLV and churn predictions completed")
        return clv_predictions, churn_probabilities

    def _predict_lightgbm_clv_and_churn(self) -> Optional[pd.DataFrame]:
        """Predict CLV and churn using fitted LightGBM models."""
        if not self.config.get('use_lightgbm', True):
            return None

        if not self.advanced_model.is_fitted or self.lightgbm_features is None:
            return None

        logger.info("Predicting CLV and churn with LightGBM models")
        return self.advanced_model.predict(self.lightgbm_features)

    def predict_customer_features(self, customer_features: pd.DataFrame,
                                  time_periods: Optional[int] = None,
                                  discount_rate: Optional[float] = None) -> pd.DataFrame:
        """
        Predict CLV and churn for customer-level lifetimes features.

        customer_features must be indexed by CustomerID and include:
        frequency, recency, T, and monetary_value.
        """
        required_cols = {'frequency', 'recency', 'T', 'monetary_value'}
        missing_cols = required_cols.difference(customer_features.columns)
        if missing_cols:
            raise ValueError(f"Missing prediction feature columns: {sorted(missing_cols)}")

        clv_predictions = self.clv_model.predict_clv(
            customer_features,
            time_periods=time_periods or self.config['time_periods'],
            discount_rate=discount_rate or self.config['discount_rate']
        )
        churn_probabilities = self.clv_model.predict_churn_probability(customer_features)

        results = clv_predictions.copy()
        results['churn_probability'] = churn_probabilities.reindex(customer_features.index).values
        return results

    def predict_customer_features_lightgbm(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict CLV and churn with the supervised LightGBM models.

        customer_features should be indexed by CustomerID and contain rolling
        snapshot features. Missing LightGBM features are filled with 0.
        """
        return self.advanced_model.predict(customer_features)

    def _generate_recommendations(self) -> pd.DataFrame:
        """Step 7: Generate recommendations"""
        logger.info("Step 7: Generate recommendations")

        # Segment customers by CLV and churn
        customer_segments = self.recommender.segment_customers_by_clv_and_churn(
            self.clv_predictions,
            self.churn_probabilities
        )

        # Generate personalized recommendations
        recommendations = self.recommender.generate_recommendations(customer_segments)

        logger.info("Recommendations generated")
        return recommendations

    def _save_results(self):
        """Step 8: Save results"""
        logger.info("Step 8: Save results")

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)

        # Save predictions
        self.clv_predictions.to_csv(output_dir / 'clv_predictions.csv', index=False)
        self.churn_probabilities.to_csv(output_dir / 'churn_probabilities.csv')

        if self.lightgbm_predictions is not None:
            self.lightgbm_predictions.to_csv(output_dir / 'lightgbm_predictions.csv', index=False)
            feature_importance = self.advanced_model.get_feature_importance()
            feature_importance.to_csv(output_dir / 'lightgbm_feature_importance.csv', index=False)
            self._save_lightgbm_artifacts(output_dir, feature_importance)

        # Save recommendations
        self.recommender.export_recommendations(
            self.recommendations,
            str(output_dir / 'customer_recommendations.csv')
        )

        # Save segments
        self.customer_segments.to_csv(output_dir / 'customer_segments.csv')

        logger.info(f"Results saved to {output_dir}")

    def _lightgbm_registry_dir(self) -> Path:
        """Return the local LightGBM model registry directory."""
        return Path(self.config.get('model_registry_dir', 'models/lightgbm_registry'))

    def _lightgbm_champion_dir(self) -> Path:
        """Return the local champion model directory."""
        return self._lightgbm_registry_dir() / 'champion'

    def _build_lightgbm_metadata(self) -> Dict[str, Any]:
        """Build metadata for the current LightGBM model version."""
        if not self.advanced_model.metrics:
            return {}

        return {
            'model_version': datetime.now().strftime('%Y%m%d%H%M%S'),
            'training_snapshot_date': (
                pd.to_datetime(self.config['calibration_end']) -
                pd.Timedelta(days=self.config.get('target_window_days', 90))
            ).date().isoformat(),
            'validation_snapshot_date': pd.to_datetime(
                self.config['calibration_end']
            ).date().isoformat(),
            'feature_window_days': self.config.get('feature_window_days', 90),
            'target_window_days': self.config.get('target_window_days', 90),
            'feature_columns': self.advanced_model.feature_columns,
            'feature_selection_rules': {
                'kept': [
                    'High-importance logical recency/frequency/spend features',
                    'Trend features that compare recent 30-day behavior with 90-day behavior',
                    'Tenure, basket-size, and product-diversity features',
                ],
                'removed': [
                    'total_spend removed as duplicate alias for spend_last_90d',
                    'monetary_value and avg_order_value removed because they duplicate average spend signals',
                    'monetary_log removed because it duplicated the spend magnitude signal',
                ],
                'leakage_checks': [
                    'Features use InvoiceDate <= snapshot_date',
                    'Targets use snapshot_date < InvoiceDate <= snapshot_date + target_window_days',
                ],
            },
            'metrics': self.advanced_model.metrics,
        }

    def _save_lightgbm_artifacts(self, output_dir: Path,
                                 feature_importance: pd.DataFrame) -> None:
        """Save legacy artifacts plus versioned champion/challenger artifacts."""
        if not self.advanced_model.metrics:
            return

        metadata = (
            self.advanced_model.metadata
            if self.lightgbm_loaded_from_champion and self.advanced_model.metadata
            else self.lightgbm_challenger_metadata or self._build_lightgbm_metadata()
        )

        with open(output_dir / 'lightgbm_metrics.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.advanced_model.save(str(output_dir), metadata=metadata)

        if self.lightgbm_loaded_from_champion:
            comparison = {
                'old_model_version': metadata.get('model_version'),
                'new_model_version': metadata.get('model_version'),
                'recommendation': 'loaded_existing_champion',
                'metric_changes': {},
            }
            with open(output_dir / 'lightgbm_model_comparison.json', 'w') as f:
                json.dump(comparison, f, indent=2)
            return

        registry_dir = self._lightgbm_registry_dir()
        run_dir = registry_dir / 'runs' / metadata['model_version']
        run_dir.mkdir(parents=True, exist_ok=True)

        self.lightgbm_predictions.to_csv(run_dir / 'predictions.csv', index=False)
        feature_importance.to_csv(run_dir / 'feature_importance.csv', index=False)
        with open(run_dir / 'metrics.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        self.advanced_model.save(str(run_dir), metadata=metadata)

        champion_metadata = self._load_champion_metadata()
        comparison = (
            self._compare_metric_versions(champion_metadata, metadata)
            if champion_metadata
            else {
                'old_model_version': None,
                'new_model_version': metadata['model_version'],
                'recommendation': 'promote_new_model',
                'metric_changes': {},
            }
        )

        with open(output_dir / 'lightgbm_model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        with open(run_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

        self._log_lightgbm_mlflow_run(run_dir, metadata, comparison)

        if comparison['recommendation'] == 'promote_new_model':
            self._promote_lightgbm_champion(run_dir)

    def _load_champion_metadata(self) -> Optional[Dict[str, Any]]:
        """Load champion metadata if a champion model exists."""
        metadata_path = self._lightgbm_champion_dir() / 'metadata.json'
        if not metadata_path.exists():
            return None
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _promote_lightgbm_champion(self, run_dir: Path) -> None:
        """Promote a challenger run to champion."""
        champion_dir = self._lightgbm_champion_dir()
        if champion_dir.exists():
            shutil.rmtree(champion_dir)
        shutil.copytree(run_dir, champion_dir)
        logger.info("Promoted LightGBM model %s to champion", run_dir.name)

    def _log_lightgbm_mlflow_run(self, run_dir: Path, metadata: Dict[str, Any],
                                 comparison: Dict[str, Any]) -> None:
        """Log params, metrics, tags, and artifacts to local MLflow tracking."""
        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow is not installed; skipping MLflow logging")
            return

        mlflow.set_tracking_uri(self.config.get('mlflow_tracking_uri', 'file:./mlruns'))
        mlflow.set_experiment(self.config.get('mlflow_experiment_name', 'clv-lightgbm'))

        with mlflow.start_run(run_name=f"lightgbm-{metadata['model_version']}"):
            mlflow.set_tags({
                'model_family': 'lightgbm',
                'model_version': metadata['model_version'],
                'promotion_recommendation': comparison['recommendation'],
                'stage': (
                    'champion'
                    if comparison['recommendation'] == 'promote_new_model'
                    else 'challenger'
                ),
            })
            mlflow.log_params({
                'training_snapshot_date': metadata['training_snapshot_date'],
                'validation_snapshot_date': metadata['validation_snapshot_date'],
                'feature_window_days': metadata['feature_window_days'],
                'target_window_days': metadata['target_window_days'],
                'feature_count': len(metadata['feature_columns']),
            })
            mlflow.log_metrics(metadata['metrics'])
            mlflow.log_artifacts(str(run_dir))

    def _compare_metric_versions(self, previous: Dict[str, Any],
                                 current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare old and new LightGBM metric snapshots."""
        lower_is_better = {'clv_mae', 'clv_rmse'}
        old_metrics = previous.get('metrics', {})
        new_metrics = current.get('metrics', {})
        metric_changes = {}

        for metric, new_value in new_metrics.items():
            if metric not in old_metrics:
                continue
            old_value = old_metrics[metric]
            delta = new_value - old_value
            improved = delta < 0 if metric in lower_is_better else delta > 0
            metric_changes[metric] = {
                'old': old_value,
                'new': new_value,
                'delta': delta,
                'improved': improved,
            }

        return {
            'old_model_version': previous.get('model_version'),
            'new_model_version': current.get('model_version'),
            'recommendation': self._model_promotion_recommendation(metric_changes),
            'metric_changes': metric_changes,
        }

    def _model_promotion_recommendation(self, metric_changes: Dict[str, Any]) -> str:
        """Make a conservative promotion recommendation from metric deltas."""
        if not metric_changes:
            return 'no_previous_metrics_to_compare'

        key_metrics = [
            'clv_rmse',
            'clv_mae',
            'churn_roc_auc',
            'churn_pr_auc',
        ]
        evaluated = [
            metric_changes[metric]['improved']
            for metric in key_metrics
            if metric in metric_changes
        ]
        if evaluated and all(evaluated):
            return 'promote_new_model'
        if evaluated and any(evaluated):
            return 'review_tradeoffs_before_promotion'
        return 'keep_old_model'

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Step 9: Generate summary report"""
        logger.info("Step 9: Generate summary report")

        # Basic statistics
        total_customers = len(self.clv_predictions)
        avg_clv = self.clv_predictions['CLV'].mean()
        total_clv = self.clv_predictions['CLV'].sum()
        avg_churn_risk = self.churn_probabilities.mean()

        # Segment distribution
        segment_dist = self.recommendations['CLV_Segment'].value_counts()

        # Top recommendations
        top_recommendations = self.recommender.get_top_priority_customers(self.recommendations, 10)

        summary = {
            'total_customers': total_customers,
            'average_clv': avg_clv,
            'total_clv': total_clv,
            'average_churn_risk': avg_churn_risk,
            'segment_distribution': segment_dist.to_dict(),
            'top_priority_customers': top_recommendations['CustomerID'].tolist(),
            'timestamp': datetime.now().isoformat()
        }

        logger.info("Summary report generated")
        return summary

    def get_model_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.

        Returns:
            Dictionary with model metrics
        """
        if hasattr(self, 'holdout_data') and self.holdout_data is not None:
            metrics = self.clv_model.validate_model(
                self.clean_data,
                self.holdout_data,
                self.config['calibration_end'],
                self.config['holdout_end']
            )
            return metrics
        else:
            logger.warning("No holdout data available for validation")
            return {}
