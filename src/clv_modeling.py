"""
CLV modeling module for CLV prediction system.
Implements BG/NBD and Gamma-Gamma models for CLV prediction and churn analysis.
"""

import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from typing import Dict, Tuple, Optional, List
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CLVModel:
    """
    Customer Lifetime Value prediction model using BG/NBD and Gamma-Gamma.
    """

    def __init__(self, penalizer_coef: float = 0.0, random_state: int = 42):
        self.penalizer_coef = penalizer_coef
        self.random_state = random_state
        self.bgf_model = None
        self.gg_model = None
        self.frequency_holdout = None
        self.monetary_holdout = None

    def _get_penalizer_candidates(self, penalizer: float) -> List[float]:
        """Return a small sequence of fallback penalizers for unstable fits."""
        candidates = [penalizer, 0.001, 0.01, 0.1, 1.0]
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def prepare_summary_data(self, transactions: pd.DataFrame,
                           customer_id_col: str = 'CustomerID',
                           datetime_col: str = 'InvoiceDate',
                           monetary_col: str = 'TotalSales',
                           observation_end: str = None) -> pd.DataFrame:
        """
        Prepare RFM summary data from transaction data.

        Args:
            transactions: Transaction DataFrame
            customer_id_col: Customer ID column name
            datetime_col: Date column name
            monetary_col: Monetary value column name
            observation_end: End date for observation period

        Returns:
            Summary DataFrame with RFM metrics
        """
        logger.info("Preparing summary data for CLV modeling")

        summary = summary_data_from_transaction_data(
            transactions=transactions,
            customer_id_col=customer_id_col,
            datetime_col=datetime_col,
            monetary_value_col=monetary_col,
            observation_period_end=observation_end,
            freq='D'  # Daily frequency
        )

        logger.info(f"Created summary data for {len(summary)} customers")
        return summary

    def fit_bgf_model(self, summary_data: pd.DataFrame, penalizer_coef: Optional[float] = None) -> BetaGeoFitter:
        """
        Fit Beta-Geometric/Negative Binomial Distribution model.

        Args:
            summary_data: RFM summary data
            penalizer_coef: Regularization parameter

        Returns:
            Fitted BG/NBD model
        """
        penalizer = penalizer_coef if penalizer_coef is not None else self.penalizer_coef

        last_error = None

        for candidate in self._get_penalizer_candidates(penalizer):
            logger.info(f"Fitting BG/NBD model with penalizer_coef={candidate}")

            try:
                self.bgf_model = BetaGeoFitter(penalizer_coef=candidate)
                self.bgf_model.fit(
                    frequency=summary_data['frequency'],
                    recency=summary_data['recency'],
                    T=summary_data['T']
                )
                logger.info("BG/NBD model fitted successfully")
                return self.bgf_model
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "BG/NBD fit failed with penalizer_coef=%s. Retrying with a stronger penalizer.",
                    candidate
                )

        raise last_error

    def fit_gamma_gamma_model(self, summary_data: pd.DataFrame, penalizer_coef: Optional[float] = None) -> GammaGammaFitter:
        """
        Fit Gamma-Gamma model for monetary value prediction.

        Args:
            summary_data: RFM summary data (only customers with frequency > 0)
            penalizer_coef: Regularization parameter

        Returns:
            Fitted Gamma-Gamma model
        """
        penalizer = penalizer_coef if penalizer_coef is not None else self.penalizer_coef

        # Filter customers with at least one repeat purchase
        gg_data = summary_data[summary_data['frequency'] > 0]

        last_error = None

        for candidate in self._get_penalizer_candidates(penalizer):
            logger.info(
                f"Fitting Gamma-Gamma model with penalizer_coef={candidate} on {len(gg_data)} customers"
            )

            try:
                self.gg_model = GammaGammaFitter(penalizer_coef=candidate)
                self.gg_model.fit(
                    frequency=gg_data['frequency'],
                    monetary_value=gg_data['monetary_value']
                )
                logger.info("Gamma-Gamma model fitted successfully")
                return self.gg_model
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gamma-Gamma fit failed with penalizer_coef=%s. Retrying with a stronger penalizer.",
                    candidate
                )

        raise last_error

    def predict_clv(self, summary_data: pd.DataFrame, time_periods: int = 90,
                   discount_rate: float = 0.01) -> pd.DataFrame:
        """
        Predict Customer Lifetime Value.

        Args:
            summary_data: RFM summary data
            time_periods: Number of periods to predict (days)
            discount_rate: Discount rate for CLV calculation

        Returns:
            DataFrame with CLV predictions
        """
        if not self.bgf_model or not self.gg_model:
            raise ValueError("Both BG/NBD and Gamma-Gamma models must be fitted first")

        logger.info(f"Predicting CLV for {time_periods} periods with discount rate {discount_rate}")

        # Predict future purchases
        future_purchases = self.bgf_model.conditional_expected_number_of_purchases_up_to_time(
            t=time_periods,
            frequency=summary_data['frequency'],
            recency=summary_data['recency'],
            T=summary_data['T']
        )

        # Predict monetary value (only for customers with purchases)
        monetary_data = summary_data[summary_data['frequency'] > 0]
        predicted_monetary = self.gg_model.conditional_expected_average_profit(
            frequency=monetary_data['frequency'],
            monetary_value=monetary_data['monetary_value']
        )

        # Calculate CLV
        clv_predictions = future_purchases * predicted_monetary

        # Apply discount factor
        discount_factor = (1 - discount_rate) ** (time_periods / 365)  # Assuming annual discount
        clv_predictions *= discount_factor

        # Create results DataFrame
        results = pd.DataFrame({
            'CustomerID': summary_data.index,
            'predicted_purchases': future_purchases,
            'predicted_monetary': pd.Series(predicted_monetary.values, index=monetary_data.index),
            'CLV': clv_predictions
        }).fillna(0)  # Fill NaN for customers with no purchases

        logger.info("CLV prediction completed")
        return results

    def predict_churn_probability(self, summary_data: pd.DataFrame) -> pd.Series:
        """
        Predict churn probability using BG/NBD model.

        Args:
            summary_data: RFM summary data

        Returns:
            Series with churn probabilities
        """
        if not self.bgf_model:
            raise ValueError("BG/NBD model must be fitted first")

        logger.info("Predicting churn probabilities")

        # Churn probability = 1 - P(alive)
        churn_prob = 1 - self.bgf_model.conditional_probability_alive(
            frequency=summary_data['frequency'],
            recency=summary_data['recency'],
            T=summary_data['T']
        )

        logger.info("Churn probability prediction completed")
        return pd.Series(churn_prob, index=summary_data.index, name='churn_probability')

    def validate_model(self, calibration_data: pd.DataFrame, holdout_data: pd.DataFrame,
                      observation_end: str, holdout_end: str) -> Dict[str, float]:
        """
        Validate model performance using calibration and holdout data.

        Args:
            calibration_data: Calibration period transactions
            holdout_data: Holdout period transactions
            observation_end: End of calibration period
            holdout_end: End of holdout period

        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating CLV model")

        # Prepare calibration and holdout data
        cal_holdout = calibration_and_holdout_data(
            transactions=calibration_data,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            calibration_period_end=observation_end,
            observation_period_end=holdout_end,
            freq='D'
        )

        # Fit model on calibration data
        self.fit_bgf_model(cal_holdout[['frequency_cal', 'recency_cal', 'T_cal']])

        # Predict on holdout period
        predicted_purchases = self.bgf_model.predict(
            t=(pd.to_datetime(holdout_end) - pd.to_datetime(observation_end)).days,
            frequency=cal_holdout['frequency_cal'],
            recency=cal_holdout['recency_cal'],
            T=cal_holdout['T_cal']
        )

        # Calculate RMSE
        rmse = np.sqrt(np.mean((cal_holdout['frequency_holdout'] - predicted_purchases) ** 2))

        # Calculate MAE
        mae = np.mean(np.abs(cal_holdout['frequency_holdout'] - predicted_purchases))

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mean_actual': cal_holdout['frequency_holdout'].mean(),
            'mean_predicted': predicted_purchases.mean()
        }

        logger.info(f"Model validation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return metrics

    def plot_model_diagnostics(self, summary_data: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create diagnostic plots for the fitted models.

        Args:
            summary_data: RFM summary data
            save_path: Optional path to save plots
        """
        if not self.bgf_model:
            raise ValueError("BG/NBD model must be fitted first")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot frequency vs recency
        self.bgf_model.plot_frequency_recency_matrix(ax=axes[0,0])
        axes[0,0].set_title('Frequency vs Recency Matrix')

        # Plot probability alive
        self.bgf_model.plot_probability_alive_matrix(ax=axes[0,1])
        axes[0,1].set_title('Probability Alive Matrix')

        # Plot period transactions
        self.bgf_model.plot_period_transactions(ax=axes[1,0])
        axes[1,0].set_title('Period Transactions')

        # Plot calibration purchases vs holdout
        if self.frequency_holdout is not None:
            axes[1,1].scatter(self.frequency_holdout['frequency_cal'],
                            self.frequency_holdout['frequency_holdout'],
                            alpha=0.5)
            axes[1,1].set_xlabel('Calibration Period Frequency')
            axes[1,1].set_ylabel('Holdout Period Frequency')
            axes[1,1].set_title('Calibration vs Holdout Frequency')
        else:
            axes[1,1].text(0.5, 0.5, 'No holdout data available',
                         ha='center', va='center', transform=axes[1,1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model diagnostics plot saved to {save_path}")
        else:
            plt.show()
