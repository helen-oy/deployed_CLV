"""
Data preprocessing module for CLV prediction system.
Handles data cleaning, type conversions, outlier detection, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing operations for CLV analysis.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, type conversions, etc.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")

        # Make a copy to avoid modifying original
        df_clean = df.copy()

        # Drop rows with missing CustomerID
        initial_shape = df_clean.shape
        df_clean = df_clean[df_clean['CustomerID'].notna()]
        logger.info(f"Dropped {initial_shape[0] - df_clean.shape[0]} rows with missing CustomerID")

        # Convert CustomerID to string
        df_clean['CustomerID'] = df_clean['CustomerID'].astype(int).astype(str)

        # Convert InvoiceDate to datetime
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

        # Fill missing countries with mode
        if df_clean['Country'].isnull().any():
            mode_country = df_clean['Country'].mode()[0]
            df_clean['Country'] = df_clean['Country'].fillna(mode_country)
            logger.info(f"Filled missing countries with {mode_country}")

        # Remove negative quantities (returns)
        df_clean = df_clean[df_clean['Quantity'] > 0]

        # Create transaction value for downstream CLV modeling.
        df_clean['TotalSales'] = df_clean['Quantity'] * df_clean['UnitPrice']

        # Remove outliers using z-score
        df_clean = self._remove_outliers_zscore(df_clean, ['Quantity', 'UnitPrice'], threshold=3)

        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean

    def _remove_outliers_zscore(self, df: pd.DataFrame, columns: list, threshold: float = 3) -> pd.DataFrame:
        """
        Remove outliers using z-score method.

        Args:
            df: DataFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold

        Returns:
            DataFrame with outliers removed
        """
        initial_shape = df.shape

        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]

        logger.info(f"Removed {initial_shape[0] - df.shape[0]} outlier rows")
        return df

    def create_rfm_features(self, df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.

        Args:
            df: Cleaned transaction DataFrame
            cutoff_date: Date string for calculating recency

        Returns:
            DataFrame with RFM features
        """
        logger.info("Creating RFM features")

        cutoff = pd.to_datetime(cutoff_date)

        # Calculate total sales
        df = df.copy()
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']

        # RFM calculation
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (cutoff - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalSales': 'sum'  # Monetary
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalSales': 'Monetary'
        })

        # Handle log transformation for Monetary (to handle skewness)
        rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

        logger.info(f"Created RFM features for {len(rfm)} customers")
        return rfm

    def prepare_model_data(self, rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for modeling by scaling features.

        Args:
            rfm_df: RFM DataFrame

        Returns:
            Tuple of (scaled DataFrame, scaler object)
        """
        features = ['Recency', 'Frequency', 'Monetary_log']
        X = rfm_df[features].copy()

        # Scale the features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        logger.info("Prepared model data with scaling")
        return X_scaled, self.scaler

    def split_calibration_holdout(self, df: pd.DataFrame, calibration_end: str, holdout_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into calibration and holdout periods.

        Args:
            df: Transaction DataFrame
            calibration_end: End date for calibration period
            holdout_end: End date for holdout period

        Returns:
            Tuple of (calibration_df, holdout_df)
        """
        calibration_df = df[df['InvoiceDate'] <= pd.to_datetime(calibration_end)]
        holdout_df = df[(df['InvoiceDate'] > pd.to_datetime(calibration_end)) &
                       (df['InvoiceDate'] <= pd.to_datetime(holdout_end))]

        logger.info(f"Split data: {len(calibration_df)} calibration, {len(holdout_df)} holdout records")
        return calibration_df, holdout_df
