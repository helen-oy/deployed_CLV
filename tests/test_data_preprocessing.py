"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()

        # Create sample data
        self.sample_data = pd.DataFrame({
            'CustomerID': [1.0, 2.0, np.nan, 4.0],
            'InvoiceNo': ['A001', 'A002', 'A003', 'A004'],
            'StockCode': ['ABC', 'DEF', 'GHI', 'JKL'],
            'Description': ['Item A', 'Item B', 'Item C', 'Item D'],
            'Quantity': [1, 2, -5, 4],  # Include negative quantity
            'UnitPrice': [10.0, 20.0, 1000.0, 40.0],  # Include outlier
            'InvoiceDate': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']),
            'Country': ['UK', 'US', 'Unspecified', 'CA']
        })

    def test_clean_data_removes_missing_customer_id(self):
        """Test that rows with missing CustomerID are removed."""
        result = self.preprocessor.clean_data(self.sample_data)

        # Should have 3 rows (one with NaN CustomerID removed)
        assert len(result) == 3
        assert result['CustomerID'].notna().all()

    def test_clean_data_converts_customer_id_to_string(self):
        """Test that CustomerID is converted to string."""
        result = self.preprocessor.clean_data(self.sample_data)

        assert result['CustomerID'].dtype == 'object'
        assert result['CustomerID'].iloc[0] == '1'

    def test_clean_data_converts_invoice_date(self):
        """Test that InvoiceDate is converted to datetime."""
        result = self.preprocessor.clean_data(self.sample_data)

        assert pd.api.types.is_datetime64_any_dtype(result['InvoiceDate'])

    def test_clean_data_fills_missing_countries(self):
        """Test that missing countries are filled with mode."""
        result = self.preprocessor.clean_data(self.sample_data)

        assert 'Unspecified' not in result['Country'].values
        assert result['Country'].iloc[2] == 'UK'  # Mode of original data

    def test_clean_data_removes_negative_quantities(self):
        """Test that negative quantities are removed."""
        result = self.preprocessor.clean_data(self.sample_data)

        assert (result['Quantity'] > 0).all()

    def test_remove_outliers_zscore(self):
        """Test outlier removal using z-score."""
        data_with_outliers = pd.DataFrame({
            'CustomerID': ['1', '2', '3', '4', '5'],
            'Quantity': [1, 2, 3, 100, 5],  # 100 is outlier
            'UnitPrice': [10, 20, 30, 40, 50]
        })

        result = self.preprocessor._remove_outliers_zscore(data_with_outliers, ['Quantity'], threshold=2)

        # Should remove the outlier row
        assert len(result) == 4
        assert 100 not in result['Quantity'].values

    def test_create_rfm_features(self):
        """Test RFM feature creation."""
        # Create sample transaction data
        transactions = pd.DataFrame({
            'CustomerID': ['1', '1', '2', '2', '2'],
            'InvoiceDate': pd.to_datetime(['2021-01-01', '2021-01-05', '2021-01-02', '2021-01-03', '2021-01-10']),
            'TotalSales': [100, 200, 50, 75, 150]
        })

        cutoff_date = '2021-01-15'
        result = self.preprocessor.create_rfm_features(transactions, cutoff_date)

        assert len(result) == 2  # Two customers
        assert 'Recency' in result.columns
        assert 'Frequency' in result.columns
        assert 'Monetary' in result.columns
        assert 'Monetary_log' in result.columns

        # Check customer 1: recency = 10 days (2021-01-15 - 2021-01-05)
        assert result.loc['1', 'Recency'] == 10
        assert result.loc['1', 'Frequency'] == 2
        assert result.loc['1', 'Monetary'] == 300

    def test_prepare_model_data(self):
        """Test model data preparation with scaling."""
        rfm_data = pd.DataFrame({
            'Recency': [1, 2, 3],
            'Frequency': [10, 20, 30],
            'Monetary': [100, 200, 300],
            'Monetary_log': [4.6, 5.3, 5.7]
        }, index=['1', '2', '3'])

        X_scaled, scaler = self.preprocessor.prepare_model_data(rfm_data)

        assert X_scaled.shape == (3, 3)
        assert X_scaled.index.equals(rfm_data.index)
        assert list(X_scaled.columns) == ['Recency', 'Frequency', 'Monetary_log']

        # Check that data is scaled (mean should be close to 0)
        assert abs(X_scaled.values.mean()) < 0.1