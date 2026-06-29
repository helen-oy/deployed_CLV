"""
Unit tests for data ingestion module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_ingestion import DataIngestion


class TestDataIngestion:
    """Test cases for DataIngestion class."""

    def test_initialization(self):
        """Test DataIngestion initialization."""
        config = {'test': 'value'}
        ingestion = DataIngestion(config)
        assert ingestion.config == config
        assert ingestion.engine is None

    @patch('src.data_ingestion.create_engine')
    def test_connect_to_sql_success(self, mock_create_engine):
        """Test successful SQL connection."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        ingestion = DataIngestion()
        ingestion.connect_to_sql('sqlite:///test.db')

        mock_create_engine.assert_called_once_with('sqlite:///test.db')
        assert ingestion.engine == mock_engine

    @patch('src.data_ingestion.create_engine')
    def test_connect_to_sql_failure(self, mock_create_engine):
        """Test SQL connection failure."""
        mock_create_engine.side_effect = Exception('Connection failed')

        ingestion = DataIngestion()

        with pytest.raises(Exception, match='Connection failed'):
            ingestion.connect_to_sql('invalid://connection')

    def test_load_from_csv_success(self):
        """Test successful CSV loading."""
        test_data = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceNo': ['A', 'B', 'C'],
            'Quantity': [1, 2, 3]
        })

        with patch('pandas.read_csv', return_value=test_data) as mock_read_csv:
            ingestion = DataIngestion()
            result = ingestion.load_from_csv('test.csv')

            mock_read_csv.assert_called_once_with('test.csv', encoding='ISO-8859-1')
            pd.testing.assert_frame_equal(result, test_data)

    def test_load_from_csv_failure(self):
        """Test CSV loading failure."""
        with patch('pandas.read_csv', side_effect=Exception('File not found')):
            ingestion = DataIngestion()

            with pytest.raises(Exception, match='File not found'):
                ingestion.load_from_csv('nonexistent.csv')

    def test_validate_data_success(self):
        """Test successful data validation."""
        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceNo': ['A', 'B', 'C'],
            'StockCode': ['X', 'Y', 'Z'],
            'Description': ['Desc1', 'Desc2', 'Desc3'],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0],
            'InvoiceDate': pd.date_range('2021-01-01', periods=3),
            'Country': ['UK', 'US', 'CA']
        })

        ingestion = DataIngestion()
        required_cols = ['CustomerID', 'InvoiceNo', 'StockCode', 'Description',
                        'Quantity', 'UnitPrice', 'InvoiceDate', 'Country']

        assert ingestion.validate_data(df, required_cols) == True

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceNo': ['A', 'B', 'C']
        })

        ingestion = DataIngestion()
        required_cols = ['CustomerID', 'InvoiceNo', 'MissingColumn']

        assert ingestion.validate_data(df, required_cols) == False