"""
Data ingestion module for CLV prediction system.
Handles data loading from SQL databases and CSV files.
"""

import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data ingestion from various sources.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine = None

    def connect_to_sql(self, connection_string: str) -> None:
        """
        Establish connection to SQL database.

        Args:
            connection_string: SQLAlchemy connection string
        """
        try:
            self.engine = create_engine(connection_string)
            logger.info("Connected to SQL database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def load_from_sql(self, query: str) -> pd.DataFrame:
        """
        Load data from SQL database using a query.

        Args:
            query: SQL query string

        Returns:
            DataFrame with the query results
        """
        if not self.engine:
            raise ValueError("Database connection not established. Call connect_to_sql first.")

        try:
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Loaded {len(df)} records from SQL database")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from SQL: {e}")
            raise

    def load_from_csv(self, file_path: str, encoding: str = 'ISO-8859-1', **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            encoding: File encoding
            **kwargs: Additional pandas read_csv arguments

        Returns:
            DataFrame with CSV data
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.info(f"Loaded {len(df)} records from CSV file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise

    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if validation passes
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        logger.info("Data validation passed")
        return True