"""Data validation module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
from utils.exceptions import DataValidationError
from utils.config_loader import get_config

logger = get_logger(__name__)


class DataValidator:
    """Validates diabetes prediction data."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data validator.

        Args:
            config: Configuration dictionary. If None, loads from config file.
        """
        self.validation_config = config or get_config('data.validation', {})
        logger.info("DataValidator initialized")

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate entire dataframe.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required columns
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return False, errors

        # Validate each column
        for col in required_columns:
            col_errors = self.validate_column(df[col], col)
            errors.extend(col_errors)

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("DataFrame validation passed")
        else:
            logger.error(f"DataFrame validation failed with {len(errors)} errors")

        return is_valid, errors

    def validate_column(self, series: pd.Series, column_name: str) -> List[str]:
        """
        Validate a single column.

        Args:
            series: Pandas Series to validate
            column_name: Name of the column

        Returns:
            List of error messages
        """
        errors = []
        config_key = column_name.lower().replace('_', '')

        # Get validation config for this column
        col_config = self.validation_config.get(config_key.lower(), {})

        if not col_config:
            logger.warning(f"No validation config found for column: {column_name}")
            return errors

        # Check for null values
        null_count = series.isnull().sum()
        if null_count > 0:
            errors.append(f"{column_name}: Contains {null_count} null values")

        # Check value range
        min_val = col_config.get('min')
        max_val = col_config.get('max')

        if min_val is not None:
            below_min = (series < min_val).sum()
            if below_min > 0:
                errors.append(
                    f"{column_name}: {below_min} values below minimum ({min_val})"
                )

        if max_val is not None:
            above_max = (series > max_val).sum()
            if above_max > 0:
                errors.append(
                    f"{column_name}: {above_max} values above maximum ({max_val})"
                )

        return errors

    def validate_input(self, input_data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate user input data.

        Args:
            input_data: Dictionary of feature values

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        required_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Check for missing features
        missing_features = set(required_features) - set(input_data.keys())
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")
            return False, errors

        # Validate each feature
        for feature, value in input_data.items():
            config_key = feature.lower().replace('_', '')
            col_config = self.validation_config.get(config_key, {})

            if not col_config:
                continue

            # Check type
            if not isinstance(value, (int, float, np.number)):
                errors.append(f"{feature}: Invalid type (expected number, got {type(value)})")
                continue

            # Check range
            min_val = col_config.get('min')
            max_val = col_config.get('max')

            if min_val is not None and value < min_val:
                errors.append(f"{feature}: Value {value} below minimum ({min_val})")

            if max_val is not None and value > max_val:
                errors.append(f"{feature}: Value {value} above maximum ({max_val})")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Input validation passed")
        else:
            logger.warning(f"Input validation failed: {errors}")

        return is_valid, errors

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing data quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'zero_values': {},
            'duplicates': df.duplicated().sum(),
            'statistics': {}
        }

        for col in df.columns:
            # Missing values
            missing = df[col].isnull().sum()
            report['missing_values'][col] = {
                'count': int(missing),
                'percentage': float(missing / len(df) * 100)
            }

            # Zero values (for numeric columns)
            if df[col].dtype in [np.float64, np.int64]:
                zeros = (df[col] == 0).sum()
                report['zero_values'][col] = {
                    'count': int(zeros),
                    'percentage': float(zeros / len(df) * 100)
                }

                # Basic statistics
                report['statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }

        logger.info("Data quality report generated")
        return report

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers in the dataset.

        Args:
            df: DataFrame to analyze
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary mapping column names to boolean Series indicating outliers
        """
        outliers = {}

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = z_scores > threshold

        total_outliers = sum(o.sum() for o in outliers.values())
        logger.info(f"Detected {total_outliers} outliers using {method} method")

        return outliers
