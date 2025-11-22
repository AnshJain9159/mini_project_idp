"""Unit tests for data validation."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_validator import DataValidator


class TestDataValidator:
    """Test cases for DataValidator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DataValidator()
        assert validator is not None
        assert validator.validation_config is not None

    def test_validate_input_valid_data(self):
        """Test validation with valid input data."""
        validator = DataValidator()

        input_data = {
            'Pregnancies': 1,
            'Glucose': 120.0,
            'BloodPressure': 72.0,
            'SkinThickness': 20.0,
            'Insulin': 79.0,
            'BMI': 32.0,
            'DiabetesPedigreeFunction': 0.47,
            'Age': 29
        }

        is_valid, errors = validator.validate_input(input_data)
        assert is_valid == True
        assert len(errors) == 0

    def test_validate_input_missing_features(self):
        """Test validation with missing features."""
        validator = DataValidator()

        input_data = {
            'Pregnancies': 1,
            'Glucose': 120.0
        }

        is_valid, errors = validator.validate_input(input_data)
        assert is_valid == False
        assert len(errors) > 0

    def test_validate_input_out_of_range(self):
        """Test validation with out of range values."""
        validator = DataValidator()

        input_data = {
            'Pregnancies': 1,
            'Glucose': 600.0,  # Out of range
            'BloodPressure': 72.0,
            'SkinThickness': 20.0,
            'Insulin': 79.0,
            'BMI': 32.0,
            'DiabetesPedigreeFunction': 0.47,
            'Age': 29
        }

        is_valid, errors = validator.validate_input(input_data)
        assert is_valid == False
        assert any('Glucose' in error for error in errors)

    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        validator = DataValidator()

        # Create valid DataFrame
        df = pd.DataFrame({
            'Pregnancies': [1, 2, 0],
            'Glucose': [120, 130, 110],
            'BloodPressure': [72, 80, 70],
            'SkinThickness': [20, 25, 18],
            'Insulin': [79, 85, 70],
            'BMI': [32.0, 28.5, 30.2],
            'DiabetesPedigreeFunction': [0.47, 0.52, 0.38],
            'Age': [29, 35, 42]
        })

        is_valid, errors = validator.validate_dataframe(df)
        assert is_valid == True

    def test_get_data_quality_report(self):
        """Test data quality report generation."""
        validator = DataValidator()

        df = pd.DataFrame({
            'Pregnancies': [1, 2, 0, 3],
            'Glucose': [120, 0, 110, 140],  # One zero value
            'BloodPressure': [72, 80, 70, 85],
            'SkinThickness': [20, 25, 18, 22],
            'Insulin': [79, 85, 70, 90],
            'BMI': [32.0, 28.5, 30.2, 31.5],
            'DiabetesPedigreeFunction': [0.47, 0.52, 0.38, 0.45],
            'Age': [29, 35, 42, 38],
            'Outcome': [0, 1, 0, 1]
        })

        report = validator.get_data_quality_report(df)

        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'zero_values' in report
        assert report['total_rows'] == 4
        assert report['zero_values']['Glucose']['count'] == 1

    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        validator = DataValidator()

        df = pd.DataFrame({
            'Glucose': [100, 110, 120, 130, 140, 500],  # 500 is an outlier
            'BMI': [25, 28, 30, 32, 35, 100]  # 100 is an outlier
        })

        outliers = validator.detect_outliers(df, method='iqr', threshold=1.5)

        assert 'Glucose' in outliers
        assert 'BMI' in outliers
        assert outliers['Glucose'].sum() > 0  # At least one outlier detected
        assert outliers['BMI'].sum() > 0
