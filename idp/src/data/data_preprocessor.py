"""Data preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import joblib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
from utils.config_loader import get_config
from utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses data for diabetes prediction."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data preprocessor.

        Args:
            config: Configuration dictionary. If None, loads from config file.
        """
        self.config = config or get_config('data.preprocessing', {})
        self.scaler = None
        self.imputer = None
        logger.info("DataPreprocessor initialized")

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        handle_zeros: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        In the PIMA diabetes dataset, zeros in certain columns represent missing values.

        Args:
            df: Input DataFrame
            handle_zeros: Whether to treat zeros as missing values

        Returns:
            DataFrame with handled missing values
        """
        df_processed = df.copy()

        if handle_zeros or self.config.get('handle_zeros_as_missing', True):
            # Columns where zero is likely a missing value
            zero_not_allowed = [
                'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
            ]

            for col in zero_not_allowed:
                if col in df_processed.columns:
                    original_zeros = (df_processed[col] == 0).sum()
                    df_processed[col] = df_processed[col].replace(0, np.nan)
                    logger.info(f"Replaced {original_zeros} zeros with NaN in {col}")

        # Count missing values
        missing_counts = df_processed.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")

        return df_processed

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Impute missing values.

        Args:
            df: Input DataFrame with missing values
            strategy: Imputation strategy ('mean', 'median', 'knn')
            fit: Whether to fit the imputer (True for training, False for inference)

        Returns:
            DataFrame with imputed values
        """
        if strategy is None:
            strategy = self.config.get('imputation_strategy', 'median')

        if strategy in ['mean', 'median', 'most_frequent']:
            if fit or self.imputer is None:
                self.imputer = SimpleImputer(strategy=strategy)
                imputed_data = self.imputer.fit_transform(df)
                logger.info(f"Fitted SimpleImputer with strategy: {strategy}")
            else:
                imputed_data = self.imputer.transform(df)
                logger.info(f"Applied SimpleImputer with strategy: {strategy}")

        elif strategy == 'knn':
            if fit or self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                imputed_data = self.imputer.fit_transform(df)
                logger.info("Fitted KNNImputer")
            else:
                imputed_data = self.imputer.transform(df)
                logger.info("Applied KNNImputer")

        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        # Create DataFrame with original column names
        df_imputed = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
        return df_imputed

    def remove_outliers(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series] = None,
        method: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Remove outliers from the dataset.

        Args:
            df: Input DataFrame
            y: Target variable (optional)
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            Tuple of (DataFrame without outliers, target without outliers)
        """
        if method is None:
            method = self.config.get('outlier_detection', 'iqr')
        if threshold is None:
            threshold = self.config.get('outlier_threshold', 1.5)

        mask = pd.Series([True] * len(df), index=df.index)

        if method == 'iqr':
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

        elif method == 'zscore':
            for col in df.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask &= z_scores <= threshold

        outliers_removed = (~mask).sum()
        logger.info(f"Removed {outliers_removed} outliers using {method} method")

        df_clean = df[mask]
        y_clean = y[mask] if y is not None else None

        return df_clean, y_clean

    def scale_features(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features.

        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            Scaled DataFrame
        """
        if method is None:
            method = self.config.get('scaling_method', 'standard')

        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if fit or self.scaler is None:
            self.scaler = scaler_class()
            scaled_data = self.scaler.fit_transform(df)
            logger.info(f"Fitted {method} scaler")
        else:
            scaled_data = self.scaler.transform(df)
            logger.info(f"Applied {method} scaler")

        df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        return df_scaled

    def handle_class_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using oversampling.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Oversampling method ('smote', 'adasyn', 'random_oversample')

        Returns:
            Tuple of (resampled features, resampled target)
        """
        imbalance_config = get_config('data.imbalance', {})
        if not imbalance_config.get('enabled', True):
            logger.info("Class imbalance handling disabled")
            return X, y

        if method is None:
            method = imbalance_config.get('method', 'smote')

        original_counts = y.value_counts()
        logger.info(f"Original class distribution:\n{original_counts}")

        sampling_strategy = imbalance_config.get('sampling_strategy', 'auto')

        if method == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'random_oversample':
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        else:
            raise ValueError(f"Unknown oversampling method: {method}")

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        new_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"Resampled class distribution using {method}:\n{new_counts}")

        # Convert back to DataFrame/Series with proper column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        return X_resampled, y_resampled

    def preprocess_pipeline(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = True,
        handle_imbalance: bool = True,
        remove_outliers_flag: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Full preprocessing pipeline.

        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            fit: Whether to fit transformers (True for training, False for inference)
            handle_imbalance: Whether to handle class imbalance (only for training)
            remove_outliers_flag: Whether to remove outliers

        Returns:
            Tuple of (processed features, processed target)
        """
        logger.info("Starting preprocessing pipeline")

        # 1. Handle missing values
        X_processed = self.handle_missing_values(X)

        # 2. Impute missing values
        X_processed = self.impute_missing_values(X_processed, fit=fit)

        # 3. Remove outliers (only during training if enabled)
        if remove_outliers_flag and fit:
            X_processed, y = self.remove_outliers(X_processed, y)

        # 4. Scale features
        X_processed = self.scale_features(X_processed, fit=fit)

        # 5. Handle class imbalance (only during training)
        if handle_imbalance and fit and y is not None:
            X_processed, y = self.handle_class_imbalance(X_processed, y)

        logger.info("Preprocessing pipeline completed")
        return X_processed, y

    def save_transformers(self, output_dir: str = 'models'):
        """
        Save fitted transformers.

        Args:
            output_dir: Directory to save transformers
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

        if self.imputer is not None:
            imputer_path = os.path.join(output_dir, 'imputer.joblib')
            joblib.dump(self.imputer, imputer_path)
            logger.info(f"Imputer saved to {imputer_path}")

    def load_transformers(self, input_dir: str = 'models'):
        """
        Load fitted transformers.

        Args:
            input_dir: Directory containing saved transformers
        """
        scaler_path = os.path.join(input_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")

        imputer_path = os.path.join(input_dir, 'imputer.joblib')
        if os.path.exists(imputer_path):
            self.imputer = joblib.load(imputer_path)
            logger.info(f"Imputer loaded from {imputer_path}")
