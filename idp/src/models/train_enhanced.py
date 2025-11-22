"""Enhanced model training module with advanced features."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import os
import sys
from typing import Tuple, Optional, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
from utils.config_loader import get_config, load_config
from utils.exceptions import ModelError, DataValidationError
from data.data_validator import DataValidator
from data.data_preprocessor import DataPreprocessor
from data.eda import EDAAnalyzer

logger = get_logger(__name__)


def train_model_enhanced(
    data_path: Optional[str] = None,
    config_path: Optional[str] = None,
    run_eda: bool = True,
    save_models: bool = True
) -> Tuple[Any, DataPreprocessor, Dict[str, Any]]:
    """
    Enhanced model training with comprehensive preprocessing and validation.

    Args:
        data_path: Path to training data CSV
        config_path: Path to configuration file
        run_eda: Whether to run EDA and generate visualizations
        save_models: Whether to save trained models

    Returns:
        Tuple of (trained model, fitted preprocessor, training metadata)

    Raises:
        ModelError: If model training fails
        DataValidationError: If data validation fails
    """
    logger.info("="*80)
    logger.info("STARTING ENHANCED MODEL TRAINING PIPELINE")
    logger.info("="*80)

    # Load configuration
    if config_path:
        load_config(config_path)

    config = get_config('data', {})
    model_config = get_config('model', {})

    # Define paths
    if data_path is None:
        data_path = config.get('raw_data_path', 'data/diabetes.csv')

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    metadata = {
        'data_path': data_path,
        'model_type': model_config.get('model_type', 'xgboost'),
        'preprocessing_steps': [],
        'model_params': {}
    }

    try:
        # ========================================
        # STEP 1: LOAD AND VALIDATE DATA
        # ========================================
        logger.info("\n[1/8] Loading and validating data...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

        # Initialize validator
        validator = DataValidator()

        # Get data quality report
        quality_report = validator.get_data_quality_report(df)
        logger.info(f"Data quality report generated:")
        logger.info(f"  - Total rows: {quality_report['total_rows']}")
        logger.info(f"  - Total columns: {quality_report['total_columns']}")
        logger.info(f"  - Duplicate rows: {quality_report['duplicates']}")

        # ========================================
        # STEP 2: EXPLORATORY DATA ANALYSIS
        # ========================================
        if run_eda:
            logger.info("\n[2/8] Running Exploratory Data Analysis...")
            eda_analyzer = EDAAnalyzer(df, target_col='Outcome')
            eda_analyzer.generate_full_report(output_dir='visualizations/eda')
            logger.info("EDA completed. Visualizations saved to 'visualizations/eda'")
            metadata['eda_completed'] = True
        else:
            logger.info("\n[2/8] Skipping EDA")
            metadata['eda_completed'] = False

        # ========================================
        # STEP 3: SPLIT DATA
        # ========================================
        logger.info("\n[3/8] Splitting data into train/test sets...")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Class distribution in train:")
        logger.info(f"  - Class 0 (No Diabetes): {(y_train == 0).sum()}")
        logger.info(f"  - Class 1 (Diabetes): {(y_train == 1).sum()}")

        # ========================================
        # STEP 4: DATA PREPROCESSING
        # ========================================
        logger.info("\n[4/8] Preprocessing data...")
        preprocessor = DataPreprocessor()

        # Preprocess training data (fit transformers)
        X_train_processed, y_train_processed = preprocessor.preprocess_pipeline(
            X_train, y_train,
            fit=True,
            handle_imbalance=True,
            remove_outliers_flag=False  # Keep outliers for robustness
        )

        # Preprocess test data (apply transformers)
        X_test_processed, _ = preprocessor.preprocess_pipeline(
            X_test, None,
            fit=False,
            handle_imbalance=False,
            remove_outliers_flag=False
        )

        logger.info(f"After preprocessing:")
        logger.info(f"  - Train set: {len(X_train_processed)} samples (after SMOTE)")
        logger.info(f"  - Test set: {len(X_test_processed)} samples")

        metadata['preprocessing_steps'] = [
            'handle_missing_values',
            'impute_missing_values',
            'scale_features',
            'handle_class_imbalance'
        ]
        metadata['train_samples_after_preprocessing'] = len(X_train_processed)

        # ========================================
        # STEP 5: HYPERPARAMETER TUNING
        # ========================================
        logger.info("\n[5/8] Training model with hyperparameter tuning...")

        tuning_config = model_config.get('xgboost', {}).get('hyperparameter_tuning', {})
        enabled = tuning_config.get('enabled', True)

        if enabled:
            method = tuning_config.get('method', 'grid_search')
            cv_folds = tuning_config.get('cv_folds', 5)

            base_params = model_config.get('xgboost', {}).get('base_params', {})
            base_params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': random_state,
                'n_jobs': -1
            })

            xgb_model = XGBClassifier(**base_params)

            param_grid = tuning_config.get('param_grid', {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.01],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0]
            })

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

            if method == 'grid_search':
                logger.info(f"Running GridSearchCV with {cv_folds}-fold cross-validation...")
                search = GridSearchCV(
                    estimator=xgb_model,
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=1,  # Avoid multiprocessing issues on Windows
                    verbose=1,
                    scoring='roc_auc'
                )
            elif method == 'random_search':
                n_iter = tuning_config.get('n_iter', 50)
                logger.info(f"Running RandomizedSearchCV with {n_iter} iterations...")
                search = RandomizedSearchCV(
                    estimator=xgb_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    n_jobs=1,  # Avoid multiprocessing issues on Windows
                    verbose=1,
                    scoring='roc_auc',
                    random_state=random_state
                )
            else:
                raise ValueError(f"Unknown tuning method: {method}")

            search.fit(X_train_processed, y_train_processed)

            best_model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_

            logger.info(f"\nBest parameters found: {best_params}")
            logger.info(f"Best CV ROC-AUC score: {best_score:.4f}")

            metadata['best_params'] = best_params
            metadata['best_cv_score'] = float(best_score)

        else:
            logger.info("Hyperparameter tuning disabled. Using default parameters.")
            base_params = model_config.get('xgboost', {}).get('base_params', {})
            best_model = XGBClassifier(**base_params)
            best_model.fit(X_train_processed, y_train_processed)

        # ========================================
        # STEP 6: MODEL CALIBRATION
        # ========================================
        calibration_config = model_config.get('calibration', {})
        if calibration_config.get('enabled', True):
            logger.info("\n[6/8] Calibrating model probabilities...")
            method = calibration_config.get('method', 'isotonic')
            cv_folds = calibration_config.get('cv_folds', 5)

            calibrated_model = CalibratedClassifierCV(
                best_model,
                method=method,
                cv=cv_folds
            )
            calibrated_model.fit(X_train_processed, y_train_processed)

            logger.info(f"Model calibrated using {method} method")
            final_model = calibrated_model
            metadata['calibration_method'] = method
        else:
            logger.info("\n[6/8] Skipping model calibration")
            final_model = best_model
            metadata['calibration_method'] = None

        # ========================================
        # STEP 7: SAVE MODELS AND PREPROCESSORS
        # ========================================
        if save_models:
            logger.info("\n[7/8] Saving models and preprocessors...")

            # Save model
            model_path = os.path.join(model_dir, 'xgboost_diabetes_model.joblib')
            joblib.dump(final_model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Save preprocessor (scaler and imputer)
            preprocessor.save_transformers(model_dir)

            # Save metadata
            metadata_path = os.path.join(model_dir, 'training_metadata.joblib')
            joblib.dump(metadata, metadata_path)
            logger.info(f"Training metadata saved to {metadata_path}")

        # ========================================
        # STEP 8: FINAL SUMMARY
        # ========================================
        logger.info("\n[8/8] Training pipeline completed successfully!")
        logger.info("="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Model type: {metadata['model_type']}")
        logger.info(f"Training samples: {len(X_train)} → {metadata['train_samples_after_preprocessing']} (after preprocessing)")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Preprocessing steps: {', '.join(metadata['preprocessing_steps'])}")
        if metadata.get('best_cv_score'):
            logger.info(f"Best CV ROC-AUC: {metadata['best_cv_score']:.4f}")
        logger.info("="*80)

        # Store test data for evaluation
        metadata['X_test'] = X_test_processed
        metadata['y_test'] = y_test

        return final_model, preprocessor, metadata

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise ModelError(f"Training failed: {str(e)}")


if __name__ == '__main__':
    train_model_enhanced()
