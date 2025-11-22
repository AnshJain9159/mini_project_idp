"""Enhanced main pipeline with comprehensive ML workflow."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.train_enhanced import train_model_enhanced
from models.evaluate_enhanced import evaluate_model_enhanced
from utils.logger import setup_logger, get_logger

# Setup logger
setup_logger('diabetes_predictor', log_level='INFO')
logger = get_logger(__name__)


def main():
    """
    Run the complete enhanced ML pipeline:
    1. EDA
    2. Data validation and preprocessing
    3. Model training with hyperparameter tuning
    4. Model calibration
    5. Comprehensive evaluation
    """
    logger.info("🚀 Starting Enhanced Diabetes Prediction ML Pipeline")
    logger.info("")

    try:
        # ========================================
        # TRAINING PIPELINE
        # ========================================
        logger.info("Phase 1: Training")
        logger.info("-" * 80)

        model, preprocessor, metadata = train_model_enhanced(
            data_path='data/diabetes.csv',
            run_eda=True,
            save_models=True
        )

        logger.info("✅ Training completed successfully\n")

        # ========================================
        # EVALUATION PIPELINE
        # ========================================
        logger.info("Phase 2: Evaluation")
        logger.info("-" * 80)

        X_test = metadata.get('X_test')
        y_test = metadata.get('y_test')

        if X_test is not None and y_test is not None:
            results = evaluate_model_enhanced(
                model=model,
                X_test=X_test,
                y_test=y_test,
                output_dir='visualizations/evaluation'
            )

            logger.info("✅ Evaluation completed successfully\n")

            # ========================================
            # FINAL SUMMARY
            # ========================================
            logger.info("=" * 80)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info("\nKey Metrics:")
            logger.info(f"  • Accuracy:        {results['accuracy']:.4f}")
            logger.info(f"  • Precision:       {results['precision']:.4f}")
            logger.info(f"  • Recall:          {results['recall']:.4f}")
            logger.info(f"  • F1-Score:        {results['f1_score']:.4f}")
            logger.info(f"  • ROC-AUC:         {results['roc_auc']:.4f}")
            logger.info(f"  • Avg Precision:   {results['average_precision']:.4f}")
            logger.info("\nOutputs:")
            logger.info("  • Models: models/")
            logger.info("  • EDA Visualizations: visualizations/eda/")
            logger.info("  • Evaluation Visualizations: visualizations/evaluation/")
            logger.info("  • Logs: logs/app.log")
            logger.info("=" * 80)

        else:
            logger.error("❌ Test data not found in metadata")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
