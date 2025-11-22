"""Enhanced model evaluation with comprehensive metrics and visualizations."""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
from utils.config_loader import get_config

logger = get_logger(__name__)


def evaluate_model_enhanced(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = 'visualizations/evaluation'
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with metrics and visualizations.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save visualizations

    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("="*80)
    logger.info("STARTING ENHANCED MODEL EVALUATION")
    logger.info("="*80)

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    try:
        # ========================================
        # STEP 1: PREDICTIONS
        # ========================================
        logger.info("\n[1/6] Generating predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # ========================================
        # STEP 2: CLASSIFICATION METRICS
        # ========================================
        logger.info("\n[2/6] Calculating classification metrics...")

        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred)
        results['recall'] = recall_score(y_test, y_pred)
        results['f1_score'] = f1_score(y_test, y_pred)
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        results['average_precision'] = average_precision_score(y_test, y_pred_proba)

        logger.info("Classification Metrics:")
        logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  F1-Score:  {results['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        logger.info(f"  Avg Precision: {results['average_precision']:.4f}")

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report

        logger.info("\nDetailed Classification Report:")
        logger.info(classification_report(y_test, y_pred))

        # ========================================
        # STEP 3: CONFUSION MATRIX
        # ========================================
        logger.info("\n[3/6] Generating confusion matrix...")

        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {cm_path}")
        plt.close()

        # ========================================
        # STEP 4: ROC CURVE
        # ========================================
        logger.info("\n[4/6] Generating ROC curve...")

        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
        results['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds_roc.tolist()
        }

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC Curve (AUC = {results["roc_auc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)

        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {roc_path}")
        plt.close()

        # ========================================
        # STEP 5: PRECISION-RECALL CURVE
        # ========================================
        logger.info("\n[5/6] Generating Precision-Recall curve...")

        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
        results['precision_recall_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds_pr.tolist()
        }

        # Plot Precision-Recall curve
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, color='#3498db', linewidth=2,
                label=f'PR Curve (AP = {results["average_precision"]:.4f})')
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(alpha=0.3)

        pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {pr_path}")
        plt.close()

        # ========================================
        # STEP 6: CALIBRATION CURVE
        # ========================================
        logger.info("\n[6/6] Generating calibration curve...")

        try:
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')
            results['calibration_curve'] = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }

            # Plot calibration curve
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#e74c3c', label='Model')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Perfectly Calibrated')
            ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
            ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(alpha=0.3)

            cal_path = os.path.join(output_dir, 'calibration_curve.png')
            plt.savefig(cal_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {cal_path}")
            plt.close()

        except Exception as e:
            logger.warning(f"Could not generate calibration curve: {e}")

        # ========================================
        # SUMMARY
        # ========================================
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Test samples: {len(y_test)}")
        logger.info(f"Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"ROC-AUC:   {results['roc_auc']:.4f}")
        logger.info(f"F1-Score:  {results['f1_score']:.4f}")
        logger.info("="*80)

        # Save results
        results_path = os.path.join(output_dir, 'evaluation_results.joblib')
        joblib.dump(results, results_path)
        logger.info(f"\nEvaluation results saved to {results_path}")

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    # Load model and test data
    model = joblib.load('models/xgboost_diabetes_model.joblib')
    metadata = joblib.load('models/training_metadata.joblib')

    X_test = metadata.get('X_test')
    y_test = metadata.get('y_test')

    if X_test is None or y_test is None:
        logger.error("Test data not found in metadata. Please run training first.")
    else:
        evaluate_model_enhanced(model, X_test, y_test)
