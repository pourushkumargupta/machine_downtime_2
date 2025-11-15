"""
Model Evaluation Script for Machine Downtime Prediction
=======================================================

This script evaluates the trained model:
- Loads trained model and test data
- Computes comprehensive evaluation metrics
- Generates confusion matrix
- Saves evaluation results

Author: Bharani Kumar
Date: November 2025
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate machine downtime prediction model')
    
    # Input paths
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-path', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/evaluation')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5)
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Find model file
    model_file = os.path.join(model_path, 'model.joblib')
    
    if not os.path.exists(model_file):
        # Try alternative location
        model_file = os.path.join(model_path, 'model.tar.gz')
        if os.path.exists(model_file):
            # Extract tar.gz
            import tarfile
            with tarfile.open(model_file, 'r:gz') as tar:
                tar.extractall(path=model_path)
            model_file = os.path.join(model_path, 'model.joblib')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found in {model_path}")
    
    logger.info(f"Loading model from: {model_file}")
    model = joblib.load(model_file)
    logger.info("Model loaded successfully!")
    logger.info(f"Model type: {type(model).__name__}")
    
    return model


def load_test_data(test_path):
    """
    Load test data
    
    Args:
        test_path: Path to test data directory
        
    Returns:
        tuple: (X_test, y_test) feature matrix and target vector
    """
    logger.info(f"Loading test data from {test_path}")
    
    # Find CSV file
    csv_files = [f for f in os.listdir(test_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {test_path}")
    
    test_file = os.path.join(test_path, csv_files[0])
    logger.info(f"Reading test file: {test_file}")
    
    df = pd.read_csv(test_file)
    logger.info(f"Test data shape: {df.shape}")
    
    # Separate features and target
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in test data")
    
    X_test = df.drop('label', axis=1)
    y_test = df['label']
    
    logger.info(f"Test samples: {X_test.shape[0]}")
    logger.info(f"Test downtime rate: {y_test.mean():.2%}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("="*80)
    logger.info("Evaluating Model Performance")
    logger.info("="*80)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Compile metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'specificity': float(specificity),
        'npv': float(npv),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'threshold': float(threshold),
        'test_samples': int(len(y_test)),
        'downtime_rate': float(y_test.mean())
    }
    
    # Log metrics
    logger.info("\nEvaluation Metrics:")
    logger.info(f"  Accuracy:    {accuracy:.4f}")
    logger.info(f"  Precision:   {precision:.4f}")
    logger.info(f"  Recall:      {recall:.4f}")
    logger.info(f"  F1-Score:    {f1:.4f}")
    logger.info(f"  ROC-AUC:     {roc_auc:.4f}")
    logger.info(f"  Specificity: {specificity:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {tn}")
    logger.info(f"  False Positives: {fp}")
    logger.info(f"  False Negatives: {fn}")
    logger.info(f"  True Positives:  {tp}")
    
    # Classification report
    logger.info("\nClassification Report:")
    class_report = classification_report(y_test, y_pred, target_names=['Normal', 'Downtime'])
    logger.info("\n" + class_report)
    
    return metrics


def assess_model_quality(metrics, quality_gates=None):
    """
    Assess if model meets quality gates
    
    Args:
        metrics: Evaluation metrics dictionary
        quality_gates: Dict of minimum acceptable metrics
        
    Returns:
        dict: Quality assessment results
    """
    if quality_gates is None:
        quality_gates = {
            'min_accuracy': 0.80,
            'min_f1_score': 0.75,
            'min_recall': 0.85
        }
    
    logger.info("="*80)
    logger.info("Model Quality Assessment")
    logger.info("="*80)
    
    assessment = {
        'quality_gates': quality_gates,
        'passes_all_gates': True,
        'gate_results': {}
    }
    
    # Check accuracy
    accuracy_pass = metrics['accuracy'] >= quality_gates['min_accuracy']
    assessment['gate_results']['accuracy'] = {
        'value': metrics['accuracy'],
        'threshold': quality_gates['min_accuracy'],
        'passed': accuracy_pass
    }
    logger.info(f"Accuracy Gate: {metrics['accuracy']:.4f} >= {quality_gates['min_accuracy']:.4f} : {'✓ PASS' if accuracy_pass else '✗ FAIL'}")
    
    # Check F1-score
    f1_pass = metrics['f1_score'] >= quality_gates['min_f1_score']
    assessment['gate_results']['f1_score'] = {
        'value': metrics['f1_score'],
        'threshold': quality_gates['min_f1_score'],
        'passed': f1_pass
    }
    logger.info(f"F1-Score Gate: {metrics['f1_score']:.4f} >= {quality_gates['min_f1_score']:.4f} : {'✓ PASS' if f1_pass else '✗ FAIL'}")
    
    # Check recall
    recall_pass = metrics['recall'] >= quality_gates['min_recall']
    assessment['gate_results']['recall'] = {
        'value': metrics['recall'],
        'threshold': quality_gates['min_recall'],
        'passed': recall_pass
    }
    logger.info(f"Recall Gate:   {metrics['recall']:.4f} >= {quality_gates['min_recall']:.4f} : {'✓ PASS' if recall_pass else '✗ FAIL'}")
    
    # Overall assessment
    assessment['passes_all_gates'] = all([accuracy_pass, f1_pass, recall_pass])
    
    if assessment['passes_all_gates']:
        logger.info("\n✓ Model PASSES all quality gates!")
    else:
        logger.warning("\n✗ Model FAILS one or more quality gates")
    
    return assessment


def save_evaluation_results(metrics, assessment, output_path):
    """
    Save evaluation results to JSON
    
    Args:
        metrics: Evaluation metrics
        assessment: Quality assessment results
        output_path: Directory to save results
    """
    logger.info(f"Saving evaluation results to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Combine all results
    results = {
        'metrics': metrics,
        'quality_assessment': assessment
    }
    
    # Save metrics
    metrics_file = os.path.join(output_path, 'evaluation.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {metrics_file}")
    
    # Also save just metrics for model registry
    metrics_only_file = os.path.join(output_path, 'metrics.json')
    with open(metrics_only_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_only_file}")


def main():
    """Main evaluation pipeline"""
    logger.info("="*80)
    logger.info("Starting Model Evaluation Pipeline")
    logger.info("="*80)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Load model
        model = load_model(args.model_path)
        
        # Load test data
        X_test, y_test = load_test_data(args.test_path)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, threshold=args.threshold)
        
        # Assess quality
        assessment = assess_model_quality(metrics)
        
        # Save results
        save_evaluation_results(metrics, assessment, args.output_path)
        
        logger.info("="*80)
        logger.info("Evaluation completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
