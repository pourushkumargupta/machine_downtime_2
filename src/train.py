"""
Model Training Script for Machine Downtime Prediction
=====================================================

This script trains a RandomForestClassifier for downtime prediction:
- Loads preprocessed training data
- Trains the model with specified hyperparameters
- Saves the trained model
- Outputs training metrics

Author: Bharani Kumar
Date: November 2025
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    parser = argparse.ArgumentParser(description='Train machine downtime prediction model')
    
    # SageMaker specific paths
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--max-features', type=str, default='sqrt')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    
    return parser.parse_args()


def load_training_data(train_path):
    """
    Load preprocessed training data
    
    Args:
        train_path: Path to training data directory
        
    Returns:
        tuple: (X_train, y_train) feature matrix and target vector
    """
    logger.info(f"Loading training data from {train_path}")
    
    # Find CSV file
    csv_files = [f for f in os.listdir(train_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {train_path}")
    
    train_file = os.path.join(train_path, csv_files[0])
    logger.info(f"Reading training file: {train_file}")
    
    df = pd.read_csv(train_file)
    logger.info(f"Training data shape: {df.shape}")
    
    # Separate features and target
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in training data")
    
    X_train = df.drop('label', axis=1)
    y_train = df['label']
    
    logger.info(f"Feature count: {X_train.shape[1]}")
    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Downtime rate: {y_train.mean():.2%}")
    logger.info(f"Class distribution:\n{y_train.value_counts()}")
    
    return X_train, y_train


def train_model(X_train, y_train, hyperparameters):
    """
    Train RandomForestClassifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameters: Dict of hyperparameters
        
    Returns:
        Trained model
    """
    logger.info("="*80)
    logger.info("Training RandomForestClassifier")
    logger.info("="*80)
    logger.info(f"Hyperparameters: {hyperparameters}")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        min_samples_split=hyperparameters['min_samples_split'],
        min_samples_leaf=hyperparameters['min_samples_leaf'],
        max_features=hyperparameters['max_features'],
        random_state=hyperparameters['random_state'],
        n_jobs=hyperparameters['n_jobs'],
        class_weight='balanced',  # Handle class imbalance
        verbose=1
    )
    
    # Train model
    logger.info("Starting model training...")
    model.fit(X_train, y_train)
    logger.info("Model training completed!")
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Feature Importances:")
    logger.info("\n" + feature_importances.head(10).to_string(index=False))
    
    return model


def evaluate_training(model, X_train, y_train):
    """
    Evaluate model on training data
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Training metrics
    """
    logger.info("="*80)
    logger.info("Evaluating model on training data")
    logger.info("="*80)
    
    # Make predictions
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_pred)),
        'train_precision': float(precision_score(y_train, y_pred, zero_division=0)),
        'train_recall': float(recall_score(y_train, y_pred, zero_division=0)),
        'train_f1_score': float(f1_score(y_train, y_pred, zero_division=0)),
        'train_roc_auc': float(roc_auc_score(y_train, y_pred_proba))
    }
    
    # Log metrics
    logger.info("Training Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics


def save_model(model, model_dir):
    """
    Save trained model
    
    Args:
        model: Trained model
        model_dir: Directory to save model
    """
    logger.info(f"Saving model to {model_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    
    # Verify model file size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"Model file size: {model_size_mb:.2f} MB")


def save_metrics(metrics, output_dir):
    """
    Save training metrics
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save metrics
    """
    logger.info(f"Saving metrics to {output_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")


def main():
    """Main training pipeline"""
    logger.info("="*80)
    logger.info("Starting Model Training Pipeline")
    logger.info("="*80)
    
    # Parse arguments
    args = parse_args()
    
    # Prepare hyperparameters
    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'random_state': args.random_state,
        'n_jobs': args.n_jobs
    }
    
    try:
        # Load training data
        X_train, y_train = load_training_data(args.train)
        
        # Train model
        model = train_model(X_train, y_train, hyperparameters)
        
        # Evaluate on training data
        metrics = evaluate_training(model, X_train, y_train)
        
        # Save model
        save_model(model, args.model_dir)
        
        # Save metrics
        if args.output_data_dir:
            save_metrics(metrics, args.output_data_dir)
        
        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
