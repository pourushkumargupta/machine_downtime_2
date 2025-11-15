"""
Data Preprocessing Script for Machine Downtime Prediction
==========================================================

This script handles data preprocessing for the SageMaker pipeline:
- Reads raw CSV data
- Cleans and validates data
- Creates binary labels
- Splits into train/test sets
- Outputs processed data to S3

Author: Bharani Kumar
Date: November 2025
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess machine downtime data')
    
    # SageMaker specific arguments
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-train', type=str, default='/opt/ml/processing/train')
    parser.add_argument('--output-test', type=str, default='/opt/ml/processing/test')
    
    # Processing parameters
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    
    return parser.parse_args()


def load_data(input_path):
    """
    Load raw data from input path
    
    Args:
        input_path: Path to input data directory
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    logger.info(f"Loading data from {input_path}")
    
    # Find CSV file in input directory
    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {input_path}")
    
    csv_path = os.path.join(input_path, csv_files[0])
    logger.info(f"Reading CSV file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    return df


def validate_data(df):
    """
    Validate data quality
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: Validated dataframe
    """
    logger.info("Validating data...")
    
    # Check for required columns
    required_columns = ['Downtime', 'Hydraulic_Pressure(bar)', 'Air_System_Pressure(bar)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values. Dropping rows with missing values.")
        df = df.dropna()
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows. Removing duplicates.")
        df = df.drop_duplicates()
    
    logger.info(f"Data after validation: {df.shape}")
    return df


def create_label(df):
    """
    Create binary label from Downtime column
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with binary label
    """
    logger.info("Creating binary label...")
    
    # Create binary label: 1 if 'yes', 0 if 'no'
    df['label'] = (df['Downtime'] == 'yes').astype(int)
    
    # Log label distribution
    label_dist = df['label'].value_counts()
    logger.info(f"Label distribution:\n{label_dist}")
    logger.info(f"Downtime rate: {df['label'].mean():.2%}")
    
    return df


def engineer_features(df):
    """
    Create additional engineered features
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with engineered features
    """
    logger.info("Engineering features...")
    
    # Create pressure differential (normalized)
    if 'Hydraulic_Pressure(bar)' in df.columns:
        df['Pressure_Differential'] = df['Hydraulic_Pressure(bar)'] - df['Hydraulic_Pressure(bar)'].median()
    
    # Create vibration ratio
    if 'Tool_Vibration(µm)' in df.columns and 'Spindle_Vibration(µm)' in df.columns:
        # Avoid division by zero
        df['Vibration_Ratio'] = df['Tool_Vibration(µm)'] / (df['Spindle_Vibration(µm)'] + 1e-6)
    
    # Create pressure-force interaction
    if 'Hydraulic_Pressure(bar)' in df.columns and 'Cutting_Force(kN)' in df.columns:
        df['Pressure_Force_Interaction'] = df['Hydraulic_Pressure(bar)'] * df['Cutting_Force(kN)']
    
    logger.info(f"Features after engineering: {df.shape[1]} columns")
    
    return df


def prepare_features(df):
    """
    Prepare final feature set for modeling
    
    Args:
        df: Input dataframe
        
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    logger.info("Preparing features for modeling...")
    
    # Columns to drop
    drop_columns = ['Date', 'Machine_ID', 'Downtime', 'label']
    drop_columns = [col for col in drop_columns if col in df.columns]
    
    # Separate features and target
    y = df['label']
    X = df.drop(columns=drop_columns)
    
    logger.info(f"Feature set shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    logger.info(f"Train downtime rate: {y_train.mean():.2%}")
    logger.info(f"Test downtime rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test, train_path, test_path):
    """
    Save processed data to output paths
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        train_path: Path to save training data
        test_path: Path to save test data
    """
    logger.info("Saving processed data...")
    
    # Create output directories if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Combine features and target
    train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='label')], axis=1)
    test_data = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='label')], axis=1)
    
    # Save to CSV
    train_file = os.path.join(train_path, 'train.csv')
    test_file = os.path.join(test_path, 'test.csv')
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    logger.info(f"Training data saved to: {train_file}")
    logger.info(f"Test data saved to: {test_file}")
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")


def main():
    """Main preprocessing pipeline"""
    logger.info("="*80)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("="*80)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Load data
        df = load_data(args.input_data)
        
        # Validate data
        df = validate_data(df)
        
        # Create label
        df = create_label(df)
        
        # Engineer features
        df = engineer_features(df)
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        # Save data
        save_data(X_train, X_test, y_train, y_test, args.output_train, args.output_test)
        
        logger.info("="*80)
        logger.info("Preprocessing completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
