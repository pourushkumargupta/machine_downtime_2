"""
Model Registration Script for Machine Downtime Prediction
=========================================================

This script registers the trained model in SageMaker Model Registry:
- Loads model artifacts and evaluation metrics
- Creates model package in Model Registry
- Attaches metrics and metadata
- Sets approval status

Author: Bharani Kumar
Date: November 2025
"""

import argparse
import json
import os
import time
import boto3
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Register model in SageMaker Model Registry')
    
    # Model Registry parameters
    parser.add_argument('--model-package-group-name', type=str, default='machine-downtime-model-group')
    parser.add_argument('--model-approval-status', type=str, default='PendingManualApproval',
                       choices=['Approved', 'Rejected', 'PendingManualApproval'])
    
    # Paths
    parser.add_argument('--model-data', type=str, required=True,
                       help='S3 URI to model artifacts (model.tar.gz)')
    parser.add_argument('--metrics-path', type=str, default='/opt/ml/processing/evaluation')
    
    # Model metadata
    parser.add_argument('--inference-instance-type', type=str, default='ml.m5.large')
    parser.add_argument('--transform-instance-type', type=str, default='ml.m5.large')
    
    # AWS configuration
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--role-arn', type=str, help='SageMaker execution role ARN')
    
    return parser.parse_args()


def get_sagemaker_client(region):
    """
    Create SageMaker client
    
    Args:
        region: AWS region
        
    Returns:
        boto3 SageMaker client
    """
    logger.info(f"Creating SageMaker client for region: {region}")
    return boto3.client('sagemaker', region_name=region)


def load_metrics(metrics_path):
    """
    Load evaluation metrics
    
    Args:
        metrics_path: Path to metrics directory
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Loading metrics from {metrics_path}")
    
    # Try different metric file names
    metric_files = ['metrics.json', 'evaluation.json']
    
    for metric_file in metric_files:
        full_path = os.path.join(metrics_path, metric_file)
        if os.path.exists(full_path):
            logger.info(f"Reading metrics from: {full_path}")
            with open(full_path, 'r') as f:
                data = json.load(f)
                
                # Handle nested structure
                if 'metrics' in data:
                    metrics = data['metrics']
                else:
                    metrics = data
                
                logger.info("Metrics loaded successfully")
                logger.info(f"Available metrics: {list(metrics.keys())}")
                return metrics
    
    logger.warning(f"No metrics file found in {metrics_path}")
    return {}


def ensure_model_package_group_exists(client, group_name):
    """
    Ensure model package group exists, create if not
    
    Args:
        client: SageMaker client
        group_name: Model package group name
        
    Returns:
        str: Model package group ARN
    """
    logger.info(f"Checking if model package group exists: {group_name}")
    
    try:
        response = client.describe_model_package_group(
            ModelPackageGroupName=group_name
        )
        logger.info(f"Model package group already exists: {response['ModelPackageGroupArn']}")
        return response['ModelPackageGroupArn']
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            logger.info(f"Model package group does not exist. Creating: {group_name}")
            
            response = client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription="Model package group for machine downtime prediction models"
            )
            
            logger.info(f"Model package group created: {response['ModelPackageGroupArn']}")
            return response['ModelPackageGroupArn']
        else:
            raise


def prepare_model_metrics(metrics):
    """
    Prepare metrics for model registry
    
    Args:
        metrics: Dictionary of evaluation metrics
        
    Returns:
        dict: Formatted metrics for model registry
    """
    logger.info("Preparing metrics for model registry")
    
    # Define core metrics for registration
    core_metrics = {
        'accuracy': metrics.get('accuracy', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1_score': metrics.get('f1_score', 0.0),
        'roc_auc': metrics.get('roc_auc', 0.0)
    }
    
    # Create metric specification list
    metric_list = []
    for metric_name, metric_value in core_metrics.items():
        metric_list.append({
            'Name': metric_name,
            'Type': 'Number',
            'Value': metric_value
        })
    
    logger.info(f"Prepared {len(metric_list)} metrics for registration")
    
    return metric_list


def register_model(client, args, metrics):
    """
    Register model in Model Registry
    
    Args:
        client: SageMaker client
        args: Command line arguments
        metrics: Evaluation metrics
        
    Returns:
        str: Model package ARN
    """
    logger.info("="*80)
    logger.info("Registering Model in Model Registry")
    logger.info("="*80)
    
    # Ensure model package group exists
    group_arn = ensure_model_package_group_exists(client, args.model_package_group_name)
    
    # Prepare metrics
    metric_list = prepare_model_metrics(metrics)
    
    # Get inference image (using sklearn container)
    region = args.region
    account_id = boto3.client('sts').get_caller_identity()['Account']
    image_uri = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    
    # Get role ARN
    role_arn = args.role_arn
    if not role_arn:
        # Try to get from SageMaker session
        try:
            import sagemaker
            role_arn = sagemaker.get_execution_role()
        except Exception:
            raise ValueError("role-arn must be provided or SageMaker execution role must be available")
    
    logger.info(f"Using execution role: {role_arn}")
    logger.info(f"Using inference image: {image_uri}")
    logger.info(f"Model data location: {args.model_data}")
    
    # Prepare model package input
    model_package_input = {
        'ModelPackageGroupName': args.model_package_group_name,
        'ModelPackageDescription': f'Machine downtime prediction model - {time.strftime("%Y-%m-%d %H:%M:%S")}',
        'ModelApprovalStatus': args.model_approval_status,
        'InferenceSpecification': {
            'Containers': [{
                'Image': image_uri,
                'ModelDataUrl': args.model_data,
                'Framework': 'SKLEARN',
                'FrameworkVersion': '1.2-1'
            }],
            'SupportedContentTypes': ['text/csv', 'application/json'],
            'SupportedResponseMIMETypes': ['text/csv', 'application/json'],
            'SupportedRealtimeInferenceInstanceTypes': [
                args.inference_instance_type,
                'ml.t2.medium',
                'ml.m5.xlarge'
            ],
            'SupportedTransformInstanceTypes': [
                args.transform_instance_type,
                'ml.m5.xlarge'
            ]
        },
        'ModelMetrics': {
            'ModelQuality': {
                'Statistics': {
                    'ContentType': 'application/json',
                    'S3Uri': f"{os.path.dirname(args.model_data)}/evaluation/metrics.json"
                }
            }
        },
        'CustomerMetadataProperties': {
            'Project': 'MachineDowntime',
            'Algorithm': 'RandomForestClassifier',
            'Accuracy': str(metrics.get('accuracy', 0.0)),
            'F1Score': str(metrics.get('f1_score', 0.0)),
            'Recall': str(metrics.get('recall', 0.0)),
            'CreatedBy': 'SageMaker-Pipeline'
        }
    }
    
    # Create model package
    logger.info("Creating model package...")
    response = client.create_model_package(**model_package_input)
    
    model_package_arn = response['ModelPackageArn']
    logger.info(f"Model package created successfully!")
    logger.info(f"Model Package ARN: {model_package_arn}")
    
    # Wait for model package to be completed
    logger.info("Waiting for model package to complete...")
    waiter = client.get_waiter('model_package_ready')
    waiter.wait(ModelPackageName=model_package_arn)
    
    logger.info("Model package is ready!")
    
    return model_package_arn


def get_model_package_details(client, model_package_arn):
    """
    Get model package details
    
    Args:
        client: SageMaker client
        model_package_arn: Model package ARN
        
    Returns:
        dict: Model package details
    """
    logger.info("Retrieving model package details...")
    
    response = client.describe_model_package(
        ModelPackageName=model_package_arn
    )
    
    logger.info("\nModel Package Details:")
    logger.info(f"  Status: {response['ModelPackageStatus']}")
    logger.info(f"  Approval Status: {response['ModelApprovalStatus']}")
    logger.info(f"  Creation Time: {response['CreationTime']}")
    
    if 'CustomerMetadataProperties' in response:
        logger.info("  Metadata:")
        for key, value in response['CustomerMetadataProperties'].items():
            logger.info(f"    {key}: {value}")
    
    return response


def main():
    """Main model registration pipeline"""
    logger.info("="*80)
    logger.info("Starting Model Registration Pipeline")
    logger.info("="*80)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Create SageMaker client
        client = get_sagemaker_client(args.region)
        
        # Load metrics
        metrics = load_metrics(args.metrics_path)
        
        # Register model
        model_package_arn = register_model(client, args, metrics)
        
        # Get details
        details = get_model_package_details(client, model_package_arn)
        
        # Save model package ARN to file
        output_dir = args.metrics_path
        arn_file = os.path.join(output_dir, 'model_package_arn.txt')
        with open(arn_file, 'w') as f:
            f.write(model_package_arn)
        logger.info(f"Model package ARN saved to: {arn_file}")
        
        logger.info("="*80)
        logger.info("Model Registration completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
