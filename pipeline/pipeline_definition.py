"""
SageMaker Pipeline Definition for Machine Downtime Prediction
=============================================================

This script defines the complete MLOps pipeline:
- Step 1: Data Preprocessing
- Step 2: Model Training
- Step 3: Model Evaluation
- Step 4: Model Registration

Author: Bharani Kumar
Date: November 2025
"""

import os
import yaml
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.sklearn.processing import SKLearnProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='config/pipeline_config.yaml'):
    """
    Load pipeline configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    # Handle both absolute and relative paths
    if not os.path.isabs(config_path):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    return config


def get_session_and_role():
    """
    Get SageMaker session and execution role
    
    Returns:
        tuple: (sagemaker_session, role_arn)
    """
    logger.info("Setting up SageMaker session and role")
    
    # Create SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Get execution role
    try:
        role = sagemaker.get_execution_role()
    except Exception:
        # If running locally, get role from environment or config
        role = os.environ.get('SAGEMAKER_ROLE_ARN')
        if not role:
            raise ValueError("SageMaker execution role not found. Set SAGEMAKER_ROLE_ARN environment variable.")
    
    logger.info(f"Using SageMaker role: {role}")
    logger.info(f"Region: {sagemaker_session.boto_region_name}")
    
    return sagemaker_session, role


def create_pipeline_parameters(config):
    """
    Create pipeline parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Pipeline parameters
    """
    logger.info("Creating pipeline parameters")
    
    params = {
        # Instance types
        'processing_instance_type': ParameterString(
            name='ProcessingInstanceType',
            default_value=config['instance_types']['processing']
        ),
        'training_instance_type': ParameterString(
            name='TrainingInstanceType',
            default_value=config['instance_types']['training']
        ),
        
        # Instance counts
        'processing_instance_count': ParameterInteger(
            name='ProcessingInstanceCount',
            default_value=config['instance_counts']['processing']
        ),
        'training_instance_count': ParameterInteger(
            name='TrainingInstanceCount',
            default_value=config['instance_counts']['training']
        ),
        
        # Hyperparameters
        'n_estimators': ParameterInteger(
            name='NEstimators',
            default_value=config['hyperparameters']['n_estimators']
        ),
        'max_depth': ParameterInteger(
            name='MaxDepth',
            default_value=config['hyperparameters']['max_depth']
        ),
        'random_state': ParameterInteger(
            name='RandomState',
            default_value=config['hyperparameters']['random_state']
        ),
        'test_size': ParameterFloat(
            name='TestSize',
            default_value=config['hyperparameters']['test_size']
        ),
        
        # S3 paths
        'input_data': ParameterString(
            name='InputData',
            default_value=config['s3_paths']['input_dataset']
        ),
        'model_approval_status': ParameterString(
            name='ModelApprovalStatus',
            default_value=config['model_registry']['approval_status']
        )
    }
    
    logger.info(f"Created {len(params)} pipeline parameters")
    return params


def create_preprocessing_step(sagemaker_session, role, config, params):
    """
    Create preprocessing step
    
    Args:
        sagemaker_session: SageMaker session
        role: Execution role ARN
        config: Configuration dictionary
        params: Pipeline parameters
        
    Returns:
        ProcessingStep
    """
    logger.info("Creating preprocessing step")
    
    # Create SKLearn processor
    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        instance_type=params['processing_instance_type'],
        instance_count=params['processing_instance_count'],
        base_job_name='machine-downtime-preprocessing',
        role=role,
        sagemaker_session=sagemaker_session
    )
    
    # Define step
    step_preprocess = ProcessingStep(
        name='PreprocessData',
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=params['input_data'],
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='train',
                source='/opt/ml/processing/train',
                destination=config['s3_paths']['processed_data'] + 'train'
            ),
            ProcessingOutput(
                output_name='test',
                source='/opt/ml/processing/test',
                destination=config['s3_paths']['processed_data'] + 'test'
            )
        ],
        code='src/preprocess.py',
        job_arguments=[
            '--test-size', str(config['hyperparameters']['test_size']),
            '--random-state', str(config['hyperparameters']['random_state'])
        ]
    )
    
    logger.info("Preprocessing step created")
    return step_preprocess


def create_training_step(sagemaker_session, role, config, params, step_preprocess):
    """
    Create training step
    
    Args:
        sagemaker_session: SageMaker session
        role: Execution role ARN
        config: Configuration dictionary
        params: Pipeline parameters
        step_preprocess: Preprocessing step
        
    Returns:
        TrainingStep
    """
    logger.info("Creating training step")
    
    # Create SKLearn estimator
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        source_dir='src',
        framework_version='1.2-1',
        instance_type=params['training_instance_type'],
        instance_count=params['training_instance_count'],
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name='machine-downtime-training',
        hyperparameters={
            'n-estimators': params['n_estimators'],
            'max-depth': params['max_depth'],
            'random-state': params['random_state']
        },
        output_path=config['s3_paths']['model_artifacts']
    )
    
    # Define step
    step_train = TrainingStep(
        name='TrainModel',
        estimator=sklearn_estimator,
        inputs={
            'train': sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )
    
    logger.info("Training step created")
    return step_train


def create_evaluation_step(sagemaker_session, role, config, params, step_preprocess, step_train):
    """
    Create evaluation step
    
    Args:
        sagemaker_session: SageMaker session
        role: Execution role ARN
        config: Configuration dictionary
        params: Pipeline parameters
        step_preprocess: Preprocessing step
        step_train: Training step
        
    Returns:
        ProcessingStep
    """
    logger.info("Creating evaluation step")
    
    # Create SKLearn processor
    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        instance_type=params['processing_instance_type'],
        instance_count=1,
        base_job_name='machine-downtime-evaluation',
        role=role,
        sagemaker_session=sagemaker_session
    )
    
    # Property file for evaluation metrics
    evaluation_report = PropertyFile(
        name='EvaluationReport',
        output_name='evaluation',
        path='evaluation.json'
    )
    
    # Define step
    step_evaluate = ProcessingStep(
        name='EvaluateModel',
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='evaluation',
                source='/opt/ml/processing/evaluation',
                destination=config['s3_paths']['evaluation']
            )
        ],
        code='src/evaluate.py',
        property_files=[evaluation_report]
    )
    
    logger.info("Evaluation step created")
    return step_evaluate


def create_register_model_step(sagemaker_session, role, config, params, step_train, step_evaluate):
    """
    Create model registration step
    
    Args:
        sagemaker_session: SageMaker session
        role: Execution role ARN
        config: Configuration dictionary
        params: Pipeline parameters
        step_train: Training step
        step_evaluate: Evaluation step
        
    Returns:
        RegisterModel step
    """
    logger.info("Creating model registration step")
    
    # Model metrics
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{config['s3_paths']['evaluation']}metrics.json",
            content_type='application/json'
        )
    )
    
    # Register model step
    step_register = RegisterModel(
        name='RegisterModel',
        estimator=None,  # We'll use the model from training step
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=['text/csv', 'application/json'],
        response_types=['text/csv', 'application/json'],
        inference_instances=config['model_registry']['inference_instances'],
        transform_instances=config['model_registry']['transform_instances'],
        model_package_group_name=config['model_registry']['model_group_name'],
        approval_status=params['model_approval_status'],
        model_metrics=model_metrics,
        description=config['model_registry']['model_description']
    )
    
    logger.info("Model registration step created")
    return step_register


def create_pipeline(config_path='config/pipeline_config.yaml'):
    """
    Create complete SageMaker pipeline
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Pipeline object
    """
    logger.info("="*80)
    logger.info("Creating SageMaker Pipeline")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Get session and role
    sagemaker_session, role = get_session_and_role()
    
    # Create pipeline parameters
    params = create_pipeline_parameters(config)
    
    # Create pipeline steps
    step_preprocess = create_preprocessing_step(sagemaker_session, role, config, params)
    step_train = create_training_step(sagemaker_session, role, config, params, step_preprocess)
    step_evaluate = create_evaluation_step(sagemaker_session, role, config, params, step_preprocess, step_train)
    step_register = create_register_model_step(sagemaker_session, role, config, params, step_train, step_evaluate)
    
    # Create pipeline
    pipeline = Pipeline(
        name=config['pipeline']['name'],
        parameters=list(params.values()),
        steps=[step_preprocess, step_train, step_evaluate, step_register],
        sagemaker_session=sagemaker_session
    )
    
    logger.info(f"Pipeline '{config['pipeline']['name']}' created successfully")
    logger.info(f"Pipeline contains {len(pipeline.steps)} steps")
    
    return pipeline


def upsert_pipeline(pipeline):
    """
    Create or update pipeline
    
    Args:
        pipeline: Pipeline object
        
    Returns:
        dict: Pipeline response
    """
    logger.info("Upserting pipeline...")
    
    try:
        response = pipeline.upsert(role_arn=pipeline.role)
        logger.info(f"Pipeline upserted successfully: {response['PipelineArn']}")
        return response
    except Exception as e:
        logger.error(f"Failed to upsert pipeline: {str(e)}")
        raise


def start_pipeline_execution(pipeline, execution_display_name=None):
    """
    Start pipeline execution
    
    Args:
        pipeline: Pipeline object
        execution_display_name: Display name for execution
        
    Returns:
        Pipeline execution
    """
    logger.info("Starting pipeline execution...")
    
    if execution_display_name is None:
        import time
        execution_display_name = f"Execution-{int(time.time())}"
    
    execution = pipeline.start(
        execution_display_name=execution_display_name
    )
    
    logger.info(f"Pipeline execution started: {execution.arn}")
    logger.info(f"Execution display name: {execution_display_name}")
    
    return execution


def main():
    """Main pipeline definition"""
    logger.info("="*80)
    logger.info("Machine Downtime Pipeline - Main Execution")
    logger.info("="*80)
    
    try:
        # Create pipeline
        pipeline = create_pipeline()
        
        # Upsert pipeline
        upsert_response = upsert_pipeline(pipeline)
        
        # Print pipeline details
        logger.info("\nPipeline Details:")
        logger.info(f"  Name: {pipeline.name}")
        logger.info(f"  ARN: {upsert_response['PipelineArn']}")
        logger.info(f"  Steps: {len(pipeline.steps)}")
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline created/updated successfully!")
        logger.info("Use notebooks/run_pipeline.ipynb to execute the pipeline")
        logger.info("="*80)
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Pipeline creation failed: {str(e)}")
        raise


if __name__ == '__main__':
    pipeline = main()
