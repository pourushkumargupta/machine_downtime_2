# Machine Downtime Prediction - AWS SageMaker MLOps Pipeline

Complete end-to-end MLOps pipeline for predicting machine downtime using AWS SageMaker.

## 🎯 Project Overview

This project implements a production-ready machine learning pipeline with:
- **Data Preprocessing**: Automated data cleaning and feature engineering
- **Model Training**: RandomForestClassifier with optimized hyperparameters
- **Model Evaluation**: Comprehensive metrics computation
- **Model Registry**: Automated model registration with approval workflow

## 📊 Dataset

- **Source**: `s3://machine-downtime-2/Machine_downtime_cleaned.csv`
- **Target**: Predict machine downtime events
- **Features**: 16 parameters including hydraulic pressure, cutting force, vibrations, etc.

## 📁 Project Structure

```
mlops-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/
│   └── pipeline_config.yaml          # Pipeline configuration
├── src/
│   ├── preprocess.py                 # Data preprocessing script
│   ├── train.py                      # Model training script
│   ├── evaluate.py                   # Model evaluation script
│   └── register_model.py             # Model registration script
├── pipeline/
│   └── pipeline_definition.py        # SageMaker Pipeline definition
├── notebooks/
│   └── run_pipeline.ipynb            # Pipeline execution notebook
└── .github/
    └── workflows/
        └── pipeline_trigger.yml      # CI/CD automation
```

## 🚀 Quick Start

### Prerequisites

1. **AWS Account** with SageMaker access
2. **IAM Role** with permissions for:
   - SageMaker (full access)
   - S3 (read/write to machine-downtime-2 bucket)
   - Model Registry access
3. **Python 3.8+**

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-project

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

### Running the Pipeline

#### Option 1: Using Jupyter Notebook (Recommended for first run)

```bash
jupyter notebook notebooks/run_pipeline.ipynb
```

Follow the notebook cells to:
1. Load configuration
2. Create/update the pipeline
3. Start execution
4. Monitor progress
5. View registered model

#### Option 2: Using Python Script

```python
from pipeline.pipeline_definition import create_pipeline
import boto3

# Create pipeline
pipeline = create_pipeline()

# Start execution
execution = pipeline.start()
print(f"Pipeline Execution ARN: {execution.arn}")

# Wait for completion
execution.wait()
print(f"Pipeline Status: {execution.describe()['PipelineExecutionStatus']}")
```

#### Option 3: Using AWS CLI

```bash
# Create/Update pipeline
python pipeline/pipeline_definition.py

# Start execution
aws sagemaker start-pipeline-execution \
    --pipeline-name MachineDowntimePipeline \
    --region us-east-1
```

## 🔧 Configuration

All configuration is centralized in `config/pipeline_config.yaml`:

```yaml
s3_paths:
  raw_data: s3://machine-downtime-2/raw/
  processed_data: s3://machine-downtime-2/processed/
  model_artifacts: s3://machine-downtime-2/model-artifacts/
  evaluation: s3://machine-downtime-2/evaluation/
  
model_registry:
  model_group_name: machine-downtime-model-group
  
instance_types:
  processing: ml.m5.large
  training: ml.m5.large
  evaluation: ml.m5.large
  
hyperparameters:
  n_estimators: 200
  max_depth: 10
  random_state: 42
```

## 📊 Pipeline Steps

### Step 1: Data Preprocessing
- **Script**: `src/preprocess.py`
- **Instance**: ml.m5.large
- **Input**: Raw CSV from S3
- **Output**: Cleaned train/test CSVs
- **Operations**:
  - Data cleaning and validation
  - Feature engineering
  - Label creation (Downtime: yes/no → 1/0)
  - Train/test split (80/20)

### Step 2: Model Training
- **Script**: `src/train.py`
- **Instance**: ml.m5.large
- **Algorithm**: RandomForestClassifier
- **Hyperparameters**:
  - n_estimators: 200
  - max_depth: 10
  - random_state: 42
- **Output**: Trained model (model.joblib)

### Step 3: Model Evaluation
- **Script**: `src/evaluate.py`
- **Instance**: ml.m5.large
- **Metrics Computed**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- **Output**: metrics.json

### Step 4: Model Registration
- **Script**: `src/register_model.py`
- **Model Group**: machine-downtime-model-group
- **Status**: PendingManualApproval
- **Includes**: Model artifacts + evaluation metrics

## 📈 Monitoring & Logging

### View Pipeline Execution

```python
import boto3

client = boto3.client('sagemaker')

# List executions
response = client.list_pipeline_executions(
    PipelineName='MachineDowntimePipeline'
)

for execution in response['PipelineExecutionSummaries']:
    print(f"Execution ARN: {execution['PipelineExecutionArn']}")
    print(f"Status: {execution['PipelineExecutionStatus']}")
    print(f"Start Time: {execution['StartTime']}")
```

### View Model Registry

```python
# List model packages
response = client.list_model_packages(
    ModelPackageGroupName='machine-downtime-model-group'
)

for package in response['ModelPackageSummaryList']:
    print(f"Model ARN: {package['ModelPackageArn']}")
    print(f"Status: {package['ModelApprovalStatus']}")
```

## 🔄 CI/CD Integration

The project includes GitHub Actions workflow for automated pipeline execution:

**Trigger**: Push to `main` branch  
**Workflow**: `.github/workflows/pipeline_trigger.yml`

### Setup GitHub Secrets

Add these secrets to your GitHub repository:

```
AWS_ACCESS_KEY_ID: <your-access-key>
AWS_SECRET_ACCESS_KEY: <your-secret-key>
AWS_REGION: us-east-1
```

Or use OIDC for better security (recommended).

## 🎯 Model Approval Workflow

1. Pipeline completes successfully
2. Model registered with status: `PendingManualApproval`
3. Review metrics in Model Registry
4. Approve or reject model:

```python
client.update_model_package(
    ModelPackageArn='arn:aws:sagemaker:...',
    ModelApprovalStatus='Approved'  # or 'Rejected'
)
```

5. Deploy approved models to endpoints

## 📊 Expected Results

Based on the analysis:
- **Accuracy**: 88-92%
- **F1-Score**: >0.85
- **Recall**: >0.90
- **Training Time**: ~5-10 minutes
- **Total Pipeline Duration**: ~15-20 minutes

## 🔍 Troubleshooting

### Common Issues

**Issue**: Pipeline fails at preprocessing step  
**Solution**: Check S3 bucket permissions and CSV file path

**Issue**: Training takes too long  
**Solution**: Use larger instance type (ml.m5.xlarge) or reduce n_estimators

**Issue**: Model registration fails  
**Solution**: Verify IAM role has ModelRegistry permissions

### Logs Location

- **CloudWatch Logs**: `/aws/sagemaker/ProcessingJobs` and `/aws/sagemaker/TrainingJobs`
- **S3 Logs**: `s3://machine-downtime-2/logs/`

## 🚀 Next Steps

1. **Deploy Model**: Create SageMaker endpoint from approved model
2. **Monitoring**: Set up CloudWatch alarms for endpoint metrics
3. **Retraining**: Schedule periodic pipeline executions
4. **A/B Testing**: Deploy multiple model versions for comparison
5. **Data Drift Detection**: Monitor input data distribution

## 📝 License

This project is licensed under the MIT License.

## 👥 Contributors

- **Bharani Kumar** - Garbha.ai / 360DigiTMG / AiSPRY

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Contact: [Your contact information]

---

**Last Updated**: November 15, 2025  
**Pipeline Version**: 1.0  
**SageMaker SDK Version**: 2.x
