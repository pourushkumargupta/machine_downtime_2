# ⚡ Quick Start Guide
## Get Your MLOps Pipeline Running in 5 Minutes

---

## 📋 Prerequisites Checklist

- [ ] AWS Account with SageMaker access
- [ ] IAM Role with SageMaker permissions
- [ ] Python 3.8+ installed
- [ ] AWS CLI configured
- [ ] S3 bucket: `machine-downtime-2` created
- [ ] Dataset uploaded to: `s3://machine-downtime-2/Machine_downtime_cleaned.csv`

---

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### Step 2: Configure AWS (1 minute)

```bash
# If not already configured
aws configure

# Verify configuration
aws sts get-caller-identity
```

### Step 3: Create Pipeline (1 minute)

```bash
python pipeline/pipeline_definition.py
```

Expected output:
```
Pipeline 'MachineDowntimePipeline' created successfully
Pipeline ARN: arn:aws:sagemaker:...
```

### Step 4: Start Execution (1 minute)

**Option A: Using Jupyter Notebook**
```bash
jupyter notebook notebooks/run_pipeline.ipynb
# Run all cells
```

**Option B: Using Python**
```bash
python -c "
from pipeline.pipeline_definition import create_pipeline, start_pipeline_execution
pipeline = create_pipeline()
execution = start_pipeline_execution(pipeline)
print(f'✓ Started! Execution ARN: {execution.arn}')
"
```

**Option C: Using AWS CLI**
```bash
aws sagemaker start-pipeline-execution \
    --pipeline-name MachineDowntimePipeline \
    --region us-east-1
```

### Step 5: Monitor (1 minute setup, then wait 15-20 min)

```bash
# View in SageMaker Console
echo "Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/pipelines/MachineDowntimePipeline"

# Or use AWS CLI
aws sagemaker list-pipeline-executions \
    --pipeline-name MachineDowntimePipeline \
    --max-results 1
```

---

## 📊 Expected Timeline

```
┌──────────────────────────────────────────────┐
│ Pipeline Execution Timeline (15-20 minutes)  │
├──────────────────────────────────────────────┤
│ Preprocessing:     3-5 minutes               │
│ Training:          5-8 minutes               │
│ Evaluation:        2-3 minutes               │
│ Registration:      1-2 minutes               │
└──────────────────────────────────────────────┘
```

---

## ✅ Success Checklist

After pipeline completes, verify:

- [ ] All 4 steps show "Succeeded"
- [ ] Model registered in Model Registry
- [ ] Metrics available in evaluation output
- [ ] Model status: PendingManualApproval

---

## 🎯 What You Get

After successful execution:

1. **Processed Data**
   - `s3://machine-downtime-2/processed/train/train.csv`
   - `s3://machine-downtime-2/processed/test/test.csv`

2. **Trained Model**
   - `s3://machine-downtime-2/model-artifacts/model.tar.gz`

3. **Evaluation Metrics**
   - `s3://machine-downtime-2/evaluation/evaluation.json`
   - Accuracy: ~88-92%
   - F1-Score: >0.85
   - Recall: >0.90

4. **Model Registry Entry**
   - Model Package ARN
   - Status: PendingManualApproval
   - Attached metrics

---

## 🔄 Quick Commands Reference

```bash
# Create/Update Pipeline
python pipeline/pipeline_definition.py

# Start Execution
aws sagemaker start-pipeline-execution \
    --pipeline-name MachineDowntimePipeline

# Check Status
aws sagemaker list-pipeline-executions \
    --pipeline-name MachineDowntimePipeline \
    --max-results 1

# View Model Registry
aws sagemaker list-model-packages \
    --model-package-group-name machine-downtime-model-group

# Approve Model
aws sagemaker update-model-package \
    --model-package-arn <YOUR_MODEL_ARN> \
    --model-approval-status Approved
```

---

## 🆘 Quick Troubleshooting

### Problem: "No module named 'sagemaker'"
**Solution:**
```bash
pip install sagemaker boto3
```

### Problem: "AccessDenied" error
**Solution:**
```bash
# Verify IAM role has SageMaker permissions
aws iam get-role --role-name YourSageMakerRole
```

### Problem: "Pipeline not found"
**Solution:**
```bash
# Create pipeline first
python pipeline/pipeline_definition.py
```

### Problem: Dataset not found
**Solution:**
```bash
# Upload dataset to S3
aws s3 cp Machine_downtime_cleaned.csv \
    s3://machine-downtime-2/Machine_downtime_cleaned.csv
```

---

## 📞 Need Help?

1. Check logs:
   ```bash
   # CloudWatch Logs
   aws logs tail /aws/sagemaker/ProcessingJobs --follow
   ```

2. Review README.md for detailed documentation

3. Check PROJECT_SUMMARY.md for architecture details

---

## 🎉 Next Steps After First Run

1. **Approve the Model**
   ```bash
   aws sagemaker update-model-package \
       --model-package-arn <ARN> \
       --model-approval-status Approved
   ```

2. **Deploy to Endpoint**
   ```python
   from sagemaker import ModelPackage
   model = ModelPackage(role=role, model_package_arn='<ARN>')
   predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.large')
   ```

3. **Make Predictions**
   ```python
   predictions = predictor.predict(test_data)
   ```

4. **Setup CI/CD**
   - Add AWS credentials to GitHub Secrets
   - Push code to trigger automated pipeline

---

**⏱️ Total Time: 5 minutes setup + 15-20 minutes execution = 20-25 minutes to full pipeline!**

**🚀 You're ready to go! Start with Step 1 above.**
