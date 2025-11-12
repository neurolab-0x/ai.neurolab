# Training API Documentation

Complete guide for using the model training endpoints.

## Overview

The Training API allows administrators to train, retrain, and compare machine learning models directly through the API. All training operations run as background jobs to avoid blocking the API.

## Authentication

All training endpoints require authentication with **admin role**.

```bash
# Login as admin
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin_password"}'

# Use the returned access_token in subsequent requests
Authorization: Bearer <access_token>
```

---

## Endpoints

### 1. Train Model with Data

**POST** `/api/train`

Train a new model with provided training data.

**Request Body:**
```json
{
  "X_train": [
    [0.5, 0.3, 0.2, 0.1, 0.4],
    [0.6, 0.4, 0.3, 0.2, 0.5]
  ],
  "y_train": [0, 1],
  "X_test": [
    [0.4, 0.2, 0.1, 0.05, 0.3]
  ],
  "y_test": [0],
  "config": {
    "model_type": "enhanced_cnn_lstm",
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.3,
    "use_separable": true,
    "use_relative_pos": true,
    "l1_reg": 0.00001,
    "l2_reg": 0.0001,
    "subject_id": "SUBJ001",
    "session_id": "SESS001"
  }
}
```

**Configuration Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| model_type | string | enhanced_cnn_lstm | original, enhanced_cnn_lstm, resnet_lstm, transformer | Model architecture |
| epochs | int | 30 | 1-200 | Number of training epochs |
| batch_size | int | 32 | 1-256 | Batch size |
| learning_rate | float | 0.001 | 0-1 | Learning rate |
| dropout_rate | float | 0.3 | 0-0.9 | Dropout rate |
| use_separable | bool | true | - | Use separable convolutions |
| use_relative_pos | bool | true | - | Use relative positional encoding |
| l1_reg | float | 1e-5 | ≥0 | L1 regularization |
| l2_reg | float | 1e-4 | ≥0 | L2 regularization |
| subject_id | string | null | - | Subject identifier |
| session_id | string | null | - | Session identifier |

**Response:**
```json
{
  "job_id": "train_20240115_103000_admin",
  "status": "queued",
  "message": "Training job started in background",
  "started_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `202`: Training job accepted
- `400`: Invalid data
- `401`: Unauthorized
- `403`: Forbidden (not admin)
- `500`: Server error

**Example:**
```python
import requests

# Login
login_response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"username": "admin", "password": "admin_password"}
)
token = login_response.json()["access_token"]

# Prepare training data
training_data = {
    "X_train": [[0.5, 0.3, 0.2, 0.1, 0.4]] * 100,
    "y_train": [0, 1, 2] * 33 + [0],
    "X_test": [[0.4, 0.2, 0.1, 0.05, 0.3]] * 20,
    "y_test": [0, 1, 2] * 6 + [0, 1],
    "config": {
        "model_type": "enhanced_cnn_lstm",
        "epochs": 10,
        "batch_size": 16
    }
}

# Start training
response = requests.post(
    "http://localhost:8000/api/train",
    json=training_data,
    headers={"Authorization": f"Bearer {token}"}
)

job_id = response.json()["job_id"]
print(f"Training job started: {job_id}")
```

---

### 2. Train Model from File

**POST** `/api/train/file`

Train a model from an uploaded CSV file.

**Request:**
- **Form Data:**
  - `file`: CSV file with EEG data (required)
  - `config`: JSON string with training configuration (optional)

**CSV Format:**
```csv
timestamp,alpha,beta,theta,delta,gamma,state
2024-01-15T10:00:00,0.5,0.3,0.2,0.1,0.4,0
2024-01-15T10:00:01,0.6,0.4,0.3,0.2,0.5,1
```

**Response:**
```json
{
  "job_id": "train_file_20240115_103000_admin",
  "status": "queued",
  "message": "Training job started from file training_data.csv",
  "started_at": "2024-01-15T10:30:00Z"
}
```

**Example:**
```python
import requests

# Prepare file and config
files = {'file': open('training_data.csv', 'rb')}
config = {
    "model_type": "enhanced_cnn_lstm",
    "epochs": 20
}

# Upload and train
response = requests.post(
    "http://localhost:8000/api/train/file",
    files=files,
    data={'config': json.dumps(config)},
    headers={"Authorization": f"Bearer {token}"}
)

job_id = response.json()["job_id"]
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/train/file \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@training_data.csv" \
  -F 'config={"model_type":"enhanced_cnn_lstm","epochs":20}'
```

---

### 3. Get Training Status

**GET** `/api/train/status/{job_id}`

Check the status of a training job.

**Response:**
```json
{
  "job_id": "train_20240115_103000_admin",
  "status": "training",
  "progress": 0.45,
  "message": "Training epoch 14/30",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": null,
  "metrics": null,
  "error": null
}
```

**Status Values:**
- `queued`: Job is waiting to start
- `training`: Model is currently training
- `completed`: Training finished successfully
- `failed`: Training failed with error

**Progress Values:**
- `0.0`: Job queued
- `0.1-0.8`: Training in progress
- `0.8-1.0`: Evaluating model
- `1.0`: Completed

**Example:**
```python
# Poll for status
import time

while True:
    response = requests.get(
        f"http://localhost:8000/api/train/status/{job_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    status_data = response.json()
    print(f"Status: {status_data['status']} - Progress: {status_data['progress']:.1%}")
    
    if status_data['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)

# Get final metrics
if status_data['status'] == 'completed':
    print("Training Metrics:")
    print(json.dumps(status_data['metrics'], indent=2))
```

---

### 4. List Training Jobs

**GET** `/api/train/jobs`

List training jobs for the current user (or all jobs for admins).

**Query Parameters:**
- `limit`: Maximum number of jobs to return (default: 10)

**Response:**
```json
[
  {
    "job_id": "train_20240115_103000_admin",
    "status": "completed",
    "progress": 1.0,
    "message": "Training completed successfully",
    "started_at": "2024-01-15T10:30:00Z",
    "completed_at": "2024-01-15T10:45:00Z",
    "metrics": {
      "final_train_accuracy": 0.95,
      "final_val_accuracy": 0.89,
      "final_train_loss": 0.15,
      "final_val_loss": 0.28
    },
    "error": null
  }
]
```

**Example:**
```python
response = requests.get(
    "http://localhost:8000/api/train/jobs?limit=5",
    headers={"Authorization": f"Bearer {token}"}
)

jobs = response.json()
for job in jobs:
    print(f"{job['job_id']}: {job['status']}")
```

---

### 5. Delete Training Job

**DELETE** `/api/train/job/{job_id}`

Delete a training job record (admin only).

**Response:**
```json
{
  "status": "success",
  "message": "Training job train_20240115_103000_admin deleted"
}
```

**Example:**
```python
response = requests.delete(
    f"http://localhost:8000/api/train/job/{job_id}",
    headers={"Authorization": f"Bearer {token}"}
)
```

---

### 6. Compare Models

**POST** `/api/train/compare`

Compare multiple model architectures.

**Request Body:**
```json
{
  "X_train": [[0.5, 0.3, 0.2, 0.1, 0.4]],
  "y_train": [0],
  "X_test": [[0.4, 0.2, 0.1, 0.05, 0.3]],
  "y_test": [0],
  "n_repeats": 3
}
```

**Note:** Test data is required for model comparison.

**Response:**
```json
{
  "job_id": "compare_20240115_103000_admin",
  "status": "queued",
  "message": "Model comparison started in background",
  "started_at": "2024-01-15T10:30:00Z"
}
```

**Comparison Results:**
```json
{
  "original": {
    "accuracy": [0.85, 0.87, 0.86],
    "auc": [0.88, 0.89, 0.88],
    "training_time": [45.2, 44.8, 45.5],
    "inference_time": [0.012, 0.011, 0.012],
    "model_size": [12.5, 12.5, 12.5]
  },
  "enhanced_cnn_lstm": {
    "accuracy": [0.92, 0.93, 0.91],
    "auc": [0.94, 0.95, 0.93],
    "training_time": [78.3, 79.1, 77.8],
    "inference_time": [0.018, 0.019, 0.018],
    "model_size": [24.8, 24.8, 24.8]
  }
}
```

---

## Complete Workflow Example

```python
import requests
import json
import time

# 1. Login
login_response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"username": "admin", "password": "admin_password"}
)
token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload training file
with open('training_data.csv', 'rb') as f:
    train_response = requests.post(
        "http://localhost:8000/api/train/file",
        files={'file': f},
        data={'config': json.dumps({"epochs": 20, "batch_size": 32})},
        headers=headers
    )

job_id = train_response.json()["job_id"]
print(f"Training started: {job_id}")

# 3. Monitor progress
while True:
    status_response = requests.get(
        f"http://localhost:8000/api/train/status/{job_id}",
        headers=headers
    )
    
    status = status_response.json()
    print(f"Progress: {status['progress']:.1%} - {status['message']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(10)

# 4. Check results
if status['status'] == 'completed':
    print("\nTraining completed successfully!")
    print(f"Final accuracy: {status['metrics']['final_val_accuracy']:.2%}")
    print(f"Model saved to: processed/trained_model_{config['model_type']}.h5")
else:
    print(f"\nTraining failed: {status['error']}")

# 5. List all jobs
jobs_response = requests.get(
    "http://localhost:8000/api/train/jobs",
    headers=headers
)
print(f"\nTotal jobs: {len(jobs_response.json())}")
```

---

## Best Practices

### Data Preparation
1. **Normalize features** before training
2. **Balance classes** to avoid bias
3. **Split data** properly (80/20 train/test)
4. **Validate data** format and values

### Training Configuration
1. **Start with defaults** and tune incrementally
2. **Use early stopping** to prevent overfitting
3. **Monitor validation loss** during training
4. **Save best model** based on validation metrics

### Resource Management
1. **Limit concurrent jobs** to avoid resource exhaustion
2. **Clean up old jobs** regularly
3. **Monitor memory usage** during training
4. **Use appropriate batch sizes** for your hardware

### Model Selection
1. **Start with enhanced_cnn_lstm** (best balance)
2. **Use original** for quick prototyping
3. **Use resnet_lstm** for deeper networks
4. **Use transformer** for attention-based learning

---

## Troubleshooting

### Training Fails Immediately
- Check data format and dimensions
- Verify labels are valid (0, 1, 2)
- Ensure sufficient training samples (>50)

### Training Stalls
- Reduce batch size
- Lower learning rate
- Check for NaN/Inf values in data

### Poor Performance
- Increase epochs
- Adjust learning rate
- Try different model architecture
- Check data quality and balance

### Out of Memory
- Reduce batch size
- Use smaller model architecture
- Reduce number of epochs
- Clear old training jobs

---

## Rate Limits

Training endpoints have special rate limits:
- **Training jobs**: 5 per hour per user
- **Status checks**: 60 per minute
- **Job listing**: 10 per minute

---

## Support

For issues with training:
- Check logs: `docker-compose logs neurolab-api`
- Review job status for error messages
- Contact support: support@neurolab.cc
