# EEG Data Format Specification

Complete guide for preparing and formatting EEG data for training and analysis in NeuroLab.

## Supported File Formats

NeuroLab supports the following file formats:

1. **CSV** (Comma-Separated Values) - Recommended for training
2. **EDF** (European Data Format) - Standard EEG format
3. **BDF** (BioSemi Data Format) - BioSemi device format
4. **JSON** (JavaScript Object Notation) - For API submissions

---

## CSV Format (Recommended for Training)

### Required Columns

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `alpha` | float | 0.0 - 1.0 | Alpha wave power (8-13 Hz) |
| `beta` | float | 0.0 - 1.0 | Beta wave power (13-30 Hz) |
| `theta` | float | 0.0 - 1.0 | Theta wave power (4-8 Hz) |
| `delta` | float | 0.0 - 1.0 | Delta wave power (0.5-4 Hz) |
| `gamma` | float | 0.0 - 1.0 | Gamma wave power (30-100 Hz) |
| `state` | int | 0, 1, or 2 | Mental state label |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `confidence` | float | Model confidence (0.0 - 1.0) |
| `timestamp` | string | ISO 8601 timestamp |
| `metadata` | string | JSON string with additional info |
| `subject_id` | string | Subject identifier |
| `session_id` | string | Session identifier |
| `device` | string | Recording device name |
| `signal_quality` | float | Signal quality (0.0 - 1.0) |

### Mental State Labels

| Label | Value | Description |
|-------|-------|-------------|
| Relaxed/Calm | 0 | Low stress, high alpha waves |
| Focused/Attentive | 1 | Moderate activity, balanced waves |
| Stressed/Anxious | 2 | High beta waves, elevated activity |

### Example CSV Format

```csv
alpha,beta,theta,delta,gamma,state,confidence,timestamp,metadata
0.647,0.080,0.219,0.132,0.117,0,0.869,2025-11-12T07:07:28.116583,"{""device"":""synthetic"",""signal_quality"":0.87}"
0.407,0.424,0.290,0.300,0.153,1,0.830,2025-11-12T07:07:29.116583,"{""device"":""synthetic"",""signal_quality"":0.83}"
0.502,0.266,0.323,0.278,0.176,1,0.769,2025-11-12T07:07:30.116583,"{""device"":""synthetic"",""signal_quality"":0.77}"
0.835,0.147,0.076,0.146,0.169,0,0.825,2025-11-12T07:07:31.116583,"{""device"":""synthetic"",""signal_quality"":0.83}"
0.231,0.598,0.517,0.487,0.336,2,0.826,2025-11-12T07:07:32.116583,"{""device"":""synthetic"",""signal_quality"":0.83}"
```

### Minimal CSV Format (Training Only)

```csv
alpha,beta,theta,delta,gamma,state
0.647,0.080,0.219,0.132,0.117,0
0.407,0.424,0.290,0.300,0.153,1
0.502,0.266,0.323,0.278,0.176,1
0.835,0.147,0.076,0.146,0.169,0
0.231,0.598,0.517,0.487,0.336,2
```

---

## JSON Format (API Submissions)

### Single Sample

```json
{
  "alpha": 0.647,
  "beta": 0.080,
  "theta": 0.219,
  "delta": 0.132,
  "gamma": 0.117,
  "state": 0,
  "confidence": 0.869,
  "timestamp": "2025-11-12T07:07:28.116583",
  "subject_id": "SUBJ001",
  "session_id": "SESS001",
  "metadata": {
    "device": "synthetic",
    "signal_quality": 0.87
  }
}
```

### Multiple Samples (Training)

```json
{
  "X_train": [
    [0.647, 0.080, 0.219, 0.132, 0.117],
    [0.407, 0.424, 0.290, 0.300, 0.153],
    [0.502, 0.266, 0.323, 0.278, 0.176]
  ],
  "y_train": [0, 1, 1],
  "X_test": [
    [0.835, 0.147, 0.076, 0.146, 0.169]
  ],
  "y_test": [0]
}
```

---

## EDF/BDF Format

### Structure

EDF and BDF files contain:
- **Header**: Patient info, recording details, channel configuration
- **Data Records**: Time-series EEG data for each channel
- **Annotations**: Event markers and timestamps

### Channel Names

Standard 10-20 system channel names:
- Frontal: Fp1, Fp2, F3, F4, F7, F8, Fz
- Central: C3, C4, Cz
- Temporal: T3, T4, T5, T6
- Parietal: P3, P4, Pz
- Occipital: O1, O2, Oz
- Reference: A1, A2, M1, M2

### Conversion to CSV

NeuroLab automatically converts EDF/BDF files to the required format:

1. **Load file** using MNE library
2. **Extract channels** (8, 16, 32, or 64 channels)
3. **Compute band powers** (alpha, beta, theta, delta, gamma)
4. **Normalize values** to 0-1 range
5. **Generate CSV** with required columns

---

## Data Validation Rules

### Value Ranges

| Feature | Minimum | Maximum | Notes |
|---------|---------|---------|-------|
| alpha | 0.0 | 1.0 | Normalized power |
| beta | 0.0 | 1.0 | Normalized power |
| theta | 0.0 | 1.0 | Normalized power |
| delta | 0.0 | 1.0 | Normalized power |
| gamma | 0.0 | 1.0 | Normalized power |
| state | 0 | 2 | Integer only |
| confidence | 0.0 | 1.0 | Optional |
| signal_quality | 0.0 | 1.0 | Optional |

### Data Quality Checks

✅ **Valid Data:**
- All values are finite (no NaN or Inf)
- Values within specified ranges
- Consistent number of features per sample
- At least 50 samples for training
- Balanced class distribution (recommended)

❌ **Invalid Data:**
- Missing required columns
- NaN or Inf values
- Values outside valid ranges
- Inconsistent feature dimensions
- Empty or corrupted files

---

## Data Preparation Guidelines

### 1. Data Collection

```python
# Example: Collecting EEG data
import numpy as np
import pandas as pd
from datetime import datetime

# Collect raw EEG data
raw_data = {
    'timestamp': [],
    'alpha': [],
    'beta': [],
    'theta': [],
    'delta': [],
    'gamma': [],
    'state': []
}

# Add samples
for i in range(100):
    raw_data['timestamp'].append(datetime.now().isoformat())
    raw_data['alpha'].append(np.random.rand())
    raw_data['beta'].append(np.random.rand())
    raw_data['theta'].append(np.random.rand())
    raw_data['delta'].append(np.random.rand())
    raw_data['gamma'].append(np.random.rand())
    raw_data['state'].append(np.random.randint(0, 3))

# Create DataFrame
df = pd.DataFrame(raw_data)

# Save to CSV
df.to_csv('training_data.csv', index=False)
```

### 2. Data Normalization

```python
# Normalize features to 0-1 range
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = ['alpha', 'beta', 'theta', 'delta', 'gamma']
df[features] = scaler.fit_transform(df[features])
```

### 3. Class Balancing

```python
# Check class distribution
print(df['state'].value_counts())

# Balance classes using oversampling
from imblearn.over_sampling import SMOTE

X = df[features].values
y = df['state'].values

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

### 4. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## File Size Limits

| Format | Maximum Size | Maximum Samples | Notes |
|--------|-------------|-----------------|-------|
| CSV | 500 MB | 100,000 | Recommended for training |
| EDF | 500 MB | Varies | Depends on channels/duration |
| BDF | 500 MB | Varies | Depends on channels/duration |
| JSON | 50 MB | 10,000 | For API submissions |

---

## Example Data Generation Script

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_training_data(n_samples=1000, output_file='training_data.csv'):
    """
    Generate synthetic EEG training data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_file : str
        Output CSV file path
    """
    np.random.seed(42)
    
    data = {
        'alpha': [],
        'beta': [],
        'theta': [],
        'delta': [],
        'gamma': [],
        'state': [],
        'confidence': [],
        'timestamp': [],
        'metadata': []
    }
    
    start_time = datetime.now()
    
    for i in range(n_samples):
        # Generate state (0: relaxed, 1: focused, 2: stressed)
        state = np.random.randint(0, 3)
        
        # Generate features based on state
        if state == 0:  # Relaxed
            alpha = np.random.uniform(0.6, 0.9)
            beta = np.random.uniform(0.0, 0.3)
            theta = np.random.uniform(0.1, 0.3)
            delta = np.random.uniform(0.0, 0.2)
            gamma = np.random.uniform(0.0, 0.2)
        elif state == 1:  # Focused
            alpha = np.random.uniform(0.3, 0.7)
            beta = np.random.uniform(0.2, 0.6)
            theta = np.random.uniform(0.1, 0.4)
            delta = np.random.uniform(0.1, 0.3)
            gamma = np.random.uniform(0.1, 0.3)
        else:  # Stressed
            alpha = np.random.uniform(0.1, 0.4)
            beta = np.random.uniform(0.5, 0.9)
            theta = np.random.uniform(0.3, 0.6)
            delta = np.random.uniform(0.2, 0.5)
            gamma = np.random.uniform(0.2, 0.4)
        
        # Add to data
        data['alpha'].append(alpha)
        data['beta'].append(beta)
        data['theta'].append(theta)
        data['delta'].append(delta)
        data['gamma'].append(gamma)
        data['state'].append(state)
        data['confidence'].append(np.random.uniform(0.7, 0.99))
        data['timestamp'].append((start_time + timedelta(seconds=i)).isoformat())
        data['metadata'].append(f'{{"device":"synthetic","signal_quality":{np.random.uniform(0.7, 0.99):.2f}}}')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {n_samples} samples and saved to {output_file}")
    
    # Print statistics
    print(f"\nClass distribution:")
    print(df['state'].value_counts())
    print(f"\nFeature statistics:")
    print(df[['alpha', 'beta', 'theta', 'delta', 'gamma']].describe())

# Generate training data
generate_training_data(n_samples=1000, output_file='training_data.csv')
```

---

## Common Issues and Solutions

### Issue 1: Missing Columns

**Error:** `KeyError: 'alpha'`

**Solution:**
```python
# Check required columns
required_columns = ['alpha', 'beta', 'theta', 'delta', 'gamma', 'state']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

### Issue 2: Invalid Values

**Error:** `ValueError: Input contains NaN`

**Solution:**
```python
# Remove NaN values
df = df.dropna()

# Or fill with mean
df = df.fillna(df.mean())

# Check for Inf
df = df.replace([np.inf, -np.inf], np.nan).dropna()
```

### Issue 3: Imbalanced Classes

**Warning:** `Class imbalance detected`

**Solution:**
```python
from imblearn.over_sampling import SMOTE

# Balance classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

### Issue 4: Wrong Data Types

**Error:** `TypeError: float() argument must be a string or a number`

**Solution:**
```python
# Convert to correct types
df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
df['beta'] = pd.to_numeric(df['beta'], errors='coerce')
df['state'] = df['state'].astype(int)
```

---

## Best Practices

### Data Collection
1. ✅ Use consistent sampling rate (e.g., 256 Hz)
2. ✅ Record in controlled environment
3. ✅ Minimize artifacts (eye blinks, muscle movement)
4. ✅ Use proper electrode placement (10-20 system)
5. ✅ Record sufficient duration (minimum 5 minutes per state)

### Data Preprocessing
1. ✅ Remove artifacts before feature extraction
2. ✅ Apply band-pass filtering (0.5-50 Hz)
3. ✅ Normalize features to 0-1 range
4. ✅ Balance class distribution
5. ✅ Split data properly (80/20 train/test)

### Data Quality
1. ✅ Check for missing values
2. ✅ Validate value ranges
3. ✅ Verify class labels
4. ✅ Ensure sufficient samples (>50 per class)
5. ✅ Monitor signal quality

---

## Validation Checklist

Before uploading data for training:

- [ ] File format is CSV, EDF, or BDF
- [ ] All required columns are present
- [ ] No missing values (NaN)
- [ ] No infinite values (Inf)
- [ ] Values are within valid ranges (0-1 for features)
- [ ] State labels are 0, 1, or 2
- [ ] At least 50 samples per class
- [ ] Classes are reasonably balanced
- [ ] File size is under 500 MB
- [ ] Data is properly normalized

---

## Support

For data format questions:
- Check examples in `test_data/` directory
- Review API documentation: `/docs`
- Contact support: support@neurolab.cc
