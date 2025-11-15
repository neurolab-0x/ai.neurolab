# 🧠 NeuroLab: Algorithms and Models Documentation

## Overview

This document provides a comprehensive overview of all algorithms, models, and techniques used in the NeuroLab EEG Analysis Platform.

---

## 📊 Table of Contents

1. [Deep Learning Models](#deep-learning-models)
2. [Signal Processing Algorithms](#signal-processing-algorithms)
3. [Feature Extraction](#feature-extraction)
4. [Preprocessing Techniques](#preprocessing-techniques)
5. [Model Interpretability](#model-interpretability)
6. [Optimization Techniques](#optimization-techniques)
7. [Evaluation Metrics](#evaluation-metrics)

---

## 🤖 Deep Learning Models

### 1. **Enhanced CNN-LSTM (Default)**

**Architecture:**
```
Input (5 features, 1 channel)
    ↓
GaussianNoise (0.01) - Regularization
    ↓
SeparableConv1D (32 filters, kernel=3, padding='same')
    ↓
BatchNormalization
    ↓
MaxPooling1D (pool_size=2)
    ↓
SpatialDropout1D (0.3)
    ↓
SeparableConv1D (64 filters, kernel=3, padding='same')
    ↓
BatchNormalization
    ↓
MaxPooling1D (pool_size=2)
    ↓
SpatialDropout1D (0.3)
    ↓
SeparableConv1D (128 filters, kernel=3, padding='same')
    ↓
BatchNormalization
    ↓
SpatialDropout1D (0.3)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Multi-Head Attention (4 heads)
    ↓
LayerNormalization
    ↓
GlobalAveragePooling1D
    ↓
Dense (128 units, activation='relu')
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (3 units, activation='softmax')
```

**Key Features:**
- **Separable Convolutions**: Reduces parameters while maintaining performance
- **Bidirectional LSTM**: Captures temporal dependencies in both directions
- **Multi-Head Attention**: Focuses on important features
- **Spatial Dropout**: Better regularization for sequential data
- **Batch Normalization**: Stabilizes training
- **L1/L2 Regularization**: Prevents overfitting

**Use Case:** Default model for EEG state classification

---

### 2. **ResNet-LSTM**

**Architecture:**
```
Input → Residual Blocks → LSTM → Attention → Dense → Output
```

**Residual Block Structure:**
```
Input
  ↓
SeparableConv1D + BatchNorm + ReLU + SpatialDropout
  ↓
SeparableConv1D + BatchNorm
  ↓
Add (Skip Connection)
  ↓
ReLU
```

**Key Features:**
- **Residual Connections**: Enables deeper networks without vanishing gradients
- **Skip Connections**: Preserves information flow
- **Separable Convolutions**: Efficient parameter usage

**Use Case:** When deeper networks are needed for complex patterns

---

### 3. **Transformer Model**

**Architecture:**
```
Input
  ↓
Positional Encoding
  ↓
Transformer Block (Multi-Head Attention + Feed-Forward)
  ↓
LayerNormalization
  ↓
GlobalAveragePooling
  ↓
Dense Layers
  ↓
Output
```

**Transformer Block:**
- **Multi-Head Self-Attention**: Captures relationships between all time steps
- **Relative Positional Encoding**: Better temporal understanding
- **Feed-Forward Network**: GELU activation for better gradients
- **Residual Connections**: Two per block
- **Layer Normalization**: Stabilizes training

**Key Features:**
- **Attention Mechanism**: Focuses on relevant time steps
- **Parallel Processing**: Faster than RNNs
- **Long-Range Dependencies**: Better than LSTMs for long sequences

**Use Case:** For capturing complex temporal patterns in EEG data

---

### 4. **Original CNN-LSTM (Legacy)**

**Architecture:**
```
Input → Conv1D → MaxPooling → BatchNorm → LSTM → Dense → Output
```

**Key Features:**
- Simpler architecture
- Faster training
- Good baseline performance

**Use Case:** Quick prototyping and baseline comparisons

---

## 🔊 Signal Processing Algorithms

### 1. **Frequency Band Extraction**

**Bands:**
- **Delta (0.5-4 Hz)**: Deep sleep, unconscious processes
- **Theta (4-8 Hz)**: Drowsiness, meditation, creativity
- **Alpha (8-13 Hz)**: Relaxation, calmness, closed eyes
- **Beta (13-30 Hz)**: Active thinking, focus, anxiety
- **Gamma (25-100 Hz)**: High-level information processing

**Method:** Bandpass filtering using Butterworth filters

---

### 2. **Artifact Removal**

**Techniques:**
- **Baseline Correction**: Removes DC offset
- **Notch Filtering**: Removes power line noise (50/60 Hz)
- **Outlier Detection**: Statistical methods to identify artifacts
- **Wavelet Denoising**: Removes high-frequency noise

**Implementation:**
```python
from utils.artifacts import clean_eeg
cleaned_data = clean_eeg(raw_data)
```

---

### 3. **Filtering**

**Filters Used:**
- **Butterworth Bandpass Filter**: For frequency band extraction
- **Notch Filter**: For power line noise removal
- **Low-pass Filter**: For smoothing
- **High-pass Filter**: For baseline drift removal

**Parameters:**
- Order: 4
- Sampling Rate: 250 Hz (default)
- Zero-phase filtering (filtfilt)

---

## 🎯 Feature Extraction

### 1. **Time-Domain Features**

- **Mean**: Average signal amplitude
- **Standard Deviation**: Signal variability
- **Variance**: Signal power
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution peakedness
- **Zero-Crossing Rate**: Signal complexity

---

### 2. **Frequency-Domain Features**

- **Power Spectral Density (PSD)**: Power distribution across frequencies
- **Band Power**: Power in specific frequency bands
- **Spectral Entropy**: Frequency distribution complexity
- **Peak Frequency**: Dominant frequency in each band

**Method:** Welch's method for PSD estimation

---

### 3. **Statistical Features**

- **Hjorth Parameters**:
  - Activity: Signal variance
  - Mobility: Mean frequency
  - Complexity: Change in frequency
  
- **Entropy Measures**:
  - Shannon Entropy
  - Approximate Entropy
  - Sample Entropy

---

### 4. **Cognitive Metrics**

Calculated from frequency bands:

```python
attention_index = beta / (theta + alpha)
relaxation_index = alpha / beta
stress_index = (beta + theta) / alpha
cognitive_load = (beta + gamma) / (alpha + theta)
mental_fatigue = theta / (alpha + beta)
alertness = (beta + gamma) / (delta + theta)
```

---

## 🔧 Preprocessing Techniques

### 1. **Data Normalization**

**Methods:**
- **Z-Score Normalization**: `(X - mean) / std`
- **Min-Max Scaling**: `(X - min) / (max - min)`
- **Robust Scaling**: Uses median and IQR (resistant to outliers)

**Default:** Z-Score Normalization

---

### 2. **Data Augmentation**

**Techniques:**
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **Time Warping**: Temporal distortion
- **Noise Injection**: Gaussian noise addition

**Use Case:** Handling imbalanced datasets

---

### 3. **Feature Selection**

**Methods:**
- **SelectKBest**: Statistical tests (f_classif, mutual_info)
- **RFE** (Recursive Feature Elimination): With Random Forest
- **Mutual Information**: Information-theoretic approach

---

### 4. **Missing Data Handling**

**Strategies:**
- **Mean Imputation**: Replace with column mean
- **Median Imputation**: Replace with column median
- **Forward Fill**: Use previous value
- **Interpolation**: Linear or spline interpolation

---

## 🔍 Model Interpretability

### 1. **SHAP (SHapley Additive exPlanations)**

**Algorithm:** Game-theoretic approach to explain predictions

**Features:**
- Global feature importance
- Local explanations for individual predictions
- Interaction effects
- Consistency and accuracy guarantees

**Implementation:**
```python
from utils.interpretability import ModelInterpretability

interpreter = ModelInterpretability(model)
shap_results = interpreter.explain_with_shap(X_data)
```

**Output:**
- Feature importance scores
- SHAP values for each prediction
- Summary plots

---

### 2. **LIME (Local Interpretable Model-agnostic Explanations)**

**Algorithm:** Local linear approximation of model behavior

**Features:**
- Model-agnostic (works with any model)
- Local explanations
- Human-interpretable
- Fast computation

**Implementation:**
```python
lime_results = interpreter.explain_with_lime(X_data, sample_idx=0)
```

**Output:**
- Top contributing features
- Feature weights
- Explanation confidence

---

### 3. **Confidence Calibration**

**Methods:**

**a) Temperature Scaling**
```
calibrated_prob = softmax(logits / T)
```
- Single parameter T (temperature)
- Post-processing method
- Preserves accuracy

**b) Platt Scaling**
- Logistic regression on validation set
- Maps scores to probabilities

**c) Isotonic Regression**
- Non-parametric calibration
- Monotonic transformation

**Use Case:** Ensuring prediction confidence matches true probability

---

## ⚡ Optimization Techniques

### 1. **Learning Rate Schedules**

**Cosine Annealing:**
```python
lr = min_lr + (initial_lr - min_lr) * (1 + cos(π * epoch / max_epochs)) / 2
```

**Benefits:**
- Smooth learning rate decay
- Periodic restarts possible
- Better convergence

---

### 2. **Regularization**

**Techniques:**
- **L1 Regularization**: Sparse weights (1e-5)
- **L2 Regularization**: Weight decay (1e-4)
- **Dropout**: Random neuron deactivation (0.2-0.3)
- **Spatial Dropout**: For convolutional layers
- **Batch Normalization**: Reduces internal covariate shift
- **Gaussian Noise**: Input noise injection (0.01)

---

### 3. **Callbacks**

**Early Stopping:**
- Monitor: validation loss
- Patience: 10 epochs
- Restore best weights

**Model Checkpoint:**
- Save best model based on validation loss
- Prevents overfitting

**ReduceLROnPlateau:**
- Reduces learning rate when plateau detected
- Factor: 0.5
- Patience: 5 epochs

**TensorBoard:**
- Real-time training visualization
- Loss curves, metrics, histograms

---

### 4. **Class Imbalance Handling**

**Techniques:**
- **Class Weights**: Computed using sklearn's `compute_class_weight`
- **SMOTE**: Synthetic sample generation
- **ADASYN**: Adaptive synthetic sampling
- **Focal Loss**: Focuses on hard examples

**Formula:**
```python
class_weight = n_samples / (n_classes * n_samples_per_class)
```

---

## 📈 Evaluation Metrics

### 1. **Classification Metrics**

**Accuracy:**
```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**
```
precision = TP / (TP + FP)
```

**Recall (Sensitivity):**
```
recall = TP / (TP + FN)
```

**F1 Score:**
```
F1 = 2 * (precision * recall) / (precision + recall)
```

**ROC-AUC:**
- Area Under Receiver Operating Characteristic curve
- Measures discrimination ability

---

### 2. **Confusion Matrix**

```
              Predicted
           0    1    2
Actual 0  TP   FP   FP
       1  FN   TP   FP
       2  FN   FN   TP
```

**Per-Class Metrics:**
- Precision per class
- Recall per class
- F1 score per class

---

### 3. **Temporal Metrics**

**State Duration:**
- Time spent in each state
- State transition count
- State stability

**Temporal Smoothing:**
- Moving average of predictions
- Reduces noise in state transitions

---

## 🎛️ Hyperparameters

### Model Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Conv Filters | [32, 64, 128] | [16-512] | Number of convolutional filters |
| LSTM Units | 64 | [32-512] | LSTM hidden units |
| Dense Units | 128 | [64-1024] | Dense layer neurons |
| Attention Heads | 4 | [2-16] | Multi-head attention heads |
| Dropout Rate | 0.3 | [0.1-0.5] | Dropout probability |

### Training

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Learning Rate | 0.001 | [1e-5 - 1e-2] | Initial learning rate |
| Batch Size | 32 | [8-128] | Training batch size |
| Epochs | 30 | [10-200] | Training epochs |
| L1 Regularization | 1e-5 | [0-1e-3] | L1 penalty |
| L2 Regularization | 1e-4 | [0-1e-2] | L2 penalty |

### Preprocessing

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Sampling Rate | 250 Hz | [100-1000] | EEG sampling frequency |
| Smoothing Window | 5 | [3-10] | Temporal smoothing window |
| Notch Frequency | 50/60 Hz | - | Power line noise frequency |

---

## 🔬 Advanced Techniques

### 1. **Multi-Channel Support**

**Configurations:**
- **8 channels**: Minimal setup (frontal, temporal)
- **16 channels**: Standard 10-20 system
- **32 channels**: High-density EEG
- **64 channels**: Ultra-high-density research

**Adaptive Architecture:**
- Automatically scales model complexity
- Channel-specific feature extraction
- Spatial attention mechanisms

---

### 2. **Real-Time Processing**

**Streaming Buffer:**
- Circular buffer for continuous data
- Adaptive window sizing
- Overlap processing (25%)

**Latency Optimization:**
- Model caching
- Batch processing
- GPU acceleration (when available)

**Target Performance:**
- Inference time: < 100ms
- Throughput: > 10 samples/second

---

### 3. **Ensemble Methods**

**Techniques:**
- Model averaging
- Weighted voting
- Stacking

**Benefits:**
- Improved accuracy
- Reduced variance
- Better generalization

---

## 📚 Libraries and Frameworks

### Core ML/DL
- **TensorFlow/Keras 3.0+**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

### Signal Processing
- **SciPy**: Scientific computing, filtering
- **MNE**: EEG-specific processing
- **PyWavelets**: Wavelet transforms

### Machine Learning
- **scikit-learn**: Classical ML algorithms
- **imbalanced-learn**: Handling imbalanced data

### Interpretability
- **SHAP**: Model explanations
- **LIME**: Local interpretability

### Visualization
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive plots

---

## 🎯 Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| General EEG Classification | Enhanced CNN-LSTM | Best balance of accuracy and speed |
| Complex Temporal Patterns | Transformer | Superior long-range dependencies |
| Deep Feature Learning | ResNet-LSTM | Handles very deep networks |
| Quick Prototyping | Original CNN-LSTM | Fast training, simple architecture |
| Limited Data | Enhanced CNN-LSTM | Better regularization |
| Real-time Processing | Original CNN-LSTM | Fastest inference |

---

## 🔄 Training Pipeline

```
1. Data Loading
   ↓
2. Preprocessing
   - Artifact removal
   - Filtering
   - Normalization
   ↓
3. Feature Extraction
   - Time-domain
   - Frequency-domain
   - Statistical
   ↓
4. Data Augmentation (if needed)
   - SMOTE
   - ADASYN
   ↓
5. Model Training
   - Forward pass
   - Loss calculation
   - Backpropagation
   - Weight update
   ↓
6. Validation
   - Metrics calculation
   - Early stopping check
   ↓
7. Model Calibration
   - Temperature scaling
   ↓
8. Evaluation
   - Test set metrics
   - Confusion matrix
   - ROC curves
   ↓
9. Model Saving
```

---

## 📊 Performance Benchmarks

### Expected Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Accuracy | > 85% | 87-92% |
| Precision | > 80% | 83-89% |
| Recall | > 80% | 82-88% |
| F1 Score | > 80% | 83-89% |
| Inference Time | < 100ms | 50-80ms |
| Model Size | < 50MB | 5-20MB |

### Hardware Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 1GB

**Recommended:**
- CPU: 4+ cores
- RAM: 8GB+
- GPU: NVIDIA with CUDA support
- Storage: 5GB+

---

## 🔮 Future Enhancements

### Planned Algorithms

1. **Graph Neural Networks (GNN)**
   - Model spatial relationships between EEG channels
   - Better capture brain connectivity

2. **Variational Autoencoders (VAE)**
   - Unsupervised feature learning
   - Anomaly detection

3. **Temporal Convolutional Networks (TCN)**
   - Alternative to RNNs
   - Parallel processing

4. **Meta-Learning**
   - Few-shot learning
   - Rapid adaptation to new subjects

5. **Federated Learning**
   - Privacy-preserving training
   - Distributed model updates

---

## 📖 References

### Papers
1. "Deep Learning for EEG-based Brain-Computer Interfaces" (2019)
2. "Attention Is All You Need" - Transformer architecture (2017)
3. "Deep Residual Learning for Image Recognition" - ResNet (2015)
4. "A Unified Approach to Interpreting Model Predictions" - SHAP (2017)

### Libraries Documentation
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- SHAP: https://shap.readthedocs.io/
- MNE: https://mne.tools/

---

## 💡 Best Practices

1. **Always preprocess data** before training
2. **Use cross-validation** for robust evaluation
3. **Monitor for overfitting** with validation set
4. **Calibrate confidence scores** for reliable predictions
5. **Use interpretability tools** to understand model decisions
6. **Regular model retraining** with new data
7. **Version control** for models and data
8. **Document hyperparameters** for reproducibility

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Maintainer:** NeuroLab Team
