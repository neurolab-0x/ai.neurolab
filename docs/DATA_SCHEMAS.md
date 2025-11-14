# NeuroLab Data Schemas & Models

## Table of Contents
- [Overview](#overview)
- [Core Data Models](#core-data-models)
- [Request Schemas](#request-schemas)
- [Response Schemas](#response-schemas)
- [Database Schemas](#database-schemas)
- [Validation Rules](#validation-rules)
- [Data Flow Diagrams](#data-flow-diagrams)

---

## Overview

This document defines all data structuresor all data structures used in the NeuroLab EEG Analysis platform.

---

## EEG Data Structures

### EEGFeatures

Represents frequency band powers extracted from EEG signals.

```typescript
interface EEGFeatures {
  alpha: number;        // Alpha band (8-13 Hz) - Relaxation, calmness
  beta: number;         // Beta band (13-30 Hz) - Active thinking, focus
  theta: number;        // Theta band (4-8 Hz) - Drowsiness, meditation
  delta: number;        // Delta band (0.5-4 Hz) - Deep sleep
  gamma: number;        // Gamma band (30-45 Hz) - High-level cognition
  signal_quality?: number;  // Optional: 0-1 quality score
  metadata?: {
    channel?: string;
    timestamp?: string;
    [key: string]: any;
  };
}
```

**Example:**
```json
{
  "alpha": 0.45,
  "beta": 0.32,
  "theta": 0.18,
  "delta": 0.12,
  "gamma": 0.28,
  "signal_quality": 0.92
}
```

---

### EEGDataPoint

Complete EEG data point with features and classification.

```typescript
interface EEGDataPoint {
  timestamp: string;        // ISO 8601 format
  features: EEGFeatures;
  state: number;            // 0=Relaxed, 1=Focused, 2=Stressed
  confidence: number;       // 0-100 percentage
  subject_id: string;
  session_id: string;
  metadata?: {
    signal_quality: number;
    device: string;
    sampling_rate?: number;
    [key: string]: any;
  };
}
```

**Example:**
```json
{
  "timestamp": "2025-11-13T10:30:00.000Z",
  "features": {
    "alpha": 0.45,
    "beta": 0.32,
    "theta": 0.18,
    "delta": 0.12,
    "gamma": 0.28
  },
  "state": 0,
  "confidence": 87.5,
  "subject_id": "subject_001",
  "session_id": "session_20251113_001",
  "metadata": {
    "signal_quality": 0.92,
    "device": "OpenBCI_Cyton",
    "sampling_rate": 250
  }
}
```

---

### EEGSession

Complete session data with multiple data points.

```typescript
interface EEGSession {
  session_id: string;
  subject_id: string;
  start_time: string;       // ISO 8601
  end_time: string;         // ISO 8601
  data_points: EEGDataPoint[];
  summary?: SessionSummary;
  metadata?: {
    device: string;
    location?: string;
    notes?: string;
    [key: string]: any;
  };
}
```

---

### SessionSummary

Statistical summary of a session.

```typescript
interface SessionSummary {
  duration_seconds: number;
  total_points: number;
  state_distribution: {
    [state: string]: number;  // Percentage
  };
  mean_confidence: number;
  mean_signal_quality: number;
  state_transitions: number;
  dominant_state: number;
  recommendations: Recommendation[];
}
```

**Example:**
```json
{
  "duration_seconds": 3600,
  "total_points": 3600,
  "state_distribution": {
    "0": 45.5,
    "1": 35.2,
    "2": 19.3
  },
  "mean_confidence": 85.7,
  "mean_signal_quality": 0.89,
  "state_transitions": 42,
  "dominant_state": 0,
  "recommendations": [...]
}
```

---

## Request Schemas

### LoginRequest

```typescript
interface LoginRequest {
  username: string;     // 3-50 chars, alphanumeric + _-.
  password: string;     // 8-100 chars
}
```

**Validation:**
- Username: `/^[a-zA-Z0-9_.-]{3,50}$/`
- Password: Minimum 8 characters

---

### StreamingRequest

```typescript
interface StreamingRequest {
  eeg_data: number[][];     // [channels][samples]
  client_id?: string;       // Optional client identifier
  model_type?: string;      // Model to use
  clean_artifacts?: boolean; // Default: true
  encrypt_response?: boolean; // Default: false
  include_interpretability?: boolean; // Default: false
}
```

**Constraints:**
- Max channels: 64
- Max samples per channel: 10,000
- Max amplitude: ±1000 μV
- No NaN or Inf values

**Example:**
```json
{
  "eeg_data": [
    [0.45, 0.32, 0.18, 0.12, 0.28],
    [0.50, 0.35, 0.20, 0.15, 0.30],
    [0.42, 0.30, 0.16, 0.10, 0.25]
  ],
  "client_id": "web_client_001",
  "model_type": "enhanced_cnn_lstm",
  "clean_artifacts": true,
  "include_interpretability": true
}
```

---

### TrainingRequest

```typescript
interface TrainingRequest {
  X_train: number[][];      // Training features
  y_train: number[];        // Training labels
  X_test?: number[][];      // Optional test features
  y_test?: number[];        // Optional test labels
  config?: TrainingConfig;
}

interface TrainingConfig {
  model_type?: string;      // Default: "enhanced_cnn_lstm"
  epochs?: number;          // 1-200, default: 30
  batch_size?: number;      // 1-256, default: 32
  learning_rate?: number;   // 0-1, default: 0.001
  dropout_rate?: number;    // 0-0.9, default: 0.3
  use_separable?: boolean;  // Default: true
  use_relative_pos?: boolean; // Default: true
  l1_reg?: number;          // Default: 0.00001
  l2_reg?: number;          // Default: 0.0001
  subject_id?: string;
  session_id?: string;
}
```

**Validation:**
- X_train: Non-empty, max 100,000 samples
- y_train: Same length as X_train
- model_type: One of ["original", "enhanced_cnn_lstm", "resnet_lstm", "transformer"]

---

## Response Schemas

### StreamingResponse

```typescript
interface StreamingResponse {
  predicted_states: number[];
  dominant_state: number;
  confidence: number;
  processing_time_ms: number;
  timestamp: string;
  encrypted: boolean;
  interpretability?: InterpretabilityData;
}

interface InterpretabilityData {
  method: "lime" | "shap";
  feature_importance: {
    [feature: string]: number;
  };
  predicted_class: number;
}
```

**Example:**
```json
{
  "predicted_states": [0, 0, 1],
  "dominant_state": 0,
  "confidence": 87.5,
  "processing_time_ms": 45.2,
  "timestamp": "2025-11-13T10:30:00.000Z",
  "encrypted": false,
  "interpretability": {
    "method": "lime",
    "feature_importance": {
      "alpha": 0.35,
      "beta": 0.25,
      "theta": 0.20,
      "delta": 0.10,
      "gamma": 0.10
    },
    "predicted_class": 0
  }
}
```

---

### AnalysisResponse

```typescript
interface AnalysisResponse {
  status: string;
  state_label: string;
  dominant_state: number;
  confidence: number;
  state_percentages: {
    [state: string]: number;
  };
  cognitive_metrics: CognitiveMetrics;
  recommendations: Recommendation[];
  temporal_analysis?: TemporalAnalysis;
}

interface CognitiveMetrics {
  attention_score: number;      // 0-1
  stress_level: number;         // 0-1
  relaxation_index: number;     // 0-1
  mental_workload?: number;     // 0-1
  engagement_level?: number;    // 0-1
}

interface Recommendation {
  type: string;
  message: string;
  priority: "low" | "medium" | "high" | "info";
  severity?: string;
  confidence?: number;
}

interface TemporalAnalysis {
  total_samples: number;
  state_transitions: number;
  average_state_duration: number;
  state_sequence?: number[];
}
```

---

### TrainingStatusResponse

```typescript
interface TrainingStatusResponse {
  job_id: string;
  status: "queued" | "training" | "completed" | "failed";
  progress: number;         // 0-1
  message: string;
  started_at: string;
  completed_at?: string;
  metrics?: TrainingMetrics;
  error?: string;
}

interface TrainingMetrics {
  final_train_accuracy: number;
  final_val_accuracy: number;
  final_train_loss: number;
  final_val_loss: number;
  test_metrics?: {
    accuracy: number;
    precision: number[];
    recall: number[];
    confusion_matrix?: number[][];
  };
}
```

---

### TokenResponse

```typescript
interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: "bearer";
  expires_in: number;       // Seconds
  user_info: UserInfo;
}

interface UserInfo {
  username: string;
  roles: string[];
  created_at?: string;
}
```