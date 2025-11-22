# NeuroLab: EEG & Voice Analysis Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Voice Processing](#voice-processing)
- [Model Interpretability](#model-interpretability)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Contributing](#contributing)
- [Contact](#contact)

## 🔭 Overview

NeuroLab is a sophisticated multimodal analysis platform that combines EEG (Electroencephalogram) data processing with voice emotion detection to provide comprehensive mental state classification. The system leverages machine learning to identify mental states such as relaxed, focused, and stressed, making it valuable for applications in mental health monitoring, neurofeedback, and brain-computer interfaces.

## ✨ Features

### Core Capabilities
- **Real-time EEG Processing**: Stream and analyze EEG data in real-time
- **Voice Emotion Detection**: Analyze audio for emotion and mental state classification
- **Multimodal Analysis**: Combine EEG and voice data for comprehensive assessment
- **Multiple File Format Support**: Compatible with .edf, .bdf, .gdf, .csv, and audio formats
- **Advanced Signal Processing**: Comprehensive preprocessing and feature extraction
- **Machine Learning Integration**: Hybrid model approach with automated calibration
- **NLP-based Recommendations**: AI-driven personalized insights and recommendations
- **RESTful API**: FastAPI-powered endpoints for seamless integration
- **Scalable Architecture**: Modular design for easy extension and maintenance

### Mental State Classification
- **Relaxed** (State 0): Calm, neutral emotional states
- **Focused** (State 1): Alert, positive, engaged states
- **Stressed** (State 2): Anxious, fearful, negative states

## 🏗 System Architecture

```
neurolab_model/
├── api/                    # API endpoints and routing
│   ├── auth.py            # Authentication endpoints
│   ├── training.py        # Model training endpoints
│   ├── voice.py           # Voice processing endpoints
│   └── streaming_endpoint.py
├── config/                # Configuration files
│   ├── database.py
│   └── settings.py
├── core/                  # Core functionality
│   ├── config/
│   ├── data/
│   ├── ml/
│   ├── models/
│   └── services/
├── preprocessing/         # Data preprocessing modules
│   ├── features.py
│   ├── labeling.py
│   ├── load_data.py
│   └── preprocess.py
├── utils/                 # Utility functions
│   ├── ml_processor.py
│   ├── nlp_recommendations.py
│   ├── voice_processor.py
│   └── model_manager.py
├── data/                  # Raw data storage
├── processed/             # Processed data and trained models
├── main.py               # Application entry point
├── requirements.txt      # Project dependencies
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) MongoDB for data storage
- (Optional) InfluxDB for time-series data

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/neurolab-0x/ai.neurolab.git neurolab_model
   cd neurolab_model
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Voice Processing Dependencies** (Optional)
   ```bash
   pip install transformers torch scipy
   ```

5. **Environment Setup**
   ```bash
   cp .env.example .env
   # Configure your .env file with appropriate settings
   ```

## 🎯 Quick Start

### 1. Start the Server
```bash
uvicorn main:app --reload
```
Server will run on: http://localhost:8000

### 2. Access API Documentation
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### 3. Test Voice Processing
```bash
# Generate test audio files
python generate_test_audio.py

# Run test suite
python test_voice_api.py
```

## 📚 API Documentation

### Core Endpoints

#### Health & Status
- `GET /health` - System health check and diagnostics
- `GET /` - API information and available endpoints

#### EEG Analysis
- `POST /upload` - Upload and process EEG files
  - Supports files up to 500MB
  - Returns mental state classification and analysis
  
- `POST /analyze` - Analyze EEG data
  - Real-time EEG data processing
  - Returns mental state, confidence, and metrics

- `POST /detailed-report` - Generate comprehensive analysis report
  - Includes cognitive metrics
  - Provides NLP-based recommendations
  - Optional report saving

#### Recommendations
- `POST /recommendations` - Get personalized recommendations
  - Based on mental state analysis
  - NLP-powered insights
  - Customizable recommendation count

#### Model Management
- `POST /calibrate` - Calibrate model with new data
- `POST /train` - Train model with custom dataset (requires auth)

## 🎤 Voice Processing

### Overview
The voice processing module analyzes audio for emotion detection and maps emotions to mental states compatible with EEG analysis.

### Supported Emotions
- **Angry** → Stressed (State 2)
- **Fear** → Stressed (State 2)
- **Sad** → Stressed (State 2)
- **Neutral** → Relaxed (State 0)
- **Calm** → Relaxed (State 0)
- **Happy** → Focused (State 1)
- **Surprise** → Focused (State 1)

### Voice API Endpoints

#### Health Check
```bash
GET /voice/health
```
Check if voice processor is initialized and ready.

#### Get Supported Emotions
```bash
GET /voice/emotions
```
List all supported emotions and their mental state mappings.

#### Analyze Audio File
```bash
POST /voice/analyze
```
Upload and analyze an audio file for emotion detection.

**Example:**
```python
import requests

with open('audio.wav', 'rb') as f:
    files = {'file': ('audio.wav', f, 'audio/wav')}
    response = requests.post('http://localhost:8000/voice/analyze', files=files)
    result = response.json()
    
print(f"Emotion: {result['data']['emotion']}")
print(f"Mental State: {result['data']['mental_state']}")
print(f"Confidence: {result['data']['confidence']}")
```

#### Batch Analysis
```bash
POST /voice/analyze-batch
```
Analyze multiple audio files with pattern analysis.

**Features:**
- Process up to 50 files simultaneously
- Aggregate emotion distribution
- Calculate average mental state
- Identify dominant emotions

#### Raw Audio Analysis
```bash
POST /voice/analyze-raw
```
Analyze raw audio data (base64 or bytes array).

**Example:**
```python
import base64
import requests

with open('audio.wav', 'rb') as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

payload = {
    "audio_data": {
        "data": audio_base64,
        "format": "base64"
    },
    "sample_rate": 16000
}

response = requests.post('http://localhost:8000/voice/analyze-raw', json=payload)
```

### Multimodal Analysis

Combine EEG and voice data for comprehensive mental state assessment:

```python
import requests

# Analyze EEG data
eeg_response = requests.post('http://localhost:8000/analyze', json=eeg_data)
eeg_state = eeg_response.json()['mental_state']

# Analyze voice data
with open('audio.wav', 'rb') as f:
    voice_response = requests.post('http://localhost:8000/voice/analyze', 
                                   files={'file': f})
voice_state = voice_response.json()['data']['mental_state']

# Combine results
combined_state = (eeg_state + voice_state) / 2
print(f"Combined Mental State: {combined_state}")
```

## 🔍 Model Interpretability

### SHAP (SHapley Additive exPlanations)
- Explains model predictions by attributing feature importance
- Identifies which EEG features contribute most to classifications
- Available via: `/interpretability/explain?explanation_type=shap`

### LIME (Local Interpretable Model-agnostic Explanations)
- Provides local explanations for individual predictions
- Available via: `/interpretability/explain?explanation_type=lime`
- Can be included in streaming responses with `include_interpretability=true`

### Confidence Calibration
- Ensures confidence scores accurately reflect true probabilities
- Methods: temperature scaling, Platt scaling, isotonic regression
- Available via: `/interpretability/calibrate?method=temperature_scaling`

**Usage Example:**
```python
from utils.interpretability import ModelInterpretability

interpreter = ModelInterpretability(model)

# Get SHAP explanations
shap_results = interpreter.explain_with_shap(X_data)

# Calibrate confidence
cal_results = interpreter.calibrate_confidence(X_val, y_val, 
                                               method='temperature_scaling')

# Make predictions with calibrated confidence
predictions = interpreter.predict_with_calibration(X_test)
```

## 🔄 Data Processing Pipeline

### EEG Processing
1. **Data Loading** - File validation and format checking
2. **Preprocessing** - Artifact removal, filtering, normalization
3. **Feature Extraction** - Temporal, frequency domain, statistical features
4. **State Classification** - Mental state prediction with confidence scoring

### Voice Processing
1. **Audio Loading** - Multiple format support (WAV, MP3, etc.)
2. **Preprocessing** - Normalization, resampling to 16kHz
3. **Feature Extraction** - RMS energy, zero-crossing rate, spectral features
4. **Emotion Detection** - Wav2Vec2-based emotion classification
5. **State Mapping** - Convert emotions to mental states

## 🧠 Model Training

### Training Process
1. Data preparation and splitting
2. Feature engineering
3. Model selection and hyperparameter tuning
4. Cross-validation
5. Model calibration
6. Performance evaluation

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confidence calibration metrics

## 📖 Additional Documentation

- [Voice API Documentation](VOICE_API_README.md) - Detailed voice processing API guide
- [Voice Setup Guide](VOICE_SETUP.md) - Installation and troubleshooting
- [API Documentation](API_DOCUMENTATION.md) - Complete API reference

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**AI Model Maintainer**: Mugisha Prosper  
Email: nelsonprox92@gmail.com

**Project**: [Neurolabs Inc](https://neurolab.cc)  
Repository: [GitHub](https://github.com/neurolab-0x/ai.neurolab)

---

**Built with ❤️ by the NeuroLab Team**
