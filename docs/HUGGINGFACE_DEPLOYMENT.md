# Deploying NeuroLab to Hugging Face

This guide explains how to deploy your NeuroLab model to Hugging Face for testing and API access.

## 📋 Table of Contents
- [Option 1: Hugging Face Spaces (Gradio)](#option-1-hugging-face-spaces-gradio)
- [Option 2: Hugging Face Model Hub](#option-2-hugging-face-model-hub)
- [Option 3: Docker Space (FastAPI)](#option-3-docker-space-fastapi)

---

## Option 1: Hugging Face Spaces (Gradio)

This is the **easiest and recommended** approach for testing your model with a web interface.

### Prerequisites
1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Install Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   ```
3. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

### Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `nlpt_2-preview`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU (free tier) or GPU (if needed)
4. Click **"Create Space"**

### Step 2: Prepare Your Repository

Create a new directory for the Space:

```bash
mkdir neurolab-space
cd neurolab-space
```

Copy essential files:
```bash
# Copy your Gradio app
cp ../gradio_app.py app.py

# Copy required modules
cp -r ../src ./
cp -r ../utils ./
cp -r ../core ./
cp -r ../preprocessing ./

# Copy model files (if you have trained models)
mkdir -p model
cp ../model/trained_model.h5 model/ 2>/dev/null || echo "No EEG model found"
cp ../model/voice_emotion_model.h5 model/ 2>/dev/null || echo "No voice model found"
```

### Step 3: Create Required Files

#### `requirements.txt`
```txt
gradio>=4.0.0
numpy>=1.21.0
pandas>=1.3.0
tensorflow>=2.12.0
scikit-learn>=1.0.0
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.7.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

#### `README.md`
```markdown
---
title: NeuroLab EEG Analysis
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# NeuroLab: EEG & Voice Analysis Platform

Analyze EEG data to detect mental states: **Relaxed**, **Focused**, or **Stressed**

## Features
- 📝 Manual EEG input with sliders
- 🎲 Sample data generation
- 📁 CSV file upload
- 🎤 Voice emotion detection
- 💡 AI-powered recommendations

## Usage
1. Choose an input method (Manual, Sample, or CSV)
2. Enter or upload your EEG data
3. Click "Analyze" to get results
4. View mental state classification and recommendations

## Model
- **Input**: 5 EEG frequency bands (alpha, beta, theta, delta, gamma)
- **Output**: 3 mental states (relaxed, focused, stressed)
- **Architecture**: CNN with BatchNormalization

## About
Built by the NeuroLab Team for mental state monitoring and neurofeedback applications.
```

#### `app.py` (Modified Gradio App)
Create a simplified version that works on Hugging Face:

```python
"""
NeuroLab Gradio Interface for Hugging Face Spaces
"""
import gradio as gr
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components (with error handling)
try:
    from utils.ml_processor import MLProcessor
    ml_processor = MLProcessor()
    MODEL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not load ML processor: {e}")
    MODEL_AVAILABLE = False
    ml_processor = None

def generate_sample_eeg_data(state: str = "relaxed") -> Dict[str, float]:
    """Generate sample EEG data for testing"""
    if state.lower() == "relaxed":
        return {
            "alpha": np.random.uniform(8, 13),
            "beta": np.random.uniform(5, 10),
            "theta": np.random.uniform(4, 8),
            "delta": np.random.uniform(0.5, 4),
            "gamma": np.random.uniform(25, 35)
        }
    elif state.lower() == "focused":
        return {
            "alpha": np.random.uniform(5, 8),
            "beta": np.random.uniform(15, 25),
            "theta": np.random.uniform(3, 6),
            "delta": np.random.uniform(0.5, 3),
            "gamma": np.random.uniform(30, 45)
        }
    elif state.lower() == "stressed":
        return {
            "alpha": np.random.uniform(4, 7),
            "beta": np.random.uniform(20, 30),
            "theta": np.random.uniform(6, 10),
            "delta": np.random.uniform(1, 4),
            "gamma": np.random.uniform(35, 50)
        }
    else:
        return {
            "alpha": np.random.uniform(6, 12),
            "beta": np.random.uniform(10, 20),
            "theta": np.random.uniform(4, 8),
            "delta": np.random.uniform(1, 4),
            "gamma": np.random.uniform(25, 40)
        }

def analyze_eeg(alpha, beta, theta, delta, gamma):
    """Analyze EEG data"""
    try:
        data = {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "delta": delta,
            "gamma": gamma
        }
        
        if MODEL_AVAILABLE and ml_processor:
            result = ml_processor.process_eeg_data(
                data,
                subject_id="hf_user",
                session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Format results
            state_label = result.get('state_label', 'unknown')
            confidence = result.get('confidence', 0)
            
            output = f"""
## 🧠 Analysis Result

**Mental State:** {state_label.upper()}  
**Confidence:** {confidence:.1f}%

### Input Values
- Alpha: {alpha:.2f} Hz
- Beta: {beta:.2f} Hz
- Theta: {theta:.2f} Hz
- Delta: {delta:.2f} Hz
- Gamma: {gamma:.2f} Hz
"""
            
            # Add recommendations if available
            if 'recommendations' in result:
                output += "\n### 💡 Recommendations\n"
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    output += f"{i}. {rec}\n"
            
            return output
        else:
            # Fallback rule-based classification
            if alpha > 10 and beta < 15:
                state = "RELAXED"
            elif beta > 20:
                state = "STRESSED"
            else:
                state = "FOCUSED"
                
            return f"""
## 🧠 Analysis Result (Rule-based)

**Mental State:** {state}  
**Note:** Using rule-based classification (model not available)

### Input Values
- Alpha: {alpha:.2f} Hz
- Beta: {beta:.2f} Hz
- Theta: {theta:.2f} Hz
- Delta: {delta:.2f} Hz
- Gamma: {gamma:.2f} Hz
"""
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="NeuroLab EEG Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 NeuroLab EEG Analysis Platform
    
    Analyze EEG data to detect mental states: **Relaxed**, **Focused**, or **Stressed**
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 Manual Input"):
            gr.Markdown("### Enter EEG frequency band values")
            
            with gr.Row():
                with gr.Column():
                    alpha = gr.Slider(0, 50, value=10, label="Alpha (8-13 Hz)", info="Relaxation")
                    beta = gr.Slider(0, 50, value=15, label="Beta (13-30 Hz)", info="Focus/Anxiety")
                    theta = gr.Slider(0, 50, value=6, label="Theta (4-8 Hz)", info="Drowsiness")
                    delta = gr.Slider(0, 50, value=2, label="Delta (0.5-4 Hz)", info="Deep sleep")
                    gamma = gr.Slider(0, 100, value=30, label="Gamma (25-100 Hz)", info="Processing")
                    
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column():
                    output = gr.Markdown()
            
            analyze_btn.click(
                fn=analyze_eeg,
                inputs=[alpha, beta, theta, delta, gamma],
                outputs=output
            )
        
        with gr.Tab("🎲 Sample Data"):
            gr.Markdown("### Test with pre-generated data")
            
            state_choice = gr.Radio(
                choices=["Relaxed", "Focused", "Stressed"],
                value="Relaxed",
                label="Select Mental State"
            )
            sample_btn = gr.Button("🎲 Generate & Analyze", variant="primary")
            sample_output = gr.Markdown()
            
            def analyze_sample(state):
                data = generate_sample_eeg_data(state)
                return analyze_eeg(**data)
            
            sample_btn.click(
                fn=analyze_sample,
                inputs=state_choice,
                outputs=sample_output
            )
    
    gr.Markdown("""
    ---
    ### 📖 About
    
    **NeuroLab** uses machine learning to classify mental states from EEG data.
    
    - **Model:** CNN with BatchNormalization
    - **Input:** 5 EEG frequency bands
    - **Output:** 3 mental states
    
    Built with ❤️ by the NeuroLab Team
    """)

if __name__ == "__main__":
    demo.launch()
```

### Step 4: Initialize Git and Push to Hugging Face

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit: NeuroLab EEG Analysis"

# Add Hugging Face remote (replace YOUR_USERNAME)
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/neurolab-eeg-analysis

# Push to Hugging Face
git push -u origin main
```

### Step 5: Access Your Space

Your Space will be available at:
```
https://huggingface.co/spaces/neurolab-0x/nlpt_2-preview
```

The Space will automatically build and deploy. You can monitor the build logs in the Space's "Logs" tab.

---

## Option 2: Hugging Face Model Hub

Upload your trained model to the Model Hub for inference.

### Step 1: Prepare Model Files

```bash
mkdir neurolab-model
cd neurolab-model

# Copy your trained model
cp ../processed/trained_model.h5 ./
```

### Step 2: Create Model Card

Create `README.md`:

```markdown
---
license: mit
tags:
- eeg
- neuroscience
- mental-state
- classification
library_name: tensorflow
---

# NeuroLab EEG Mental State Classifier

## Model Description

This model classifies mental states from EEG frequency band data.

### Model Details
- **Architecture**: CNN with BatchNormalization
- **Input**: 5 EEG frequency bands (alpha, beta, theta, delta, gamma)
- **Output**: 3 mental states (0=relaxed, 1=focused, 2=stressed)
- **Framework**: TensorFlow/Keras

### Usage

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/neurolab-eeg-model",
    filename="trained_model.h5"
)

# Load model
model = tf.keras.models.load_model(model_path)

# Prepare input (example)
eeg_data = np.array([[10.5, 15.2, 6.3, 2.1, 30.5]])  # alpha, beta, theta, delta, gamma

# Predict
prediction = model.predict(eeg_data)
state = np.argmax(prediction)

states = {0: "Relaxed", 1: "Focused", 2: "Stressed"}
print(f"Mental State: {states[state]}")
```

## Training Data

The model was trained on EEG data with labeled mental states.

## Limitations

- Requires calibrated EEG frequency band values
- Performance may vary across different EEG devices
- Should not be used for medical diagnosis

## Citation

```
@misc{neurolab2024,
  author = {NeuroLab Team},
  title = {NeuroLab EEG Mental State Classifier},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/neurolab-0x/nlpt_2-preview}}
}
```
```

### Step 3: Upload to Model Hub

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create repository
huggingface-cli repo create nlpt_2-preview --type model

# Initialize git
git init
git add .
git commit -m "Upload NeuroLab EEG model"

# Add remote and push
git remote add origin https://huggingface.co/neurolab-0x/nlpt_2-preview
git push -u origin main
```

---

## Option 3: Docker Space (FastAPI)

Deploy your FastAPI application as a Docker Space.

### Step 1: Create Dockerfile for Hugging Face

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Step 2: Create README for Docker Space

```markdown
---
title: NeuroLab API
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# NeuroLab EEG Analysis API

FastAPI-based REST API for EEG and voice analysis.

## Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /analyze` - Analyze EEG data
- `POST /voice/analyze` - Analyze voice emotion
- `GET /docs` - Interactive API documentation

## Usage

Visit `/docs` for interactive API documentation.
```

### Step 3: Push to Hugging Face

```bash
git init
git add .
git commit -m "Initial commit: NeuroLab API"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/neurolab-api
git push -u origin main
```

---

## Testing Your Deployment

### Test Gradio Space

Visit your Space URL and interact with the interface.

### Test API Endpoints

```python
import requests

# Replace with your Space URL
API_URL = "https://YOUR_USERNAME-neurolab-api.hf.space"

# Test health endpoint
response = requests.get(f"{API_URL}/health")
print(response.json())

# Test EEG analysis
eeg_data = {
    "alpha": 10.5,
    "beta": 15.2,
    "theta": 6.3,
    "delta": 2.1,
    "gamma": 30.5
}

response = requests.post(f"{API_URL}/analyze", json=eeg_data)
print(response.json())
```

### Test Model from Hub

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/neurolab-eeg-model",
    filename="trained_model.h5"
)

# Load and use model
model = tf.keras.models.load_model(model_path)
```

---

## Troubleshooting

### Space Build Fails

1. Check the build logs in the "Logs" tab
2. Ensure all dependencies are in `requirements.txt`
3. Verify file paths are correct
4. Check that model files are included (or use Git LFS for large files)

### Model Files Too Large

Use Git LFS for files > 10MB:

```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add trained_model.h5
git commit -m "Add model with LFS"
git push
```

### Import Errors

Ensure all required modules are copied to the Space directory and paths are adjusted for the new structure.

---

## Next Steps

1. **Monitor Usage**: Check Space analytics in your Hugging Face dashboard
2. **Upgrade Hardware**: If needed, upgrade to GPU for faster inference
3. **Add Authentication**: Implement API keys for production use
4. **Enable Persistence**: Use Hugging Face Datasets for data storage
5. **Create Inference Endpoint**: For production-grade API with auto-scaling

---

## Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs)
- [Hugging Face Model Hub](https://huggingface.co/docs/hub/models)
- [Git LFS Guide](https://git-lfs.github.com/)

---

**Built with ❤️ by the NeuroLab Team**
