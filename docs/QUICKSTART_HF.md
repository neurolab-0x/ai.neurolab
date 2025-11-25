# Quick Start: Deploy NeuroLab to Hugging Face

This is a quick reference guide to get your NeuroLab model on Hugging Face in minutes.

## 🚀 Fastest Method: Gradio Space (Recommended)

### Prerequisites
```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 1: Prepare Your Space
```bash
# Run the automated preparation script
python scripts/prepare_hf_space.py
```

This creates a `neurolab-hf-space/` directory with everything you need.

### Step 2: Create Space on Hugging Face

**Option A: Via Web Interface**
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `neurolab-eeg-analysis`
4. SDK: Gradio
5. Click "Create Space"

**Option B: Via CLI**
```bash
huggingface-cli repo create neurolab-eeg-analysis --type space --space_sdk gradio
```

### Step 3: Deploy
```bash
cd neurolab-hf-space
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/neurolab-eeg-analysis
git push -u origin main
```

### Step 4: Access Your Space
Visit: `https://huggingface.co/spaces/YOUR_USERNAME/neurolab-eeg-analysis`

---

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Create Space Directory
```bash
mkdir neurolab-hf-space
cd neurolab-hf-space
```

### 2. Copy Essential Files
```bash
# Copy source code
cp -r ../src ./
cp -r ../utils ./
cp -r ../core ./
cp -r ../preprocessing ./

# Copy models (if available)
mkdir -p processed model
cp ../processed/trained_model.h5 processed/ 2>/dev/null || true
cp ../model/voice_emotion_model.h5 model/ 2>/dev/null || true

# Copy Gradio app
cp ../gradio_app.py app.py
```

### 3. Create requirements.txt
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

### 4. Create README.md
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

# NeuroLab: EEG Analysis Platform

Analyze EEG data to detect mental states.
```

### 5. Deploy
```bash
git init
git add .
git commit -m "Deploy NeuroLab"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/neurolab-eeg-analysis
git push -u origin main
```

---

## 🧪 Testing Your Deployed API

### Test via Web Interface
Simply visit your Space URL and use the Gradio interface.

### Test via Python API
```python
from gradio_client import Client

# Connect to your Space
client = Client("YOUR_USERNAME/neurolab-eeg-analysis")

# Test analysis
result = client.predict(
    alpha=10.5,
    beta=15.2,
    theta=6.3,
    delta=2.1,
    gamma=30.5,
    api_name="/analyze_eeg"
)

print(result)
```

### Test via cURL
```bash
curl -X POST "https://YOUR_USERNAME-neurolab-eeg-analysis.hf.space/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [10.5, 15.2, 6.3, 2.1, 30.5]
  }'
```

---

## 📊 Deploy FastAPI Instead

If you want to deploy the FastAPI backend:

### 1. Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. Update README.md
```markdown
---
title: NeuroLab API
emoji: 🧠
sdk: docker
pinned: false
---
```

### 3. Deploy
```bash
git init
git add .
git commit -m "Deploy FastAPI"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/neurolab-api
git push -u origin main
```

### 4. Access API
- Docs: `https://YOUR_USERNAME-neurolab-api.hf.space/docs`
- Health: `https://YOUR_USERNAME-neurolab-api.hf.space/health`

---

## 🔍 Common Issues

### Build Fails
- Check logs in Space's "Logs" tab
- Verify all imports are in requirements.txt
- Ensure file paths are correct

### Model Not Loading
- Check if model files are included
- For files >10MB, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.h5"
  git add .gitattributes
  ```

### Import Errors
- Ensure all source directories are copied
- Check Python path in app.py
- Verify module structure matches imports

### Space Timeout
- Reduce model size
- Optimize initialization
- Consider upgrading to GPU hardware

---

## 💡 Pro Tips

1. **Use Git LFS for large models**
   ```bash
   git lfs track "*.h5" "*.pkl" "*.pt"
   ```

2. **Enable Gradio Analytics**
   ```python
   demo.launch(analytics_enabled=True)
   ```

3. **Add Examples**
   ```python
   demo.launch(
       examples=[
           [10.5, 15.2, 6.3, 2.1, 30.5],  # Relaxed
           [5.2, 22.1, 4.5, 1.8, 42.3],   # Focused
       ]
   )
   ```

4. **Enable Queue for Heavy Models**
   ```python
   demo.queue().launch()
   ```

5. **Monitor Usage**
   - Check Space analytics dashboard
   - View logs for errors
   - Monitor API calls

---

## 🎯 Next Steps

After deployment:

1. ✅ Test all functionality
2. 📊 Monitor Space analytics
3. 🔒 Add authentication if needed
4. 📈 Upgrade hardware if slow
5. 🌐 Share with community
6. 📝 Update documentation
7. 🔄 Set up CI/CD for updates

---

## 📚 Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs)
- [Full Deployment Guide](./HUGGINGFACE_DEPLOYMENT.md)

---

**Questions?** Check the full guide at `docs/HUGGINGFACE_DEPLOYMENT.md`
