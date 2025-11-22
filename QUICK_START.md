# NeuroLab Quick Start Guide

## Initial Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Directories
```bash
python setup_directories.py
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

## Generate Training Data

```bash
# Generate synthetic training data
python src/scripts/generation/generate_training_data.py

# Output: data/train_data/training.csv
```

## Train Model

```bash
# Train the model
python src/scripts/training/train_model.py

# Outputs:
# - model/trained_model_improved.h5
# - checkpoints/
# - training_results/
```

## Run API Server

```bash
# Start the server
uvicorn main:app --reload

# Server runs on: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Test Voice API

```bash
# Generate test audio files
python src/scripts/generation/generate_test_audio.py

# Run voice API tests
python src/tests/test_voice_api.py
```

## Common Commands

### Check Health
```bash
curl http://localhost:8000/health
```

### Analyze EEG Data
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"eeg_data": {...}}'
```

### Analyze Voice
```bash
curl -X POST http://localhost:8000/voice/analyze \
  -F "file=@test_audio.wav"
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "state_durations": {0: 10.5, 1: 15.2, 2: 5.3},
    "total_duration": 31.0,
    "confidence": 0.85
  }'
```

## Project Structure

```
neurolab_model/
├── main.py              # API entry point
├── src/                 # Source code
│   ├── api/            # API endpoints
│   ├── utils/          # Utilities
│   ├── preprocessing/  # Data processing
│   ├── scripts/        # Scripts
│   └── tests/          # Tests
├── data/               # Data storage
├── model/              # Trained models
├── output/             # Generated outputs
└── docs/               # Documentation
```

## Import Pattern

Always use `src.` prefix for imports:

```python
# Correct
from src.utils.ml_processor import MLProcessor
from src.api.voice import router
from src.preprocessing import load_data

# Incorrect
from utils.ml_processor import MLProcessor
from api.voice import router
from preprocessing import load_data
```

## Troubleshooting

### Import Errors
```bash
# If you see: ModuleNotFoundError: No module named 'utils'
# Fix: Update imports to use src. prefix
```

### Directory Not Found
```bash
# If you see: FileNotFoundError: No such file or directory
# Fix: Run setup script
python setup_directories.py
```

### Model Not Found
```bash
# If you see: Model file not found
# Fix: Train the model first
python src/scripts/training/train_model.py
```

## Documentation

- [README.md](README.md) - Main documentation
- [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md) - Import fixes details
- [VOICE_API_README.md](docs/VOICE_API_README.md) - Voice API documentation
- [VOICE_SETUP.md](docs/VOICE_SETUP.md) - Voice setup guide
- [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) - Complete API reference

## Support

For issues or questions:
- Check documentation in `docs/` folder
- Review `IMPORT_FIX_SUMMARY.md` for import issues
- Check API docs at http://localhost:8000/docs when server is running

## Next Steps

1. ✓ Setup complete
2. Generate training data
3. Train model
4. Start API server
5. Test endpoints
6. Integrate with your application

Happy coding! 🚀
