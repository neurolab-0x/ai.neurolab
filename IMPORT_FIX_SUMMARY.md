# Import Fix Summary

## Overview
This document summarizes all import fixes and directory structure corrections made to the NeuroLab codebase.

## Directory Structure

```
neurolab_model/
├── data/                      # Data storage
│   ├── train_data/           # Training datasets
│   ├── test_data/            # Test datasets
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data
├── model/                     # Model storage
│   ├── trained_model.h5      # Main trained model
│   ├── checkpoints/          # Training checkpoints
│   └── archived/             # Archived models
├── src/                       # Source code (main package)
│   ├── api/                  # API endpoints
│   │   ├── auth.py
│   │   ├── training.py
│   │   ├── voice.py
│   │   ├── streaming_endpoint.py
│   │   └── ...
│   ├── config/               # Configuration
│   │   ├── database.py
│   │   └── settings.py
│   ├── core/                 # Core functionality
│   │   ├── ml/              # ML models
│   │   ├── data/            # Data handlers
│   │   ├── models/          # Data models
│   │   └── services/        # Services
│   ├── preprocessing/        # Data preprocessing
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── labeling.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── scripts/              # Utility scripts
│   │   ├── generation/      # Data generation
│   │   └── training/        # Training scripts
│   ├── tests/                # Test files
│   ├── utils/                # Utility modules
│   │   ├── ml_processor.py
│   │   ├── voice_processor.py
│   │   ├── nlp_recommendations.py
│   │   └── ...
│   └── models/               # Model definitions
├── output/                    # Output files
│   ├── reports/              # Generated reports
│   ├── plots/                # Visualization plots
│   └── logs/                 # Output logs
├── training_results/          # Training outputs
│   ├── plots/                # Training plots
│   └── metrics/              # Training metrics
├── logs/                      # Application logs
│   ├── training/             # Training logs
│   ├── api/                  # API logs
│   └── processing/           # Processing logs
├── processed/                 # Legacy processed data
├── checkpoints/               # Legacy checkpoints
├── docs/                      # Documentation
├── main.py                    # Application entry point
├── setup_directories.py       # Directory setup script
└── requirements.txt           # Dependencies
```

## Import Fixes Applied

### 1. Main Application (main.py)
**Before:**
```python
from utils.file_handler import validate_file, save_uploaded_file
from utils.model_manager import ModelManager
from utils.ml_processor import MLProcessor
from api.training import router as training_router
from api.auth import router as auth_router
from api.voice import router as voice_router
from api.streaming_endpoint import router as streaming_router
from utils.nlp_recommendations import get_recommendations
```

**After:**
```python
from src.utils.file_handler import validate_file, save_uploaded_file
from src.utils.model_manager import ModelManager
from src.utils.ml_processor import MLProcessor
from src.api.training import router as training_router
from src.api.auth import router as auth_router
from src.api.voice import router as voice_router
from src.api.streaming_endpoint import router as streaming_router
from src.utils.nlp_recommendations import get_recommendations
```

### 2. ML Processor (src/utils/ml_processor.py)
**Before:**
```python
from preprocessing import load_data, extract_features, preprocess_data
from preprocessing.labeling import label_eeg_states
from core.ml.model import load_calibrated_model
from utils.temporal_processing import temporal_smoothing, calculate_state_durations
from utils.nlp_recommendations import NLPRecommendationEngine
from config.settings import PROCESSING_CONFIG, THRESHOLDS
```

**After:**
```python
from src.preprocessing import load_data, extract_features, preprocess_data
from src.preprocessing.labeling import label_eeg_states
from src.core.ml.model import load_calibrated_model
from src.utils.temporal_processing import temporal_smoothing, calculate_state_durations
from src.utils.nlp_recommendations import NLPRecommendationEngine
from src.config.settings import PROCESSING_CONFIG, THRESHOLDS
```

### 3. Utility Modules
Fixed imports in:
- `src/utils/session_summary.py`
- `src/utils/recommendations.py`
- `src/utils/model_manager.py`
- `src/utils/event_detector.py`
- `src/utils/duration_calculation.py`
- `src/utils/database_service.py`

All changed from `from utils.X` or `from config.X` to `from src.utils.X` or `from src.config.X`

### 4. Test Files
Fixed imports in all test files:
- `src/tests/test_nlp_integration.py`
- `src/tests/test_ml_processor.py`
- `src/tests/test_explanation_generator.py`
- `src/tests/test_data_processing.py`
- `src/tests/test_data_handler.py`

All changed from `from utils.X` to `from src.utils.X`

### 5. API Files
All API files already had correct imports using `src.` prefix:
- `src/api/auth.py` ✓
- `src/api/training.py` ✓
- `src/api/voice.py` ✓
- `src/api/streaming_endpoint.py` ✓

### 6. Scripts
Training and generation scripts already had correct imports:
- `src/scripts/training/train_model.py` ✓
- `src/scripts/generation/generate_training_data.py` ✓

## Output Directory Fixes

### 1. Training Data Generation
**File:** `src/scripts/generation/generate_training_data.py`

**Before:**
```python
def save_training_data(df, filename='train_data/training.csv'):
```

**After:**
```python
def save_training_data(df, filename='data/train_data/training.csv'):
```

### 2. Model Training
**File:** `src/scripts/training/train_model.py`

Default paths already correct:
- Model save: `model/trained_model_improved.h5` ✓
- Checkpoints: `./checkpoints` ✓
- Results: `./training_results` ✓

### 3. Created Directories
All necessary directories created via `setup_directories.py`:
- ✓ data/train_data
- ✓ data/test_data
- ✓ data/raw
- ✓ data/processed
- ✓ model/checkpoints
- ✓ model/archived
- ✓ output/reports
- ✓ output/plots
- ✓ output/logs
- ✓ training_results/plots
- ✓ training_results/metrics
- ✓ logs/training
- ✓ logs/api
- ✓ logs/processing
- ✓ processed
- ✓ checkpoints

## Import Pattern

### Standard Import Pattern
All imports within the `src/` package should use the full path from the project root:

```python
# Correct
from src.utils.ml_processor import MLProcessor
from src.api.voice import router
from src.preprocessing import load_data
from src.config.settings import PROCESSING_CONFIG

# Incorrect
from utils.ml_processor import MLProcessor
from api.voice import router
from preprocessing import load_data
from config.settings import PROCESSING_CONFIG
```

### Relative Imports (within same package)
Can use relative imports within the same package:

```python
# In src/preprocessing/features.py
from .load_data import load_data  # OK
from src.preprocessing.load_data import load_data  # Also OK
```

## Verification

### Test Imports
Run these commands to verify imports work:

```bash
# Test main application
python -c "import main; print('✓ Main import successful')"

# Test ML processor
python -c "from src.utils.ml_processor import MLProcessor; print('✓ MLProcessor import successful')"

# Test voice API
python -c "from src.api.voice import router; print('✓ Voice API import successful')"

# Test preprocessing
python -c "from src.preprocessing import load_data; print('✓ Preprocessing import successful')"
```

### Run Application
```bash
# Start the server
uvicorn main:app --reload

# Should start without import errors
```

### Generate Training Data
```bash
# Generate training data
python src/scripts/generation/generate_training_data.py

# Output should be saved to: data/train_data/training.csv
```

### Train Model
```bash
# Train model
python src/scripts/training/train_model.py

# Outputs:
# - Model: model/trained_model_improved.h5
# - Checkpoints: checkpoints/
# - Results: training_results/
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError
**Error:** `ModuleNotFoundError: No module named 'utils'`

**Solution:** Update import to use `src.` prefix:
```python
from src.utils.module_name import ClassName
```

### Issue 2: Directory Not Found
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'train_data/...'`

**Solution:** Run directory setup:
```bash
python setup_directories.py
```

### Issue 3: Circular Import
**Error:** `ImportError: cannot import name 'X' from partially initialized module`

**Solution:** Check for circular dependencies and use lazy imports if needed:
```python
# Instead of top-level import
def function():
    from src.utils.module import function_name
    return function_name()
```

## Best Practices

1. **Always use absolute imports** from project root with `src.` prefix
2. **Run setup_directories.py** before first use
3. **Check output paths** in scripts to ensure they use correct directories
4. **Test imports** after making changes
5. **Keep __init__.py files** in all packages for proper module recognition

## Files Modified

### Core Files
- ✓ main.py
- ✓ setup_directories.py (new)

### Source Files
- ✓ src/utils/ml_processor.py
- ✓ src/utils/session_summary.py
- ✓ src/utils/recommendations.py
- ✓ src/utils/model_manager.py
- ✓ src/utils/event_detector.py
- ✓ src/utils/duration_calculation.py
- ✓ src/utils/database_service.py

### Scripts
- ✓ src/scripts/generation/generate_training_data.py

### Tests
- ✓ src/tests/test_nlp_integration.py
- ✓ src/tests/test_ml_processor.py
- ✓ src/tests/test_explanation_generator.py
- ✓ src/tests/test_data_processing.py
- ✓ src/tests/test_data_handler.py

## Summary

✓ **21 directories** created/verified
✓ **15 files** updated with correct imports
✓ **All imports** now use `src.` prefix
✓ **Output paths** corrected to use proper directory structure
✓ **Import verification** passed

The codebase is now properly structured with consistent imports and organized output directories.
