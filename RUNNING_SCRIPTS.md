# Running Scripts Guide

## Overview

This guide explains how to run scripts in the NeuroLab project correctly, avoiding import errors.

## The Import Issue

Python scripts inside the `src/` directory need to import modules using the `src.` prefix. However, when running scripts directly, Python doesn't automatically add the project root to the path.

## Solutions

### Solution 1: Use Wrapper Scripts (Recommended)

We've created wrapper scripts in the `scripts/` directory that handle the path setup automatically:

```bash
# Generate training data
python scripts/generate_data.py

# Train model
python scripts/train_model.py

# Test voice API
python scripts/test_voice.py
```

### Solution 2: Run from Project Root

Always run scripts from the project root directory:

```bash
# ✓ Correct - from project root
python src/scripts/training/train_model.py

# ✗ Wrong - from inside src/
cd src/scripts/training
python train_model.py  # This will fail!
```

### Solution 3: Set PYTHONPATH

Set the PYTHONPATH environment variable:

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "C:\Users\pc\Documents\Neurolab\neurolab_model"
python src/scripts/training/train_model.py
```

**Windows (CMD):**
```cmd
set PYTHONPATH=C:\Users\pc\Documents\Neurolab\neurolab_model
python src/scripts/training/train_model.py
```

**Linux/Mac:**
```bash
export PYTHONPATH=/path/to/neurolab_model
python src/scripts/training/train_model.py
```

## Available Wrapper Scripts

### 1. Generate Training Data
```bash
python scripts/generate_data.py
```
- Generates synthetic EEG training data
- Output: `data/train_data/training.csv`
- Creates 10,000 samples per state (relaxed, focused, stressed)

### 2. Train Model
```bash
python scripts/train_model.py
```
- Trains the EEG classification model
- Outputs:
  - Model: `model/trained_model_improved.h5`
  - Checkpoints: `checkpoints/`
  - Results: `training_results/`
  - Plots: `training_results/plots/`

### 3. Test Voice API
```bash
python scripts/test_voice.py
```
- Tests voice processing endpoints
- Requires: API server running (`uvicorn main:app`)
- Tests: health, emotions, audio analysis

## Direct Script Execution

If you prefer to run scripts directly from `src/`, they now include automatic path setup:

```python
# This is added to all scripts in src/scripts/
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

So you can run:
```bash
python src/scripts/training/train_model.py
```

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup directories
python setup_directories.py
```

### Data Generation
```bash
# Generate training data
python scripts/generate_data.py

# Generate test audio
python src/scripts/generation/generate_test_audio.py
```

### Training
```bash
# Train model
python scripts/train_model.py

# Or with custom parameters (edit the script first)
python src/scripts/training/train_model.py
```

### Testing
```bash
# Test voice API
python scripts/test_voice.py

# Run all tests
python -m pytest src/tests/

# Run specific test
python -m pytest src/tests/test_voice_api.py
```

### API Server
```bash
# Start server
uvicorn main:app --reload

# Start on specific port
uvicorn main:app --port 8080

# Start with host binding
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'src'

**Cause:** Running script from wrong directory or PYTHONPATH not set

**Solutions:**
1. Use wrapper scripts: `python scripts/train_model.py`
2. Run from project root: `python src/scripts/training/train_model.py`
3. Set PYTHONPATH (see Solution 3 above)

### Error: No such file or directory: 'train_data/...'

**Cause:** Output directories don't exist

**Solution:**
```bash
python setup_directories.py
```

### Error: Model file not found

**Cause:** Model hasn't been trained yet

**Solution:**
```bash
# Generate data first
python scripts/generate_data.py

# Then train model
python scripts/train_model.py
```

### Error: Connection refused (testing voice API)

**Cause:** API server not running

**Solution:**
```bash
# Start server in another terminal
uvicorn main:app --reload

# Then run tests
python scripts/test_voice.py
```

## Project Structure

```
neurolab_model/
├── scripts/              # Wrapper scripts (use these!)
│   ├── generate_data.py
│   ├── train_model.py
│   └── test_voice.py
├── src/
│   ├── scripts/         # Actual implementation
│   │   ├── generation/
│   │   └── training/
│   ├── tests/           # Test files
│   └── ...
├── main.py              # API entry point
└── setup_directories.py # Directory setup
```

## Best Practices

1. **Always use wrapper scripts** when possible
2. **Run from project root** if running directly
3. **Check current directory** before running scripts:
   ```bash
   pwd  # Linux/Mac
   cd   # Windows
   ```
4. **Use virtual environment**:
   ```bash
   # Activate venv first
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

## IDE Configuration

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

### PyCharm
1. Right-click project root
2. Mark Directory as → Sources Root

## Summary

✓ **Use wrapper scripts** in `scripts/` directory (easiest)
✓ **Run from project root** when using direct paths
✓ **Set PYTHONPATH** for advanced usage
✓ **Check documentation** when in doubt

For more information, see:
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md) - Import details
- [README.md](README.md) - Main documentation
