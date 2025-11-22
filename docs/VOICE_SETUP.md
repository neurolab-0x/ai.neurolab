# Voice API Setup Guide

## Quick Start

### 1. Install Dependencies

The voice processor needs these packages:
```bash
pip install transformers torch scipy
```

Optional (for better audio support):
```bash
pip install torchaudio librosa
```

### 2. Generate Test Audio Files

```bash
python generate_test_audio.py
```

This creates:
- `test_audio.wav` - Single test file
- `test_audio_1.wav`, `test_audio_2.wav`, `test_audio_3.wav` - For batch testing

### 3. Start the Server

```bash
uvicorn main:app
```

Or with auto-reload for development:
```bash
uvicorn main:app --reload
```

Server will run on: http://localhost:8000

### 4. Test the API

```bash
python test_voice_api.py
```

Or visit the interactive docs: http://localhost:8000/docs

## Current Status

✓ **Working**: The API is functional and accepts audio files
✓ **Fallback Mode**: Currently running in fallback mode (returns neutral emotion)

⚠️ **Model Loading**: The emotion recognition model needs to be properly configured

## Fixing Model Loading

The voice processor tries to load emotion recognition models from Hugging Face. If you see warnings about model loading, you have two options:

### Option 1: Use Fallback Mode (Current)
The API works in fallback mode, returning neutral emotion. Good for testing the API structure.

### Option 2: Install Proper Model
To get real emotion detection:

1. Install transformers with all dependencies:
```bash
pip install transformers[torch] accelerate
```

2. Download a working emotion model:
```bash
python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification; Wav2Vec2Processor.from_pretrained('superb/wav2vec2-base-superb-er'); Wav2Vec2ForSequenceClassification.from_pretrained('superb/wav2vec2-base-superb-er')"
```

3. Restart the server

## Testing with cURL

```bash
# Health check
curl http://localhost:8000/voice/health

# Get emotions
curl http://localhost:8000/voice/emotions

# Analyze audio
curl -X POST http://localhost:8000/voice/analyze \
  -F "file=@test_audio.wav"
```

## Testing with Python

```python
import requests

# Analyze audio file
with open('test_audio.wav', 'rb') as f:
    files = {'file': ('test_audio.wav', f, 'audio/wav')}
    response = requests.post('http://localhost:8000/voice/analyze', files=files)
    print(response.json())
```

## Troubleshooting

### "Model not loaded, using fallback emotion detection"
This is expected if the Hugging Face model couldn't be downloaded. The API still works but returns neutral emotion.

**Solution**: Follow "Option 2: Install Proper Model" above

### "Error loading with torchaudio: Could not load libtorchcodec"
This warning is normal if FFmpeg is not installed. The API falls back to scipy for audio loading.

**Solution**: Install scipy (already in requirements): `pip install scipy`

### "Connection refused" when testing
The server is not running.

**Solution**: Start it with `uvicorn main:app`

## API Endpoints

All endpoints are under `/voice`:

- `GET /voice/health` - Check processor status
- `GET /voice/emotions` - List supported emotions
- `POST /voice/analyze` - Analyze single audio file
- `POST /voice/analyze-batch` - Analyze multiple files
- `POST /voice/analyze-raw` - Analyze raw audio data

See `VOICE_API_README.md` for detailed documentation.

## Integration with EEG Analysis

The voice processor outputs mental states (0=relaxed, 1=focused, 2=stressed) that match your EEG analysis states. You can:

1. Analyze EEG data: `POST /analyze`
2. Analyze voice data: `POST /voice/analyze`
3. Combine results for multimodal assessment

Example:
```python
# Get EEG mental state
eeg_response = requests.post('http://localhost:8000/analyze', json=eeg_data)
eeg_state = eeg_response.json()['mental_state']

# Get voice mental state
with open('audio.wav', 'rb') as f:
    voice_response = requests.post('http://localhost:8000/voice/analyze', 
                                   files={'file': f})
voice_state = voice_response.json()['data']['mental_state']

# Combine (simple average)
combined_state = (eeg_state + voice_state) / 2
```
