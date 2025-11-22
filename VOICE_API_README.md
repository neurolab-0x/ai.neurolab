# Voice Processing API

Voice emotion detection and mental state analysis integrated into the NeuroLab API.

## Features

- **Emotion Detection**: Detects 8 emotions (angry, calm, disgust, fear, happy, neutral, sad, surprise)
- **Mental State Mapping**: Maps emotions to mental states (0=relaxed, 1=focused, 2=stressed)
- **Audio Feature Extraction**: Extracts RMS energy, zero-crossing rate, spectral features
- **Batch Processing**: Analyze multiple audio files with pattern analysis
- **Multiple Input Formats**: Supports file upload, raw bytes, and base64 encoded audio

## API Endpoints

### 1. Health Check
```
GET /voice/health
```
Check if the voice processor is initialized and ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "processor_loaded": true,
  "device": "cpu",
  "sample_rate": 16000,
  "timestamp": "2024-11-22T10:30:00"
}
```

### 2. Get Supported Emotions
```
GET /voice/emotions
```
Get list of supported emotions and mental state mappings.

**Response:**
```json
{
  "emotions": ["angry", "fear", "sad", "neutral", "calm", "happy", "surprise"],
  "emotion_to_state_mapping": {
    "angry": 2,
    "fear": 2,
    "sad": 2,
    "neutral": 0,
    "calm": 0,
    "happy": 1,
    "surprise": 1
  },
  "mental_states": {
    "0": "relaxed",
    "1": "focused",
    "2": "stressed"
  }
}
```

### 3. Analyze Audio File
```
POST /voice/analyze
```
Upload and analyze an audio file.

**Parameters:**
- `file` (form-data): Audio file (WAV, MP3, etc.)
- `sample_rate` (query, optional): Audio sample rate if known

**Response:**
```json
{
  "status": "success",
  "data": {
    "emotion": "happy",
    "confidence": 0.87,
    "mental_state": 1,
    "emotion_probabilities": {
      "angry": 0.02,
      "calm": 0.05,
      "happy": 0.87,
      "neutral": 0.03,
      "sad": 0.01,
      "fear": 0.01,
      "surprise": 0.01
    },
    "features": {
      "rms_energy": 0.15,
      "zero_crossing_rate": 0.08,
      "mean_amplitude": 0.12,
      "max_amplitude": 0.95,
      "duration": 3.5
    },
    "timestamp": "2024-11-22T10:30:00"
  },
  "filename": "audio.wav",
  "file_size": 112000
}
```

### 4. Analyze Multiple Audio Files (Batch)
```
POST /voice/analyze-batch
```
Analyze multiple audio files and get pattern analysis.

**Parameters:**
- `files` (form-data): Multiple audio files (max 50)
- `sample_rate` (query, optional): Audio sample rate if known

**Response:**
```json
{
  "status": "success",
  "total_files": 3,
  "processed_files": 3,
  "results": [
    {
      "filename": "audio1.wav",
      "result": { "emotion": "happy", "confidence": 0.85, ... }
    },
    ...
  ],
  "pattern_analysis": {
    "total_segments": 3,
    "dominant_emotion": "happy",
    "emotion_distribution": {
      "happy": 2,
      "neutral": 1
    },
    "average_confidence": 0.82,
    "average_mental_state": 0.67,
    "state_variability": 0.47
  }
}
```

### 5. Analyze Raw Audio Data
```
POST /voice/analyze-raw
```
Analyze raw audio data (base64 or bytes array).

**Request Body:**
```json
{
  "audio_data": {
    "data": "base64_encoded_audio_data",
    "format": "base64"
  },
  "sample_rate": 16000
}
```

## Usage Examples

### Python with requests
```python
import requests

# Analyze audio file
with open('audio.wav', 'rb') as f:
    files = {'file': ('audio.wav', f, 'audio/wav')}
    response = requests.post('http://localhost:8000/voice/analyze', files=files)
    result = response.json()
    print(f"Emotion: {result['data']['emotion']}")
    print(f"Mental State: {result['data']['mental_state']}")
```

### cURL
```bash
# Health check
curl http://localhost:8000/voice/health

# Analyze audio file
curl -X POST http://localhost:8000/voice/analyze \
  -F "file=@audio.wav"

# Get supported emotions
curl http://localhost:8000/voice/emotions
```

### JavaScript/Fetch
```javascript
// Analyze audio file
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:8000/voice/analyze', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Emotion:', data.data.emotion);
  console.log('Mental State:', data.data.mental_state);
});
```

## Testing

Run the test script:
```bash
python test_voice_api.py
```

Make sure the API server is running first:
```bash
python main.py
```

## Requirements

The voice processor requires these additional packages:
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch
- `torchaudio` - Audio processing
- `numpy` - Numerical operations

Install with:
```bash
pip install transformers torch torchaudio numpy
```

## Model Information

- **Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Base Architecture**: Wav2Vec2
- **Sample Rate**: 16kHz
- **Emotions**: 8 classes (angry, calm, disgust, fear, happy, neutral, sad, surprise)

## Integration with EEG Analysis

The voice processor can be used alongside EEG analysis to provide multimodal mental state assessment:

1. Analyze EEG data for brain activity patterns
2. Analyze voice for emotional state
3. Combine both for comprehensive mental state assessment

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Server error
- `503`: Service unavailable (model not loaded)

Error responses include details:
```json
{
  "detail": "Error message describing the issue"
}
```
