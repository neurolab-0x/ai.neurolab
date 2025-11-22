"""
Test script for Voice Processing API
Tests the voice analysis endpoints
"""

import requests
import json
import base64
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_voice_health():
    """Test voice processor health endpoint"""
    print("\n=== Testing Voice Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/voice/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_get_emotions():
    """Test get supported emotions endpoint"""
    print("\n=== Testing Get Emotions Endpoint ===")
    response = requests.get(f"{BASE_URL}/voice/emotions")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_analyze_audio(audio_file_path):
    """Test audio analysis with file upload"""
    print(f"\n=== Testing Audio Analysis with {audio_file_path} ===")
    
    if not Path(audio_file_path).exists():
        print(f"Error: File {audio_file_path} not found")
        return False
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': (Path(audio_file_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/voice/analyze", files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Print key results
        if 'data' in result:
            data = result['data']
            print(f"\n--- Key Results ---")
            print(f"Emotion: {data.get('emotion')}")
            print(f"Confidence: {data.get('confidence'):.2f}")
            print(f"Mental State: {data.get('mental_state')} (0=relaxed, 1=focused, 2=stressed)")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_analyze_raw_audio():
    """Test raw audio analysis with base64 data"""
    print("\n=== Testing Raw Audio Analysis ===")
    
    # Create a simple sine wave as test audio (1 second at 16kHz)
    import numpy as np
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    payload = {
        "audio_data": {
            "data": audio_base64,
            "format": "base64"
        },
        "sample_rate": sample_rate
    }
    
    response = requests.post(
        f"{BASE_URL}/voice/analyze-raw",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_batch_analysis(audio_files):
    """Test batch audio analysis"""
    print(f"\n=== Testing Batch Audio Analysis with {len(audio_files)} files ===")
    
    files = []
    for audio_file in audio_files:
        if Path(audio_file).exists():
            files.append(('files', (Path(audio_file).name, open(audio_file, 'rb'), 'audio/wav')))
        else:
            print(f"Warning: File {audio_file} not found, skipping")
    
    if not files:
        print("Error: No valid files to upload")
        return False
    
    response = requests.post(f"{BASE_URL}/voice/analyze-batch", files=files)
    
    # Close file handles
    for _, (_, f, _) in files:
        f.close()
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result.get('processed_files')} files")
        if result.get('pattern_analysis'):
            print(f"\n--- Pattern Analysis ---")
            print(f"Dominant Emotion: {result['pattern_analysis'].get('dominant_emotion')}")
            print(f"Average Confidence: {result['pattern_analysis'].get('average_confidence'):.2f}")
            print(f"Average Mental State: {result['pattern_analysis'].get('average_mental_state'):.2f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 60)
    print("Voice Processing API Test Suite")
    print("=" * 60)
    print("\nMake sure the API server is running on http://localhost:8000")
    print("Start it with: uvicorn main:app")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"✓ Server is running")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Server is not running!")
        print("  Start it with: uvicorn main:app")
        return
    except Exception as e:
        print(f"✗ Error connecting to server: {e}")
        return
    
    # Test health endpoint
    try:
        test_voice_health()
    except Exception as e:
        print(f"Error testing health: {e}")
    
    # Test emotions endpoint
    try:
        test_get_emotions()
    except Exception as e:
        print(f"Error testing emotions: {e}")
    
    # Test with sample audio file (if available)
    sample_audio = "test_audio.wav"
    if Path(sample_audio).exists():
        try:
            test_analyze_audio(sample_audio)
        except Exception as e:
            print(f"Error testing audio analysis: {e}")
    else:
        print(f"\n=== Skipping file upload test (no {sample_audio} found) ===")
        print("Generate test audio with: python generate_test_audio.py")
    
    # Test raw audio analysis
    try:
        test_analyze_raw_audio()
    except Exception as e:
        print(f"Error testing raw audio: {e}")
    
    # Test batch analysis if multiple files exist
    batch_files = [f"test_audio_{i}.wav" for i in range(1, 4)]
    existing_files = [f for f in batch_files if Path(f).exists()]
    if len(existing_files) >= 2:
        try:
            test_batch_analysis(existing_files)
        except Exception as e:
            print(f"Error testing batch analysis: {e}")
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
