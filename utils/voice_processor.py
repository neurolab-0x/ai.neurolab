"""
Voice Processing and Analysis Module
Uses lightweight Hugging Face models for audio detection and speech analysis
"""

import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import io
import tempfile
import os

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """
    Voice processor for emotion detection and speech analysis
    Uses lightweight HuggingFace models for efficient processing
    """
    
    def __init__(self, device: str = None):
        """
        Initialize voice processor with models
        
        Args:
            device: Device to run models on ('cpu', 'cuda', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing VoiceProcessor on device: {self.device}")
        
        self.model = None
        self.processor = None
        self.sample_rate = 16000  # Standard for wav2vec2
        
        # Emotion to mental state mapping
        self.emotion_to_state = {
            'angry': 2,      # Stressed
            'fear': 2,       # Stressed
            'sad': 2,        # Stressed
            'neutral': 0,    # Relaxed
            'calm': 0,       # Relaxed
            'happy': 1,      # Focused/Positive
            'surprise': 1    # Focused/Alert
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load emotion recognition model from HuggingFace"""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            
            # Try multiple model options
            model_options = [
                "superb/wav2vec2-base-superb-er",  # Emotion recognition model
                "facebook/wav2vec2-base",  # Fallback to base model
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Voice processing model loaded successfully: {model_name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
                    continue
            
            raise Exception("All model loading attempts failed")
            
        except Exception as e:
            logger.error(f"Error loading voice models: {str(e)}")
            logger.warning("Voice processor will run in fallback mode")

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int = None) -> np.ndarray:
        """Resample audio to target sample rate"""
        if target_sr is None:
            target_sr = self.sample_rate
        
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            logger.warning("librosa not available, using simple resampling")
            # Simple linear interpolation
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def _predict_emotion(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from audio
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (predicted_emotion, confidence, emotion_probabilities)
        """
        if self.model is None or self.processor is None:
            logger.warning("Model not loaded, using fallback emotion detection")
            return 'neutral', 1.0, {'neutral': 1.0}
        
        try:
            # Process audio
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get emotion labels
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
            
            # Map to emotion labels
            emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            predicted_emotion = emotion_labels[predicted_id] if predicted_id < len(emotion_labels) else 'neutral'
            
            # Create probability dict
            emotion_probs = {label: probabilities[0][i].item() for i, label in enumerate(emotion_labels)}
            
            return predicted_emotion, confidence, emotion_probs
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {str(e)}")
            return 'neutral', 0.5, {'neutral': 1.0}
    
    def process_audio(self, audio_data: bytes, sample_rate: int = None) -> Dict[str, Any]:
        """
        Process audio data and extract features
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Original sample rate (if known)
            
        Returns:
            Dictionary with emotion, mental state, and audio features
        """
        try:
            # Load audio from bytes
            audio_array, sr = self._load_audio_from_bytes(audio_data, sample_rate)
            
            # Normalize audio
            audio_array = self._normalize_audio(audio_array)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio_array = self._resample_audio(audio_array, sr, self.sample_rate)
            
            # Predict emotion
            emotion, confidence, emotion_probs = self._predict_emotion(audio_array)
            
            # Map to mental state
            mental_state = self.emotion_to_state.get(emotion, 0)
            
            # Extract audio features
            features = self._extract_audio_features(audio_array)
            
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'emotion_probabilities': emotion_probs,
                'mental_state': mental_state,
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Processed audio: emotion={emotion}, confidence={confidence:.2f}, state={mental_state}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'mental_state': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _load_audio_from_bytes(self, audio_data: bytes, sample_rate: int = None) -> Tuple[np.ndarray, int]:
        """Load audio from bytes"""
        # Try multiple methods to load audio
        
        # Method 1: Try scipy.io.wavfile (most reliable for WAV files)
        try:
            from scipy.io import wavfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                sr, audio_array = wavfile.read(tmp_path)
                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                return audio_array, sr
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            logger.debug(f"scipy.io.wavfile failed: {str(e)}")
        
        # Method 2: Try torchaudio (if FFmpeg is available)
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                waveform, sr = torchaudio.load(tmp_path)
                audio_array = waveform.numpy()[0]  # Take first channel
                return audio_array, sr
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            logger.debug(f"torchaudio failed: {str(e)}")
        
        # Method 3: Fallback - assume raw PCM data
        logger.warning("Using fallback audio loading (raw PCM)")
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return audio_array, sample_rate
    
    def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract basic audio features"""
        try:
            features = {
                'rms_energy': float(np.sqrt(np.mean(audio**2))),
                'zero_crossing_rate': float(np.mean(np.abs(np.diff(np.sign(audio))) / 2)),
                'mean_amplitude': float(np.mean(np.abs(audio))),
                'max_amplitude': float(np.max(np.abs(audio))),
                'duration': len(audio) / self.sample_rate
            }
            
            # Spectral features if possible
            try:
                fft = np.fft.rfft(audio)
                magnitude = np.abs(fft)
                features['spectral_centroid'] = float(np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude))
                features['spectral_rolloff'] = float(np.percentile(magnitude, 85))
            except:
                pass
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def analyze_speech_patterns(self, audio_segments: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze patterns across multiple audio segments
        
        Args:
            audio_segments: List of audio arrays
            
        Returns:
            Analysis results including trends and patterns
        """
        if not audio_segments:
            return {'error': 'No audio segments provided'}
        
        emotions = []
        confidences = []
        states = []
        
        for segment in audio_segments:
            emotion, confidence, _ = self._predict_emotion(segment)
            emotions.append(emotion)
            confidences.append(confidence)
            states.append(self.emotion_to_state.get(emotion, 0))
        
        # Calculate statistics
        from collections import Counter
        emotion_counts = Counter(emotions)
        
        analysis = {
            'total_segments': len(audio_segments),
            'dominant_emotion': emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral',
            'emotion_distribution': dict(emotion_counts),
            'average_confidence': float(np.mean(confidences)),
            'average_mental_state': float(np.mean(states)),
            'state_variability': float(np.std(states)),
            'emotions': emotions,
            'confidences': confidences,
            'mental_states': states
        }
        
        return analysis