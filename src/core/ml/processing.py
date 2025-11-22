"""
DEPRECATED: This module has been consolidated into the preprocessing package.

Please update your imports:
- load_data -> from preprocessing import load_data
- extract_features -> from preprocessing import extract_features  
- label_eeg_states -> from preprocessing.labeling import label_eeg_states
- preprocess_data -> from preprocessing import preprocess_data
- temporal_smoothing -> from utils.temporal_processing import temporal_smoothing
- calculate_state_durations -> from utils.temporal_processing import calculate_state_durations

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings
import numpy as np
from typing import Dict, List

warnings.warn(
    "core.ml.processing is deprecated. Use preprocessing package instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility wrappers
def load_data(file_path: str) -> np.ndarray:
    """DEPRECATED: Use preprocessing.load_data instead"""
    warnings.warn("Use preprocessing.load_data instead", DeprecationWarning)
    from preprocessing import load_data as new_load_data
    result = new_load_data(file_path)
    if hasattr(result, 'values'):
        return result.values
    return result

def extract_features(data: np.ndarray):
    """DEPRECATED: Use preprocessing.extract_features instead"""
    warnings.warn("Use preprocessing.extract_features instead", DeprecationWarning)
    from preprocessing import extract_features as new_extract_features
    return new_extract_features(data)

def label_eeg_states(data: np.ndarray) -> np.ndarray:
    """DEPRECATED: Use preprocessing.labeling.label_eeg_states instead"""
    warnings.warn("Use preprocessing.labeling.label_eeg_states instead", DeprecationWarning)
    from preprocessing.labeling import label_eeg_states as new_label_eeg_states
    import pandas as pd
    
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=['alpha', 'beta', 'theta', 'delta', 'gamma'])
        result_df = new_label_eeg_states(df)
        return result_df['eeg_state'].values
    return new_label_eeg_states(data)

def preprocess_data(features):
    """DEPRECATED: Use preprocessing.preprocess_data instead"""
    warnings.warn("Use preprocessing.preprocess_data instead", DeprecationWarning)
    from preprocessing import preprocess_data as new_preprocess_data
    return new_preprocess_data(features)

def temporal_smoothing(predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """DEPRECATED: Use utils.temporal_processing.temporal_smoothing instead"""
    warnings.warn("Use utils.temporal_processing.temporal_smoothing instead", DeprecationWarning)
    from utils.temporal_processing import temporal_smoothing as new_temporal_smoothing
    return new_temporal_smoothing(predictions, window_size)

def calculate_state_durations(states: np.ndarray) -> Dict[int, float]:
    """DEPRECATED: Use utils.temporal_processing.calculate_state_durations instead"""
    warnings.warn("Use utils.temporal_processing.calculate_state_durations instead", DeprecationWarning)
    from utils.temporal_processing import calculate_state_durations as new_calculate_state_durations
    return new_calculate_state_durations(states)

def generate_recommendations(state_durations: Dict[int, float], total_duration: float, confidence: float) -> List[Dict]:
    """DEPRECATED: Use utils.recommendations.generate_recommendations instead"""
    warnings.warn("Use utils.recommendations.generate_recommendations instead", DeprecationWarning)
    from utils.recommendations import generate_recommendations as new_generate_recommendations
    return new_generate_recommendations(state_durations, total_duration, confidence)
