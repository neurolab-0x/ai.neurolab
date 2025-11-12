"""Generate synthetic EEG training data for model development"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_eeg_sample(state, noise_level=0.05):
    """Generate a synthetic EEG sample for a given mental state"""
    if state == 0:  # Relaxed
        alpha = np.random.uniform(0.6, 0.9)
        beta = np.random.uniform(0.1, 0.3)
        theta = np.random.uniform(0.1, 0.25)
        delta = np.random.uniform(0.05, 0.15)
        gamma = np.random.uniform(0.05, 0.15)
    elif state == 1:  # Neutral
        alpha = np.random.uniform(0.4, 0.6)
        beta = np.random.uniform(0.3, 0.5)
        theta = np.random.uniform(0.2, 0.35)
        delta = np.random.uniform(0.15, 0.25)
        gamma = np.random.uniform(0.15, 0.25)
    else:  # Stressed
        alpha = np.random.uniform(0.2, 0.4)
        beta = np.random.uniform(0.5, 0.8)
        theta = np.random.uniform(0.3, 0.5)
        delta = np.random.uniform(0.25, 0.45)
        gamma = np.random.uniform(0.25, 0.4)
    
    # Add noise
    alpha += np.random.normal(0, noise_level)
    beta += np.random.normal(0, noise_level)
    theta += np.random.normal(0, noise_level)
    delta += np.random.normal(0, noise_level)
    gamma += np.random.normal(0, noise_level)
    
    return {
        'alpha': max(0, alpha),
        'beta': max(0, beta),
        'theta': max(0, theta),
        'delta': max(0, delta),
        'gamma': max(0, gamma),
        'state': state,
        'confidence': np.random.uniform(0.7, 0.98)
    }

def generate_dataset(n_samples=1000):
    """Generate a synthetic EEG dataset"""
    logger.info(f"Generating {n_samples} samples...")
    data = []
    start_time = datetime.now()
    
    samples_per_class = n_samples // 3
    states = [0] * samples_per_class + [1] * samples_per_class + [2] * samples_per_class
    remaining = n_samples - len(states)
    states.extend([i % 3 for i in range(remaining)])
    np.random.shuffle(states)
    
    for i, state in enumerate(states):
        sample = generate_eeg_sample(state)
        sample['timestamp'] = (start_time + timedelta(seconds=i)).isoformat()
        sample['metadata'] = f'{{"device":"synthetic","signal_quality":{sample["confidence"]:.2f}}}'
        data.append(sample)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    train_df = generate_dataset(1000)
    train_df.to_csv("test_data/training_eeg.csv", index=False)
    logger.info(f"Saved training data: {train_df.shape}")
    print(train_df.head())