import os
import csv
import numpy as np

def generate_eeg_samples(num_samples=300, save_path="test_data/sample_egg.csv"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Channel names
    channels = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
                "F7","F8","T3","T4","T5","T6"]
    
    # Open file and write CSV
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["timestamp"] + channels)
        
        # Generate data
        timestamp = 0.0
        for i in range(num_samples):
            # Simulate EEG data as small random variations
            row = [f"{timestamp:.3f}"] + list(np.round(np.random.normal(0, 10, size=len(channels)), 1))
            writer.writerow(row)
            timestamp += 0.004  # 250Hz sampling interval

    print(f"{num_samples} EEG samples saved to {save_path}")

# Example usage
generate_eeg_samples()