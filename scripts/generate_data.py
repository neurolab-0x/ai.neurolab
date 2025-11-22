"""
Wrapper script to generate training data from project root
Usage: python scripts/generate_data.py
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now run the data generation script
if __name__ == "__main__":
    # Import the generation module
    from src.scripts.generation import generate_train_datasets
    
    print("=" * 60)
    print("EEG Training Data Generator")
    print("=" * 60)
    
    # Generate training data
    df = generate_train_datasets.generate_training_data(
        samples_per_state=10000,
        include_transitions=True
    )
    
    # Save to file
    generate_train_datasets.save_training_data(df)
    
    print("\n✓ Data generation complete!")
