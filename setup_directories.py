"""
Setup script to create all necessary directories for the NeuroLab project
"""

import os

# Define all required directories
DIRECTORIES = [
    # Data directories
    'data',
    'data/training_data',
    'data/testing_data',
    'data/raw',
    'data/processed',
    
    # Model directories
    'model',
    'model/checkpoints',
    'model/archived',
    
    # Output directories
    'output',
    'output/reports',
    'output/plots',
    'output/logs',
    
    # Training results
    'training_results',
    'training_results/plots',
    'training_results/metrics',
    
    # Logs
    'logs',
    'logs/training',
    'logs/api',
    'logs/processing',
    
    # Processed data
    'processed',
    
    # Checkpoints
    'checkpoints',
]

def setup_directories():
    """Create all required directories"""
    print("Setting up NeuroLab directory structure...")
    print("=" * 60)
    
    created_count = 0
    existing_count = 0
    
    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created: {directory}")
            created_count += 1
        else:
            print(f"  Exists:  {directory}")
            existing_count += 1
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Created: {created_count} directories")
    print(f"  Existing: {existing_count} directories")
    print(f"  Total: {len(DIRECTORIES)} directories")
    print("\n✓ Directory setup complete!")

if __name__ == "__main__":
    setup_directories()
