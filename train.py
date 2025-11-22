"""
Wrapper script to run model training from project root
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the training script
if __name__ == "__main__":
    from src.scripts.training import train_model
    
    # The train_model module will execute when imported
    print("Training script completed!")
