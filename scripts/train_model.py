"""
Wrapper script to train the model from project root
Usage: python scripts/train_model.py
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now run the actual training script
if __name__ == "__main__":
    # Import the training module
    from src.scripts.training.train_model import main
    
    # Run training
    main()
