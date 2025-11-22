"""
Wrapper script to test voice API from project root
Usage: python scripts/test_voice.py
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now run the voice test script
if __name__ == "__main__":
    # Import the test module
    from src.tests import test_voice_api
    
    # Run tests
    test_voice_api.main()
