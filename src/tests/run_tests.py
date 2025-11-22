#!/usr/bin/env python
"""
Test runner script for the NeuroLab project.
Runs all tests and generates a coverage report.
"""
import sys
import os
import unittest
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_all_tests():
    """Discover and run all tests"""
    logger.info("="*60)
    logger.info("Starting Neurolab Test Suite")
    logger.info("="*60)
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    logger.info("="*60)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
