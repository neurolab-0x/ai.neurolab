"""
Test script for ML Processor integration
"""
import numpy as np
import logging
from src.utils.ml_processor import MLProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_processor():
    """Test the ML Processor with different input types"""
    
    print("=" * 60)
    print("Testing ML Processor Integration")
    print("=" * 60)
    
    # Initialize processor
    processor = MLProcessor()
    
    # Check status
    status = processor.get_status()
    print(f"\nProcessor Status:")
    print(f"  Model Loaded: {status['model_loaded']}")
    print(f"  Model Path: {status['model_path']}")
    print(f"  Model Exists: {status['model_exists']}")
    
    # Test 1: Dictionary input (single sample)
    print("\n" + "=" * 60)
    print("Test 1: Dictionary Input (Single Sample)")
    print("=" * 60)
    
    test_data_dict = {
        'alpha': 0.5,
        'beta': 0.3,
        'theta': 0.2,
        'delta': 0.1,
        'gamma': 0.4
    }
    
    try:
        result = processor.process_eeg_data(
            test_data_dict,
            subject_id="test_subject_001",
            session_id="test_session_001"
        )
        
        print(f"\nResults:")
        print(f"  Dominant State: {result['state_label']} (state {result['dominant_state']})")
        print(f"  Confidence: {result['confidence']:.2f}%")
        print(f"  State Percentages: {result['state_percentages']}")
        print(f"  Recommendations: {result['recommendations']}")
        print(f"  Cognitive Metrics:")
        for metric, value in result['cognitive_metrics'].items():
            print(f"    {metric}: {value:.4f}")
        
        print("\n✓ Test 1 PASSED")
        
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Numpy array input (multiple samples)
    print("\n" + "=" * 60)
    print("Test 2: Numpy Array Input (Multiple Samples)")
    print("=" * 60)
    
    test_data_array = np.array([
        [0.5, 0.3, 0.2, 0.1, 0.4],  # Sample 1
        [0.6, 0.4, 0.3, 0.2, 0.5],  # Sample 2
        [0.4, 0.2, 0.1, 0.05, 0.3], # Sample 3
        [0.7, 0.5, 0.4, 0.3, 0.6],  # Sample 4
        [0.3, 0.1, 0.05, 0.02, 0.2] # Sample 5
    ])
    
    try:
        result = processor.process_eeg_data(
            test_data_array,
            subject_id="test_subject_002",
            session_id="test_session_002"
        )
        
        print(f"\nResults:")
        print(f"  Total Samples: {result['temporal_analysis']['total_samples']}")
        print(f"  Dominant State: {result['state_label']} (state {result['dominant_state']})")
        print(f"  Confidence: {result['confidence']:.2f}%")
        print(f"  State Transitions: {result['temporal_analysis']['state_transitions']}")
        print(f"  State Distribution: {result['state_percentages']}")
        print(f"  Recommendations: {result['recommendations']}")
        
        print("\n✓ Test 2 PASSED")
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 3: File input (if training data exists)
    print("\n" + "=" * 60)
    print("Test 3: File Input (CSV)")
    print("=" * 60)
    
    try:
        result = processor.process_eeg_data(
            "test_data/training_eeg.csv",
            subject_id="test_subject_003",
            session_id="test_session_003"
        )
        
        print(f"\nResults:")
        print(f"  Total Samples: {result['temporal_analysis']['total_samples']}")
        print(f"  Dominant State: {result['state_label']} (state {result['dominant_state']})")
        print(f"  Confidence: {result['confidence']:.2f}%")
        print(f"  State Transitions: {result['temporal_analysis']['state_transitions']}")
        print(f"  Recommendations: {result['recommendations']}")
        
        print("\n✓ Test 3 PASSED")
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {str(e)}")
        print("  (This is expected if training_eeg.csv doesn't exist)")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_ml_processor()
