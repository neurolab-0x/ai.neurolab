import time
import tensorflow as tf
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load your trained model
model_path = "processed/trained_model.h5"
if not os.path.exists(model_path):
    logger.error(f"Model not found at {model_path}")
    sys.exit(1)

try:
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sys.exit(1)

def generate_dummy_data(num_samples=100, sequence_length=5, num_classes=3):
    """
    Generate dummy EEG data with correct shape for Conv1D layer.
    
    Args:
        num_samples: Number of samples to generate
        sequence_length: Length of each EEG sequence (default: 5 features)
        num_classes: Number of output classes
    
    Returns:
        X: Data with shape (num_samples, sequence_length, 1)
        y: Labels with shape (num_samples,)
    """
    X = np.random.rand(num_samples, sequence_length, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    logger.info(f"Generated {num_samples} samples with shape {X.shape}")
    return X, y

def evaluate_model(model, X_test, y_test, batch_size=16):
    """
    Evaluate model performance with various metrics.
    
    Parameters:
    -----------
    model : Keras model
        Trained model to evaluate
    X_test : array
        Test data
    y_test : array
        Test labels
    batch_size : int
        Batch size for inference
        
    Returns:
    --------
    dict : Performance metrics
    """
    inference_times = []
    y_preds = []
    
    logger.info(f"Evaluating model on {len(X_test)} samples with batch size {batch_size}")
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        start_time = time.time()
        outputs = model.predict(batch, verbose=0)
        inference_times.append(time.time() - start_time)
        
        preds = np.argmax(outputs, axis=1)
        y_preds.extend(preds)
    
    # Compute performance metrics with 'weighted' average for multiclass
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_preds, average='weighted', zero_division=0)
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    throughput = len(X_test) / total_inference_time  # samples per second
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_preds)
    
    # Model size
    temp_model_path = "temp_model_benchmark.h5"
    model.save(temp_model_path)
    model_size = round((os.path.getsize(temp_model_path) / 1e6), 2)  # Size in MB
    os.remove(temp_model_path)
    
    # Count model parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Avg Inference Time (s)": round(avg_inference_time, 4),
        "Total Inference Time (s)": round(total_inference_time, 4),
        "Throughput (samples/s)": round(throughput, 2),
        "Model Size (MB)": model_size,
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Non-trainable Parameters": non_trainable_params,
        "Confusion Matrix": conf_matrix.tolist()
    }

def run_benchmark():
    """Run comprehensive benchmark tests"""
    logger.info("="*60)
    logger.info("Starting Model Performance Benchmark")
    logger.info("="*60)
    
    # Generate test data
    X_test, y_test = generate_dummy_data(num_samples=100, sequence_length=5)
    
    # Run evaluation
    results = evaluate_model(model, X_test, y_test, batch_size=16)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("Benchmark Results:")
    logger.info("="*60)
    for key, value in results.items():
        if key != "Confusion Matrix":
            logger.info(f"{key:.<40} {value}")
    
    logger.info("\nConfusion Matrix:")
    for row in results["Confusion Matrix"]:
        logger.info(f"  {row}")
    
    logger.info("="*60)
    
    return results

if __name__ == "__main__":
    try:
        results = run_benchmark()
        logger.info("Benchmark completed successfully")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
