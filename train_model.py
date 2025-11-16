import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os
import logging
from core.data.handler import DataHandler
from preprocessing import load_data, extract_features, preprocess_data
from preprocessing.labeling import label_eeg_states
from core.ml.model import create_model as create_base_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model creation moved to core/ml/model.py - using create_base_model imported above

def prepare_training_data(data_path: str):
    """Prepare data for training"""
    try:
        logger.info("Loading data...")
        # Load data using pandas
        df = pd.read_csv(data_path)
        
        # Check dataset size
        logger.info(f"Dataset size: {len(df)} samples")
        if len(df) < 50:
            logger.warning(f"Dataset is very small ({len(df)} samples). Consider using more data for better results.")
        
        # Extract features and labels
        X = df[['alpha', 'beta', 'theta', 'delta', 'gamma']].values
        y = df['state'].values
        
        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-10)
        
        # Split into train/test sets (80/20)
        split_idx = int(0.8 * len(X_normalized))
        X_train = X_normalized[:split_idx]
        X_test = X_normalized[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Reshape for CNN
        X_train = X_train.reshape(-1, X_train.shape[1], 1)
        X_test = X_test.reshape(-1, X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_model(data_path: str, model_save_path: str):
    """Train the model using the provided data"""
    try:
        logger.info("Preparing training data...")
        X_train, X_test, y_train, y_test = prepare_training_data(data_path)
        
        # Adjust batch size based on dataset size
        batch_size = min(32, max(1, len(X_train) // 4))
        logger.info(f"Using batch size: {batch_size}")
        
        # Create and train model
        logger.info("Creating model...")
        model = create_base_model(input_shape=(X_train.shape[1], 1))
        
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        # Make predictions to see what the model is outputting
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        logger.info(f"Predicted classes: {predicted_classes}")
        logger.info(f"Actual classes: {y_test}")
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def main():
    """Main training script"""
    try:
        # Use training data
        data_path = "train_data/training.csv"
        model_save_path = "processed/trained_model.h5"
        
        logger.info(f"Starting model training with data from {data_path}")
        model, history = train_model(data_path, model_save_path)
        
        # Print training summary
        logger.info("Training completed successfully")
        logger.info(f"Final model saved to {model_save_path}")
        logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()