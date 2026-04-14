"""
CNN Model for AMOLED Defect Detection
Classifies display images as "good" or "defective"
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class DefectDetector:
    """
    CNN-based defect detector for AMOLED displays
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 3)):
        """
        Initialize the detector
        
        Args:
            input_shape: Image dimensions (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build CNN architecture
        
        Architecture:
        - 3 convolutional blocks for feature extraction
        - Dropout for regularization
        - Dense layers for classification
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block (deeper features)
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and classify
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 20,
              batch_size: int = 32,
              verbose: int = 1) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: 0=silent, 1=progress bar, 2=one line per epoch
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Data augmentation for better generalization
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Apply augmentation only to training data
        X_train_augmented = data_augmentation(X_train, training=True)
        
        # Train with callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        self.history = self.model.fit(
            X_train_augmented, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Predict if a single image has defects
        
        Args:
            image: Input image (height, width, channels)
            
        Returns:
            Tuple of (confidence_score, prediction_label)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() or load_model() first.")
        
        # Preprocess
        if image.shape[:2] != self.input_shape[:2]:
            image = tf.image.resize(image, self.input_shape[:2])
        
        image = np.expand_dims(image, axis=0) / 255.0
        
        # Predict
        confidence = self.model.predict(image, verbose=0)[0][0]
        
        label = "DEFECTIVE" if confidence > 0.5 else "CLEAN"
        confidence_percent = confidence if confidence > 0.5 else 1 - confidence
        
        return confidence, label
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        return metrics
    
    def save_model(self, filepath: str = "models/defect_detector.h5"):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/defect_detector.h5"):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
    
    def plot_training_history(self):
        """
        Plot accuracy and loss curves
        """
        if self.history is None:
            print("No training history found.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.history['accuracy'], label='Train')
        axes[0].plot(self.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history['loss'], label='Train')
        axes[1].plot(self.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


# ============= DEMO CODE =============
if __name__ == "__main__":
    print("=" * 50)
    print("AMOLED Defect Detector - Model Demo")
    print("=" * 50)
    
    # Create detector
    detector = DefectDetector(input_shape=(128, 128, 3))
    
    # Build and show model summary
    print("\n1. Building model architecture...")
    detector.build_model()
    detector.model.summary()
    
    print("\n✅ Model ready for training!")
    print("\nNext step: Run src/train.py to train on generated data")
