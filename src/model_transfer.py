"""
Transfer Learning Model for AMOLED Defect Detection
Uses MobileNetV2 pre-trained on ImageNet for better feature extraction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class TransferDefectDetector:
    """
    Transfer learning-based defect detector using MobileNetV2
    More reliable than training from scratch, especially with limited data
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
        self.base_model = None
        
    def build_model(self, fine_tune: bool = False, fine_tune_at: int = 100) -> keras.Model:
        """
        Build transfer learning model with MobileNetV2 base
        
        Args:
            fine_tune: Whether to unfreeze some base layers
            fine_tune_at: Unfreeze layers from this index onward (if fine_tune=True)
        """
        # Load pre-trained MobileNetV2 (no top classification layers)
        self.base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model initially
        self.base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation for better generalization
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Pass through base model
        x = self.base_model(x, training=False)
        
        # Global pooling and classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile with lower learning rate for fine-tuning
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        return model
    
    def fine_tune(self, unfreeze_layers: int = 30):
        """
        Unfreeze some layers for fine-tuning after initial training
        
        Args:
            unfreeze_layers: Number of top layers to unfreeze
        """
        if self.base_model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Unfreeze the top N layers
        self.base_model.trainable = True
        
        # Freeze all layers first
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Unfreeze the last 'unfreeze_layers' layers
        for layer in self.base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # 10x lower
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Fine-tuning enabled: Last {unfreeze_layers} layers unfrozen")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 20,
              batch_size: int = 32,
              fine_tune_epochs: int = 10,
              verbose: int = 1) -> Dict:
        """
        Train the model with two phases:
        1. Train only the top layers (base model frozen)
        2. Optional: Fine-tune some base layers
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Initial training epochs
            batch_size: Batch size
            fine_tune_epochs: Additional epochs for fine-tuning (0 = skip)
            verbose: 0=silent, 1=progress bar
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks for better training
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
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Phase 1: Train only the top layers
        print("\n" + "=" * 50)
        print("PHASE 1: Training classification head")
        print("=" * 50)
        
        history1 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Phase 2: Fine-tuning (optional)
        history2 = None
        if fine_tune_epochs > 0:
            print("\n" + "=" * 50)
            print("PHASE 2: Fine-tuning base model")
            print("=" * 50)
            
            self.fine_tune(unfreeze_layers=30)
            
            history2 = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        # Combine histories (FIXED: handle both History objects correctly)
        history2_dict = history2.history if history2 is not None else {}
        
        self.history = {
            'loss': history1.history['loss'] + history2_dict.get('loss', []),
            'val_loss': history1.history['val_loss'] + history2_dict.get('val_loss', []),
            'accuracy': history1.history['accuracy'] + history2_dict.get('accuracy', []),
            'val_accuracy': history1.history['val_accuracy'] + history2_dict.get('val_accuracy', []),
            'precision': history1.history.get('precision', []) + history2_dict.get('precision', []),
            'val_precision': history1.history.get('val_precision', []) + history2_dict.get('val_precision', []),
            'recall': history1.history.get('recall', []) + history2_dict.get('recall', []),
            'val_recall': history1.history.get('val_recall', []) + history2_dict.get('val_recall', [])
        }
        
        return self.history
    
    def predict(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Predict if a single image has defects
        
        Args:
            image: Input image (height, width, channels)
            
        Returns:
            Tuple of (confidence_score, prediction_label)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess
        if image.shape[:2] != self.input_shape[:2]:
            image = tf.image.resize(image, self.input_shape[:2])
        
        # Ensure float32 and normalize if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize if not already (assuming [0,255] input)
        if image.max() > 1.0:
            image = image / 255.0
        
        image = np.expand_dims(image, axis=0)
        
        # Predict
        confidence = self.model.predict(image, verbose=0)[0][0]
        
        label = "DEFECTIVE" if confidence > 0.5 else "CLEAN"
        
        return confidence, label
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Predict on a batch of images
        
        Args:
            images: Batch of images (n, height, width, channels)
            
        Returns:
            Array of confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Preprocess batch
        if images.shape[1:3] != self.input_shape[:2]:
            import tensorflow as tf
            images = tf.image.resize(images, self.input_shape[:2])
        
        if images.dtype != np.float32:
            images = images.astype(np.float32)
        
        if images.max() > 1.0:
            images = images / 255.0
        
        return self.model.predict(images, verbose=0).flatten()
    
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
    
    def save_model(self, filepath: str = "models/transfer_defect_detector.keras"):
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
    
    def load_model(self, filepath: str = "models/transfer_defect_detector.keras"):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
        
        # Rebuild base_model reference for fine-tuning (optional)
        # The loaded model doesn't have base_model attribute, so we set to None
        self.base_model = None
    
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
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            print("Model not built yet.")
        else:
            self.model.summary()


# ============= QUICK TEST =============
if __name__ == "__main__":
    print("=" * 50)
    print("Transfer Learning Defect Detector - Test")
    print("=" * 50)
    
    # Create detector
    detector = TransferDefectDetector(input_shape=(128, 128, 3))
    
    # Build and show model
    print("\n1. Building transfer learning model...")
    detector.build_model()
    
    print("\n2. Model summary:")
    detector.summary()
    
    print("\n✅ Transfer learning model ready!")
    print("\n📊 Model characteristics:")
    print("   - Base: MobileNetV2 (pre-trained on ImageNet)")
    print("   - Trainable parameters: Only classification head initially")
    print("   - Can be fine-tuned for better performance")
    
    # Test with random data
    print("\n3. Testing with random data...")
    random_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    try:
        output = detector.model.predict(random_input, verbose=0)
        print(f"   ✅ Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
