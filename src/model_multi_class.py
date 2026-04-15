"""
Multi-Class Defect Classifier for AMOLED Displays
Classes: Clean, Dead Pixel, Stuck Pixel, Mura, Scratch, Dust
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class MultiClassDefectDetector:
    def __init__(self, input_shape=(128, 128, 3), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.base_model = None
        self.class_names = ['Clean', 'Dead_Pixel', 'Stuck_Pixel', 'Mura', 'Scratch', 'Dust']
    
    def build_model(self):
        # Load pre-trained MobileNetV2
        self.base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        x = self.base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model
    
    def fine_tune(self, unfreeze_layers=30):
        self.base_model.trainable = True
        for layer in self.base_model.layers:
            layer.trainable = False
        for layer in self.base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Fine-tuning enabled: Last {unfreeze_layers} layers unfrozen")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, fine_tune_epochs=5):
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        print("\nPhase 1: Training classification head")
        history1 = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1)
        
        if fine_tune_epochs > 0:
            print("\nPhase 2: Fine-tuning")
            self.fine_tune()
            history2 = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                      epochs=fine_tune_epochs, batch_size=32, callbacks=callbacks, verbose=1)
        else:
            history2 = None
        
        # Combine histories
        hist2_dict = history2.history if history2 else {}
        self.history = {
            'accuracy': history1.history['accuracy'] + hist2_dict.get('accuracy', []),
            'val_accuracy': history1.history['val_accuracy'] + hist2_dict.get('val_accuracy', []),
            'loss': history1.history['loss'] + hist2_dict.get('loss', []),
            'val_loss': history1.history['val_loss'] + hist2_dict.get('val_loss', [])
        }
        return self.history
    
    def predict(self, image: np.ndarray) -> Tuple[int, str, float]:
        """Returns (class_index, class_name, confidence)"""
        if self.model is None:
            raise ValueError("Model not trained")
        # Preprocess
        if image.shape[:2] != self.input_shape[:2]:
            import cv2
            image = cv2.resize(image, self.input_shape[:2])
        if image.max() > 1.0:
            image = image / 255.0
        img_batch = np.expand_dims(image, axis=0)
        probs = self.model.predict(img_batch, verbose=0)[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        return class_idx, self.class_names[class_idx], float(confidence)
    
    def save_model(self, path):
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        print(f"✅ Model loaded from {path}")
