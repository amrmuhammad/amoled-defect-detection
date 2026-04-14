"""
Self-contained training with V2 generator (128x128 native resolution)
No import issues - everything in one file
"""

import numpy as np
import cv2
import random
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================
# V2 DEFECT GENERATOR (128x128 native)
# ============================================================

class AMOLEDDefectGeneratorV2:
    """Generate defects directly at 128x128 resolution"""
    
    def __init__(self, width: int = 128, height: int = 128):
        self.width = width
        self.height = height
        
    def generate_clean_display(self, brightness: int = 128) -> np.ndarray:
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * brightness
        noise = np.random.normal(0, 2, (self.height, self.width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return image
    
    def add_dead_pixel_cluster(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        center = (random.randint(10, w-10), random.randint(10, h-10))
        radius = random.randint(2, 5)
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx*dx + dy*dy <= radius*radius:
                    x = center[0] + dx
                    y = center[1] + dy
                    if 0 <= x < w and 0 <= y < h:
                        distance = np.sqrt(dx*dx + dy*dy)
                        darkness = int(255 * (1 - distance / radius) * random.uniform(0.6, 0.95))
                        img[y, x] = [darkness, darkness, darkness]
        return img
    
    def add_stuck_pixel(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        center = (random.randint(10, w-10), random.randint(10, h-10))
        radius = random.randint(2, 4)
        
        color = (random.choice([255, 200]), random.choice([255, 200]), random.choice([255, 200]))
        channel = random.randint(0, 2)
        color_list = list(color)
        color_list[channel] = 255
        color = tuple(color_list)
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx*dx + dy*dy <= radius*radius:
                    x = center[0] + dx
                    y = center[1] + dy
                    if 0 <= x < w and 0 <= y < h:
                        distance = np.sqrt(dx*dx + dy*dy)
                        intensity = 1.0 - (distance / radius) * 0.3
                        pixel_color = tuple(int(c * intensity) for c in color)
                        img[y, x] = pixel_color
        return img
    
    def add_mura(self, image: np.ndarray) -> np.ndarray:
        img = image.copy().astype(np.float32)
        h, w = img.shape[:2]
        intensity = random.uniform(0.5, 0.9)
        
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        pattern = np.sin(3 * xx) * np.cos(3 * yy)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        pattern = (pattern - 0.5) * intensity * 2.0
        
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 + pattern)
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def add_scratch(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        start = (random.randint(5, w-5), random.randint(5, h-5))
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(15, 50)
        end = (int(start[0] + length * np.cos(angle)), int(start[1] + length * np.sin(angle)))
        color = (0, 0, 0) if random.random() > 0.3 else (255, 255, 255)
        cv2.line(img, start, end, color, 1)
        return img
    
    def add_dust(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        num_particles = random.randint(2, 8)
        for _ in range(num_particles):
            x = random.randint(2, w-2)
            y = random.randint(2, h-2)
            radius = random.randint(1, 2)
            color = random.choice([(0, 0, 0), (30, 30, 30), (200, 200, 200)])
            cv2.circle(img, (x, y), radius, color, -1)
        return img
    
    def generate_defective_image(self) -> tuple:
        image = self.generate_clean_display()
        defects = ['dead_pixel', 'stuck_pixel', 'mura', 'scratch', 'dust']
        num_defects = random.randint(2, 4)
        selected = random.sample(defects, num_defects)
        
        for defect in selected:
            if defect == 'dead_pixel':
                image = self.add_dead_pixel_cluster(image)
            elif defect == 'stuck_pixel':
                image = self.add_stuck_pixel(image)
            elif defect == 'mura':
                image = self.add_mura(image)
            elif defect == 'scratch':
                image = self.add_scratch(image)
            elif defect == 'dust':
                image = self.add_dust(image)
        
        return image, selected

# ============================================================
# TRANSFER LEARNING MODEL
# ============================================================

class TransferDefectDetector:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights='imagenet'
        )
        base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                             epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1)
    
    def save_model(self, path):
        self.model.save(path)
        print(f"✅ Model saved to {path}")

# ============================================================
# MAIN TRAINING
# ============================================================

print("=" * 60)
print("TRAINING V2 - 128x128 Native Resolution")
print("=" * 60)

# Initialize
gen = AMOLEDDefectGeneratorV2()
detector = TransferDefectDetector()

# Generate dataset
print("\n📊 Generating 3000 images at 128x128...")
images = []
labels = []

for i in range(3000):
    if i % 500 == 0:
        print(f"   Progress: {i}/3000")
    
    if random.random() > 0.5:
        img, _ = gen.generate_defective_image()
        labels.append(1)
    else:
        img = gen.generate_clean_display()
        labels.append(0)
    
    img_normalized = img.astype(np.float32) / 255.0
    images.append(img_normalized)

images = np.array(images)
labels = np.array(labels)
print(f"✅ Generated {len(images)} images")

# Split
print("\n📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

print(f"   Training: {len(X_train)}")
print(f"   Validation: {len(X_val)}")
print(f"   Testing: {len(X_test)}")

# Build and train
print("\n📊 Building model...")
detector.build_model()
print(f"   Model parameters: {detector.model.count_params():,}")

print("\n📊 Training...")
history = detector.train(X_train, y_train, X_val, y_val, epochs=20)

# Evaluate
print("\n📊 Evaluating on test set...")
y_pred_proba = detector.model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
accuracy = np.mean(y_pred == y_test)
print(f"\n{'='*60}")
print(f"✅ TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

# Save model
detector.save_model('models/defect_detector_v2.keras')

# Quick test
print("\n📊 Quick validation on random images:")
for i in range(5):
    if random.random() > 0.5:
        img, defects = gen.generate_defective_image()
        expected = "DEFECTIVE"
    else:
        img = gen.generate_clean_display()
        defects = []
        expected = "CLEAN"
    
    img_norm = img.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    pred_proba = detector.model.predict(img_batch, verbose=0)[0][0]
    prediction = "DEFECTIVE" if pred_proba > 0.5 else "CLEAN"
    status = "✅" if prediction == expected else "❌"
    print(f"   {status} Expected: {expected}, Got: {prediction} ({pred_proba:.2%})")

print("\n✅ Training complete! Model saved to models/defect_detector_v2.keras")
