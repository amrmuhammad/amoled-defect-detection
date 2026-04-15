"""
Memory-efficient multi-class training using batch generation
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from src.data_generator_v2 import AMOLEDDefectGeneratorV2
from src.model_multi_class import MultiClassDefectDetector

defect_to_class = {
    'clean': 0,
    'dead_pixel': 1,
    'stuck_pixel': 2,
    'mura': 3,
    'scratch': 4,
    'dust': 5
}

def generate_data_in_batches(num_samples=1500, batch_size=500):
    """
    Generate data in batches to avoid loading everything into memory.
    Returns X, y as numpy arrays (still in memory but fewer samples).
    """
    gen = AMOLEDDefectGeneratorV2(width=128, height=128)
    all_images = []
    all_labels = []
    
    print(f"Generating {num_samples} images in batches of {batch_size}...")
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        print(f"  Batch {batch_start}-{batch_end}")
        
        batch_images = []
        batch_labels = []
        for i in range(batch_start, batch_end):
            defect_type = random.choice(['clean', 'dead_pixel', 'stuck_pixel', 'mura', 'scratch', 'dust'])
            if defect_type == 'clean':
                img = gen.generate_clean_display()
            else:
                img, _ = gen.generate_defective_image(defect_types=[defect_type])
            img_norm = img.astype(np.float32) / 255.0
            batch_images.append(img_norm)
            batch_labels.append(defect_to_class[defect_type])
        
        all_images.extend(batch_images)
        all_labels.extend(batch_labels)
    
    return np.array(all_images), np.array(all_labels)

if __name__ == "__main__":
    print("="*60)
    print("Multi-Class Training (Memory Efficient)")
    print("Classes: Clean(0), Dead_Pixel(1), Stuck_Pixel(2), Mura(3), Scratch(4), Dust(5)")
    print("="*60)
    
    # Use fewer samples: 1500 instead of 3000
    NUM_SAMPLES = 1500   # Adjust down to 1000 if still killed
    X, y = generate_data_in_batches(num_samples=NUM_SAMPLES, batch_size=500)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    detector = MultiClassDefectDetector(input_shape=(128,128,3), num_classes=6)
    detector.build_model()
    print(f"Model parameters: {detector.model.count_params():,}")
    
    # Train with fewer epochs initially
    detector.train(X_train, y_train, X_val, y_val, epochs=15, fine_tune_epochs=3)
    
    # Evaluate
    y_pred_proba = detector.model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    detector.save_model('models/multi_class_defect_detector.keras')
    
    # Quick test
    gen = AMOLEDDefectGeneratorV2()
    print("\nQuick test on random images:")
    for _ in range(5):
        defect = random.choice(['clean', 'dead_pixel', 'stuck_pixel', 'mura', 'scratch', 'dust'])
        if defect == 'clean':
            img = gen.generate_clean_display()
        else:
            img, _ = gen.generate_defective_image(defect_types=[defect])
        idx, name, conf = detector.predict(img)
        expected = defect.replace('_', ' ').title()
        status = "✅" if name.replace('_', ' ').lower() == defect else "❌"
        print(f"  {status} Expected: {expected}, Got: {name} ({conf:.2%})")
