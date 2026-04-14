"""
Training Pipeline with Transfer Learning - Memory Efficient Version
Processes data in batches to avoid RAM exhaustion
FIXED: Deterministic training and proper tensor handling
"""

import numpy as np
import cv2
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import AMOLEDDefectGenerator
from src.model_transfer import TransferDefectDetector

class TransferTrainingPipeline:
    """
    Memory-efficient training pipeline using transfer learning
    Processes data in batches to avoid memory issues
    """
    
    def __init__(self, img_size: int = 128):
        """
        Initialize pipeline
        
        Args:
            img_size: Size to resize images to (square)
        """
        self.img_size = img_size
        self.generator = AMOLEDDefectGenerator(width=400, height=400)
        self.detector = TransferDefectDetector(input_shape=(img_size, img_size, 3))
        
    def prepare_data(self, 
                     num_samples: int = 2000,
                     defect_ratio: float = 0.5,
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     batch_size: int = 200) -> dict:
        """
        Generate and prepare data IN BATCHES to save memory
        """
        print("=" * 60)
        print("STEP 1: Generating Synthetic Dataset (Batch Mode)")
        print("=" * 60)
        print(f"   Total samples: {num_samples}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {self.img_size}x{self.img_size}")
        
        num_defective = int(num_samples * defect_ratio)
        num_clean = num_samples - num_defective
        
        print(f"   Defective: {num_defective}")
        print(f"   Clean: {num_clean}")
        print()
        
        all_images = []
        all_labels = []
        
        # Process defective images in batches
        print("Generating defective images:")
        for batch_start in range(0, num_defective, batch_size):
            batch_end = min(batch_start + batch_size, num_defective)
            print(f"  Batch {batch_start}-{batch_end}/{num_defective}")
            
            batch_images = []
            for i in range(batch_start, batch_end):
                img, info = self.generator.generate_defective_image()
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_normalized = img_resized / 255.0
                batch_images.append(img_normalized)
            
            all_images.extend(batch_images)
            all_labels.extend([1] * len(batch_images))
        
        # Process clean images in batches
        print("\nGenerating clean images:")
        for batch_start in range(0, num_clean, batch_size):
            batch_end = min(batch_start + batch_size, num_clean)
            print(f"  Batch {batch_start}-{batch_end}/{num_clean}")
            
            batch_images = []
            for i in range(batch_start, batch_end):
                img = self.generator.generate_clean_display()
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_normalized = img_resized / 255.0
                batch_images.append(img_normalized)
            
            all_images.extend(batch_images)
            all_labels.extend([0] * len(batch_images))
        
        # Convert to numpy arrays
        images = np.array(all_images, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)
        
        memory_mb = images.nbytes / 1024 / 1024
        print(f"\n✅ Generated {len(images)} images")
        print(f"   Memory usage: {memory_mb:.1f} MB")
        print(f"   Image shape: {images[0].shape}")
        print(f"   Data range: [{images.min():.2f}, {images.max():.2f}]")
        
        # Split data: train+val vs test
        print("\n📊 Splitting dataset...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=42, stratify=y_temp
        )
        
        print(f"\n✅ Data splits complete:")
        print(f"   - Training:   {len(X_train)} images ({len(X_train)/num_samples*100:.1f}%)")
        print(f"   - Validation: {len(X_val)} images ({len(X_val)/num_samples*100:.1f}%)")
        print(f"   - Testing:    {len(X_test)} images ({len(X_test)/num_samples*100:.1f}%)")
        
        # Show class balance
        train_defective = sum(y_train)
        val_defective = sum(y_val)
        test_defective = sum(y_test)
        
        print(f"\n📊 Class balance:")
        print(f"   Training:   {train_defective}/{len(y_train)} defective ({train_defective/len(y_train)*100:.1f}%)")
        print(f"   Validation: {val_defective}/{len(y_val)} defective ({val_defective/len(y_val)*100:.1f}%)")
        print(f"   Testing:    {test_defective}/{len(y_test)} defective ({test_defective/len(y_test)*100:.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def run(self, 
            num_samples: int = 1500,
            epochs: int = 15,
            fine_tune_epochs: int = 5,
            batch_size: int = 200) -> TransferDefectDetector:
        """
        Run complete training pipeline
        """
        print("\n" + "=" * 60)
        print("🚀 TRANSFER LEARNING - AMOLED DEFECT DETECTION")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"   - Samples: {num_samples}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Fine-tune epochs: {fine_tune_epochs}")
        print(f"   - Batch size: {batch_size}")
        
        # Prepare data
        data = self.prepare_data(
            num_samples=num_samples, 
            batch_size=batch_size
        )
        
        # Build and train model
        print("\n" + "=" * 60)
        print("STEP 2: Building Transfer Learning Model")
        print("=" * 60)
        
        self.detector.build_model()
        print("\n✅ Model built successfully!")
        print(f"   Total parameters: {self.detector.model.count_params():,}")
        
        # Train
        history = self.detector.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            epochs=epochs,
            batch_size=32,
            fine_tune_epochs=fine_tune_epochs
        )
        
        # Evaluate
        print("\n" + "=" * 60)
        print("STEP 3: Evaluating Model")
        print("=" * 60)
        
        # Get predictions
        print("Running predictions on test set...")
        y_pred_proba = self.detector.model.predict(data['X_test'], verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        print("\n" + "-" * 50)
        print("CLASSIFICATION REPORT")
        print("-" * 50)
        print(classification_report(data['y_test'], y_pred, 
                                   target_names=['Clean', 'Defective']))
        
        # Confusion matrix
        cm = confusion_matrix(data['y_test'], y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Clean', 'Defective'],
                   yticklabels=['Clean', 'Defective'])
        plt.title('Confusion Matrix - Transfer Learning')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/transfer_confusion_matrix.png', dpi=150)
        plt.show()
        
        # Calculate metrics manually
        accuracy = np.mean(y_pred == data['y_test'])
        tp = np.sum((y_pred == 1) & (data['y_test'] == 1))
        fp = np.sum((y_pred == 1) & (data['y_test'] == 0))
        fn = np.sum((y_pred == 0) & (data['y_test'] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n📊 Test Metrics:")
        print(f"   ┌─────────────────┬────────────┐")
        print(f"   │ Metric          │ Value      │")
        print(f"   ├─────────────────┼────────────┤")
        print(f"   │ Accuracy        │ {accuracy:.4f}     │")
        print(f"   │ Precision       │ {precision:.4f}     │")
        print(f"   │ Recall          │ {recall:.4f}     │")
        print(f"   └─────────────────┴────────────┘")
        
        # Plot training history
        print("\n📈 Generating training history plots...")
        self.detector.plot_training_history()
        
        # Save model
        self.detector.save_model('models/transfer_defect_detector.keras')
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        # Test on a few random images
        print("\n🎯 Quick test on random images:")
        for i in range(5):
            # Random clean or defective
            if np.random.random() > 0.5:
                img, info = self.generator.generate_defective_image()
                expected = "DEFECTIVE"
            else:
                img = self.generator.generate_clean_display()
                info = {'defects': ['none']}
                expected = "CLEAN"
            
            # Convert to numpy if needed
            if not isinstance(img, np.ndarray):
                img = img.numpy()
            
            confidence, prediction = self.detector.predict(img)
            status = "✅" if prediction == expected else "❌"
            print(f"   {status} Expected: {expected}, Got: {prediction} ({confidence:.2%} confidence)")
        
        print("\n🎯 Next steps:")
        print("   1. Run: streamlit run dashboard/app.py")
        print("   2. Upload test images to the web interface")
        print("   3. Record demo video for Jingce/Jingzhida")
        
        return self.detector


# ============= MAIN =============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AMOLED Defect Detector with Transfer Learning')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate (default: 1000)')
    parser.add_argument('--epochs', type=int, default=12,
                        help='Number of training epochs (default: 12)')
    parser.add_argument('--fine-tune', type=int, default=3,
                        help='Fine-tuning epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Batch size for data generation (default: 200)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMOLED DEFECT DETECTION - TRANSFER LEARNING")
    print("=" * 60)
    print(f"\nRunning with:")
    print(f"   --samples {args.samples}")
    print(f"   --epochs {args.epochs}")
    print(f"   --fine-tune {args.fine_tune}")
    print(f"   --batch-size {args.batch_size}")
    
    # Run pipeline
    pipeline = TransferTrainingPipeline()
    detector = pipeline.run(
        num_samples=args.samples,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune,
        batch_size=args.batch_size
    )
