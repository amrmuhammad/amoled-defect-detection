"""
Training Pipeline for AMOLED Defect Detection
Combines data generation with model training
"""

import numpy as np
import cv2
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import AMOLEDDefectGenerator
from src.model import DefectDetector

class TrainingPipeline:
    """
    Complete training pipeline for defect detection
    """
    
    def __init__(self, img_size: int = 128):
        """
        Initialize pipeline
        
        Args:
            img_size: Size to resize images to (square)
        """
        self.img_size = img_size
        self.generator = AMOLEDDefectGenerator(width=400, height=400)
        self.detector = DefectDetector(input_shape=(img_size, img_size, 3))
        
    def prepare_data(self, 
                     num_samples: int = 2000,
                     defect_ratio: float = 0.5,
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> dict:
        """
        Generate and prepare data for training
        
        Args:
            num_samples: Total number of images to generate
            defect_ratio: Proportion of defective images
            test_size: Proportion for testing
            val_size: Proportion for validation
            
        Returns:
            Dictionary with train/val/test splits
        """
        print("=" * 60)
        print("STEP 1: Generating Synthetic Dataset")
        print("=" * 60)
        
        # Generate raw images
        images_raw, labels_raw = self.generator.generate_dataset(
            num_samples=num_samples,
            defect_ratio=defect_ratio
        )
        
        print(f"\n✅ Generated {len(images_raw)} images")
        print(f"   - Defective: {sum(labels_raw)}")
        print(f"   - Clean: {len(labels_raw) - sum(labels_raw)}")
        
        # Resize and normalize images
        print("\n📊 Preprocessing images...")
        images = []
        for img in images_raw:
            img_resized = cv2.resize(img, (self.img_size, self.img_size))
            img_normalized = img_resized / 255.0
            images.append(img_normalized)
        
        images = np.array(images)
        labels = np.array(labels_raw)
        
        print(f"   Image shape: {images[0].shape}")
        print(f"   Data range: [{images.min():.2f}, {images.max():.2f}]")
        
        # Split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=42, stratify=y_temp
        )
        
        print(f"\n✅ Data splits complete:")
        print(f"   - Training: {len(X_train)} images")
        print(f"   - Validation: {len(X_val)} images")
        print(f"   - Testing: {len(X_test)} images")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train(self, data: dict, epochs: int = 30) -> dict:
        """
        Train the model
        
        Args:
            data: Dictionary from prepare_data()
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        print("\n" + "=" * 60)
        print("STEP 2: Training Model")
        print("=" * 60)
        
        history = self.detector.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def evaluate(self, data: dict) -> dict:
        """
        Evaluate model on test set
        
        Args:
            data: Dictionary from prepare_data()
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("STEP 3: Evaluating Model")
        print("=" * 60)
        
        # Get predictions
        y_pred_proba = self.detector.model.predict(data['X_test'])
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Classification report
        print("\nClassification Report:")
        print("-" * 40)
        print(classification_report(data['y_test'], y_pred, 
                                     target_names=['Clean', 'Defective']))
        
        # Confusion matrix
        cm = confusion_matrix(data['y_test'], y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Clean', 'Defective'],
                    yticklabels=['Clean', 'Defective'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png', dpi=150)
        plt.show()
        
        # Metrics
        metrics = self.detector.evaluate(data['X_test'], data['y_test'])
        
        print(f"\n📊 Test Metrics:")
        print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   - Precision: {metrics['precision']:.4f}")
        print(f"   - Recall:    {metrics['recall']:.4f}")
        
        return metrics
    
    def run(self, 
            num_samples: int = 2000,
            epochs: int = 30,
            save_model: bool = True) -> DefectDetector:
        """
        Run complete training pipeline
        
        Args:
            num_samples: Number of images to generate
            epochs: Training epochs
            save_model: Whether to save the trained model
            
        Returns:
            Trained detector
        """
        print("\n" + "=" * 60)
        print("🚀 AMOLED DEFECT DETECTION - TRAINING PIPELINE")
        print("=" * 60)
        
        # Step 1: Prepare data
        data = self.prepare_data(num_samples=num_samples)
        
        # Step 2: Train model
        history = self.train(data, epochs=epochs)
        
        # Step 3: Evaluate
        metrics = self.evaluate(data)
        
        # Step 4: Plot training history
        self.detector.plot_training_history()
        
        # Step 5: Save model
        if save_model:
            self.detector.save_model('models/defect_detector.h5')
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        return self.detector


def quick_test():
    """
    Quick test function to verify everything works
    """
    print("Running quick test with small dataset...")
    
    pipeline = TrainingPipeline(img_size=128)
    
    # Small test run
    data = pipeline.prepare_data(num_samples=200, test_size=0.2, val_size=0.1)
    history = pipeline.train(data, epochs=5)
    metrics = pipeline.evaluate(data)
    
    print("\n✅ Quick test passed!")
    return pipeline


# ============= MAIN =============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AMOLED Defect Detector')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick test with small dataset')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of samples to generate (default: 2000)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test
        quick_test()
    else:
        # Full training
        pipeline = TrainingPipeline()
        detector = pipeline.run(
            num_samples=args.samples,
            epochs=args.epochs,
            save_model=True
        )
        
        print("\n🎯 Next steps:")
        print("   1. Run: streamlit run dashboard/app.py")
        print("   2. Upload test images to the web interface")
        print("   3. Record demo video for Jingce/Jingzhida")
