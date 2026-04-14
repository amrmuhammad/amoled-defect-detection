"""
Final test for the trained defect detector
Tests the model on clean and defective images with various defect types
"""

import sys
import os
import numpy as np
import cv2
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_transfer import TransferDefectDetector
from src.data_generator import AMOLEDDefectGenerator

def print_header(text: str, char: str = "=", width: int = 60):
    """Print a formatted header"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)

def print_success(text: str):
    """Print success message"""
    print(f"   ✅ {text}")

def print_error(text: str):
    """Print error message"""
    print(f"   ❌ {text}")

def print_info(text: str):
    """Print info message"""
    print(f"   📊 {text}")

def test_model():
    """Main test function"""
    
    print_header("AMOLED DEFECT DETECTOR - FINAL MODEL TEST", "=", 60)
    
    # ============================================================
    # STEP 1: Load the trained model
    # ============================================================
    print_header("STEP 1: Loading Trained Model", "-", 60)
    
    model_path = 'models/transfer_defect_detector.keras'
    
    if not os.path.exists(model_path):
        print_error(f"Model not found at {model_path}")
        print_info("Please run training first: python src/train_transfer.py")
        return False
    
    try:
        detector = TransferDefectDetector(input_shape=(128, 128, 3))
        detector.load_model(model_path)
        print_success(f"Model loaded from {model_path}")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return False
    
    # ============================================================
    # STEP 2: Test on Clean Images
    # ============================================================
    print_header("STEP 2: Testing Clean Images (Expected: CLEAN)", "-", 60)
    
    gen = AMOLEDDefectGenerator(width=400, height=400)
    
    clean_results = []
    for i in range(5):
        img = gen.generate_clean_display()
        try:
            confidence, prediction = detector.predict(img)
            is_correct = (prediction == "CLEAN")
            clean_results.append(is_correct)
            
            if is_correct:
                print_success(f"Image {i+1}: {prediction} ({confidence:.2%} confidence)")
            else:
                print_error(f"Image {i+1}: {prediction} ({confidence:.2%} confidence) - Expected CLEAN")
        except Exception as e:
            print_error(f"Image {i+1}: Prediction failed - {e}")
            clean_results.append(False)
    
    clean_accuracy = sum(clean_results) / len(clean_results) if clean_results else 0
    print_info(f"Clean image accuracy: {clean_accuracy:.1%} ({sum(clean_results)}/{len(clean_results)})")
    
    # ============================================================
    # STEP 3: Test on Defective Images (Various Defect Types)
    # ============================================================
    print_header("STEP 3: Testing Defective Images (Expected: DEFECTIVE)", "-", 60)
    
    defect_types = ['dead_pixel', 'stuck_pixel', 'mura', 'scratch', 'dust']
    defect_results = {}
    
    for defect in defect_types:
        print(f"\n   Testing {defect.upper()} defects:")
        defect_correct = []
        
        for i in range(3):  # 3 examples per defect type
            img, info = gen.generate_defective_image(defect_types=[defect])
            try:
                confidence, prediction = detector.predict(img)
                is_correct = (prediction == "DEFECTIVE")
                defect_correct.append(is_correct)
                
                if is_correct:
                    print_success(f"   {i+1}: {prediction} ({confidence:.2%} confidence) - {', '.join(info['defects'])}")
                else:
                    print_error(f"   {i+1}: {prediction} ({confidence:.2%} confidence) - Expected DEFECTIVE")
            except Exception as e:
                print_error(f"   {i+1}: Prediction failed - {e}")
                defect_correct.append(False)
        
        defect_results[defect] = sum(defect_correct) / len(defect_correct) if defect_correct else 0
        print_info(f"   {defect.upper()} accuracy: {defect_results[defect]:.1%}")
    
    # ============================================================
    # STEP 4: Test on Mixed Random Images
    # ============================================================
    print_header("STEP 4: Testing Mixed Random Images", "-", 60)
    
    mixed_results = []
    for i in range(10):
        # Randomly choose clean or defective
        if np.random.random() > 0.5:
            img, info = gen.generate_defective_image()
            expected = "DEFECTIVE"
            defect_info = f" - {', '.join(info['defects'])}"
        else:
            img = gen.generate_clean_display()
            expected = "CLEAN"
            defect_info = ""
        
        try:
            confidence, prediction = detector.predict(img)
            is_correct = (prediction == expected)
            mixed_results.append(is_correct)
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {i+1:2d}: Expected {expected:10s} | Got {prediction:10s} | Confidence: {confidence:.2%}{defect_info}")
        except Exception as e:
            print_error(f"   Image {i+1}: Prediction failed - {e}")
            mixed_results.append(False)
    
    mixed_accuracy = sum(mixed_results) / len(mixed_results) if mixed_results else 0
    print_info(f"Mixed image accuracy: {mixed_accuracy:.1%} ({sum(mixed_results)}/{len(mixed_results)})")
    
    # ============================================================
    # STEP 5: Performance Summary
    # ============================================================
    print_header("STEP 5: Performance Summary", "=", 60)
    
    # Calculate overall metrics
    total_tests = len(clean_results) + sum(len([1 for _ in range(3)]) for _ in defect_types) + len(mixed_results)
    total_correct = sum(clean_results) + sum(sum(1 for _ in range(3) for r in [defect_results.get(d, 0)] if r == 1) for d in defect_types) + sum(mixed_results)
    
    print(f"\n   📊 Test Results:")
    print(f"   ┌─────────────────────────────────────┬────────────┐")
    print(f"   │ Test Category                       │ Accuracy   │")
    print(f"   ├─────────────────────────────────────┼────────────┤")
    print(f"   │ Clean Images (5 samples)            │ {clean_accuracy:.1%}         │")
    print(f"   │ Mixed Random Images (10 samples)    │ {mixed_accuracy:.1%}         │")
    print(f"   └─────────────────────────────────────┴────────────┘")
    
    print(f"\n   📊 Defect Type Breakdown:")
    print(f"   ┌─────────────────────────────────────┬────────────┐")
    for defect, acc in defect_results.items():
        print(f"   │ {defect.upper().replace('_', ' '):30s} │ {acc:.1%}         │")
    print(f"   └─────────────────────────────────────┴────────────┘")
    
    print(f"\n   📊 Overall Performance:")
    print(f"   ┌─────────────────────────────────────┬────────────┐")
    print(f"   │ Total Tests                         │ {total_tests}           │")
    print(f"   │ Correct Predictions                 │ {total_correct}           │")
    print(f"   │ Overall Accuracy                    │ {total_correct/total_tests:.1%}         │")
    print(f"   └─────────────────────────────────────┴────────────┘")
    
    # ============================================================
    # STEP 6: Save Results to File
    # ============================================================
    print_header("STEP 6: Saving Test Results", "-", 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.txt"
    
    try:
        with open(results_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AMOLED DEFECT DETECTOR - TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Clean Image Accuracy: {:.1%}\n".format(clean_accuracy))
            f.write("Mixed Image Accuracy: {:.1%}\n\n".format(mixed_accuracy))
            
            f.write("Defect Type Breakdown:\n")
            for defect, acc in defect_results.items():
                f.write(f"  - {defect.upper()}: {acc:.1%}\n")
            
            f.write(f"\nOverall Accuracy: {total_correct/total_tests:.1%}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Correct Predictions: {total_correct}\n")
        
        print_success(f"Results saved to {results_file}")
    except Exception as e:
        print_error(f"Failed to save results: {e}")
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print_header("FINAL VERDICT", "=", 60)
    
    if total_correct / total_tests > 0.95:
        print("\n   🎉 EXCELLENT! Model is ready for production demo!")
        print("   📊 Model consistently achieves >95% accuracy")
        print("   🚀 Ready to present to Jingce/Jingzhida")
    elif total_correct / total_tests > 0.85:
        print("\n   ✅ GOOD! Model is acceptable for demo")
        print("   📊 Model achieves >85% accuracy")
        print("   🔧 Some improvements possible but ready for presentation")
    else:
        print("\n   ⚠️ NEEDS IMPROVEMENT! Model accuracy is below 85%")
        print("   🔧 Consider retraining with more data or different parameters")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE".center(60))
    print("=" * 60)
    
    return total_correct / total_tests > 0.85

# ============================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================

def test_single_image(image_path: str):
    """
    Test a single image file
    
    Args:
        image_path: Path to the image file
    """
    print_header(f"Testing Single Image: {image_path}", "-", 60)
    
    # Load model
    detector = TransferDefectDetector(input_shape=(128, 128, 3))
    detector.load_model('models/transfer_defect_detector.keras')
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print_error(f"Could not load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Predict
    confidence, prediction = detector.predict(img)
    
    print(f"\n   📊 Prediction Results:")
    print(f"   ┌─────────────────────────────────────┬────────────────────┐")
    print(f"   │ Prediction                          │ {prediction}                 │")
    print(f"   │ Confidence                          │ {confidence:.2%}                  │")
    print(f"   └─────────────────────────────────────┴────────────────────┘")
    
    if prediction == "DEFECTIVE":
        print(f"\n   ⚠️ Defect detected with {confidence:.2%} confidence")
    else:
        print(f"\n   ✅ No defects detected with {confidence:.2%} confidence")

def generate_demo_images(num_images: int = 10, output_dir: str = "demo_images"):
    """
    Generate sample images for demo
    
    Args:
        num_images: Number of images to generate
        output_dir: Directory to save images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    gen = AMOLEDDefectGenerator()
    detector = TransferDefectDetector(input_shape=(128, 128, 3))
    detector.load_model('models/transfer_defect_detector.keras')
    
    print_header(f"Generating {num_images} Demo Images", "-", 60)
    
    for i in range(num_images):
        # Alternate between clean and defective
        if i % 2 == 0:
            img, info = gen.generate_defective_image()
            expected = "DEFECTIVE"
        else:
            img = gen.generate_clean_display()
            info = {'defects': ['none']}
            expected = "CLEAN"
        
        confidence, prediction = detector.predict(img)
        
        # Save image
        filename = f"{output_dir}/demo_{i+1:02d}_{prediction}_{confidence:.2f}.png"
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        status = "✅" if prediction == expected else "❌"
        print(f"   {status} {filename}")
        print(f"       Expected: {expected}, Got: {prediction} ({confidence:.2%})")
        if 'defects' in info and info['defects'] != ['none']:
            print(f"       Defects: {', '.join(info['defects'])}")
    
    print_success(f"Demo images saved to {output_dir}/")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AMOLED Defect Detector')
    parser.add_argument('--image', type=str, help='Path to a single image to test')
    parser.add_argument('--generate-demo', type=int, default=0, 
                        help='Number of demo images to generate (default: 0)')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick test only')
    
    args = parser.parse_args()
    
    if args.image:
        # Test a single image
        test_single_image(args.image)
    elif args.generate_demo > 0:
        # Generate demo images
        generate_demo_images(args.generate_demo)
    else:
        # Run full test suite
        success = test_model()
        sys.exit(0 if success else 1)
