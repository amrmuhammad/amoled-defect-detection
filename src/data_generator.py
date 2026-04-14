"""
Synthetic AMOLED Defect Generator
Creates realistic display defects for training AI models

Defect types:
- Dead pixels (dark spots)
- Stuck pixels (bright color spots)
- Mura (brightness non-uniformity) - IMPROVED VISIBILITY
- Scratches (line defects)
- Dust contamination
"""

import numpy as np
import cv2
import random
import os
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

class AMOLEDDefectGenerator:
    """
    Generate synthetic AMOLED display images with various defects
    """
    
    def __init__(self, width: int = 400, height: int = 400):
        """
        Initialize generator
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self.width = width
        self.height = height
        
    def generate_clean_display(self, brightness: int = 128) -> np.ndarray:
        """
        Generate a clean display image (no defects)
        
        Args:
            brightness: Gray value (0-255), default 128 = mid-gray
            
        Returns:
            RGB image array of shape (height, width, 3)
        """
        # Start with uniform color
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * brightness
        
        # Add slight natural variation (real displays aren't perfectly uniform)
        noise = np.random.normal(0, 2, (self.height, self.width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def add_dead_pixel_cluster(self, image: np.ndarray, 
                                num_pixels: int = None,
                                center: Tuple[int, int] = None) -> np.ndarray:
        """
        Add dead pixel cluster (dark spots)
        
        Args:
            image: Input image
            num_pixels: Number of dead pixels in cluster (random if None)
            center: Center of cluster (random if None)
            
        Returns:
            Image with dead pixels added
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        if num_pixels is None:
            num_pixels = random.randint(5, 30)  # Increased for better visibility
        
        if center is None:
            center = (random.randint(20, w-20), random.randint(20, h-20))
        
        for _ in range(num_pixels):
            # Add slight spread around center
            x = center[0] + random.randint(-10, 10)
            y = center[1] + random.randint(-10, 10)
            
            if 0 <= x < w and 0 <= y < h:
                # Dead pixel = black
                img[y, x] = [0, 0, 0]
                
        return img
    
    def add_stuck_pixel(self, image: np.ndarray,
                        color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Add stuck pixel (bright color spot)
        
        Args:
            image: Input image
            color: RGB color tuple (random if None)
            
        Returns:
            Image with stuck pixel added
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        if color is None:
            # Random bright color
            color = (
                random.choice([0, 255]),
                random.choice([0, 255]),
                random.choice([0, 255])
            )
        
        x = random.randint(10, w-10)
        y = random.randint(10, h-10)
        
        # Make it a larger cluster for visibility (3-5 pixels)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2 and 0 <= x+dx < w and 0 <= y+dy < h:
                    img[y+dy, x+dx] = color
                    
        return img
    
    def add_mura(self, image: np.ndarray,
                 intensity: float = None,
                 pattern_type: str = None) -> np.ndarray:
        """
        Add Mura defect with HIGH visibility for training
        
        Mura types:
        - 'cloud': Cloud-like uneven brightness
        - 'line': Linear gradient
        - 'circle': Circular pattern
        - 'random': Random noise pattern
        - 'band': Horizontal or vertical band
        - 'spot': Localized spot (common Mura type)
        
        Args:
            image: Input image
            intensity: Defect intensity (0-1), random if None
            pattern_type: Type of Mura pattern
            
        Returns:
            Image with Mura added
        """
        img = image.copy().astype(np.float32)
        h, w = img.shape[:2]
        
        # Much higher intensity for Mura to make it visible
        if intensity is None:
            intensity = random.uniform(0.6, 0.95)  # Very visible
        
        if pattern_type is None:
            pattern_type = random.choice(['cloud', 'line', 'circle', 'random', 'band', 'spot'])
        
        # Create coordinate grid
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        if pattern_type == 'cloud':
            # Cloud-like pattern using sine/cosine waves
            pattern = np.sin(4 * xx) * np.cos(4 * yy)
            pattern += 0.5 * np.sin(12 * xx * yy)
            
        elif pattern_type == 'line':
            # Linear gradient
            pattern = xx * random.choice([-1, 1])
            
        elif pattern_type == 'circle':
            # Circular pattern
            r = np.sqrt(xx**2 + yy**2)
            pattern = np.cos(6 * r)
            
        elif pattern_type == 'band':
            # Horizontal or vertical band (common Mura type)
            if random.random() > 0.5:
                pattern = np.sin(2 * np.pi * 3 * yy)
            else:
                pattern = np.sin(2 * np.pi * 3 * xx)
                
        elif pattern_type == 'spot':
            # Localized spot (very common Mura type)
            cx = random.uniform(-0.5, 0.5)
            cy = random.uniform(-0.5, 0.5)
            r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            pattern = np.exp(-r**2 * 20) * random.choice([-1, 1])
            
        else:  # random
            # Random Perlin-like noise
            pattern = np.random.normal(0, 1.2, (h, w))
            pattern = cv2.GaussianBlur(pattern, (31, 31), 0)
        
        # Normalize pattern to range [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        # Apply with VERY HIGH intensity multiplier
        pattern = (pattern - 0.5) * intensity * 2.5
        
        # Apply pattern to each channel
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 + pattern)
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def add_scratch(self, image: np.ndarray,
                    length: int = None) -> np.ndarray:
        """
        Add scratch line defect
        
        Args:
            image: Input image
            length: Scratch length in pixels (random if None)
            
        Returns:
            Image with scratch added
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        if length is None:
            length = random.randint(40, 200)  # Longer scratches for visibility
        
        # Random start point
        start_x = random.randint(20, w-20)
        start_y = random.randint(20, h-20)
        
        # Random direction
        angle = random.uniform(0, 2 * np.pi)
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))
        
        # Draw line with thicker line for visibility
        thickness = random.randint(2, 4)
        color = (0, 0, 0) if random.random() > 0.3 else (255, 255, 255)
        
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        
        return img
    
    def add_dust(self, image: np.ndarray,
                 num_particles: int = None) -> np.ndarray:
        """
        Add dust particle contamination
        
        Args:
            image: Input image
            num_particles: Number of dust particles (random if None)
            
        Returns:
            Image with dust added
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        if num_particles is None:
            num_particles = random.randint(3, 25)  # More dust particles
        
        for _ in range(num_particles):
            x = random.randint(5, w-5)
            y = random.randint(5, h-5)
            radius = random.randint(2, 6)  # Larger dust particles
            color = random.choice([
                (0, 0, 0),           # Black dust
                (30, 30, 30),        # Dark gray
                (200, 200, 200)      # Light dust
            ])
            
            cv2.circle(img, (x, y), radius, color, -1)
            
        return img
    
    def generate_defective_image(self, 
                                  defect_types: List[str] = None,
                                  brightness: int = 128) -> Tuple[np.ndarray, Dict]:
        """
        Generate a single image with random defects
        
        Args:
            defect_types: List of defects to add (random selection if None)
            brightness: Base brightness level (0-255)
            
        Returns:
            Tuple of (image, defect_info dictionary)
        """
        # Start with clean display
        image = self.generate_clean_display(brightness)
        
        # Track what defects were added
        defects_added = []
        
        # Choose random defects if not specified
        if defect_types is None:
            all_defects = ['dead_pixel', 'stuck_pixel', 'mura', 'scratch', 'dust']
            # Add 2-5 defects (more variety for training)
            num_defects = random.randint(2, 5)
            defect_types = random.sample(all_defects, min(num_defects, len(all_defects)))
        
        # Apply each defect
        for defect in defect_types:
            if defect == 'dead_pixel':
                image = self.add_dead_pixel_cluster(image)
                defects_added.append('dead_pixel_cluster')
            elif defect == 'stuck_pixel':
                image = self.add_stuck_pixel(image)
                defects_added.append('stuck_pixel')
            elif defect == 'mura':
                image = self.add_mura(image)
                defects_added.append('mura')
            elif defect == 'scratch':
                image = self.add_scratch(image)
                defects_added.append('scratch')
            elif defect == 'dust':
                image = self.add_dust(image)
                defects_added.append('dust')
        
        return image, {'defects': defects_added}
    
    def generate_dataset(self, 
                         num_samples: int = 1000,
                         defect_ratio: float = 0.5,
                         save_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset for training
        
        Args:
            num_samples: Total number of images to generate
            defect_ratio: Proportion of images with defects (0.5 = 50% defective)
            save_dir: If provided, save images to this directory
            
        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []
        
        num_defective = int(num_samples * defect_ratio)
        num_clean = num_samples - num_defective
        
        print(f"Generating {num_samples} images...")
        print(f"  - Defective: {num_defective}")
        print(f"  - Clean: {num_clean}")
        
        # Generate defective images
        for i in range(num_defective):
            if i % 100 == 0:
                print(f"  Progress: {i}/{num_defective} defective images")
            
            image, info = self.generate_defective_image()
            images.append(image)
            labels.append(1)  # 1 = defective
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(f"{save_dir}/defect_{i}_{'_'.join(info['defects'])}.png", image)
        
        # Generate clean images
        for i in range(num_clean):
            if i % 100 == 0:
                print(f"  Progress: {i}/{num_clean} clean images")
            
            image = self.generate_clean_display()
            images.append(image)
            labels.append(0)  # 0 = clean
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(f"{save_dir}/clean_{i}.png", image)
        
        return np.array(images), np.array(labels)
    
    def visualize_defects(self, num_examples: int = 6):
        """
        Visualize different defect types
        
        Args:
            num_examples: Number of defect examples to show
        """
        defect_types = ['dead_pixel', 'stuck_pixel', 'mura', 'scratch', 'dust']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        # Clean image
        clean = self.generate_clean_display()
        axes[0].imshow(clean)
        axes[0].set_title("Clean Display")
        axes[0].axis('off')
        
        # Defect examples
        for i, defect in enumerate(defect_types[:5]):
            img, info = self.generate_defective_image(defect_types=[defect])
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{defect.replace('_', ' ').title()}")
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============= DEMO CODE =============
if __name__ == "__main__":
    print("=" * 50)
    print("AMOLED Defect Generator - Demo (Improved Mura)")
    print("=" * 50)
    
    # Create generator
    gen = AMOLEDDefectGenerator(width=400, height=400)
    
    # Generate and display examples
    print("\n1. Visualizing defect types...")
    gen.visualize_defects()
    
    # Generate a dataset
    print("\n2. Generating small dataset...")
    images, labels = gen.generate_dataset(num_samples=100, defect_ratio=0.5)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"   - Total images: {len(images)}")
    print(f"   - Image shape: {images[0].shape}")
    print(f"   - Defective count: {sum(labels)}")
    print(f"   - Clean count: {len(labels) - sum(labels)}")
    
    # Save a sample
    print("\n3. Saving sample images to 'demo/sample_images/'...")
    os.makedirs("demo/sample_images", exist_ok=True)
    
    for i in range(5):
        img, info = gen.generate_defective_image()
        cv2.imwrite(f"demo/sample_images/defect_sample_{i}.png", img)
        print(f"   Saved: defect_sample_{i}.png - Defects: {info['defects']}")
    
    print("\n✅ All done! Ready for training with improved Mura visibility.")
