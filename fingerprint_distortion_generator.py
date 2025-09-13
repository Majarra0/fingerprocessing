#!/usr/bin/env python3
"""
Fingerprint Distortion Generator
This script creates realistic synthetic distortions from clean fingerprint images
to generate a comprehensive training dataset for the 6 distortion classes:
- Blur, Noise, Partial, Geometric, Morphed, Adversarial
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import albumentations as A
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import List, Tuple, Optional
import json

class FingerprintDistortionGenerator:
    """
    A comprehensive class for generating realistic fingerprint distortions
    """
    
    def __init__(self, seed=42):
        """Initialize the distortion generator with reproducible random seed"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define distortion parameters with realistic ranges
        self.blur_params = {
            'gaussian_sigma': (1.0, 8.0),
            'motion_blur_kernel': (5, 25),
            'defocus_radius': (2, 12)
        }
        
        self.noise_params = {
            'gaussian_var': (0.001, 0.05),
            'salt_pepper_amount': (0.01, 0.15),
            'speckle_var': (0.005, 0.08),
            'poisson_scale': (0.8, 1.2)
        }
        
        self.geometric_params = {
            'rotation_range': (-45, 45),
            'scale_range': (0.7, 1.4),
            'shear_range': (-0.3, 0.3),
            'perspective_scale': (0.05, 0.15)
        }
        
        # Initialize Albumentations transforms
        self._init_albumentations_transforms()
    
    def _init_albumentations_transforms(self):
        """Initialize Albumentations transformation pipelines"""
        
        # Blur transformations
        self.blur_transforms = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 15), p=0.4),
                A.MotionBlur(blur_limit=(3, 15), p=0.4),
                A.MedianBlur(blur_limit=(3, 15), p=0.2),
            ], p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
        ])
        
        # Noise transformations
        self.noise_transforms = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10, 80), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.3),
            ], p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)
        ])
        
        # Geometric transformations
        self.geometric_transforms = A.Compose([
            A.OneOf([
                A.Rotate(limit=45, p=0.3),
                A.Affine(scale=(0.7, 1.3), translate_percent=(-0.2, 0.2), 
                        rotate=(-30, 30), shear=(-20, 20), p=0.4),
                A.Perspective(scale=(0.05, 0.15), p=0.3),
            ], p=1.0),
        ])
        
        # Partial/Occlusion transformations
        self.partial_transforms = A.Compose([
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=50, max_width=50, p=0.4),
                A.GridDropout(ratio=0.3, random_offset=True, p=0.3),
                A.CoarseDropout(max_holes=5, hole_height=40, hole_width=40, p=0.3),
            ], p=1.0),
        ])
    
    def apply_blur_distortion(self, image: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Apply realistic blur distortions"""
        severity_multipliers = {'light': 0.5, 'medium': 1.0, 'heavy': 1.8}
        mult = severity_multipliers.get(severity, 1.0)
        
        # Choose blur type randomly
        blur_type = random.choice(['gaussian', 'motion', 'defocus', 'combined'])
        
        if blur_type == 'gaussian':
            sigma = random.uniform(*self.blur_params['gaussian_sigma']) * mult
            image = cv2.GaussianBlur(image, (0, 0), sigma)
            
        elif blur_type == 'motion':
            # Create motion blur kernel
            kernel_size = int(random.uniform(*self.blur_params['motion_blur_kernel']) * mult)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            angle = random.uniform(0, 180)
            kernel = self._get_motion_blur_kernel(kernel_size, angle)
            image = cv2.filter2D(image, -1, kernel)
            
        elif blur_type == 'defocus':
            # Simulate defocus blur using disk-shaped kernel
            radius = int(random.uniform(*self.blur_params['defocus_radius']) * mult)
            kernel = self._get_defocus_kernel(radius)
            image = cv2.filter2D(image, -1, kernel)
            
        else:  # combined
            # Apply multiple blur types
            sigma = random.uniform(1, 4) * mult
            image = cv2.GaussianBlur(image, (0, 0), sigma)
            if random.random() < 0.5:
                kernel_size = int(random.uniform(3, 9) * mult)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                image = cv2.medianBlur(image, kernel_size)
        
        # Add slight brightness/contrast variation
        if random.random() < 0.6:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-10, 10)    # brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image
    
    def apply_noise_distortion(self, image: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Apply realistic noise distortions"""
        severity_multipliers = {'light': 0.5, 'medium': 1.0, 'heavy': 2.0}
        mult = severity_multipliers.get(severity, 1.0)
        
        # Choose noise type(s) - can apply multiple
        noise_types = random.sample(['gaussian', 'salt_pepper', 'speckle', 'poisson'], 
                                  k=random.randint(1, 2))
        
        for noise_type in noise_types:
            if noise_type == 'gaussian':
                var = random.uniform(*self.noise_params['gaussian_var']) * mult
                noise = np.random.normal(0, np.sqrt(var), image.shape)
                image = np.clip(image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
                
            elif noise_type == 'salt_pepper':
                amount = random.uniform(*self.noise_params['salt_pepper_amount']) * mult
                # Salt noise
                salt_coords = tuple([np.random.randint(0, i-1, int(amount * image.size * 0.5)) 
                                   for i in image.shape])
                image[salt_coords] = 255
                # Pepper noise
                pepper_coords = tuple([np.random.randint(0, i-1, int(amount * image.size * 0.5)) 
                                     for i in image.shape])
                image[pepper_coords] = 0
                
            elif noise_type == 'speckle':
                var = random.uniform(*self.noise_params['speckle_var']) * mult
                noise = np.random.normal(0, np.sqrt(var), image.shape)
                image = np.clip(image * (1 + noise), 0, 255).astype(np.uint8)
                
            elif noise_type == 'poisson':
                # Poisson noise (photon noise)
                scale = random.uniform(*self.noise_params['poisson_scale'])
                image = np.random.poisson(image * scale) / scale
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_partial_distortion(self, image: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Apply partial/occlusion distortions"""
        severity_multipliers = {'light': 0.5, 'medium': 1.0, 'heavy': 1.5}
        mult = severity_multipliers.get(severity, 1.0)
        
        h, w = image.shape[:2]
        
        # Choose occlusion type
        occlusion_type = random.choice(['rectangular', 'circular', 'irregular', 'edge_crop'])
        
        if occlusion_type == 'rectangular':
            # Random rectangular occlusions
            num_occlusions = random.randint(1, int(5 * mult))
            for _ in range(num_occlusions):
                x1 = random.randint(0, w//2)
                y1 = random.randint(0, h//2)
                width = random.randint(int(20*mult), int(80*mult))
                height = random.randint(int(20*mult), int(80*mult))
                x2 = min(x1 + width, w)
                y2 = min(y1 + height, h)
                
                # Random occlusion color (black, white, or gray)
                color = random.choice([0, 255, random.randint(100, 150)])
                image[y1:y2, x1:x2] = color
                
        elif occlusion_type == 'circular':
            # Circular occlusions (simulate fingerprints on sensor)
            num_circles = random.randint(1, int(4 * mult))
            for _ in range(num_circles):
                center_x = random.randint(0, w)
                center_y = random.randint(0, h)
                radius = random.randint(int(15*mult), int(50*mult))
                color = random.choice([0, 255])
                cv2.circle(image, (center_x, center_y), radius, color, -1)
                
        elif occlusion_type == 'irregular':
            # Irregular shaped occlusions
            num_points = random.randint(3, 8)
            points = np.array([[random.randint(0, w), random.randint(0, h)] 
                             for _ in range(num_points)], np.int32)
            color = random.choice([0, 255])
            cv2.fillPoly(image, [points], color)
            
        else:  # edge_crop
            # Crop from edges to simulate partial capture
            crop_side = random.choice(['top', 'bottom', 'left', 'right'])
            crop_amount = int(random.uniform(0.1, 0.4) * mult * min(h, w))
            
            if crop_side == 'top':
                image[:crop_amount, :] = 0
            elif crop_side == 'bottom':
                image[h-crop_amount:, :] = 0
            elif crop_side == 'left':
                image[:, :crop_amount] = 0
            else:  # right
                image[:, w-crop_amount:] = 0
        
        return image
    
    def apply_geometric_distortion(self, image: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Apply geometric distortions"""
        h, w = image.shape[:2]
        severity_multipliers = {'light': 0.5, 'medium': 1.0, 'heavy': 1.8}
        mult = severity_multipliers.get(severity, 1.0)
        
        # Choose geometric transformation type
        transform_type = random.choice(['rotation', 'scaling', 'shearing', 'perspective', 'combined'])
        
        if transform_type == 'rotation':
            angle = random.uniform(*self.geometric_params['rotation_range']) * mult
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        elif transform_type == 'scaling':
            scale = random.uniform(*self.geometric_params['scale_range'])
            scale = 1 + (scale - 1) * mult
            M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        elif transform_type == 'shearing':
            shear_x = random.uniform(*self.geometric_params['shear_range']) * mult
            shear_y = random.uniform(*self.geometric_params['shear_range']) * mult
            M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        elif transform_type == 'perspective':
            # Perspective transformation
            scale = random.uniform(*self.geometric_params['perspective_scale']) * mult
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_points = src_points + np.random.uniform(-scale*min(w,h), scale*min(w,h), src_points.shape)
            # Ensure both are np.float32 and shape (4, 2)
            src_points = np.array(src_points, dtype=np.float32).reshape(4, 2)
            dst_points = np.array(dst_points, dtype=np.float32).reshape(4, 2)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            image = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        else:  # combined
            # Apply multiple transformations
            angle = random.uniform(-15, 15) * mult
            scale = random.uniform(0.9, 1.1) * mult
            shear = random.uniform(-0.1, 0.1) * mult
            
            # Rotation + scaling
            M1 = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
            image = cv2.warpAffine(image, M1, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Shearing
            M2 = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
            image = cv2.warpAffine(image, M2, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image
    
    def apply_morphed_distortion(self, image: np.ndarray, other_images: List[np.ndarray], 
                               severity: str = 'medium') -> np.ndarray:
        """Apply morphed distortions by blending multiple fingerprints"""
        if not other_images:
            # If no other images available, create synthetic morphing
            return self._create_synthetic_morph(image, severity)
        
        severity_multipliers = {'light': 0.3, 'medium': 0.5, 'heavy': 0.7}
        alpha = severity_multipliers.get(severity, 0.5)
        
        # Select random image to morph with
        other_image = random.choice(other_images)
        other_image = cv2.resize(other_image, (image.shape[1], image.shape[0]))
        
        # Blend the images
        morphed = cv2.addWeighted(image, 1-alpha, other_image, alpha, 0)
        
        # Add some realistic distortion to make it look more natural
        if random.random() < 0.5:
            morphed = self.apply_noise_distortion(morphed, 'light')
        
        return morphed
    
    def apply_adversarial_distortion(self, image: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Apply adversarial distortions"""
        severity_multipliers = {'light': 0.5, 'medium': 1.0, 'heavy': 1.5}
        mult = severity_multipliers.get(severity, 1.0)
        
        # Choose adversarial attack type
        attack_type = random.choice(['gradient_noise', 'pattern_overlay', 'frequency_attack', 'combined'])
        
        if attack_type == 'gradient_noise':
            # Add structured noise that mimics gradient-based attacks
            noise_strength = random.uniform(5, 25) * mult
            gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Normalize and add as adversarial noise
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * noise_strength)
            adversarial_noise = gradient_magnitude * random.choice([-1, 1])
            image = np.clip(image.astype(np.float32) + adversarial_noise, 0, 255).astype(np.uint8)
            
        elif attack_type == 'pattern_overlay':
            # Overlay subtle patterns that can fool systems
            pattern_type = random.choice(['sinusoidal', 'checkerboard', 'random_lines'])
            
            if pattern_type == 'sinusoidal':
                x = np.arange(image.shape[1])
                y = np.arange(image.shape[0])
                X, Y = np.meshgrid(x, y)
                frequency = random.uniform(0.01, 0.05) * mult
                amplitude = random.uniform(5, 20) * mult
                pattern = amplitude * np.sin(frequency * X + frequency * Y)
                
            elif pattern_type == 'checkerboard':
                size = random.randint(int(8/mult), int(32/mult))
                pattern = np.zeros_like(image, dtype=np.float32)
                for i in range(0, image.shape[0], size*2):
                    for j in range(0, image.shape[1], size*2):
                        pattern[i:i+size, j:j+size] = random.uniform(10, 30) * mult
                        pattern[i+size:i+2*size, j+size:j+2*size] = random.uniform(10, 30) * mult
                        
            else:  # random_lines
                pattern = np.zeros_like(image, dtype=np.float32)
                num_lines = random.randint(int(5*mult), int(20*mult))
                for _ in range(num_lines):
                    x1, y1 = random.randint(0, image.shape[1]), random.randint(0, image.shape[0])
                    x2, y2 = random.randint(0, image.shape[1]), random.randint(0, image.shape[0])
                    cv2.line(pattern, (x1, y1), (x2, y2), random.uniform(10, 25) * mult, 1)
            
            image = np.clip(image.astype(np.float32) + pattern, 0, 255).astype(np.uint8)
            
        elif attack_type == 'frequency_attack':
            # Frequency domain manipulation
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            
            # Add noise in frequency domain
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create frequency mask
            noise_strength = random.uniform(0.1, 0.5) * mult
            freq_noise = np.random.normal(0, noise_strength, f_shift.shape) + \
                        1j * np.random.normal(0, noise_strength, f_shift.shape)
            
            f_shift_noisy = f_shift + freq_noise
            f_ishift = np.fft.ifftshift(f_shift_noisy)
            image_back = np.fft.ifft2(f_ishift)
            image = np.clip(np.abs(image_back), 0, 255).astype(np.uint8)
            
        else:  # combined
            # Apply multiple adversarial techniques
            image = self.apply_adversarial_distortion(image, severity)
            if random.random() < 0.6:
                image = self.apply_noise_distortion(image, 'light')
        
        return image
    
    def _get_motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """Generate motion blur kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Create line in the direction of motion
        for i in range(size):
            offset = i - center
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        return kernel / kernel.sum()
    
    def _get_defocus_kernel(self, radius: int) -> np.ndarray:
        """Generate defocus blur kernel (disk-shaped)"""
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        center = radius
        
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        kernel[mask] = 1
        
        return kernel / kernel.sum()
    
    def _create_synthetic_morph(self, image: np.ndarray, severity: str) -> np.ndarray:
        """Create synthetic morphed appearance when no other images available"""
        # Create a synthetic "other" fingerprint by transforming the original
        synthetic_other = image.copy()
        
        # Apply random transformations to create variation
        synthetic_other = self.apply_geometric_distortion(synthetic_other, 'light')
        synthetic_other = self.apply_noise_distortion(synthetic_other, 'light')
        
        # Blend with original
        severity_multipliers = {'light': 0.3, 'medium': 0.5, 'heavy': 0.7}
        alpha = severity_multipliers.get(severity, 0.5)
        
        morphed = cv2.addWeighted(image, 1-alpha, synthetic_other, alpha, 0)
        return morphed
    
    def generate_distorted_dataset(self, 
                                 input_dir: str, 
                                 output_dir: str,
                                 images_per_class: int = 500,
                                 severity_distribution: dict = None):
        """
        Generate complete distorted dataset from clean fingerprint images
        
        Args:
            input_dir: Directory containing clean fingerprint images
            output_dir: Directory to save distorted images
            images_per_class: Number of images to generate per distortion class
            severity_distribution: Dict specifying severity distribution
        """
        
        if severity_distribution is None:
            severity_distribution = {'light': 0.3, 'medium': 0.5, 'heavy': 0.2}
        
        # Create output directories
        distortion_classes = ['blur', 'noise', 'partial', 'geometric', 'morphed', 'adversarial']
        output_path = Path(output_dir)
        
        for class_name in distortion_classes:
            (output_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Load all input images
        input_path = Path(input_dir)
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.bmp')) + list(input_path.glob('*.jpeg')) + \
                     list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        
        if not image_files:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(image_files)} clean images")
        print(f"Generating {images_per_class} images per class...")
        
        # Load all images for morphing
        all_images = []
        for img_file in image_files[:50]:  # Limit for memory efficiency
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_images.append(img)
        
        # Generate distortions for each class
        for class_name in distortion_classes:
            print(f"\\nGenerating {class_name} distortions...")
            
            for i in tqdm(range(images_per_class)):
                # Select random base image
                base_image_path = random.choice(image_files)
                base_image = cv2.imread(str(base_image_path), cv2.IMREAD_GRAYSCALE)
                
                if base_image is None:
                    continue
                
                # Choose severity based on distribution
                severity = np.random.choice(list(severity_distribution.keys()),
                                          p=list(severity_distribution.values()))
                
                # Apply appropriate distortion
                if class_name == 'blur':
                    distorted_image = self.apply_blur_distortion(base_image.copy(), severity)
                elif class_name == 'noise':
                    distorted_image = self.apply_noise_distortion(base_image.copy(), severity)
                elif class_name == 'partial':
                    distorted_image = self.apply_partial_distortion(base_image.copy(), severity)
                elif class_name == 'geometric':
                    distorted_image = self.apply_geometric_distortion(base_image.copy(), severity)
                elif class_name == 'morphed':
                    distorted_image = self.apply_morphed_distortion(base_image.copy(), all_images, severity)
                else:  # adversarial
                    distorted_image = self.apply_adversarial_distortion(base_image.copy(), severity)
                
                # Save distorted image
                output_filename = f"{class_name}_{i:04d}_{severity}.jpg"
                output_file_path = output_path / class_name / output_filename
                cv2.imwrite(str(output_file_path), distorted_image)
        
        print(f"\\nDataset generation complete!")
        print(f"Generated {len(distortion_classes) * images_per_class} total images")
        print(f"Output directory: {output_dir}")
        
        # Generate summary report
        self._generate_summary_report(output_dir, distortion_classes, images_per_class, severity_distribution)
    
    def _generate_summary_report(self, output_dir: str, classes: List[str], 
                            images_per_class: int, severity_dist: dict):
        """Generate a summary report of the generated dataset"""

        report = {
            'dataset_info': {
                'total_images': len(classes) * images_per_class,
                'classes': classes,
                'images_per_class': images_per_class,
                'severity_distribution': severity_dist
            },
            'distortion_parameters': {
                'blur_params': self.blur_params,
                'noise_params': self.noise_params,
                'geometric_params': self.geometric_params
            }
        }
        
        # Save JSON report
        with open(Path(output_dir) / 'dataset_summary.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Create markdown report
        markdown_report = f"""# Fingerprint Distortion Dataset Summary

    ## Dataset Statistics
    - **Total Images**: {len(classes) * images_per_class}
    - **Number of Classes**: {len(classes)}
    - **Images per Class**: {images_per_class}

    ## Distortion Classes
    {chr(10).join(f'- **{cls.title()}**: {images_per_class} images' for cls in classes)}

    ## Severity Distribution
    {chr(10).join(f'- **{sev.title()}**: {int(prob*100)}%' for sev, prob in severity_dist.items())}

    ## Usage
    This dataset can be used directly with the fingerprint classification training script.
    Make sure the folder structure matches the expected format:

    ```
    fingerprint_dataset/
    ├── blur/
    ├── noise/
    ├── partial/
    ├── geometric/
    ├── morphed/
    └── adversarial/
    ```
    """

        with open(Path(output_dir) / 'README.md', 'w', encoding='utf-8') as f:
            f.write(markdown_report)

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Generate synthetic fingerprint distortions')
    parser.add_argument('--input', '-i', required=True, 
                       help='Directory containing clean fingerprint images')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for distorted dataset')
    parser.add_argument('--count', '-c', type=int, default=500,
                       help='Number of images per distortion class (default: 500)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create distortion generator
    generator = FingerprintDistortionGenerator(seed=args.seed)
    
    # Generate dataset
    generator.generate_distorted_dataset(
        input_dir=args.input,
        output_dir=args.output,
        images_per_class=args.count
    )

if __name__ == "__main__":
    # Example usage if run directly
    print("Fingerprint Distortion Generator")
    print("="*50)
    
    # Interactive mode if no command line arguments
    import sys
    if len(sys.argv) == 1:
        print("Interactive Mode")
        print("Please provide the following information:")
        
        input_dir = input("Enter path to clean fingerprint images directory: ").strip()
        output_dir = input("Enter output directory for distorted dataset: ").strip()
        
        try:
            images_per_class = int(input("Enter number of images per class (default 500): ") or "500")
        except ValueError:
            images_per_class = 500
            
        try:
            seed = int(input("Enter random seed (default 42): ") or "42")
        except ValueError:
            seed = 42
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist!")
            sys.exit(1)
        
        print(f"\nGenerating dataset...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Images per class: {images_per_class}")
        print(f"Random seed: {seed}")
        
        # Create generator and generate dataset
        generator = FingerprintDistortionGenerator(seed=seed)
        generator.generate_distorted_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            images_per_class=images_per_class
        )
        
        print("\n" + "="*50)
        print("Dataset generation completed successfully!")
        print(f"You can now use '{output_dir}' with the fingerprint classifier training script.")
        
    else:
        # Command line mode
        main()


# Additional utility functions for testing and visualization

def visualize_distortions(generator, sample_image_path: str, output_path: str = "distortion_samples.png"):
    """
    Create a visualization showing all distortion types applied to a sample image
    """
    import matplotlib.pyplot as plt
    
    # Load sample image
    original = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Could not load image: {sample_image_path}")
    
    # Apply each distortion type
    distortions = {
        'Original': original,
        'Blur': generator.apply_blur_distortion(original.copy(), 'medium'),
        'Noise': generator.apply_noise_distortion(original.copy(), 'medium'),
        'Partial': generator.apply_partial_distortion(original.copy(), 'medium'),
        'Geometric': generator.apply_geometric_distortion(original.copy(), 'medium'),
        'Morphed': generator.apply_morphed_distortion(original.copy(), [original], 'medium'),
        'Adversarial': generator.apply_adversarial_distortion(original.copy(), 'medium')
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, img) in enumerate(distortions.items()):
        if idx < len(axes):
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
    
    # Hide the last subplot if we have fewer images than subplots
    if len(distortions) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Distortion visualization saved to: {output_path}")

def test_single_distortion(generator, image_path: str, distortion_type: str, severity: str = 'medium'):
    """
    Test a single distortion type and display the result
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply distortion
    if distortion_type == 'blur':
        distorted = generator.apply_blur_distortion(image.copy(), severity)
    elif distortion_type == 'noise':
        distorted = generator.apply_noise_distortion(image.copy(), severity)
    elif distortion_type == 'partial':
        distorted = generator.apply_partial_distortion(image.copy(), severity)
    elif distortion_type == 'geometric':
        distorted = generator.apply_geometric_distortion(image.copy(), severity)
    elif distortion_type == 'morphed':
        distorted = generator.apply_morphed_distortion(image.copy(), [image], severity)
    elif distortion_type == 'adversarial':
        distorted = generator.apply_adversarial_distortion(image.copy(), severity)
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(distorted, cmap='gray')
    ax2.set_title(f'{distortion_type.title()} ({severity})', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return distorted


# Example usage and testing functions
def run_example():
    """
    Run a complete example of dataset generation
    """
    print("Running example dataset generation...")
    
    # This would be your actual paths
    INPUT_DIR = "clean_fingerprints"  # Directory with clean fingerprint images
    OUTPUT_DIR = "fingerprint_dataset"  # Output directory for distorted dataset
    IMAGES_PER_CLASS = 100  # Generate 100 images per distortion class
    
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Images per class: {IMAGES_PER_CLASS}")
    
    # Create generator
    generator = FingerprintDistortionGenerator(seed=42)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Warning: Input directory '{INPUT_DIR}' does not exist!")
        print("Please create this directory and add your clean fingerprint images.")
        print("Supported formats: .jpg, .png, .bmp, .jpeg")
        return
    
    # Generate the dataset
    try:
        generator.generate_distorted_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            images_per_class=IMAGES_PER_CLASS
        )
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print("Please check your input directory and try again.")


# Custom severity distributions for different use cases
SEVERITY_DISTRIBUTIONS = {
    'balanced': {'light': 0.33, 'medium': 0.34, 'heavy': 0.33},
    'realistic': {'light': 0.5, 'medium': 0.35, 'heavy': 0.15},
    'challenging': {'light': 0.2, 'medium': 0.3, 'heavy': 0.5},
    'mild': {'light': 0.7, 'medium': 0.25, 'heavy': 0.05}
}

def generate_custom_dataset(input_dir: str, output_dir: str, 
                          distribution_type: str = 'realistic',
                          images_per_class: int = 500):
    """
    Generate dataset with predefined severity distributions
    """
    if distribution_type not in SEVERITY_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution type. Choose from: {list(SEVERITY_DISTRIBUTIONS.keys())}")
    
    generator = FingerprintDistortionGenerator(seed=42)
    severity_dist = SEVERITY_DISTRIBUTIONS[distribution_type]
    
    print(f"Using {distribution_type} severity distribution: {severity_dist}")
    
    generator.generate_distorted_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        images_per_class=images_per_class,
        severity_distribution=severity_dist
    )