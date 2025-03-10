"""
Precompute keypoints script with proper imports.
"""

import os
import sys
# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import cv2
from tqdm import tqdm
import logging

# Disable MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.utils.keypoint_extraction import extract_keypoints


def precompute_keypoints(data_dir, output_file="data/keypoints.npz"):
    """
    Extract keypoints from all images in the data directory and save them.
    
    Args:
        data_dir: Directory containing ASL alphabet image folders
        output_file: Path to save the keypoints file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all classes
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))])
    
    # Map classes to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Lists to store keypoints and labels
    all_keypoints = []
    all_labels = []
    
    # Process each class folder
    for cls in classes:
        print(f"Processing class: {cls}")
        cls_dir = os.path.join(data_dir, cls)
        # Get all image files
        all_image_files = [f for f in os.listdir(cls_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit to 30 images per class for quick demonstration
        image_files = all_image_files[:30]
        print(f"Using {len(image_files)} out of {len(all_image_files)} images for class {cls}")
        
        # Process each image in the class
        for img_file in tqdm(image_files, desc=f"Extracting {cls} keypoints"):
            img_path = os.path.join(cls_dir, img_file)
            
            # Load and convert image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load {img_path}, skipping.")
                continue
            
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract keypoints
            keypoints, _ = extract_keypoints(image)
            
            # Store keypoints and label
            all_keypoints.append(keypoints)
            all_labels.append(class_to_idx[cls])
    
    # Convert to numpy arrays
    all_keypoints = np.array(all_keypoints, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    # Save to compressed file
    np.savez_compressed(
        output_file,
        keypoints=all_keypoints,
        labels=all_labels,
        classes=classes
    )
    
    print(f"Keypoints saved to {output_file}")
    print(f"Extracted {len(all_keypoints)} keypoints from {len(classes)} classes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute ASL hand keypoints.")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing ASL alphabet training images")
    parser.add_argument("--output_file", type=str, default="data/keypoints.npz",
                       help="Path to save the keypoints file")
    args = parser.parse_args()
    
    precompute_keypoints(args.data_dir, args.output_file)