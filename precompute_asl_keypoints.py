#!/usr/bin/env python
"""
Precompute ASL Hand Keypoints Script
-------------------------------------
This script precomputes hand keypoints for every image in your ASL dataset and
saves the results to a pickle file. The dataset is expected to be organized as:

    dataset_dir/
        A/
            image1.jpg, image2.jpg, ...
        B/
            image1.jpg, image2.jpg, ...
        ...
        SPACE/
        DELETE/
        NOTHING/

For each image, the script:
  - Loads the image using OpenCV.
  - Converts it from BGR to RGB.
  - Uses MediaPipe’s Hands solution (in static image mode) to detect hand landmarks.
  - Extracts the 21 landmarks (each with x and y coordinates → 42 numbers).
  - If no hand is detected or the image fails to load, it stores a zero vector.
  - Stores the keypoints, label (as an index), and class name in a cache.

If the script is interrupted (e.g., via Ctrl+C), it will save the partial results.
The final output (default: 'asl_keypoints_cache.pkl') is saved in the current working directory
or at a specified path.

Usage:
    python precompute_asl_keypoints.py --data_dir archive/asl_alphabet_train/asl_alphabet_train --output_file asl_keypoints_cache.pkl
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tqdm import tqdm
import argparse
import contextlib
import sys

# Initialize MediaPipe Hands module once for use in extraction.
mp_hands = mp.solutions.hands

def extract_hand_keypoints(image):
    """
    Extracts hand keypoints from an image using MediaPipe Hands.
    Suppresses the backend's stderr output (e.g. Metal messages).

    Args:
        image (np.array): Input image in BGR format.

    Returns:
        np.array: A 42-dimensional numpy array containing [x,y] for 21 landmarks.
                  Returns a zero vector if no hand is detected.
    """
    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Suppress stderr output from MediaPipe using contextlib.redirect_stderr
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        keypoints = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            keypoints.extend([landmark.x, landmark.y])
        return np.array(keypoints, dtype=np.float32)
    else:
        return np.zeros(42, dtype=np.float32)

def precompute_keypoints(dataset_dir, output_file):
    """
    Iterates over the dataset, extracts keypoints for each image,
    and stores them in a cache. If interrupted, saves partial results.

    Args:
        dataset_dir (str): Path to the dataset directory.
        output_file (str): Path to the output pickle file.
    """
    data_cache = {}  # Will map each image's path to its keypoint info.
    
    # Get a sorted list of subdirectories (each representing a class)
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"Found classes: {classes}")
    # Map class names to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    try:
        # Process each class folder
        for cls in classes:
            cls_dir = os.path.join(dataset_dir, cls)
            # List image files in this class directory
            image_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Use tqdm to show progress for each class
            for img_file in tqdm(image_files, desc=f"Processing {cls}", unit="image"):
                img_path = os.path.join(cls_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    # Warn if image cannot be loaded and use zero vector
                    print(f"Warning: Failed to load image {img_path}. Using zero vector.", file=sys.stderr)
                    keypoints = np.zeros(42, dtype=np.float32)
                else:
                    keypoints = extract_hand_keypoints(image)
                data_cache[img_path] = {
                    "keypoints": keypoints,
                    "label": class_to_idx[cls],
                    "class": cls
                }
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving partial cache...")
    
    # Save the cache along with class information
    output_data = {
        "cache": data_cache,
        "classes": classes,
        "class_to_idx": class_to_idx
    }
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\nPrecomputed keypoints for {len(data_cache)} images saved to '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Precompute hand keypoints for an ASL dataset and save to a pickle file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root directory of the ASL dataset")
    parser.add_argument("--output_file", type=str, default="asl_keypoints_cache.pkl", help="Output pickle file (default: asl_keypoints_cache.pkl)")
    args = parser.parse_args()
    
    precompute_keypoints(args.data_dir, args.output_file)

if __name__ == "__main__":
    main()
