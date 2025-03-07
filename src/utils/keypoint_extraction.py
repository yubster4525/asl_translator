"""
Keypoint extraction utilities for ASL Translator.

This module provides functions to extract hand landmarks from images using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(image, static_mode=True, max_num_hands=1, min_detection_confidence=0.5):
    """
    Extract hand landmarks from an image using MediaPipe.
    
    Args:
        image: RGB image (numpy array)
        static_mode: Whether to treat the input as a static image (True) or video frame (False)
        max_num_hands: Maximum number of hands to detect
        min_detection_confidence: Minimum confidence value for hand detection
        
    Returns:
        keypoints: Flattened array of keypoints (x, y coordinates) or None if no hand detected
        annotated_image: Image with hand landmarks drawn (for visualization)
    """
    # Make a copy for drawing
    annotated_image = image.copy()
    
    # Process image with MediaPipe
    with mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence
    ) as hands:
        results = hands.process(image)
        
        # If hand landmarks detected
        if results.multi_hand_landmarks:
            # Take the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract keypoints
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            
            # Draw landmarks on annotated image
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            return np.array(keypoints, dtype=np.float32), annotated_image
    
    # If no hand detected, return zeros and original image
    return np.zeros(42, dtype=np.float32), annotated_image


def draw_landmarks(image, keypoints):
    """
    Draw landmarks on an image given flattened keypoints.
    
    Args:
        image: RGB image (numpy array)
        keypoints: Flattened array of keypoints (x, y coordinates)
        
    Returns:
        annotated_image: Image with hand landmarks drawn
    """
    if keypoints is None or np.all(keypoints == 0):
        return image
    
    annotated_image = image.copy()
    h, w = annotated_image.shape[:2]
    
    # Convert flat array to landmarks
    landmarks = []
    for i in range(0, len(keypoints), 2):
        x, y = keypoints[i], keypoints[i+1]
        # Scale to image dimensions
        landmarks.append((int(x * w), int(y * h)))
    
    # Draw landmarks
    for idx, point in enumerate(landmarks):
        cv2.circle(annotated_image, point, 3, (0, 255, 0), -1)
        if idx > 0:  # Connect points
            cv2.line(annotated_image, landmarks[idx-1], point, (0, 0, 255), 1)
    
    return annotated_image