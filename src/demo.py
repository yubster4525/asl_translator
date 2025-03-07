"""
Real-time ASL Translator Demo.

This script runs a live webcam demo using trained ASL models (CNN or Keypoint-based).
"""

import argparse
import cv2
import torch
import numpy as np
import os

from src.utils.keypoint_extraction import extract_keypoints
from src.models.cnn_model import ASLCNN
from src.models.keypoint_model import KeypointMLP
from src.utils.visualization import add_prediction_to_frame


def load_model(model_type, model_path, num_classes, device):
    """
    Load the specified model from disk.
    
    Args:
        model_type: Type of model ('cnn' or 'keypoint')
        model_path: Path to the model file
        num_classes: Number of output classes
        device: Device to run model on
    
    Returns:
        model: Loaded PyTorch model
    """
    if model_type == 'cnn':
        model = ASLCNN(num_classes=num_classes).to(device)
    else:  # keypoint
        model = KeypointMLP(input_dim=42, num_classes=num_classes).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def run_demo(args):
    """
    Run the ASL translator demo with webcam feed.
    
    Args:
        args: Command line arguments
    """
    # Set up device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Define class names (A-Z, space, nothing)
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'space', 'nothing']
    num_classes = len(class_names)
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        if args.model_type == 'cnn':
            model_path = os.path.join('models', 'best_asl_model.pth')
        else:
            model_path = os.path.join('models', 'best_keypoint_model.pth')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load model
    print(f"Loading {args.model_type} model from {model_path}...")
    model = load_model(args.model_type, model_path, num_classes, device)
    
    # Start webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Smoothing predictions
    last_predictions = []
    buffer_size = 5
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break
            
            # Create display frame
            display_frame = frame.copy()
            
            # Process based on model type
            if args.model_type == 'cnn':
                # Preprocess for CNN
                resized = cv2.resize(frame, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Normalize and convert to tensor
                rgb = rgb / 255.0
                rgb = rgb.transpose((2, 0, 1))  # HWC to CHW
                tensor = torch.FloatTensor(rgb).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                # Manage prediction buffer
                last_predictions.append((class_names[pred.item()], conf.item()))
                if len(last_predictions) > buffer_size:
                    last_predictions.pop(0)
                
            else:  # keypoint model
                # Convert BGR to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract keypoints and get annotated frame
                keypoints, display_frame = extract_keypoints(
                    rgb, static_mode=False, min_detection_confidence=0.5
                )
                
                # Only predict if keypoints were detected (not all zeros)
                if not np.all(keypoints == 0):
                    # Convert to tensor
                    tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)
                    
                    # Manage prediction buffer
                    last_predictions.append((class_names[pred.item()], conf.item()))
                    if len(last_predictions) > buffer_size:
                        last_predictions.pop(0)
            
            # Smooth predictions by taking most common
            if last_predictions:
                # Count predictions
                predictions = {}
                for p, c in last_predictions:
                    if p not in predictions:
                        predictions[p] = {'count': 0, 'conf': 0}
                    predictions[p]['count'] += 1
                    predictions[p]['conf'] += c
                
                # Find most common
                top_pred = max(predictions.items(), key=lambda x: x[1]['count'])
                pred_class = top_pred[0]
                avg_conf = top_pred[1]['conf'] / top_pred[1]['count']
                
                # Add prediction to frame
                display_frame = add_prediction_to_frame(
                    display_frame, pred_class, avg_conf
                )
            
            # Show frame
            cv2.imshow("ASL Translator", display_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Translator Demo")
    parser.add_argument("--model_type", type=str, choices=['cnn', 'keypoint'], 
                        default='keypoint', help="Type of model to use")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model file (optional, uses default if not specified)")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    run_demo(args)