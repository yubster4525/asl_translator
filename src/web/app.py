"""
Web interface for ASL Translator.

This module provides a Flask web application for the ASL Translator.
"""

import os
import cv2
import torch
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

from src.utils.keypoint_extraction import extract_keypoints
from src.models.cnn_model import ASLCNN
from src.models.keypoint_model import KeypointMLP


# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

# Global variables
models = {}
device = None
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'space', 'nothing']


def init_models():
    """Initialize models and device."""
    global models, device
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Define model paths
    model_dir = 'models'
    cnn_model_path = os.path.join(model_dir, 'best_asl_model.pth')
    keypoint_model_path = os.path.join(model_dir, 'best_keypoint_model.pth')
    
    # Load CNN model if available
    if os.path.exists(cnn_model_path):
        print(f"Loading CNN model from {cnn_model_path}")
        cnn_model = ASLCNN(num_classes=len(class_names)).to(device)
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        cnn_model.eval()
        models['cnn'] = cnn_model
    
    # Load Keypoint model if available
    if os.path.exists(keypoint_model_path):
        print(f"Loading Keypoint model from {keypoint_model_path}")
        keypoint_model = KeypointMLP(input_dim=42, num_classes=len(class_names)).to(device)
        keypoint_model.load_state_dict(torch.load(keypoint_model_path, map_location=device))
        keypoint_model.eval()
        models['keypoint'] = keypoint_model
    
    if not models:
        print("Warning: No models found. Please train models first.")


@app.route('/')
def index():
    """Render the main page."""
    available_models = list(models.keys())
    return render_template('index.html', available_models=available_models)


def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image."""
    # Remove the "data:image/jpeg;base64," prefix
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    image_bytes = base64.b64decode(base64_string)
    
    # Convert to numpy array
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    
    # Convert RGB to BGR for OpenCV
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    # Convert BGR to RGB
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert to base64
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_string}"


@app.route('/predict', methods=['POST'])
def predict():
    """Process webcam frame and return prediction."""
    try:
        # Get data from request
        data = request.json
        model_type = data.get('model', 'keypoint')
        image_data = data.get('image')
        
        # Check if model is available
        if model_type not in models:
            return jsonify({
                'error': f"Model '{model_type}' not available. Available models: {list(models.keys())}"
            })
        
        # Convert base64 to image
        frame = base64_to_image(image_data)
        
        # Make prediction based on model type
        if model_type == 'cnn':
            # Preprocess for CNN
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize and convert to tensor
            rgb = rgb / 255.0
            rgb = rgb.transpose((2, 0, 1))  # HWC to CHW
            tensor = torch.FloatTensor(rgb).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = models[model_type](tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            # No visualization needed for CNN
            annotated_image = frame
            
        else:  # keypoint model
            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract keypoints and get annotated frame
            keypoints, annotated_image = extract_keypoints(
                rgb, static_mode=False, min_detection_confidence=0.5
            )
            
            # Only predict if keypoints were detected (not all zeros)
            if not np.all(keypoints == 0):
                # Convert to tensor
                tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = models[model_type](tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
            else:
                # No hand detected
                pred = torch.tensor([len(class_names) - 1])  # 'nothing' class
                conf = torch.tensor([1.0])
        
        # Get class and confidence
        predicted_class = class_names[pred.item()]
        confidence = float(conf.item())
        
        # Convert annotated image to base64
        annotated_base64 = image_to_base64(annotated_image)
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'annotated_image': annotated_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    init_models()
    app.run(debug=True)