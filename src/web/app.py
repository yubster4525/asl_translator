"""
Web interface for ASL Translator.

This module provides a Flask web application for the ASL Translator.
"""

import os
import cv2
import torch
import numpy as np
import base64
import threading
import time
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import io
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.keypoint_extraction import extract_keypoints
from src.models.cnn_model import ASLCNN
from src.models.keypoint_model import KeypointMLP
from src.utils.data_utils import ASLImageDataset, ASLKeypointDataset, create_data_loaders
from src.models.train_cnn_model import train_cnn_model
from src.models.train_keypoint_model import train_keypoint_model
import torchvision.transforms as transforms


# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

# Global variables
models = {}
device = None
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'space', 'nothing', 'del']  # Added 'del' class to match the model

# Training globals
training_thread = None
training_log = []
training_status = "idle"
training_progress = 0
training_history = None


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
        try:
            # Try loading the model directly
            cnn_model = ASLCNN(num_classes=len(class_names)).to(device)
            cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
            cnn_model.eval()
            models['cnn'] = cnn_model
            print(f"Successfully loaded CNN model with {len(class_names)} classes")
        except Exception as e:
            print(f"Error loading CNN model: {str(e)}")
            # If there's still an issue, try to detect the number of classes from the model file
            try:
                state_dict = torch.load(cnn_model_path, map_location=device)
                # Get the shape of the last layer's weight to determine number of classes
                output_size = state_dict['classifier.6.weight'].shape[0]
                print(f"Detected {output_size} classes in saved CNN model")
                cnn_model = ASLCNN(num_classes=output_size).to(device)
                cnn_model.load_state_dict(state_dict)
                cnn_model.eval()
                models['cnn'] = cnn_model
                print(f"Successfully loaded CNN model with corrected classes ({output_size})")
            except Exception as nested_e:
                print(f"Failed to load CNN model even after class correction: {str(nested_e)}")
    
    # Load Keypoint model if available
    if os.path.exists(keypoint_model_path):
        print(f"Loading Keypoint model from {keypoint_model_path}")
        try:
            # Try loading the model directly
            keypoint_model = KeypointMLP(input_dim=42, num_classes=len(class_names)).to(device)
            keypoint_model.load_state_dict(torch.load(keypoint_model_path, map_location=device))
            keypoint_model.eval()
            models['keypoint'] = keypoint_model
            print(f"Successfully loaded keypoint model with {len(class_names)} classes")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If there's still an issue, try to detect the number of classes from the model file
            try:
                state_dict = torch.load(keypoint_model_path, map_location=device)
                # Get the shape of the last layer's weight to determine number of classes
                output_size = state_dict['fc.8.weight'].shape[0]
                print(f"Detected {output_size} classes in saved model")
                keypoint_model = KeypointMLP(input_dim=42, num_classes=output_size).to(device)
                keypoint_model.load_state_dict(state_dict)
                keypoint_model.eval()
                models['keypoint'] = keypoint_model
                print(f"Successfully loaded keypoint model with corrected classes ({output_size})")
            except Exception as nested_e:
                print(f"Failed to load model even after class correction: {str(nested_e)}")
    
    if not models:
        print("Warning: No models found. Please train models first.")


@app.route('/')
def index():
    """Render the main page."""
    available_models = list(models.keys())
    return render_template('index.html', available_models=available_models)


# Custom progress callback for training
class ProgressCallback:
    def __init__(self):
        self.epochs = 0
        self.current_epoch = 0
        self.step = 0
        self.steps_per_epoch = 0
        
    def set_params(self, epochs, steps_per_epoch):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.step = 0
        
    def update(self, epoch, step=None):
        global training_progress
        self.current_epoch = epoch
        if step is not None:
            self.step = step
            # Calculate overall progress (0-100)
            if self.epochs > 0 and self.steps_per_epoch > 0:
                progress = ((epoch - 1) / self.epochs) * 100
                if step > 0:
                    progress += (step / self.steps_per_epoch) * (100 / self.epochs)
                training_progress = min(99, int(progress))  # Cap at 99 until completely done
        else:
            # If only epoch is updated, calculate based on completed epochs
            training_progress = min(99, int((epoch / self.epochs) * 100))


def train_model_thread(model_type, data_dir, epochs, batch_size, learning_rate, use_precomputed=False):
    """Background thread for model training."""
    global training_status, training_log, models, device, training_progress, training_history
    
    try:
        training_status = "preparing"
        training_log.append(f"Starting {model_type} model training...")
        training_log.append(f"Data directory: {data_dir}")
        training_log.append(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Set up progress callback
        progress_callback = ProgressCallback()
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Create output directory for training artifacts
        output_dir = 'models'
        
        if model_type == 'cnn':
            # Define data transforms for CNN
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Create dataset
            training_log.append(f"Loading dataset from {data_dir}...")
            dataset = ASLImageDataset(
                root_dir=data_dir,
                transform=data_transforms,
                img_size=(224, 224)
            )
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                dataset, 
                batch_size=batch_size,
                val_split=0.15
            )
            
            # Set progress callback parameters
            progress_callback.set_params(epochs, len(train_loader))
            
            # Train CNN model
            training_status = "training"
            training_log.append(f"Starting CNN training for {epochs} epochs...")
            model, history = train_cnn_model(
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=len(dataset.classes),
                device=device,
                epochs=epochs,
                lr=learning_rate,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            
            # Update global model
            models['cnn'] = model
            
        else:  # keypoint model
            if use_precomputed and os.path.exists(use_precomputed):
                # Use precomputed keypoints
                training_log.append(f"Using precomputed keypoints from {use_precomputed}")
                dataset = ASLKeypointDataset(
                    root_dir=data_dir,
                    precomputed_keypoints=use_precomputed
                )
            else:
                # Try to find precomputed keypoints in the default location
                default_keypoints_path = os.path.join('data', 'precomputed_keypoints.npz')
                if os.path.exists(default_keypoints_path):
                    training_log.append(f"Found precomputed keypoints at {default_keypoints_path}, using these for faster training...")
                    dataset = ASLKeypointDataset(
                        root_dir=data_dir,
                        precomputed_keypoints=default_keypoints_path
                    )
                else:
                    # Extract keypoints on the fly
                    training_log.append("No precomputed keypoints found. Extracting keypoints on the fly (this might be slow)...")
                    training_log.append("Tip: Run 'python precompute.py' first to speed up future training.")
                    dataset = ASLKeypointDataset(
                        root_dir=data_dir
                    )
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                dataset, 
                batch_size=batch_size,
                val_split=0.15
            )
            
            # Set progress callback parameters
            progress_callback.set_params(epochs, len(train_loader))
            
            # Train keypoint model
            training_status = "training"
            training_log.append(f"Starting keypoint model training for {epochs} epochs...")
            model, history = train_keypoint_model(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=42,  # 21 keypoints with x,y coordinates
                num_classes=len(dataset.classes),
                device=device,
                epochs=epochs,
                lr=learning_rate,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            
            # Update global model
            models['keypoint'] = model
        
        # Save training history for plotting
        training_history = history
        
        # Training complete
        training_status = "completed"
        training_progress = 100
        training_log.append("Training completed successfully!")
        training_log.append(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
        training_log.append(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        training_log.append(f"Model saved to {output_dir}")
        
    except Exception as e:
        # Handle exceptions
        training_status = "failed"
        training_log.append(f"ERROR: Training failed: {str(e)}")
        import traceback
        training_log.append(traceback.format_exc())


@app.route('/train', methods=['POST'])
def train_model():
    """Start model training."""
    global training_thread, training_status, training_log, training_progress, training_history
    
    # Check if training already in progress
    if training_thread and training_thread.is_alive():
        return jsonify({
            'status': 'error',
            'message': 'Training already in progress'
        })
    
    # Get training parameters
    model_type = request.form.get('model_type', 'keypoint')
    data_dir = request.form.get('data_dir', 'data/asl_alphabet')
    epochs = int(request.form.get('epochs', 10))
    batch_size = int(request.form.get('batch_size', 32))
    learning_rate = float(request.form.get('learning_rate', 0.001))
    use_precomputed = request.form.get('precomputed_keypoints', False)
    
    if use_precomputed == 'false':
        use_precomputed = False
    
    # Reset training state
    training_status = "starting"
    training_log = []
    training_progress = 0
    training_history = None
    
    # Start training in a background thread
    training_thread = threading.Thread(
        target=train_model_thread,
        args=(model_type, data_dir, epochs, batch_size, learning_rate, use_precomputed)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Training started'
    })


@app.route('/training_status')
def get_training_status():
    """Get current training status."""
    global training_status, training_log, training_progress
    
    return jsonify({
        'status': training_status,
        'progress': training_progress,
        'log': training_log
    })


@app.route('/training_plot')
def get_training_plot():
    """Generate and return a training history plot."""
    global training_history
    
    if training_history is None:
        return Response("No training history available", status=404)
    
    # Create a figure with accuracy and loss subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot accuracy
    ax1.plot(training_history['train_acc'], label='Training Accuracy')
    ax1.plot(training_history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(training_history['train_loss'], label='Training Loss')
    ax2.plot(training_history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    
    # Convert plot to PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    
    # Return the image
    return Response(output.getvalue(), mimetype='image/png')


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