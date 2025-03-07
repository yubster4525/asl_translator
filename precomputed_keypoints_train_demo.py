#!/usr/bin/env python
"""
Precomputed Keypoints Training and Demo Pipeline
--------------------------------------------------
This script supports two modes:
  1. Training Mode:
     - Loads precomputed ASL keypoints from a pickle file.
     - Constructs a PyTorch Dataset from the precomputed cache.
     - Splits the dataset into training and validation sets.
     - Trains a simple MLP classifier (KeypointMLP) on the keypoint data.
     - Saves the best model (based on validation accuracy) to disk.
  2. Demo Mode:
     - Loads the best saved model.
     - Uses MediaPipe in real time (with stderr suppressed) to extract hand keypoints
       from the live webcam feed.
     - Feeds the keypoints into the trained model to predict the current ASL alphabet.
     - Displays the prediction overlaid on the video feed.

Usage:
  Training:
    python precomputed_keypoints_train_demo.py --mode train --cache_file asl_keypoints_cache.pkl --epochs 10 --batch_size 32
  Demo:
    python precomputed_keypoints_train_demo.py --mode demo --cache_file asl_keypoints_cache.pkl --output_dir output

The cache file (e.g., asl_keypoints_cache.pkl) should contain a dictionary with:
  - "cache": mapping image paths to {"keypoints": numpy array, "label": int, "class": str}
  - "classes": a sorted list of class names
  - "class_to_idx": a dictionary mapping class names to indices
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import cv2
import mediapipe as mp
import contextlib

# ---------------------------
# Custom Dataset to load precomputed keypoints from cache
# ---------------------------
class PrecomputedKeypointDataset(Dataset):
    def __init__(self, cache_file):
        """
        Initializes the dataset by loading the pickle cache file.

        Args:
            cache_file (str): Path to the pickle file containing the cache.
        """
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        self.cache = data["cache"]           # Dictionary mapping image path -> {keypoints, label, class}
        self.classes = data["classes"]         # Sorted list of class names (e.g., ["A", "B", ..., "nothing"])
        self.class_to_idx = data["class_to_idx"] # Dictionary mapping class name to index
        self.keys = list(self.cache.keys())    # List of image paths (or keys) for indexing

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.cache[key]
        keypoints = sample["keypoints"]  # NumPy array of shape (42,)
        label = sample["label"]
        # Convert keypoints and label to PyTorch tensors
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return keypoints_tensor, label_tensor

# ---------------------------
# Model: Simple MLP Classifier for Keypoints
# ---------------------------
class KeypointMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        A simple multilayer perceptron classifier.

        Args:
            input_dim (int): Dimensionality of input features (42 for 21 keypoints × 2).
            num_classes (int): Number of output classes.
        """
        super(KeypointMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# ---------------------------
# Training and Validation Functions
# ---------------------------
def train_epoch(model, device, dataloader, criterion, optimizer):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        device: Device to run on.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        epoch_loss, epoch_accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Use tqdm for a progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        total += target.size(0)
        correct += (preds == target).sum().item()
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", accuracy=f"{(correct/total)*100:.2f}%")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, device, dataloader, criterion):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model.
        device: Device to run on.
        dataloader: DataLoader for validation data.
        criterion: Loss function.

    Returns:
        epoch_loss, epoch_accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", accuracy=f"{(correct/total)*100:.2f}%")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def plot_history(train_losses, val_losses, train_acc, val_acc, output_dir="output"):
    """
    Plots training and validation loss and accuracy over epochs.
    Saves the plot in the output directory.
    """
    import matplotlib.pyplot as plt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "bo-", label="Train Loss")
    plt.plot(epochs, val_losses, "ro-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, "bo-", label="Train Acc")
    plt.plot(epochs, val_acc, "ro-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "history.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Training history plot saved to {plot_path}")

# ---------------------------
# Training Pipeline
# ---------------------------
def train_pipeline(args, device):
    """
    Loads the precomputed keypoints from cache, splits the data,
    trains the KeypointMLP classifier, and saves the best model.
    
    Args:
        args: Command line arguments.
        device: Device to use.
    
    Returns:
        model, classes (list of class names)
    """
    # Load dataset from cache file
    print(f"Loading precomputed keypoints from {args.cache_file}...")
    with open(args.cache_file, "rb") as f:
        data = pickle.load(f)
    classes = data["classes"]
    print(f"Detected classes: {classes}")
    
    dataset = PrecomputedKeypointDataset(args.cache_file)
    dataset_size = len(dataset)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(classes)
    # Create the model with input dimension 42 (keypoints) and output dimension equal to number of classes.
    model = KeypointMLP(input_dim=42, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t_loss, t_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        v_loss, v_acc = validate_epoch(model, device, val_loader, criterion)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        print(f"  Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%")
        
        # Save best model based on validation accuracy
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_path = os.path.join(args.output_dir, "best_keypoint_model.pth")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(model.state_dict(), best_model_path)
            print("  Best model updated and saved.")
    
    plot_history(train_losses, val_losses, train_accs, val_accs, output_dir=args.output_dir)
    
    final_model_path = os.path.join(args.output_dir, "final_keypoint_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved as: {final_model_path}")
    return model, classes

# ---------------------------
# Demo: Real-time Video Inference using Live Keypoint Extraction
# ---------------------------
def run_demo(model, device, classes):
    """
    Runs a live demo by opening the webcam, extracting hand keypoints in real time,
    feeding them into the trained model, and displaying the predicted ASL alphabet on the video.
    
    Args:
        model: The trained KeypointMLP model.
        device: Device to run on.
        classes: List of class names.
    """
    # For live demo, we create a MediaPipe Hands object once.
    mp_hands_instance = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    # Open the webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    
    print("Starting live demo. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame. Exiting.")
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Suppress stderr from MediaPipe to avoid clutter
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
            results = mp_hands_instance.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # If a hand is detected, extract keypoints
            keypoints = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                keypoints.extend([landmark.x, landmark.y])
            keypoints = np.array(keypoints, dtype=np.float32)
        else:
            # If no hand is detected, use a zero vector
            keypoints = np.zeros(42, dtype=np.float32)
        
        # Convert keypoints to a torch tensor and add a batch dimension
        tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
        predicted_label = classes[pred.item()]
        
        # Overlay the prediction on the frame
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("ASL Keypoint Recognition Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    mp_hands_instance.close()

# ---------------------------
# Main Function and Argument Parsing
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Keypoint Pipeline: Train a classifier on precomputed keypoints or run a live demo.")
    parser.add_argument("--mode", type=str, choices=["train", "demo"], required=True,
                        help="Mode to run: 'train' for training, 'demo' for live inference demo.")
    parser.add_argument("--cache_file", type=str, required=True,
                        help="Path to the precomputed keypoints cache file (e.g., asl_keypoints_cache.pkl)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save models and plots (default: output)")
    args = parser.parse_args()

    # Use MPS if available on macOS (or else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "train":
        model, classes = train_pipeline(args, device)
    elif args.mode == "demo":
        # In demo mode, load the best model from the output directory.
        best_model_path = os.path.join(args.output_dir, "best_keypoint_model.pth")
        if not os.path.exists(best_model_path):
            print(f"Error: Best model not found at '{best_model_path}'. Please run training mode first.")
            sys.exit(1)
        # Load classes from cache file
        with open(args.cache_file, "rb") as f:
            data = pickle.load(f)
        classes = data["classes"]
        num_classes = len(classes)
        # Initialize model and load state_dict
        model = KeypointMLP(input_dim=42, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from '{best_model_path}'. Starting demo...")
        run_demo(model, device, classes)
