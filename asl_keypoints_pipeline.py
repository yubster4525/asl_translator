"""
asl_keypoints_pipeline.py

This script trains a classifier based on hand keypoints extracted using MediaPipe,
instead of training on raw image data. It then offers a demo mode that captures live video,
extracts keypoints, and uses the trained model to predict the current ASL alphabet sign.

Usage:
    For training:
        python asl_keypoints_pipeline.py --mode train --data_dir /path/to/asl_alphabet --epochs 10
    For demo:
        python asl_keypoints_pipeline.py --mode demo --data_dir /path/to/asl_alphabet
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import mediapipe as mp

# Set up MediaPipe Hands (for keypoint extraction)
mp_hands = mp.solutions.hands

# ---------------------------
# Custom Dataset: KeypointDataset
# ---------------------------
class KeypointDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Assumes the dataset directory is structured as:
            root_dir/
                class1/ (e.g., A)
                    img1.jpg, img2.jpg, ...
                class2/ (e.g., B)
                    ...
        """
        self.samples = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load image using OpenCV
        img_path = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            # If image fails to load, return a zero vector.
            keypoints = np.zeros(42, dtype=np.float32)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            # Extract keypoints using MediaPipe Hands (static image mode)
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    # Take the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y])
                    keypoints = np.array(keypoints, dtype=np.float32)
                else:
                    keypoints = np.zeros(42, dtype=np.float32)
        label = self.labels[idx]
        return keypoints, label

# ---------------------------
# Model: MLP Classifier on Keypoints
# ---------------------------
class KeypointMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
    import matplotlib.pyplot as plt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history.png"))
    plt.show()

# ---------------------------
# Training Pipeline for Keypoints
# ---------------------------
def train_pipeline(args, device):
    # Create dataset and split into train/val
    dataset = KeypointDataset(root_dir=args.data_dir)
    dataset_size = len(dataset)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(dataset.classes)
    print("Detected classes:", dataset.classes)

    # Define model: input dimension is 42 (if one hand is detected)
    model = KeypointMLP(input_dim=42, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t_loss, t_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        v_loss, v_acc = validate_epoch(model, device, val_loader, criterion)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        print(f"  Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%")
        # Save best model
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
    print("Training complete. Final model saved as:", final_model_path)
    return model, dataset.classes

# ---------------------------
# Demo: Live Video Inference using Keypoints
# ---------------------------
def run_demo(model, device, classes):
    model.eval()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break
            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process with MediaPipe to extract keypoints
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y])
                keypoints = np.array(keypoints, dtype=np.float32)
            else:
                keypoints = np.zeros(42, dtype=np.float32)
            # Prepare tensor and predict
            tensor = torch.from_numpy(keypoints).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                _, pred = torch.max(outputs, 1)
            label = classes[pred.item()]
            # Overlay prediction on the frame
            cv2.putText(frame, f"Predicted: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ASL Keypoint Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Main Function
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Keypoint Pipeline: Train a keypoint-based classifier or run demo on live video.")
    parser.add_argument("--mode", type=str, choices=["train", "demo"], required=True,
                        help="Mode: 'train' to train the keypoint-based model; 'demo' to run live inference.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the ASL Alphabet dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs (models, plots)")
    args = parser.parse_args()

    # Set device (using MPS if available on macOS with Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    if args.mode == "train":
        model, classes = train_pipeline(args, device)
    elif args.mode == "demo":
        # In demo mode, we load the best saved model
        best_model_path = os.path.join(args.output_dir, "best_keypoint_model.pth")
        if not os.path.exists(best_model_path):
            print(f"Best model not found at {best_model_path}. Please train the model first.")
            exit(1)
        # Get class names from dataset directory (folders sorted alphabetically)
        classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
        num_classes = len(classes)
        model = KeypointMLP(input_dim=42, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best keypoint model from", best_model_path)
        run_demo(model, device, classes)
