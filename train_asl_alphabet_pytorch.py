"""
train_asl_alphabet_pytorch_mps.py

This script trains a simple CNN to classify ASL alphabet signs using PyTorch
with support for Apple's MPS backend. It uses torchvision's ImageFolder
to load and augment data, and tqdm to display a live progress bar with loss
and accuracy updates.

Assumes your dataset directory is organized like:
  asl_alphabet/
      A/
          img1.jpg
          img2.jpg
          ...
      B/
          ...
      ...
      SPACE/
      DELETE/
      NOTHING/

Usage:
    conda activate signlang
    python train_asl_alphabet_pytorch_mps.py --data_dir /path/to/asl_alphabet --epochs 10
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Set global image dimensions for model construction later
global img_height, img_width

class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super(ASLCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 x H x W
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # Downsample by factor 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_height // 8) * (img_width // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def plot_history(train_losses, val_losses, train_acc, val_acc, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    epochs = np.arange(1, len(train_losses)+1)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    
    output_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(output_path)
    plt.show()
    print(f"Training history plot saved to {output_path}")

def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Wrap the loader with tqdm to show a progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
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
        
        # Update progress bar with current batch statistics
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", accuracy=f"{(correct/total)*100:.2f}%")
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_model(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on the ASL Alphabet dataset using PyTorch with MPS support and live progress bar.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the ASL Alphabet dataset directory")
    parser.add_argument("--img_height", type=int, default=64, help="Image height (default: 64)")
    parser.add_argument("--img_width", type=int, default=64, help="Image width (default: 64)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output plots and models")
    args = parser.parse_args()
    
    img_height = args.img_height
    img_width = args.img_width
    
    # Use MPS if available, otherwise fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    # Define data transforms for training and validation
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Load dataset using ImageFolder; expects folder structure as described
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transforms)
    dataset_size = len(full_dataset)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Change validation dataset transform to val_transforms
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(full_dataset.classes)
    print("Detected classes:", full_dataset.classes)
    
    # Build the model and move it to the selected device
    model = ASLCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_model(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_model(model, device, val_loader, criterion)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, "best_asl_model.pth")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(model.state_dict(), best_model_path)
            print("  Best model updated and saved.")
    
    plot_history(train_losses, val_losses, train_accs, val_accs, output_dir=args.output_dir)
    final_model_path = os.path.join(args.output_dir, "final_asl_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Final model saved as:", final_model_path)
