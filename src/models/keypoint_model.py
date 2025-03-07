"""
MLP model for ASL Keypoint classification.

This module defines a PyTorch model and training functions for ASL keypoint classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.utils.visualization import plot_training_history


class KeypointMLP(nn.Module):
    """MLP model for ASL keypoint classification."""
    
    def __init__(self, input_dim=42, num_classes=28):
        """
        Args:
            input_dim: Number of input features (42 for hand keypoints x,y values)
            num_classes: Number of output classes
        """
        super(KeypointMLP, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


def train_epoch(model, device, train_loader, criterion, optimizer):
    """
    Train model for one epoch.
    
    Args:
        model: The neural network model
        device: Device to train on (cpu or cuda)
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
    
    Returns:
        epoch_loss, epoch_acc: Loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss/total:.4f}", 
            'acc': f"{100*correct/total:.2f}%"
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, device, val_loader, criterion):
    """
    Validate model on validation set.
    
    Args:
        model: The neural network model
        device: Device to validate on (cpu or cuda)
        val_loader: DataLoader for validation data
        criterion: Loss function
    
    Returns:
        epoch_loss, epoch_acc: Loss and accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/total:.4f}", 
                'acc': f"{100*correct/total:.2f}%"
            })
    
    # Calculate validation metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_keypoint_model(train_loader, val_loader, num_classes, device, 
                         epochs=10, lr=0.001, output_dir="models"):
    """
    Train a keypoint-based ASL classifier.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_classes: Number of output classes
        device: Device to train on (cpu or cuda)
        epochs: Number of training epochs
        lr: Learning rate
        output_dir: Directory to save model and plots
    
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    # Create model
    model = KeypointMLP(input_dim=42, num_classes=num_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Track best validation accuracy
    best_val_acc = 0.0
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, device, val_loader, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_keypoint_model.pth"))
            print(f"  New best model saved! Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_keypoint_model.pth"))
    
    # Plot training history
    plot_training_history(
        history['train_loss'], 
        history['val_loss'],
        history['train_acc'], 
        history['val_acc'],
        output_dir
    )
    
    return model, history