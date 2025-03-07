"""
Train CNN model for ASL image recognition.

This script trains a CNN model on ASL alphabet images and saves the model and training history.
"""

import os
import argparse
import torch
import torchvision.transforms as transforms

from src.utils.data_utils import ASLImageDataset, create_data_loaders
from src.models.cnn_model import train_cnn_model


def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set up device for training
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Define data transforms
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
    print(f"Loading dataset from {args.data_dir}...")
    dataset = ASLImageDataset(
        root_dir=args.data_dir,
        transform=data_transforms,
        img_size=(224, 224)
    )
    
    # Get number of classes
    num_classes = len(dataset.classes)
    print(f"Detected {num_classes} classes: {dataset.classes}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset, 
        batch_size=args.batch_size,
        val_split=0.15
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    model, history = train_cnn_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        output_dir=args.output_dir
    )
    
    print(f"Training complete. Model and history saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model for ASL image recognition")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ASL alphabet dataset directory")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model and training history")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    
    args = parser.parse_args()
    main(args)