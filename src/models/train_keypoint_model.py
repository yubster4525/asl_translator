"""
Train Keypoint model for ASL recognition.

This script trains an MLP model on ASL hand keypoints and saves the model and training history.
"""

import os
import sys
import argparse
import torch

# Add the project root to the path so we can import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.data_utils import ASLKeypointDataset, create_data_loaders
from src.models.keypoint_model import train_keypoint_model


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
    
    # Create dataset
    print(f"Loading dataset...")
    if args.precomputed_keypoints:
        print(f"Using precomputed keypoints from {args.precomputed_keypoints}")
        dataset = ASLKeypointDataset(
            root_dir=args.data_dir,
            precomputed_keypoints=args.precomputed_keypoints
        )
    else:
        print(f"Extracting keypoints on-the-fly from {args.data_dir}")
        dataset = ASLKeypointDataset(root_dir=args.data_dir)
    
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
    model, history = train_keypoint_model(
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
    parser = argparse.ArgumentParser(description="Train Keypoint model for ASL recognition")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ASL alphabet dataset directory")
    parser.add_argument("--precomputed_keypoints", type=str, default=None,
                        help="Path to precomputed keypoints file (.npz)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model and training history")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    
    args = parser.parse_args()
    main(args)