"""
Data utilities for ASL Translator.

This module provides utilities for data loading, preprocessing, and augmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from src.utils.keypoint_extraction import extract_keypoints

class ASLImageDataset(Dataset):
    """Dataset for loading ASL alphabet images."""
    
    def __init__(self, root_dir, transform=None, img_size=(224, 224)):
        """
        Args:
            root_dir: Directory with ASL alphabet folders
            transform: Optional image transformations
            img_size: Resize images to this size
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self.labels = []
        
        # Get all classes (folders)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all samples
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = cv2.imread(img_path)
        if image is None:
            # If image fails to load, return a zero tensor
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize image
            image = cv2.resize(image, self.img_size)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = image.transpose((2, 0, 1))  # HWC to CHW
            image = image / 255.0  # Normalize to [0, 1]
            image = torch.FloatTensor(image)
        
        return image, label


class ASLKeypointDataset(Dataset):
    """Dataset for loading precomputed ASL keypoints or extracting them from images."""
    
    def __init__(self, root_dir, precomputed_keypoints=None, transform=None):
        """
        Args:
            root_dir: Directory with ASL alphabet folders
            precomputed_keypoints: Path to precomputed keypoints file (optional)
            transform: Optional transformations for keypoints
        """
        self.root_dir = root_dir
        self.transform = transform
        self.precomputed = precomputed_keypoints is not None
        self.samples = []
        self.labels = []
        
        # Get all classes (folders)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        if self.precomputed:
            # Load precomputed keypoints
            data = np.load(precomputed_keypoints, allow_pickle=True)
            self.keypoints = data['keypoints']
            self.keypoint_labels = data['labels']
        else:
            # Load all image paths
            for cls in self.classes:
                cls_dir = os.path.join(root_dir, cls)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(cls_dir, fname))
                        self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        if self.precomputed:
            return len(self.keypoint_labels)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.precomputed:
            # Return precomputed keypoints
            keypoints = self.keypoints[idx]
            label = self.keypoint_labels[idx]
        else:
            # Load image and extract keypoints
            img_path = self.samples[idx]
            label = self.labels[idx]
            
            image = cv2.imread(img_path)
            if image is None:
                # If image fails to load, return a zero vector
                keypoints = np.zeros(42, dtype=np.float32)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Extract keypoints
                keypoints, _ = extract_keypoints(image)
        
        # Apply transformations if specified
        if self.transform:
            keypoints = self.transform(keypoints)
        
        keypoints = torch.FloatTensor(keypoints)
        return keypoints, label


def create_data_loaders(dataset, batch_size=32, val_split=0.15, num_workers=0):
    """
    Create training and validation data loaders from a dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for DataLoader
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for DataLoader
        
    Returns:
        train_loader, val_loader: Training and validation DataLoader objects
    """
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader