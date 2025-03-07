"""
Visualization utilities for ASL Translator.

This module provides functions for visualizing training history, 
model predictions, and keypoint visualization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_training_history(train_losses, val_losses, train_acc, val_acc, output_dir="models"):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch  
        train_acc: List of training accuracies per epoch
        val_acc: List of validation accuracies per epoch
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def add_prediction_to_frame(frame, prediction, confidence=None, position=(10, 30)):
    """
    Add prediction text overlay to a video frame.
    
    Args:
        frame: The video frame (numpy array)
        prediction: The predicted class
        confidence: Optional confidence score
        position: Position for the text (x, y)
        
    Returns:
        Modified frame with prediction text
    """
    annotated_frame = frame.copy()
    
    # Prepare text to display
    if confidence is not None:
        text = f"Predicted: {prediction} ({confidence:.2f})"
    else:
        text = f"Predicted: {prediction}"
    
    # Add text to the frame
    cv2.putText(
        annotated_frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    return annotated_frame


def create_confusion_matrix_plot(cm, class_names, output_dir="models"):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Set up axes with class names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text labels to each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()