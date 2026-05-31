"""
Helper functions for training, evaluation, and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import pandas as pd


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: 'cuda' or 'cpu'
    
    Returns:
        Average loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc='Training')
    
    for inputs, labels in progress_bar:
        # Move data to device (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: Neural network model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: 'cuda' or 'cpu'
    
    Returns:
        loss, accuracy, all predictions, all true labels
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    # No gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Accumulate metrics
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    
    return avg_loss, accuracy, all_preds, all_labels


def plot_training_history(histories, model_names, save_path='results/training_history.png'):
    """
    Plot training and validation metrics for all models.
    
    Args:
        histories: List of dictionaries containing training metrics
        model_names: List of model names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    for history, name in zip(histories, model_names):
        axes[0].plot(history['train_loss'], label=f'{name} (Train)', linestyle='--')
        axes[0].plot(history['val_loss'], label=f'{name} (Val)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    for history, name in zip(histories, model_names):
        axes[1].plot(history['train_acc'], label=f'{name} (Train)', linestyle='--')
        axes[1].plot(history['val_acc'], label=f'{name} (Val)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved training history to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, 
                         save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix with class names.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrix to {save_path}")


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate per-class and overall metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary containing metrics
    """
    # Overall metrics
    overall_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True)
    
    # Create DataFrame for easy viewing
    metrics_df = pd.DataFrame(report).transpose()
    
    return {
        'overall_f1': overall_f1,
        'classification_report': report,
        'metrics_df': metrics_df
    }


def visualize_predictions(model, data_loader, class_names, device, 
                         num_images=16, save_path='results/predictions.png'):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        class_names: List of class names
        device: 'cuda' or 'cpu'
        num_images: Number of images to display
        save_path: Path to save the plot
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].cpu().squeeze()
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        confidence = probabilities[i][predicted[i]].item() * 100
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Green if correct, red if wrong
        color = 'green' if labels[i] == predicted[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved predictions visualization to {save_path}")


class LearningRateScheduler:
    """
    Custom learning rate scheduler with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs=3, max_epochs=20, 
                 base_lr=0.001, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
