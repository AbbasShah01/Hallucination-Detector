"""
Fine-tuning script for DistilBERT on HaluEval dataset
Trains a binary classification model to detect hallucinations in LLM outputs.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class HallucinationDataset(Dataset):
    """
    PyTorch Dataset class for preprocessed hallucination detection data.
    """
    
    def __init__(self, tokenized_data, tokenizer):
        """
        Initialize dataset.
        
        Args:
            tokenized_data: List of dictionaries with tokenized samples
            tokenizer: Tokenizer object for padding
        """
        self.data = tokenized_data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        sample = self.data[idx]
        
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


def load_preprocessed_data(data_path="data/preprocessed/tokenized_data.json"):
    """
    Load preprocessed tokenized data from disk.
    
    Args:
        data_path: Path to the tokenized data JSON file
    
    Returns:
        List of tokenized samples
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}. "
            "Please run preprocess_halueval.py first."
        )
    
    print(f"Loading preprocessed data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        tokenized_data = json.load(f)
    
    print(f"Loaded {len(tokenized_data)} samples")
    return tokenized_data


def load_tokenizer(tokenizer_path="data/preprocessed/tokenizer"):
    """
    Load tokenizer from disk.
    
    Args:
        tokenizer_path: Path to the saved tokenizer directory
    
    Returns:
        Tokenizer object
    """
    if not os.path.exists(tokenizer_path):
        # Fallback to default DistilBERT tokenizer
        print(f"Tokenizer not found at {tokenizer_path}, using default DistilBERT tokenizer...")
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    return AutoTokenizer.from_pretrained(tokenizer_path)


def split_data(tokenized_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        tokenized_data: List of all tokenized samples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle data
    indices = np.random.permutation(len(tokenized_data))
    tokenized_data = [tokenized_data[i] for i in indices]
    
    # Calculate split indices
    n_total = len(tokenized_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data
    train_data = tokenized_data[:n_train]
    val_data = tokenized_data[n_train:n_train + n_val]
    test_data = tokenized_data[n_train + n_val:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/n_total*100:.1f}%)")
    
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, tokenizer, batch_size=16):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        tokenizer: Tokenizer object
        batch_size: Batch size for training
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = HallucinationDataset(train_data, tokenizer)
    val_dataset = HallucinationDataset(val_data, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def initialize_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Initialize DistilBERT model for binary classification.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels (2 for binary)
    
    Returns:
        Model object
    """
    print(f"\nInitializing model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    print(f"Model initialized with {num_labels} labels")
    return model


def train_epoch(model, train_loader, optimizer, scheduler, device, criterion):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (CPU or CUDA)
        criterion: Loss function
    
    Returns:
        Average training loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, device, criterion):
    """
    Validate the model for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device (CPU or CUDA)
        criterion: Loss function
    
    Returns:
        Average validation loss, accuracy, precision, recall, and F1 score
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=3,
    learning_rate=2e-5,
    device=None
):
    """
    Main training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device (CPU or CUDA), auto-detected if None
    
    Returns:
        Dictionary with training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model.to(device)
    
    # Initialize optimizer (AdamW) and loss function (binary cross-entropy)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Binary cross-entropy loss (handled by model's loss function)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(
            model, val_loader, device, criterion
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    
    return history


def save_model(model, tokenizer, output_dir="models/distilbert_halueval"):
    """
    Save the trained model and tokenizer.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Directory to save the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Model saved successfully!")


def plot_training_history(history, output_dir="results"):
    """
    Plot training loss and accuracy curves.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create figs subdirectory for paper figures
    figs_dir = os.path.join(output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    # Save plots (to both locations)
    loss_plot_path = os.path.join(output_dir, "training_loss_accuracy.png")
    loss_plot_path_figs = os.path.join(figs_dir, "training_loss_accuracy.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(loss_plot_path_figs, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to {loss_plot_path}")
    plt.close()
    
    # Create additional plot for precision, recall, F1
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2, marker='o')
    ax.plot(epochs, history['val_recall'], 'b-', label='Recall', linewidth=2, marker='s')
    ax.plot(epochs, history['val_f1'], 'r-', label='F1 Score', linewidth=2, marker='^')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Validation Metrics: Precision, Recall, and F1 Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    metrics_plot_path = os.path.join(output_dir, "validation_metrics.png")
    metrics_plot_path_figs = os.path.join(figs_dir, "validation_metrics.png")
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(metrics_plot_path_figs, dpi=300, bbox_inches='tight')
    print(f"Validation metrics plot saved to {metrics_plot_path}")
    plt.close()
    
    plt.close('all')


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("DistilBERT Fine-tuning for Hallucination Detection")
    print("=" * 60)
    
    # Configuration
    config = {
        'data_path': "data/preprocessed/tokenized_data.json",
        'tokenizer_path': "data/preprocessed/tokenizer",
        'model_name': "distilbert-base-uncased",
        'batch_size': 16,
        'num_epochs': 3,
        'learning_rate': 2e-5,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'output_dir': "models/distilbert_halueval",
        'results_dir': "results"
    }
    
    # Step 1: Load preprocessed data
    tokenized_data = load_preprocessed_data(config['data_path'])
    
    # Step 2: Load tokenizer
    tokenizer = load_tokenizer(config['tokenizer_path'])
    
    # Step 3: Split data
    train_data, val_data, test_data = split_data(
        tokenized_data,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )
    
    # Step 4: Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, tokenizer, batch_size=config['batch_size']
    )
    
    # Step 5: Initialize model
    model = initialize_model(config['model_name'], num_labels=2)
    
    # Step 6: Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate']
    )
    
    # Step 7: Save model
    save_model(model, tokenizer, output_dir=config['output_dir'])
    
    # Step 8: Plot training history
    plot_training_history(history, output_dir=config['results_dir'])
    
    # Save training history as JSON
    history_path = os.path.join(config['results_dir'], "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

