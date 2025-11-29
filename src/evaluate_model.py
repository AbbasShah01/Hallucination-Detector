"""
Evaluation script for Hybrid Hallucination Detection System
Computes metrics, generates confusion matrix, ROC curve, and sample outputs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def load_test_data(data_path: str) -> pd.DataFrame:
    """
    Load test dataset from JSON or CSV file.
    
    Expected format:
    - JSON: List of dicts with 'response', 'label' (0=correct, 1=hallucination)
    - CSV: Columns including 'response' and 'label'
    
    Args:
        data_path: Path to test data file
    
    Returns:
        DataFrame with test data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found at {data_path}")
    
    print(f"Loading test data from {data_path}...")
    
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Ensure required columns exist
    if 'label' not in df.columns:
        raise ValueError("Test data must contain 'label' column")
    if 'response' not in df.columns:
        raise ValueError("Test data must contain 'response' column")
    
    print(f"Loaded {len(df)} test samples")
    return df


def load_predictions(predictions_path: str) -> np.ndarray:
    """
    Load predictions from file.
    
    Expected format:
    - JSON: List of probabilities or binary predictions
    - CSV: Column with predictions
    - TXT: One prediction per line
    
    Args:
        predictions_path: Path to predictions file
    
    Returns:
        Array of predictions (probabilities or binary)
    """
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions not found at {predictions_path}")
    
    print(f"Loading predictions from {predictions_path}...")
    
    if predictions_path.endswith('.json'):
        with open(predictions_path, 'r') as f:
            data = json.load(f)
        predictions = np.array(data)
    elif predictions_path.endswith('.csv'):
        df = pd.read_csv(predictions_path)
        # Assume first column or column named 'prediction'
        if 'prediction' in df.columns:
            predictions = df['prediction'].values
        else:
            predictions = df.iloc[:, 0].values
    elif predictions_path.endswith('.txt'):
        with open(predictions_path, 'r') as f:
            predictions = np.array([float(line.strip()) for line in f])
    else:
        raise ValueError(f"Unsupported file format: {predictions_path}")
    
    print(f"Loaded {len(predictions)} predictions")
    return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (0=correct, 1=hallucination)
        y_pred: Predicted labels (0=correct, 1=hallucination)
    
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix"
):
    """
    Generate and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        title: Plot title
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['Correct', 'Hallucination'],
        yticklabels=['Correct', 'Hallucination']
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add text annotations with percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = (count / total) * 100 if total > 0 else 0
            plt.text(
                j + 0.5, i + 0.7,
                f'{percentage:.1f}%',
                ha='center',
                va='center',
                fontsize=10,
                color='red' if i != j else 'green',
                fontweight='bold'
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str,
    title: str = "ROC Curve"
):
    """
    Generate and save ROC curve plot.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        title: Plot title
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    # Plot diagonal (random classifier)
    plt.plot(
        [0, 1], [0, 1],
        color='navy',
        lw=2,
        linestyle='--',
        label='Random Classifier (AUC = 0.500)'
    )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_metrics_comparison(
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Classification Metrics"
):
    """
    Generate bar plot comparing different metrics.
    
    Args:
        metrics: Dictionary with metric names and values
        save_path: Path to save the plot
        title: Plot title
    """
    # Extract metric names and values
    metric_names = list(metrics.keys())
    metric_values = [metrics[name] for name in metric_names]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(
        metric_names,
        metric_values,
        color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    plt.close()


def get_sample_outputs(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    num_samples: int = 10
) -> Dict[str, List[Dict]]:
    """
    Extract sample outputs for different prediction categories.
    
    Args:
        df: DataFrame with test data
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        num_samples: Number of samples per category
    
    Returns:
        Dictionary with samples for each category
    """
    samples = {
        'true_positives': [],  # Correctly detected hallucinations
        'true_negatives': [],  # Correctly identified as correct
        'false_positives': [],  # Incorrectly flagged as hallucinations
        'false_negatives': []   # Missed hallucinations
    }
    
    for idx in range(len(df)):
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        pred_prob = y_pred_proba[idx]
        
        response = df.iloc[idx]['response'] if 'response' in df.columns else f"Sample {idx}"
        
        sample = {
            'index': idx,
            'response': response[:200] + '...' if len(str(response)) > 200 else response,
            'true_label': 'Hallucination' if true_label == 1 else 'Correct',
            'predicted_label': 'Hallucination' if pred_label == 1 else 'Correct',
            'probability': float(pred_prob)
        }
        
        # Categorize
        if true_label == 1 and pred_label == 1:
            if len(samples['true_positives']) < num_samples:
                samples['true_positives'].append(sample)
        elif true_label == 0 and pred_label == 0:
            if len(samples['true_negatives']) < num_samples:
                samples['true_negatives'].append(sample)
        elif true_label == 0 and pred_label == 1:
            if len(samples['false_positives']) < num_samples:
                samples['false_positives'].append(sample)
        elif true_label == 1 and pred_label == 0:
            if len(samples['false_negatives']) < num_samples:
                samples['false_negatives'].append(sample)
        
        # Check if we have enough samples
        if all(len(v) >= num_samples for v in samples.values()):
            break
    
    return samples


def save_sample_outputs(
    samples: Dict[str, List[Dict]],
    save_path: str
):
    """
    Save sample outputs to JSON file.
    
    Args:
        samples: Dictionary with sample outputs
        save_path: Path to save the file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Sample outputs saved to {save_path}")


def print_sample_outputs(samples: Dict[str, List[Dict]]):
    """
    Print sample outputs to console.
    
    Args:
        samples: Dictionary with sample outputs
    """
    print("\n" + "=" * 70)
    print("Sample Outputs")
    print("=" * 70)
    
    categories = {
        'true_positives': 'True Positives (Correctly Detected Hallucinations)',
        'true_negatives': 'True Negatives (Correctly Identified as Correct)',
        'false_positives': 'False Positives (Incorrectly Flagged as Hallucinations)',
        'false_negatives': 'False Negatives (Missed Hallucinations)'
    }
    
    for category, title in categories.items():
        if samples[category]:
            print(f"\n{title}:")
            print("-" * 70)
            for i, sample in enumerate(samples[category], 1):
                print(f"\n{i}. Index: {sample['index']}")
                print(f"   Response: {sample['response']}")
                print(f"   True Label: {sample['true_label']}")
                print(f"   Predicted: {sample['predicted_label']} (prob: {sample['probability']:.3f})")


def evaluate_model(
    test_data_path: str,
    predictions_path: str,
    output_dir: str = "results",
    threshold: float = 0.5,
    num_samples: int = 10
):
    """
    Complete evaluation pipeline.
    
    Args:
        test_data_path: Path to test dataset
        predictions_path: Path to predictions file
        output_dir: Directory to save results
        threshold: Threshold for binary classification
        num_samples: Number of sample outputs per category
    """
    print("=" * 70)
    print("Model Evaluation Pipeline")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load test data
    df = load_test_data(test_data_path)
    y_true = df['label'].values
    
    # Step 2: Load predictions
    y_pred_proba = load_predictions(predictions_path)
    
    # Convert probabilities to binary predictions if needed
    if y_pred_proba.max() <= 1.0 and y_pred_proba.min() >= 0.0:
        # Assume these are probabilities
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        # Assume these are already binary
        y_pred = y_pred_proba.astype(int)
        y_pred_proba = y_pred_proba.astype(float)  # Keep as probabilities for ROC
    
    # Ensure lengths match
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_proba = y_pred_proba[:min_len]
        df = df.iloc[:min_len]
        print(f"Warning: Truncated to {min_len} samples to match lengths")
    
    # Step 3: Compute metrics
    print("\n" + "-" * 70)
    print("Computing Metrics")
    print("-" * 70)
    metrics = compute_metrics(y_true, y_pred)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Print classification report
    print("\n" + "-" * 70)
    print("Classification Report")
    print("-" * 70)
    print(classification_report(y_true, y_pred, target_names=['Correct', 'Hallucination']))
    
    # Step 4: Generate plots
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # ROC curve
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_pred_proba, roc_path)
    
    # Metrics comparison
    metrics_path = os.path.join(output_dir, "metrics_comparison.png")
    plot_metrics_comparison(metrics, metrics_path)
    
    # Step 5: Sample outputs
    print("\n" + "-" * 70)
    print("Extracting Sample Outputs")
    print("-" * 70)
    samples = get_sample_outputs(df, y_true, y_pred, y_pred_proba, num_samples)
    
    # Save samples
    samples_path = os.path.join(output_dir, "sample_outputs.json")
    save_sample_outputs(samples, samples_path)
    
    # Print samples
    print_sample_outputs(samples)
    
    # Step 6: Save metrics
    metrics_path_json = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path_json}")
    
    # Step 7: Save confusion matrix as JSON
    cm = confusion_matrix(y_true, y_pred)
    cm_dict = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }
    cm_path_json = os.path.join(output_dir, "confusion_matrix.json")
    with open(cm_path_json, 'w') as f:
        json.dump(cm_dict, f, indent=2)
    print(f"Confusion matrix saved to {cm_path_json}")
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")


def main():
    """Main function with example usage."""
    # Configuration
    config = {
        'test_data_path': 'data/test_data.json',  # Update with your test data path
        'predictions_path': 'results/predictions.json',  # Update with your predictions path
        'output_dir': 'results',
        'threshold': 0.5,
        'num_samples': 10
    }
    
    # Check if files exist, if not, create example data
    if not os.path.exists(config['test_data_path']):
        print(f"Warning: Test data not found at {config['test_data_path']}")
        print("Creating example test data for demonstration...")
        create_example_data(config['test_data_path'], config['predictions_path'])
    
    # Run evaluation
    evaluate_model(
        test_data_path=config['test_data_path'],
        predictions_path=config['predictions_path'],
        output_dir=config['output_dir'],
        threshold=config['threshold'],
        num_samples=config['num_samples']
    )


def create_example_data(test_data_path: str, predictions_path: str):
    """Create example data for demonstration."""
    # Example test data
    example_data = [
        {"response": "Barack Obama was the 44th President of the United States.", "label": 0},
        {"response": "Dr. Quantum invented the time machine in 2025.", "label": 1},
        {"response": "Albert Einstein developed the theory of relativity.", "label": 0},
        {"response": "The moon is made of cheese according to NASA.", "label": 1},
        {"response": "Microsoft is a technology company founded in 1975.", "label": 0},
        {"response": "Harry Potter discovered quantum physics in 1997.", "label": 1},
        {"response": "The Earth orbits around the Sun.", "label": 0},
        {"response": "Water boils at 100 degrees Celsius at sea level.", "label": 0},
        {"response": "The fictional character Sherlock Holmes solved real crimes.", "label": 1},
        {"response": "Python is a programming language.", "label": 0},
    ]
    
    # Create directories if needed
    os.makedirs(os.path.dirname(test_data_path) if os.path.dirname(test_data_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(predictions_path) if os.path.dirname(predictions_path) else '.', exist_ok=True)
    
    # Save test data
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(example_data, f, indent=2, ensure_ascii=False)
    
    # Generate example predictions (probabilities)
    np.random.seed(42)
    example_predictions = []
    for item in example_data:
        true_label = item['label']
        # Simulate predictions: mostly correct, but with some noise
        if true_label == 1:
            # For hallucinations, predict high probability
            prob = np.random.uniform(0.6, 0.95)
        else:
            # For correct, predict low probability
            prob = np.random.uniform(0.05, 0.4)
        example_predictions.append(prob)
    
    # Save predictions
    with open(predictions_path, 'w') as f:
        json.dump(example_predictions, f, indent=2)
    
    print(f"Example data created:")
    print(f"  Test data: {test_data_path}")
    print(f"  Predictions: {predictions_path}")


if __name__ == "__main__":
    main()

