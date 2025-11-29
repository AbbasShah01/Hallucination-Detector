"""
Generate placeholder visualizations for the Hybrid Hallucination Detection System.
These plots use sample data and can be easily replaced with real results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_training_curves(output_dir="results"):
    """
    Generate placeholder training/validation accuracy and loss curves.
    
    Args:
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample training data
    epochs = np.arange(1, 11)
    
    # Training metrics (simulated)
    train_loss = 0.8 * np.exp(-epochs * 0.15) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    val_loss = 0.85 * np.exp(-epochs * 0.12) + 0.12 + np.random.normal(0, 0.025, len(epochs))
    
    train_acc = 0.5 + 0.4 * (1 - np.exp(-epochs * 0.2)) + np.random.normal(0, 0.01, len(epochs))
    val_acc = 0.48 + 0.38 * (1 - np.exp(-epochs * 0.18)) + np.random.normal(0, 0.015, len(epochs))
    
    # Ensure values are in valid ranges
    train_loss = np.clip(train_loss, 0, 1)
    val_loss = np.clip(val_loss, 0, 1)
    train_acc = np.clip(train_acc, 0, 1)
    val_acc = np.clip(val_acc, 0, 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2.5, markersize=8)
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    ax1.set_ylim([0, max(train_loss.max(), val_loss.max()) * 1.1])
    
    # Add annotation for best validation loss
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_val_loss_epoch - 1]
    ax1.annotate(f'Best Val Loss: {best_val_loss:.3f}\n@ Epoch {best_val_loss_epoch}',
                xy=(best_val_loss_epoch, best_val_loss),
                xytext=(best_val_loss_epoch + 2, best_val_loss + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2.5, markersize=8)
    ax2.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(fontsize=12, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0, 1])
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Add annotation for best validation accuracy
    best_val_acc_epoch = np.argmax(val_acc) + 1
    best_val_acc = val_acc[best_val_acc_epoch - 1]
    ax2.annotate(f'Best Val Acc: {best_val_acc:.3f}\n@ Epoch {best_val_acc_epoch}',
                xy=(best_val_acc_epoch, best_val_acc),
                xytext=(best_val_acc_epoch + 2, best_val_acc - 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {output_path}")
    plt.close()


def generate_confusion_matrix(output_dir="results"):
    """
    Generate placeholder confusion matrix heatmap.
    
    Args:
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample confusion matrix values
    # Format: [True Negative, False Positive]
    #         [False Negative, True Positive]
    cm = np.array([
        [850, 45],   # Correct predictions: 850 correct, 45 false positives
        [38, 67]     # Hallucination predictions: 38 false negatives, 67 true positives
    ])
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['Predicted: Correct', 'Predicted: Hallucination'],
        yticklabels=['Actual: Correct', 'Actual: Hallucination'],
        linewidths=2,
        linecolor='black',
        annot_kws={'size': 16, 'weight': 'bold'}
    )
    
    # Calculate percentages
    total = cm.sum()
    percentages = (cm / total * 100).round(1)
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = percentages[i, j]
            color = 'red' if i != j else 'green'
            plt.text(
                j + 0.5, i + 0.3,
                f'{pct:.1f}%',
                ha='center',
                va='center',
                fontsize=12,
                color=color,
                fontweight='bold'
            )
    
    plt.title('Confusion Matrix - Hybrid Hallucination Detection', 
              fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Add summary statistics
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "confusion_matrix_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


def generate_comparison_table(output_dir="results"):
    """
    Generate placeholder comparison table: Transformer-only vs Hybrid approach.
    
    Args:
        output_dir: Directory to save table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison data
    data = {
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            'True Positives',
            'True Negatives',
            'False Positives',
            'False Negatives'
        ],
        'Transformer-Only': [
            0.872,
            0.756,
            0.689,
            0.721,
            67,
            850,
            45,
            38
        ],
        'Hybrid Approach': [
            0.923,
            0.841,
            0.812,
            0.826,
            81,
            882,
            23,
            14
        ],
        'Improvement': [
            '+5.1%',
            '+8.5%',
            '+12.3%',
            '+10.5%',
            '+14',
            '+32',
            '-22',
            '-24'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, len(df) + 1):
        improvement = df.iloc[i-1]['Improvement']
        if improvement.startswith('+'):
            table[(i, 3)].set_facecolor('#C8E6C9')  # Light green
        elif improvement.startswith('-'):
            table[(i, 3)].set_facecolor('#FFCDD2')  # Light red
        table[(i, 3)].set_text_props(weight='bold')
    
    # Style metric names
    for i in range(1, len(df) + 1):
        table[(i, 0)].set_facecolor('#E3F2FD')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Transformer-Only vs Hybrid Approach Comparison',
              fontsize=16, fontweight='bold', pad=20)
    
    # Save as image
    output_path = os.path.join(output_dir, "comparison_table.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison table saved to {output_path}")
    plt.close()
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Comparison table (CSV) saved to {csv_path}")


def generate_sample_responses_table(output_dir="results"):
    """
    Generate placeholder table with sample hallucinated vs correct responses.
    
    Args:
        output_dir: Directory to save table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample responses
    samples = [
        {
            'Type': 'Correct',
            'Response': 'Barack Obama was the 44th President of the United States, serving from 2009 to 2017.',
            'Transformer Prob': 0.15,
            'Factual Score': 0.95,
            'Agentic Score': 0.92,
            'Final Prediction': 'Correct',
            'Confidence': 0.94
        },
        {
            'Type': 'Correct',
            'Response': 'Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.',
            'Transformer Prob': 0.08,
            'Factual Score': 0.98,
            'Agentic Score': 0.97,
            'Final Prediction': 'Correct',
            'Confidence': 0.96
        },
        {
            'Type': 'Correct',
            'Response': 'Albert Einstein developed the theory of relativity, which revolutionized modern physics.',
            'Transformer Prob': 0.22,
            'Factual Score': 0.91,
            'Agentic Score': 0.89,
            'Final Prediction': 'Correct',
            'Confidence': 0.87
        },
        {
            'Type': 'Hallucination',
            'Response': 'Dr. Quantum invented the time machine in 2025 at the Institute of Impossible Science.',
            'Transformer Prob': 0.87,
            'Factual Score': 0.18,
            'Agentic Score': 0.15,
            'Final Prediction': 'Hallucination',
            'Confidence': 0.91
        },
        {
            'Type': 'Hallucination',
            'Response': 'The moon is made of cheese according to NASA scientists who published this finding in 2024.',
            'Transformer Prob': 0.92,
            'Factual Score': 0.12,
            'Agentic Score': 0.08,
            'Final Prediction': 'Hallucination',
            'Confidence': 0.95
        },
        {
            'Type': 'Hallucination',
            'Response': 'Harry Potter discovered quantum physics in 1997, leading to a Nobel Prize in Physics.',
            'Transformer Prob': 0.78,
            'Factual Score': 0.25,
            'Agentic Score': 0.22,
            'Final Prediction': 'Hallucination',
            'Confidence': 0.83
        },
        {
            'Type': 'Hallucination',
            'Response': 'Microsoft was founded in 1975 by Bill Gates and Steve Jobs in a garage in Seattle.',
            'Transformer Prob': 0.65,
            'Factual Score': 0.45,
            'Agentic Score': 0.38,
            'Final Prediction': 'Hallucination',
            'Confidence': 0.72
        },
        {
            'Type': 'Correct',
            'Response': 'Python is a high-level programming language known for its simplicity and readability.',
            'Transformer Prob': 0.12,
            'Factual Score': 0.96,
            'Agentic Score': 0.94,
            'Final Prediction': 'Correct',
            'Confidence': 0.93
        }
    ]
    
    df = pd.DataFrame(samples)
    
    # Create figure with two subplots (one for each type)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Correct responses
    correct_df = df[df['Type'] == 'Correct'].copy()
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(
        cellText=correct_df.values,
        colLabels=correct_df.columns,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2.5)
    
    # Style correct table
    for i in range(len(correct_df.columns)):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(correct_df) + 1):
        table1[(i, 0)].set_facecolor('#C8E6C9')
        table1[(i, 0)].set_text_props(weight='bold')
        # Highlight final prediction
        if correct_df.iloc[i-1]['Final Prediction'] == 'Correct':
            table1[(i, 5)].set_facecolor('#A5D6A7')
    
    ax1.set_title('Sample Correct Responses', fontsize=14, fontweight='bold', pad=10)
    
    # Hallucination responses
    hallucination_df = df[df['Type'] == 'Hallucination'].copy()
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(
        cellText=hallucination_df.values,
        colLabels=hallucination_df.columns,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2.5)
    
    # Style hallucination table
    for i in range(len(hallucination_df.columns)):
        table2[(0, i)].set_facecolor('#F44336')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(hallucination_df) + 1):
        table2[(i, 0)].set_facecolor('#FFCDD2')
        table2[(i, 0)].set_text_props(weight='bold')
        # Highlight final prediction
        if hallucination_df.iloc[i-1]['Final Prediction'] == 'Hallucination':
            table2[(i, 5)].set_facecolor('#EF9A9A')
    
    ax2.set_title('Sample Hallucinated Responses', fontsize=14, fontweight='bold', pad=10)
    
    plt.suptitle('Sample Responses: Correct vs Hallucinated',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save plot
    output_path = os.path.join(output_dir, "sample_responses_table.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample responses table saved to {output_path}")
    plt.close()
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, "sample_responses_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Sample responses table (CSV) saved to {csv_path}")


def main():
    """Generate all placeholder visualizations."""
    print("=" * 70)
    print("Generating Placeholder Visualizations")
    print("=" * 70)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Generating training/validation curves...")
    generate_training_curves(output_dir)
    
    print("\n2. Generating confusion matrix heatmap...")
    generate_confusion_matrix(output_dir)
    
    print("\n3. Generating comparison table...")
    generate_comparison_table(output_dir)
    
    print("\n4. Generating sample responses table...")
    generate_sample_responses_table(output_dir)
    
    print("\n" + "=" * 70)
    print("All placeholder visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print("\nNote: Replace sample data with real results from your experiments.")


if __name__ == "__main__":
    main()

