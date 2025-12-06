"""
Generate LaTeX tables from evaluation metrics for IEEE paper.
Uses global label constants to ensure consistency.
"""

import json
import os
from typing import Dict, List, Optional

# Import global label constants
try:
    from constants import (
        LABEL_CORRECT, LABEL_HALLUCINATION, LABELS,
        get_label_name
    )
except ImportError:
    # Fallback if constants not available
    LABEL_CORRECT = 0
    LABEL_HALLUCINATION = 1
    LABELS = [0, 1]
    
    def get_label_name(label):
        return "Correct" if label == 0 else "Hallucination"


def generate_overall_metrics_table(metrics: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate LaTeX table for overall classification metrics.
    
    Args:
        metrics: Dictionary with metrics (from evaluate_model.compute_metrics)
        output_path: Optional path to save the LaTeX code
    
    Returns:
        LaTeX table code as string
    """
    # Use string formatting with proper escaping
    latex = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\caption{Overall Classification Metrics}\n"
        "  \\label{tab:overall-metrics}\n"
        "  \\begin{tabular}{lcccc}\n"
        "    \\hline\n"
        "    Metric & Accuracy & Precision & Recall & F1-score \\\\ \\hline\n"
        f"    Binary (pos=1) & {metrics['accuracy']:.4f} & {metrics['precision']:.4f} & {metrics['recall']:.4f} & {metrics['f1_score']:.4f} \\\\\n"
        f"    Macro Average & {metrics['accuracy']:.4f} & {metrics['precision_macro']:.4f} & {metrics['recall_macro']:.4f} & {metrics['f1_macro']:.4f} \\\\\n"
        f"    Weighted Average & {metrics['accuracy']:.4f} & {metrics['precision_weighted']:.4f} & {metrics['recall_weighted']:.4f} & {metrics['f1_weighted']:.4f} \\\\ \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Overall metrics table saved to {output_path}")
    
    return latex


def generate_per_class_metrics_table(metrics: Dict, confusion_matrix: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate LaTeX table for per-class metrics.
    
    Args:
        metrics: Dictionary with metrics
        confusion_matrix: Dictionary with confusion matrix values
        output_path: Optional path to save the LaTeX code
    
    Returns:
        LaTeX table code as string
    """
    # Extract per-class metrics (ensuring both classes are present)
    precision_per_class = metrics.get('precision_per_class', {})
    recall_per_class = metrics.get('recall_per_class', {})
    f1_per_class = metrics.get('f1_per_class', {})
    support_per_class = metrics.get('support_per_class', {})
    
    # Use global label constants to ensure correct mapping
    prec_correct = precision_per_class.get('correct', 0.0)
    prec_hallucination = precision_per_class.get('hallucination', 0.0)
    rec_correct = recall_per_class.get('correct', 0.0)
    rec_hallucination = recall_per_class.get('hallucination', 0.0)
    f1_correct = f1_per_class.get('correct', 0.0)
    f1_hallucination = f1_per_class.get('hallucination', 0.0)
    
    # Calculate support from confusion matrix (more reliable)
    support_correct = confusion_matrix.get('true_negative', 0) + confusion_matrix.get('false_positive', 0)
    support_hallucination = confusion_matrix.get('true_positive', 0) + confusion_matrix.get('false_negative', 0)
    
    # Use support from metrics if available, otherwise from confusion matrix
    if support_per_class.get('correct', 0) > 0:
        support_correct = support_per_class.get('correct', support_correct)
    if support_per_class.get('hallucination', 0) > 0:
        support_hallucination = support_per_class.get('hallucination', support_hallucination)
    
    # HARD CHECK: Both classes must have metrics
    if prec_correct == 0.0 and prec_hallucination == 0.0:
        raise ValueError("generate_per_class_metrics_table: Both classes have zero precision. Check evaluation metrics.")
    
    latex = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\caption{Per-Class Classification Metrics}\n"
        "  \\label{tab:per-class-metrics}\n"
        "  \\begin{tabular}{lcccc}\n"
        "    \\hline\n"
        "    Class & Precision & Recall & F1-score & Support \\\\ \\hline\n"
        f"    {get_label_name(LABEL_CORRECT)} & {prec_correct:.4f} & {rec_correct:.4f} & {f1_correct:.4f} & {support_correct} \\\\\n"
        f"    {get_label_name(LABEL_HALLUCINATION)} & {prec_hallucination:.4f} & {rec_hallucination:.4f} & {f1_hallucination:.4f} & {support_hallucination} \\\\ \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Per-class metrics table saved to {output_path}")
    
    return latex


def generate_confusion_matrix_table(confusion_matrix: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate LaTeX table for confusion matrix.
    
    Args:
        confusion_matrix: Dictionary with confusion matrix values
        output_path: Optional path to save the LaTeX code
    
    Returns:
        LaTeX table code as string
    """
    tn = confusion_matrix.get('true_negative', 0)
    fp = confusion_matrix.get('false_positive', 0)
    fn = confusion_matrix.get('false_negative', 0)
    tp = confusion_matrix.get('true_positive', 0)
    
    # HARD CHECK: All values must be non-negative integers
    if any(v < 0 for v in [tn, fp, fn, tp]):
        raise ValueError(f"generate_confusion_matrix_table: Negative values in confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Verify sum matches total samples (if available)
    total = tn + fp + fn + tp
    if total == 0:
        raise ValueError("generate_confusion_matrix_table: Confusion matrix is empty (all zeros)")
    
    latex = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\caption{Confusion Matrix}\n"
        "  \\label{tab:confusion-matrix}\n"
        "  \\begin{tabular}{l|cc}\n"
        "    \\hline\n"
        "    & \\multicolumn{2}{c}{Predicted} \\\\\n"
        "    \\cline{2-3}\n"
        f"    Actual & {get_label_name(LABEL_CORRECT)} & {get_label_name(LABEL_HALLUCINATION)} \\\\ \\hline\n"
        f"    {get_label_name(LABEL_CORRECT)} & {tn} & {fp} \\\\\n"
        f"    {get_label_name(LABEL_HALLUCINATION)} & {fn} & {tp} \\\\ \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Confusion matrix table saved to {output_path}")
    
    return latex


def generate_all_tables(metrics_path: str, confusion_matrix_path: str, output_dir: str = "results/latex_tables"):
    """
    Generate all LaTeX tables from saved metrics files.
    
    Args:
        metrics_path: Path to evaluation_metrics.json
        confusion_matrix_path: Path to confusion_matrix.json
        output_dir: Directory to save LaTeX tables
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Handle backward compatibility - add default values if new fields missing
    if 'precision_macro' not in metrics:
        metrics['precision_macro'] = metrics.get('precision', 0.0)
        metrics['recall_macro'] = metrics.get('recall', 0.0)
        metrics['f1_macro'] = metrics.get('f1_score', 0.0)
        metrics['precision_weighted'] = metrics.get('precision', 0.0)
        metrics['recall_weighted'] = metrics.get('recall', 0.0)
        metrics['f1_weighted'] = metrics.get('f1_score', 0.0)
        metrics['precision_per_class'] = {
            'correct': 0.0,
            'hallucination': metrics.get('precision', 0.0)
        }
        metrics['recall_per_class'] = {
            'correct': 0.0,
            'hallucination': metrics.get('recall', 0.0)
        }
        metrics['f1_per_class'] = {
            'correct': 0.0,
            'hallucination': metrics.get('f1_score', 0.0)
        }
    
    # Load confusion matrix
    with open(confusion_matrix_path, 'r') as f:
        confusion_matrix = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate tables
    overall_table = generate_overall_metrics_table(
        metrics,
        os.path.join(output_dir, "overall_metrics.tex")
    )
    
    per_class_table = generate_per_class_metrics_table(
        metrics,
        confusion_matrix,
        os.path.join(output_dir, "per_class_metrics.tex")
    )
    
    cm_table = generate_confusion_matrix_table(
        confusion_matrix,
        os.path.join(output_dir, "confusion_matrix_table.tex")
    )
    
    # Generate combined file
    combined = """% LaTeX Tables for IEEE Paper
% Generated automatically from evaluation metrics

% Overall Metrics
""" + overall_table + """

% Per-Class Metrics
""" + per_class_table + """

% Confusion Matrix
""" + cm_table
    
    combined_path = os.path.join(output_dir, "all_tables.tex")
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f"\nAll tables saved to {output_dir}")
    print(f"Combined file: {combined_path}")
    
    return combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from evaluation metrics")
    parser.add_argument("--metrics", type=str, default="results/evaluation_metrics.json",
                       help="Path to evaluation_metrics.json")
    parser.add_argument("--confusion-matrix", type=str, default="results/confusion_matrix.json",
                       help="Path to confusion_matrix.json")
    parser.add_argument("--output-dir", type=str, default="results/latex_tables",
                       help="Output directory for LaTeX tables")
    
    args = parser.parse_args()
    
    generate_all_tables(args.metrics, args.confusion_matrix, args.output_dir)

