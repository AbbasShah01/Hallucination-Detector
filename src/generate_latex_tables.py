"""
Generate LaTeX tables from evaluation metrics for IEEE paper.
"""

import json
import os
from typing import Dict, List, Optional


def generate_overall_metrics_table(metrics: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate LaTeX table for overall classification metrics.
    
    Args:
        metrics: Dictionary with metrics (from evaluate_model.compute_metrics)
        output_path: Optional path to save the LaTeX code
    
    Returns:
        LaTeX table code as string
    """
    latex = """\\begin{table}[h]
  \\centering
  \\caption{Overall Classification Metrics}
  \\label{tab:overall-metrics}
  \\begin{tabular}{lcccc}
    \\hline
    Metric & Accuracy & Precision & Recall & F1-score \\\\ \\hline
    Binary (pos=1) & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\
    Macro Average & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\
    Weighted Average & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \\hline
  \\end{{tabular}}
\\end{{table}}""".format(
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['accuracy'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro'],
        metrics['accuracy'],
        metrics['precision_weighted'],
        metrics['recall_weighted'],
        metrics['f1_weighted']
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
    # Calculate support from confusion matrix
    support_correct = confusion_matrix.get('true_negative', 0) + confusion_matrix.get('false_positive', 0)
    support_hallucination = confusion_matrix.get('true_positive', 0) + confusion_matrix.get('false_negative', 0)
    
    latex = """\\begin{table}[h]
  \\centering
  \\caption{Per-Class Classification Metrics}
  \\label{tab:per-class-metrics}
  \\begin{tabular}{lcccc}
    \\hline
    Class & Precision & Recall & F1-score & Support \\\\ \\hline
    Correct & {:.4f} & {:.4f} & {:.4f} & {} \\\\
    Hallucination & {:.4f} & {:.4f} & {:.4f} & {} \\\\ \\hline
  \\end{{tabular}}
\\end{{table}}""".format(
        metrics['precision_per_class']['correct'],
        metrics['recall_per_class']['correct'],
        metrics['f1_per_class']['correct'],
        support_correct,
        metrics['precision_per_class']['hallucination'],
        metrics['recall_per_class']['hallucination'],
        metrics['f1_per_class']['hallucination'],
        support_hallucination
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
    
    latex = """\\begin{table}[h]
  \\centering
  \\caption{Confusion Matrix}
  \\label{tab:confusion-matrix}
  \\begin{tabular}{l|cc}
    \\hline
    & \\multicolumn{{2}}{{c}}{{Predicted}} \\\\
    \\cline{{2-3}}
    Actual & Correct & Hallucination \\\\ \\hline
    Correct & {} & {} \\\\
    Hallucination & {} & {} \\\\ \\hline
  \\end{{tabular}}
\\end{{table}}""".format(tn, fp, fn, tp)
    
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

