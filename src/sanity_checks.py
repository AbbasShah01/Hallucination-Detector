"""
Comprehensive sanity checks for evaluation pipeline.
Called before plotting, saving results, and exporting LaTeX tables.
"""

import numpy as np
from typing import Tuple, Dict, Optional

try:
    from constants import (
        LABEL_CORRECT, LABEL_HALLUCINATION, LABELS, POS_LABEL,
        MIN_TEST_SIZE, MIN_SAMPLES_PER_CLASS, DEMO_MODE,
        validate_labels, get_label_name
    )
except ImportError:
    LABEL_CORRECT = 0
    LABEL_HALLUCINATION = 1
    LABELS = [0, 1]
    POS_LABEL = 1
    MIN_TEST_SIZE = 30
    MIN_SAMPLES_PER_CLASS = 5
    DEMO_MODE = False
    
    def validate_labels(labels, context=""):
        labels = np.asarray(labels, dtype=int).ravel()
        unique_labels = set(np.unique(labels))
        if not unique_labels.issubset(set(LABELS)):
            raise ValueError(f"{context}: Invalid labels: {unique_labels}")
        if len(unique_labels) < 2:
            raise ValueError(f"{context}: Missing classes. Found: {unique_labels}, need: {set(LABELS)}")
    
    def get_label_name(label):
        return "Correct" if label == 0 else "Hallucination"


def run_sanity_checks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    context: str = "evaluation",
    demo_mode: Optional[bool] = None
) -> Dict[str, any]:
    """
    Run comprehensive sanity checks before computing metrics, plotting, or saving results.
    
    HARD CHECKS (raise ValueError if failed):
    - Dataset size >= MIN_TEST_SIZE (unless demo_mode)
    - Both classes present in y_true
    - Labels conform to global contract
    - y_true and y_pred have same length
    
    WARNINGS (print but continue):
    - Small dataset size
    - Class imbalance
    - Model predicts only one class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        context: Context string for error messages
        demo_mode: If True, allows smaller datasets. If None, uses global DEMO_MODE.
    
    Returns:
        Dictionary with check results and statistics
    
    Raises:
        ValueError: If hard requirements are not met
    """
    if demo_mode is None:
        demo_mode = DEMO_MODE
    
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    results = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    # CHECK 1: Same length
    if len(y_true) != len(y_pred):
        error_msg = f"{context}: y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}"
        results['errors'].append(error_msg)
        results['passed'] = False
        raise ValueError(error_msg)
    
    # CHECK 2: Dataset size
    n_samples = len(y_true)
    results['stats']['n_samples'] = n_samples
    
    if n_samples < MIN_TEST_SIZE:
        if demo_mode:
            # In demo mode, this is just a warning
            results['warnings'].append(
                f"Demo mode: Test set has only {n_samples} samples (recommended: >= {MIN_TEST_SIZE})"
            )
        else:
            # In production mode, this is an error
            error_msg = (
                f"{context}: Test set has only {n_samples} samples. "
                f"Minimum required: {MIN_TEST_SIZE} (unless demo_mode=True). "
                f"Results may not be statistically reliable."
            )
            results['errors'].append(error_msg)
            results['passed'] = False
            raise ValueError(error_msg)
    
    # CHECK 3: Label validation
    try:
        validate_labels(y_true, context=f"{context}: y_true")
    except ValueError as e:
        results['errors'].append(str(e))
        results['passed'] = False
        raise
    
    try:
        validate_labels(y_pred, context=f"{context}: y_pred")
    except ValueError as e:
        results['warnings'].append(f"y_pred validation: {str(e)}")
        # Don't raise for y_pred - model might predict only one class
    
    # CHECK 4: Both classes in y_true (HARD REQUIREMENT)
    unique_true = set(np.unique(y_true))
    if unique_true != set(LABELS):
        error_msg = (
            f"{context}: y_true must contain both classes {LABELS}. "
            f"Found: {unique_true}. Cannot compute meaningful binary classification metrics."
        )
        results['errors'].append(error_msg)
        results['passed'] = False
        raise ValueError(error_msg)
    
    # CHECK 5: Class distribution
    class_counts_true = {label: np.sum(y_true == label) for label in LABELS}
    class_counts_pred = {label: np.sum(y_pred == label) for label in LABELS}
    
    results['stats']['class_counts_true'] = class_counts_true
    results['stats']['class_counts_pred'] = class_counts_pred
    
    # Check for class imbalance
    min_count = min(class_counts_true.values())
    max_count = max(class_counts_true.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 3.0:
        results['warnings'].append(
            f"Class imbalance detected: ratio = {imbalance_ratio:.2f} "
            f"(Class {LABEL_CORRECT}: {class_counts_true[LABEL_CORRECT]}, "
            f"Class {LABEL_HALLUCINATION}: {class_counts_true[LABEL_HALLUCINATION]})"
        )
    
    # CHECK 6: Minimum samples per class
    for label in LABELS:
        if class_counts_true[label] < MIN_SAMPLES_PER_CLASS:
            if demo_mode:
                results['warnings'].append(
                    f"Demo mode: Class {label} ({get_label_name(label)}) has only "
                    f"{class_counts_true[label]} samples (recommended: >= {MIN_SAMPLES_PER_CLASS})"
                )
            else:
                error_msg = (
                    f"{context}: Class {label} ({get_label_name(label)}) has only "
                    f"{class_counts_true[label]} samples. Minimum required: {MIN_SAMPLES_PER_CLASS}."
                )
                results['errors'].append(error_msg)
                results['passed'] = False
                raise ValueError(error_msg)
    
    # CHECK 7: Model predicts only one class
    unique_pred = set(np.unique(y_pred))
    if len(unique_pred) < 2:
        results['warnings'].append(
            f"Model predicts only class {list(unique_pred)[0]} ({get_label_name(list(unique_pred)[0])}). "
            f"This may indicate a threshold issue or model bias."
        )
    
    # CHECK 8: Probability range (if provided)
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba).ravel()
        if len(y_pred_proba) != len(y_true):
            results['warnings'].append(
                f"y_pred_proba length ({len(y_pred_proba)}) doesn't match y_true ({len(y_true)})"
            )
        else:
            prob_min, prob_max = y_pred_proba.min(), y_pred_proba.max()
            if prob_min < 0 or prob_max > 1:
                results['warnings'].append(
                    f"y_pred_proba values outside [0, 1]: min={prob_min:.4f}, max={prob_max:.4f}"
                )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"SANITY CHECKS: {context.upper()}")
    print(f"{'='*70}")
    
    if results['passed']:
        print("[OK] All hard checks PASSED")
    else:
        print("[ERROR] HARD CHECKS FAILED - Cannot proceed")
        for error in results['errors']:
            print(f"   ERROR: {error}")
        return results
    
    if results['warnings']:
        print(f"\n[WARNING] WARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   {warning}")
    
    print(f"\nStatistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  True label distribution:")
    for label in LABELS:
        count = class_counts_true[label]
        pct = 100 * count / n_samples
        print(f"    Class {label} ({get_label_name(label)}): {count} ({pct:.1f}%)")
    print(f"  Predicted label distribution:")
    for label in LABELS:
        count = class_counts_pred.get(label, 0)
        pct = 100 * count / n_samples if n_samples > 0 else 0
        print(f"    Class {label} ({get_label_name(label)}): {count} ({pct:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return results

