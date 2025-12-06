"""
Global constants for the Hallucination Detection System.
Enforces consistent label mapping across all modules.
"""

# LABEL MAPPING (NON-NEGOTIABLE)
# This is the SINGLE SOURCE OF TRUTH for label encoding
LABEL_CORRECT = 0  # Non-hallucination / Factual response
LABEL_HALLUCINATION = 1  # Hallucination / Non-factual response

# All valid labels
LABELS = [LABEL_CORRECT, LABEL_HALLUCINATION]

# Label names for display
LABEL_NAMES = {
    LABEL_CORRECT: "Correct",
    LABEL_HALLUCINATION: "Hallucination"
}

# For sklearn metrics: positive class is hallucination
POS_LABEL = LABEL_HALLUCINATION  # 1

# Minimum dataset requirements
MIN_TEST_SIZE = 30  # Minimum test samples for reliable evaluation
MIN_SAMPLES_PER_CLASS = 5  # Minimum samples per class in test set

# Demo mode flag (allows smaller datasets for testing)
DEMO_MODE = False  # Set to True only for quick testing with small datasets


def validate_labels(labels, context: str = "") -> None:
    """
    Validate that labels conform to the global label contract.
    
    Args:
        labels: Array of labels to validate
        context: Context string for error messages
    
    Raises:
        ValueError: If labels are invalid
    """
    import numpy as np
    
    labels = np.asarray(labels, dtype=int).ravel()
    unique_labels = np.unique(labels)
    
    # Check all labels are valid
    invalid_labels = set(unique_labels) - set(LABELS)
    if invalid_labels:
        raise ValueError(
            f"{context}: Invalid labels found: {invalid_labels}. "
            f"Valid labels are: {LABELS} (0=Correct, 1=Hallucination)"
        )
    
    # Check both classes present (for binary classification)
    if len(unique_labels) < 2:
        missing = set(LABELS) - set(unique_labels)
        raise ValueError(
            f"{context}: Missing class(es): {missing}. "
            f"Both classes ({LABELS}) must be present for binary classification. "
            f"Found only: {unique_labels}"
        )


def get_label_name(label: int) -> str:
    """Get human-readable name for a label."""
    return LABEL_NAMES.get(label, f"Unknown({label})")

