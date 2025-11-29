"""
Sentence-Level (Span-Level) Evaluation Module

Evaluates sentence-level hallucination detection performance.
Computes precision, recall, F1-score, confusion matrix, and summary reports.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will not be computed.")


@dataclass
class SpanEvaluationMetrics:
    """Evaluation metrics for sentence-level detection."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_sentences: int
    hallucinated_sentences: int
    factual_sentences: int


class SpanLevelEvaluator:
    """
    Evaluates sentence-level hallucination detection performance.
    
    Computes metrics at the sentence level, enabling fine-grained
    analysis of detection accuracy.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> SpanEvaluationMetrics:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of prediction dicts with 'label' and 'sentence'
            ground_truth: List of ground truth dicts with 'label' and 'sentence'
        
        Returns:
            SpanEvaluationMetrics object
        """
        # Align predictions and ground truth by sentence
        aligned_pairs = self._align_predictions_ground_truth(predictions, ground_truth)
        
        if not aligned_pairs:
            raise ValueError("No aligned predictions and ground truth found.")
        
        # Extract labels
        y_true = [1 if pair['ground_truth']['label'] == 'hallucinated' else 0 
                  for pair in aligned_pairs]
        y_pred = [1 if pair['prediction']['label'] == 'hallucinated' else 0 
                  for pair in aligned_pairs]
        
        # Compute metrics
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            # Manual computation
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute confusion matrix components
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        return SpanEvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_sentences=len(aligned_pairs),
            hallucinated_sentences=sum(y_true),
            factual_sentences=len(y_true) - sum(y_true)
        )
    
    def _align_predictions_ground_truth(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> List[Dict]:
        """
        Align predictions with ground truth by sentence text.
        
        Args:
            predictions: List of prediction dicts
            ground_truth: List of ground truth dicts
        
        Returns:
            List of aligned pairs
        """
        # Create lookup by sentence text
        pred_lookup = {p['sentence']: p for p in predictions}
        gt_lookup = {g['sentence']: g for g in ground_truth}
        
        # Find common sentences
        common_sentences = set(pred_lookup.keys()) & set(gt_lookup.keys())
        
        aligned = []
        for sentence in common_sentences:
            aligned.append({
                'sentence': sentence,
                'prediction': pred_lookup[sentence],
                'ground_truth': gt_lookup[sentence]
            })
        
        return aligned
    
    def compute_confusion_matrix(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: List of prediction dicts
            ground_truth: List of ground truth dicts
        
        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        aligned_pairs = self._align_predictions_ground_truth(predictions, ground_truth)
        
        y_true = [1 if pair['ground_truth']['label'] == 'hallucinated' else 0 
                  for pair in aligned_pairs]
        y_pred = [1 if pair['prediction']['label'] == 'hallucinated' else 0 
                  for pair in aligned_pairs]
        
        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        else:
            # Manual computation
            tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            cm = np.array([[tn, fp], [fn, tp]])
        
        return cm
    
    def generate_classification_report(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            predictions: List of prediction dicts
            ground_truth: List of ground truth dicts
        
        Returns:
            Classification report string
        """
        aligned_pairs = self._align_predictions_ground_truth(predictions, ground_truth)
        
        y_true = [1 if pair['ground_truth']['label'] == 'hallucinated' else 0 
                  for pair in aligned_pairs]
        y_pred = [1 if pair['prediction']['label'] == 'hallucinated' else 0 
                  for pair in aligned_pairs]
        
        if SKLEARN_AVAILABLE:
            target_names = ['factual', 'hallucinated']
            report = classification_report(
                y_true, y_pred,
                target_names=target_names,
                zero_division=0
            )
        else:
            # Manual report
            metrics = self.evaluate(predictions, ground_truth)
            report = f"""
Classification Report (Sentence-Level)
=====================================
              precision    recall  f1-score   support

     factual        {metrics.precision:.3f}       {metrics.recall:.3f}      {metrics.f1_score:.3f}        {metrics.factual_sentences}
hallucinated        {metrics.precision:.3f}       {metrics.recall:.3f}      {metrics.f1_score:.3f}        {metrics.hallucinated_sentences}

    accuracy                            {metrics.accuracy:.3f}        {metrics.total_sentences}
"""
        
        return report
    
    def generate_summary_report(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive summary report.
        
        Args:
            predictions: List of prediction dicts
            ground_truth: List of ground truth dicts
            output_path: Optional path to save report
        
        Returns:
            Summary dict
        """
        metrics = self.evaluate(predictions, ground_truth)
        cm = self.compute_confusion_matrix(predictions, ground_truth)
        classification_report_str = self.generate_classification_report(predictions, ground_truth)
        
        # Error analysis
        aligned_pairs = self._align_predictions_ground_truth(predictions, ground_truth)
        false_positives = [
            pair for pair in aligned_pairs
            if pair['ground_truth']['label'] == 'factual' and 
               pair['prediction']['label'] == 'hallucinated'
        ]
        false_negatives = [
            pair for pair in aligned_pairs
            if pair['ground_truth']['label'] == 'hallucinated' and 
               pair['prediction']['label'] == 'factual'
        ]
        
        summary = {
            "metrics": {
                "accuracy": round(metrics.accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1_score": round(metrics.f1_score, 4)
            },
            "confusion_matrix": {
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1])
            },
            "statistics": {
                "total_sentences": metrics.total_sentences,
                "hallucinated_sentences": metrics.hallucinated_sentences,
                "factual_sentences": metrics.factual_sentences,
                "hallucination_rate": round(
                    metrics.hallucinated_sentences / metrics.total_sentences 
                    if metrics.total_sentences > 0 else 0.0, 4
                )
            },
            "error_analysis": {
                "false_positives_count": len(false_positives),
                "false_negatives_count": len(false_negatives),
                "false_positive_examples": [
                    {
                        "sentence": pair['sentence'],
                        "predicted_score": pair['prediction'].get('final_hallucination_score', 0.0)
                    }
                    for pair in false_positives[:10]  # Top 10
                ],
                "false_negative_examples": [
                    {
                        "sentence": pair['sentence'],
                        "predicted_score": pair['prediction'].get('final_hallucination_score', 1.0)
                    }
                    for pair in false_negatives[:10]  # Top 10
                ]
            },
            "classification_report": classification_report_str
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def evaluate_from_files(
        self,
        predictions_path: str,
        ground_truth_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate from JSON files.
        
        Args:
            predictions_path: Path to predictions JSON file
            ground_truth_path: Path to ground truth JSON file
            output_path: Optional path to save summary report
        
        Returns:
            Summary dict
        """
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        return self.generate_summary_report(predictions, ground_truth, output_path)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Span-Level Evaluation Test")
    print("=" * 70)
    
    evaluator = SpanLevelEvaluator()
    
    # Example predictions
    predictions = [
        {
            "sentence": "Barack Obama was the 44th President.",
            "label": "factual",
            "final_hallucination_score": 0.2
        },
        {
            "sentence": "The moon is made of cheese.",
            "label": "hallucinated",
            "final_hallucination_score": 0.9
        }
    ]
    
    # Example ground truth
    ground_truth = [
        {
            "sentence": "Barack Obama was the 44th President.",
            "label": "factual"
        },
        {
            "sentence": "The moon is made of cheese.",
            "label": "hallucinated"
        }
    ]
    
    # Evaluate
    metrics = evaluator.evaluate(predictions, ground_truth)
    
    print("\nMetrics:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1-Score: {metrics.f1_score:.3f}")
    
    print("\nConfusion Matrix:")
    cm = evaluator.compute_confusion_matrix(predictions, ground_truth)
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    print("\nClassification Report:")
    print(evaluator.generate_classification_report(predictions, ground_truth))
    
    print("\n" + "=" * 70)

