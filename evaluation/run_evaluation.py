"""
Main script to run complete evaluation pipeline.
Example usage and integration.
"""

import argparse
import numpy as np
import json
from pathlib import Path

from evaluation_pipeline import EvaluationPipeline
from baseline_comparison import BaselineComparator


def load_test_data(data_path: str) -> dict:
    """Load test data from JSON or CSV."""
    path = Path(data_path)
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        # Assume format with labels and responses
        return {
            'labels': np.array(data.get('labels', [])),
            'responses': data.get('responses', [])
        }
    else:
        import pandas as pd
        df = pd.read_csv(path)
        return {
            'labels': df['label'].values,
            'responses': df['response'].tolist()
        }


def example_model_predictor(components: dict, test_data: dict):
    """
    Example model predictor function for ablation study.
    Replace with your actual model.
    """
    # Placeholder: random predictions based on components
    n = len(test_data['labels'])
    
    if components.get('transformer', True):
        base_prob = np.random.uniform(0.2, 0.8, n)
    else:
        base_prob = np.random.uniform(0.4, 0.6, n)
    
    if components.get('entity_verification', True):
        base_prob += np.random.uniform(-0.1, 0.1, n)
    
    if components.get('agentic_verification', True):
        base_prob += np.random.uniform(-0.05, 0.05, n)
    
    base_prob = np.clip(base_prob, 0, 1)
    predictions = (base_prob >= 0.5).astype(int)
    
    return predictions, base_prob


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions file")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                       help="Output directory")
    parser.add_argument("--model_name", type=str, default="model",
                       help="Model name")
    parser.add_argument("--run_baselines", action="store_true",
                       help="Run baseline comparison")
    parser.add_argument("--run_ablation", action="store_true",
                       help="Run ablation study")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading test data...")
    test_data = load_test_data(args.test_data)
    
    # Load predictions
    print("Loading predictions...")
    with open(args.predictions, 'r') as f:
        pred_data = json.load(f)
    
    y_pred = np.array(pred_data['predictions'])
    y_prob = np.array(pred_data['probabilities'])
    y_true = test_data['labels']
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(output_dir=args.output_dir)
    
    # Run evaluation
    results = pipeline.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        responses=test_data['responses'],
        model_name=args.model_name
    )
    
    # Baseline comparison
    if args.run_baselines:
        comparison_df = pipeline.compare_baselines(test_data)
        
        # Add to models_data for visualization
        models_data = {
            args.model_name: {
                'predictions': y_pred,
                'probabilities': y_prob,
                'metrics': results['standard_metrics']
            }
        }
        
        # Add baseline results
        for baseline_result in pipeline.baseline_comparator.results:
            models_data[baseline_result.model_name] = {
                'predictions': baseline_result.predictions,
                'probabilities': baseline_result.probabilities,
                'metrics': baseline_result.metrics
            }
        
        # Generate visualizations
        pipeline.generate_all_visualizations(y_true, models_data)
    
    # Ablation study
    if args.run_ablation:
        base_config = {
            'transformer': True,
            'entity_verification': True,
            'agentic_verification': True
        }
        
        ablation_results = pipeline.run_ablation_study(
            base_config, test_data, example_model_predictor
        )
        
        # Plot ablation results
        ablation_path = Path(args.output_dir) / "ablation_results.png"
        pipeline.visualizer.plot_ablation_results(
            ablation_results, str(ablation_path)
        )
    
    # Generate final report
    pipeline.generate_final_report()
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

