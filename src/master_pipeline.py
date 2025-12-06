"""
Master Pipeline Script for Hybrid Hallucination Detection System
Orchestrates the complete workflow: preprocessing, training, verification, fusion, and evaluation.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Import project modules
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import utilities for reproducibility
try:
    from utils import set_random_seeds, ensure_dir
except ImportError:
    # Fallback if utils not available
    def set_random_seeds(seed=42):
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

# Import functions from train_model (which has the loaders)
try:
    from train_model import (
        load_preprocessed_data, load_tokenizer,
        split_data, create_data_loaders, initialize_model,
        train_model, save_model, plot_training_history
    )
except ImportError as e:
    print(f"Error: Could not import train_model functions: {e}")
    print("Make sure train_model.py is in the src/ directory")
    sys.exit(1)


try:
    from entity_verification import EntityVerifier
except ImportError as e:
    print(f"Warning: Could not import EntityVerifier: {e}")
    EntityVerifier = None

try:
    from hybrid_fusion import hybrid_predict, batch_predict
except ImportError as e:
    print(f"Warning: Could not import hybrid_fusion: {e}")

try:
    from agentic_verification import AgenticVerifier, integrate_with_hybrid_fusion
except ImportError as e:
    print(f"Warning: Could not import agentic_verification: {e}")
    AgenticVerifier = None
    integrate_with_hybrid_fusion = None

try:
    from evaluate_model import evaluate_model
except ImportError as e:
    print(f"Warning: Could not import evaluate_model: {e}")

# Import sentence-level detector
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.span_level_detector import SpanInferencePipeline
    SPAN_LEVEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import span-level detector: {e}")
    SPAN_LEVEL_AVAILABLE = False
    SpanInferencePipeline = None

# Import novel research modules
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.novel_metric.shds import SHDS
    from modules.fusion.dmsf import DMSF
    NOVEL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import novel modules: {e}")
    NOVEL_MODULES_AVAILABLE = False
    SHDS = None
    DMSF = None


class PipelineLogger:
    """Custom logger for pipeline execution."""
    
    def __init__(self, log_file: str = "results/pipeline.log"):
        """Initialize logger."""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('HallucinationDetectionPipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        print(f"[INFO] {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        print(f"[WARNING] {message}")
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        print(f"[ERROR] {message}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


class MasterPipeline:
    """
    Master pipeline orchestrator for hybrid hallucination detection.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "results",
        log_file: str = "results/pipeline.log",
        random_seed: int = 42
    ):
        """
        Initialize master pipeline.
        
        Args:
            config_path: Path to configuration JSON file
            output_dir: Directory for output files
            log_file: Path to log file
            random_seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        set_random_seeds(random_seed)
        
        self.output_dir = output_dir
        self.logger = PipelineLogger(log_file)
        
        # Create output directories
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "figs"))
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # Set DEMO_MODE from config BEFORE other modules import constants
        demo_mode = self.config.get("demo_mode", False)
        try:
            import constants
            constants.DEMO_MODE = demo_mode
            self.logger.info(f"Demo mode set to {demo_mode} from config")
        except ImportError:
            pass
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.entity_verifier = None
        self.agentic_verifier = None
        self.span_pipeline = None  # Sentence-level pipeline
        self.shds_calculator = None  # SHDS metric calculator
        self.dmsf_fusion = None  # DMSF fusion calculator
        
        # Initialize novel modules if available
        if NOVEL_MODULES_AVAILABLE:
            try:
                self.shds_calculator = SHDS()
                self.dmsf_fusion = DMSF()
                self.logger.info("Novel research modules (SHDS, DMSF) initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize novel modules: {e}")
        
        self.logger.info("Master pipeline initialized")
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "data": {
                "preprocessed_path": "data/preprocessed/tokenized_data.json",
                "tokenizer_path": "data/preprocessed/tokenizer",
                "test_data_path": None  # Will use validation split if None
            },
            "training": {
                "model_name": "distilbert-base-uncased",
                "batch_size": 16,
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1
            },
            "verification": {
                "use_entity_verification": True,
                "entity_extractor_method": "spacy",
                "use_wikipedia": False,  # Set to True for real verification
                "use_agentic_verification": False,  # Set to True to enable
                "agentic_method": "local",  # "local" or "api"
                "agentic_provider": None  # "openai" or "anthropic" if using API
            },
            "fusion": {
                "method": "classic_fusion",  # "classic_fusion" or "novel_dmsf"
                "alpha": 0.7,  # Weight for transformer
                "beta": 0.2,   # Weight for entity verification
                "gamma": 0.1,  # Weight for agentic verification
                "threshold": 0.5
            },
            "evaluation": {
                "num_samples": 10
            },
            "mode": "response_level"  # "response_level" or "sentence_level"
        }
    
    def step1_load_data(self):
        """Step 1: Load preprocessed data."""
        self.logger.info("=" * 70)
        self.logger.info("STEP 1: Loading Preprocessed Data")
        self.logger.info("=" * 70)
        
        try:
            data_path = self.config["data"]["preprocessed_path"]
            tokenizer_path = self.config["data"]["tokenizer_path"]
            
            # Load tokenized data
            self.logger.info(f"Loading tokenized data from {data_path}...")
            tokenized_data = load_preprocessed_data(data_path)
            self.logger.info(f"Loaded {len(tokenized_data)} samples")
            
            # Load tokenizer
            self.logger.info(f"Loading tokenizer from {tokenizer_path}...")
            self.tokenizer = load_tokenizer(tokenizer_path)
            self.logger.info("[OK] Tokenizer loaded successfully")
            
            return tokenized_data
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def step2_train_model(self, tokenized_data: List[Dict]):
        """Step 2: Train transformer model."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 2: Training Transformer Model")
        self.logger.info("=" * 70)
        
        try:
            train_config = self.config["training"]
            
            # Split data (with demo_mode from config if available)
            self.logger.info("Splitting data into train/val/test sets...")
            demo_mode = self.config.get("demo_mode", False)
            train_data, val_data, test_data = split_data(
                tokenized_data,
                train_ratio=train_config["train_ratio"],
                val_ratio=train_config["val_ratio"],
                test_ratio=train_config["test_ratio"],
                demo_mode=demo_mode
            )
            
            # Create data loaders
            self.logger.info("Creating data loaders...")
            train_loader, val_loader = create_data_loaders(
                train_data, val_data, self.tokenizer,
                batch_size=train_config["batch_size"]
            )
            
            # Initialize model
            self.logger.info(f"Initializing model: {train_config['model_name']}...")
            self.model = initialize_model(
                train_config["model_name"],
                num_labels=2
            )
            
            # Train model
            self.logger.info("Starting training...")
            history = train_model(
                self.model,
                train_loader,
                val_loader,
                num_epochs=train_config["num_epochs"],
                learning_rate=train_config["learning_rate"]
            )
            
            # Save model
            model_dir = os.path.join(self.output_dir, "trained_model")
            self.logger.info(f"Saving model to {model_dir}...")
            save_model(self.model, self.tokenizer, output_dir=model_dir)
            
            # Plot training history
            self.logger.info("Generating training plots...")
            plot_training_history(history, output_dir=self.output_dir)
            
            # Save training history
            history_path = os.path.join(self.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            self.logger.info(f"Training history saved to {history_path}")
            
            return test_data, history
        
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def step3_initialize_verifiers(self):
        """Step 3: Initialize verification components."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 3: Initializing Verification Components")
        self.logger.info("=" * 70)
        
        verif_config = self.config["verification"]
        
        # Entity verifier
        if verif_config["use_entity_verification"]:
            try:
                self.logger.info("Initializing entity verifier...")
                self.entity_verifier = EntityVerifier(
                    extractor_method=verif_config["entity_extractor_method"],
                    use_wikipedia=verif_config["use_wikipedia"]
                )
                self.logger.info("[OK] Entity verifier initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize entity verifier: {e}")
                self.entity_verifier = None
        
        # Agentic verifier (optional)
        if verif_config["use_agentic_verification"]:
            try:
                self.logger.info("Initializing agentic verifier...")
                self.agentic_verifier = AgenticVerifier(
                    method=verif_config["agentic_method"],
                    provider=verif_config.get("agentic_provider")
                )
                self.logger.info("[OK] Agentic verifier initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize agentic verifier: {e}")
                self.agentic_verifier = None
        else:
            self.logger.info("Agentic verification disabled")
    
    def step4_make_predictions(
        self,
        test_data: List[Dict],
        use_hybrid: bool = True
    ) -> Dict:
        """Step 4: Make predictions on test data."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 4: Making Predictions")
        self.logger.info("=" * 70)
        
        predictions = {
            "transformer_probs": [],
            "factual_scores": [],
            "agentic_scores": [],
            "fusion_probs": [],
            "final_predictions": [],
            "responses": [],
            "true_labels": []
        }
        
        fusion_config = self.config["fusion"]
        
        self.logger.info(f"Processing {len(test_data)} test samples...")
        
        for i, sample in enumerate(test_data):
            if i % 10 == 0:
                self.logger.info(f"Processing sample {i+1}/{len(test_data)}...")
            
            # Extract response text
            response = sample.get('response', sample.get('prompt', ''))
            if not response:
                continue
            
            true_label = sample.get('label', 0)
            predictions["responses"].append(response)
            predictions["true_labels"].append(true_label)
            
            # Transformer prediction (simulated - in real use, run model inference)
            # For now, we'll use a placeholder. In production, you'd run:
            # transformer_prob = self._predict_with_model(response)
            transformer_prob = np.random.uniform(0.1, 0.9)  # Placeholder
            predictions["transformer_probs"].append(transformer_prob)
            
            # Entity verification
            factual_score = 0.5  # Default neutral
            if self.entity_verifier:
                try:
                    result = self.entity_verifier.verify_response(response)
                    factual_score = result.correctness_score
                except Exception as e:
                    self.logger.debug(f"Entity verification error: {e}")
            predictions["factual_scores"].append(factual_score)
            
            # Agentic verification (optional)
            agentic_score = 0.5  # Default neutral
            if self.agentic_verifier:
                try:
                    result = self.agentic_verifier.verify(response)
                    agentic_score = result.verification_score
                except Exception as e:
                    self.logger.debug(f"Agentic verification error: {e}")
            predictions["agentic_scores"].append(agentic_score)
            
            # Fusion (classic or novel DMSF)
            fusion_method = fusion_config.get("method", "classic_fusion")
            
            if fusion_method == "novel_dmsf" and NOVEL_MODULES_AVAILABLE and self.dmsf_fusion:
                # Use novel DMSF fusion
                try:
                    # Compute SHDS if needed
                    shds_result = None
                    if self.shds_calculator:
                        shds_result = self.shds_calculator.compute(
                            span=response,
                            failed_entity_checks=0,  # Would need entity verification details
                            total_entities=0,
                            agentic_verification_score=agentic_score
                        )
                    
                    # DMSF fusion
                    dmsf_result = self.dmsf_fusion.fuse(
                        classifier_score=transformer_prob,
                        entity_score=factual_score,
                        agentic_score=agentic_score,
                        shds_score=shds_result.shds_score if shds_result else None,
                        span=response,
                        compute_shds=(shds_result is None)
                    )
                    fusion_prob = dmsf_result.final_score
                except Exception as e:
                    self.logger.warning(f"DMSF fusion failed, falling back to classic: {e}")
                    fusion_method = "classic_fusion"  # Fallback
            
            if fusion_method == "classic_fusion":
                # Classic hybrid fusion
                if use_hybrid and self.entity_verifier:
                    if self.agentic_verifier:
                        # Three-way fusion
                        fusion_prob = integrate_with_hybrid_fusion(
                            transformer_prob=transformer_prob,
                            factual_score=factual_score,
                            agentic_score=agentic_score,
                            alpha=fusion_config["alpha"],
                            beta=fusion_config["beta"],
                            gamma=fusion_config["gamma"]
                        )
                    else:
                        # Two-way fusion
                        result = hybrid_predict(
                            transformer_prob=transformer_prob,
                            factual_score=factual_score,
                            alpha=fusion_config["alpha"],
                            threshold=fusion_config["threshold"]
                        )
                        fusion_prob = result.fusion_prob
                else:
                    # Transformer only
                    fusion_prob = transformer_prob
            
            predictions["fusion_probs"].append(fusion_prob)
            predictions["final_predictions"].append(
                1 if fusion_prob >= fusion_config["threshold"] else 0
            )
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        self.logger.info(f"Predictions saved to {predictions_path}")
        
        return predictions
    
    def step5_evaluate(self, predictions: Dict):
        """Step 5: Evaluate predictions."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 5: Evaluating Predictions")
        self.logger.info("=" * 70)
        
        # Create test data file for evaluation
        test_data_path = os.path.join(self.output_dir, "test_data_for_eval.json")
        test_data = [
            {"response": resp, "label": label}
            for resp, label in zip(predictions["responses"], predictions["true_labels"])
        ]
        with open(test_data_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        # Create predictions file
        predictions_path = os.path.join(self.output_dir, "predictions_for_eval.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions["fusion_probs"], f, indent=2)
        
        # Run evaluation
        try:
            evaluate_model(
                test_data_path=test_data_path,
                predictions_path=predictions_path,
                output_dir=self.output_dir,
                threshold=self.config["fusion"]["threshold"],
                num_samples=self.config["evaluation"]["num_samples"]
            )
            # Update DEMO_MODE in constants before evaluation (in case it changed)
            demo_mode = self.config.get("demo_mode", False)
            try:
                import constants
                constants.DEMO_MODE = demo_mode
            except ImportError:
                pass
            
            self.logger.info("[OK] Evaluation completed")
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
    
    def detect_sentence_level(self, text: str) -> List[Dict]:
        """
        Detect hallucinations at sentence level.
        
        Args:
            text: Input text to analyze
        
        Returns:
            List of sentence-level detection results
        """
        if not SPAN_LEVEL_AVAILABLE:
            raise ValueError("Sentence-level detection not available. Install required dependencies.")
        
        # Initialize span pipeline if not already done
        if self.span_pipeline is None:
            model_path = None
            if self.model is not None:
                # Use trained model if available
                model_path = os.path.join(self.output_dir, "trained_model")
            
            self.span_pipeline = SpanInferencePipeline(
                model_path=model_path,
                splitter_method="nltk",
                extractor_method=self.config["verification"].get("entity_extractor_method", "transformers"),
                use_entity_verification=self.config["verification"].get("use_entity_verification", True),
                use_agent_verification=self.config["verification"].get("use_agentic_verification", False),
                fusion_alpha=self.config["fusion"]["alpha"],
                fusion_beta=self.config["fusion"]["beta"],
                fusion_gamma=self.config["fusion"]["gamma"],
                fusion_threshold=self.config["fusion"]["threshold"]
            )
        
        # Detect
        results = self.span_pipeline.detect(text, return_json=True)
        
        return results
    
    def step7_generate_latex_tables(self):
        """Step 7: Generate LaTeX tables for paper."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 7: Generating LaTeX Tables")
        self.logger.info("=" * 70)
        
        try:
            from generate_latex_tables import generate_all_tables
            
            metrics_path = os.path.join(self.output_dir, "evaluation_metrics.json")
            cm_path = os.path.join(self.output_dir, "confusion_matrix.json")
            latex_output_dir = os.path.join(self.output_dir, "latex_tables")
            
            if os.path.exists(metrics_path) and os.path.exists(cm_path):
                generate_all_tables(metrics_path, cm_path, latex_output_dir)
                self.logger.info("[OK] LaTeX tables generated")
            else:
                self.logger.warning("Metrics files not found, skipping LaTeX table generation")
        except ImportError:
            self.logger.warning("Could not import generate_latex_tables, skipping")
        except Exception as e:
            self.logger.warning(f"Error generating LaTeX tables: {e}")
    
    def step6_generate_sample_outputs(self, predictions: Dict):
        """Step 6: Generate sample predictions output."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 6: Generating Sample Outputs")
        self.logger.info("=" * 70)
        
        # Create sample outputs
        samples = []
        num_samples = self.config["evaluation"]["num_samples"]
        
        # Get diverse samples
        indices = np.random.choice(
            len(predictions["responses"]),
            size=min(num_samples, len(predictions["responses"])),
            replace=False
        )
        
        for idx in indices:
            sample = {
                "response": predictions["responses"][idx],
                "true_label": "Hallucination" if predictions["true_labels"][idx] == 1 else "Correct",
                "transformer_prob": predictions["transformer_probs"][idx],
                "factual_score": predictions["factual_scores"][idx],
                "agentic_score": predictions["agentic_scores"][idx] if self.agentic_verifier else None,
                "fusion_prob": predictions["fusion_probs"][idx],
                "predicted_label": "Hallucination" if predictions["final_predictions"][idx] == 1 else "Correct",
                "correct": predictions["true_labels"][idx] == predictions["final_predictions"][idx]
            }
            samples.append(sample)
        
        # Save samples
        samples_path = os.path.join(self.output_dir, "final_sample_predictions.json")
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Sample predictions saved to {samples_path}")
        
        # Print samples
        self.logger.info("\nSample Predictions:")
        self.logger.info("-" * 70)
        for i, sample in enumerate(samples, 1):
            self.logger.info(f"\nSample {i}:")
            self.logger.info(f"  Response: {sample['response'][:100]}...")
            self.logger.info(f"  True Label: {sample['true_label']}")
            self.logger.info(f"  Predicted: {sample['predicted_label']} (prob: {sample['fusion_prob']:.3f})")
            self.logger.info(f"  Correct: {'[OK]' if sample['correct'] else '[X]'}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        start_time = datetime.now()
        self.logger.info("\n" + "=" * 70)
        self.logger.info("HYBRID HALLUCINATION DETECTION PIPELINE")
        self.logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)
        
        try:
            # Step 1: Load data
            tokenized_data = self.step1_load_data()
            
            # Step 2: Train model
            test_data, training_history = self.step2_train_model(tokenized_data)
            
            # Step 3: Initialize verifiers
            self.step3_initialize_verifiers()
            
            # Step 4: Make predictions
            predictions = self.step4_make_predictions(test_data, use_hybrid=True)
            
            # Step 5: Evaluate
            self.step5_evaluate(predictions)
            
            # Step 6: Generate sample outputs
            self.step6_generate_sample_outputs(predictions)
            
            # Step 7: Generate LaTeX tables
            self.step7_generate_latex_tables()
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total duration: {duration:.2f} seconds")
            self.logger.info(f"Results saved to: {self.output_dir}")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master pipeline for hybrid hallucination detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="results/pipeline.log",
        help="Path to log file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["response_level", "sentence_level"],
        default="response_level",
        help="Detection mode: 'response_level' or 'sentence_level'"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to analyze (for sentence-level mode)"
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=["classic_fusion", "novel_dmsf"],
        default=None,
        help="Fusion method: 'classic_fusion' or 'novel_dmsf'"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = MasterPipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    
    # Update config with mode and fusion method
    pipeline.config["mode"] = args.mode
    if args.fusion_method:
        pipeline.config["fusion"]["method"] = args.fusion_method
    
    # Run based on mode
    if args.mode == "sentence_level":
        if args.text:
            # Single text analysis
            print(f"\nAnalyzing text in sentence-level mode...")
            results = pipeline.detect_sentence_level(args.text)
            
            # Print results
            print(f"\nDetected {len(results)} sentences:")
            for result in results:
                label_icon = "[X]" if result['label'] == 'hallucinated' else "[OK]"
                print(f"\n{label_icon} [{result['sentence_index']}] {result['sentence']}")
                print(f"   Final score: {result['final_hallucination_score']:.3f} ({result['label']})")
                print(f"   Confidence: {result['confidence']:.3f}")
            
            # Save results
            output_path = os.path.join(args.output_dir, "sentence_level_results.json")
            pipeline.span_pipeline.save_results(results, output_path)
            print(f"\nResults saved to: {output_path}")
            
            # Summary
            summary = pipeline.span_pipeline.get_summary(results)
            print(f"\nSummary:")
            print(f"  Total sentences: {summary['total_sentences']}")
            print(f"  Hallucinated: {summary['hallucinated_sentences']}")
            print(f"  Factual: {summary['factual_sentences']}")
            print(f"  Hallucination rate: {summary['hallucination_rate']:.2%}")
        else:
            print("Error: --text argument required for sentence-level mode")
            print("Usage: python src/master_pipeline.py --mode sentence_level --text 'Your text here'")
    else:
        # Response-level mode (default)
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

