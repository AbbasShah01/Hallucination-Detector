# Master Pipeline Script

The `master_pipeline.py` script orchestrates the complete hybrid hallucination detection workflow.

## Overview

This script automates the entire pipeline:
1. **Load Preprocessed Data** - Loads tokenized data and tokenizer
2. **Train Transformer Model** - Fine-tunes DistilBERT for binary classification
3. **Initialize Verifiers** - Sets up entity and agentic verification components
4. **Make Predictions** - Generates predictions using hybrid fusion
5. **Evaluate** - Computes metrics and generates visualizations
6. **Generate Sample Outputs** - Creates sample predictions for review

## Usage

### Basic Usage

```bash
python src/master_pipeline.py
```

### With Configuration File

```bash
python src/master_pipeline.py --config config.json --output-dir results
```

### Command Line Options

- `--config`: Path to configuration JSON file (optional)
- `--output-dir`: Directory for output files (default: "results")
- `--log-file`: Path to log file (default: "results/pipeline.log")

## Configuration

Create a `config.json` file to customize the pipeline:

```json
{
  "data": {
    "preprocessed_path": "data/preprocessed/tokenized_data.json",
    "tokenizer_path": "data/preprocessed/tokenizer"
  },
  "training": {
    "model_name": "distilbert-base-uncased",
    "batch_size": 16,
    "num_epochs": 3,
    "learning_rate": 2e-5
  },
  "verification": {
    "use_entity_verification": true,
    "use_wikipedia": false,
    "use_agentic_verification": false
  },
  "fusion": {
    "alpha": 0.7,
    "beta": 0.2,
    "gamma": 0.1,
    "threshold": 0.5
  }
}
```

## Output Files

All results are saved to the output directory:

- `trained_model/` - Saved model and tokenizer
- `training_history.json` - Training metrics
- `training_loss_accuracy.png` - Training curves
- `validation_metrics.png` - Validation metrics
- `predictions.json` - All predictions
- `evaluation_metrics.json` - Evaluation metrics
- `confusion_matrix.png` - Confusion matrix
- `roc_curve.png` - ROC curve
- `sample_outputs.json` - Sample predictions
- `final_sample_predictions.json` - Final sample outputs
- `pipeline.log` - Execution log

## Logging

The script includes comprehensive logging:
- All steps are logged to console and file
- Progress updates for long-running operations
- Error messages with context
- Execution time tracking

## Prerequisites

1. Run preprocessing first:
   ```bash
   python src/preprocess_halueval.py
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Example Workflow

```bash
# Step 1: Preprocess data
python src/preprocess_halueval.py

# Step 2: Run master pipeline
python src/master_pipeline.py --config config.json

# Step 3: Check results
ls results/
```

## Notes

- The script handles missing components gracefully (warnings instead of failures)
- Entity verification can be disabled if spaCy is not available
- Agentic verification is optional and requires API keys if using API method
- All intermediate results are saved for debugging

