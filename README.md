# Hybrid Hallucination Detection System for LLMs

A comprehensive Python-based system for detecting hallucinations in Large Language Model (LLM) outputs using a hybrid approach that combines transformer models, entity verification, and agentic verification.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a hybrid hallucination detection system that combines multiple detection methods:

1. **Transformer-based Classification**: Fine-tuned DistilBERT model for binary classification
2. **Entity Verification**: Named Entity Recognition (NER) with Wikipedia fact-checking
3. **Agentic Verification**: LLM-based cross-verification of responses
4. **Hybrid Fusion**: Weighted combination of all detection methods

The system processes LLM responses and outputs a hallucination probability score (0-1), enabling reliable detection of factual inaccuracies in generated text.

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Hybrid Hallucination Detection System              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Input: LLM Response Text          │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────────┐
│  Transformer     │                    │  Entity Verification │
│  Model           │                    │  (NER + Wikipedia)   │
│  (DistilBERT)    │                    └──────────────────────┘
│                  │                              │
│  Output:         │                              │
│  Hallucination   │                              │
│  Probability      │                              │
└──────────────────┘                              │
        │                                         │
        │                                         ▼
        │                              ┌──────────────────────┐
        │                              │  Agentic Verification│
        │                              │  (LLM Cross-Check)    │
        │                              └──────────────────────┘
        │                                         │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Hybrid Fusion      │
              │   (Weighted Sum)     │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Final Prediction     │
              │  (Hallucination/      │
              │   Correct + Score)    │
              └───────────────────────┘
```

### Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        Master Pipeline                           │
└──────────────────────────────────────────────────────────────────┘

Step 1: Data Loading
    │
    ├─ Load preprocessed tokenized data
    ├─ Load tokenizer
    └─ Split into train/val/test sets
    │
    ▼
Step 2: Model Training
    │
    ├─ Initialize DistilBERT model
    ├─ Train for N epochs
    ├─ Validate on validation set
    └─ Save trained model
    │
    ▼
Step 3: Verification Setup
    │
    ├─ Initialize Entity Verifier (spaCy/Transformers)
    └─ Initialize Agentic Verifier (Optional)
    │
    ▼
Step 4: Prediction Generation
    │
    ├─ Get transformer predictions
    ├─ Extract entities and verify
    ├─ Get agentic verification (optional)
    └─ Apply hybrid fusion
    │
    ▼
Step 5: Evaluation
    │
    ├─ Compute metrics (Accuracy, Precision, Recall, F1)
    ├─ Generate confusion matrix
    ├─ Generate ROC curve
    └─ Create visualizations
    │
    ▼
Step 6: Sample Outputs
    │
    ├─ Extract sample predictions
    ├─ Categorize (TP, TN, FP, FN)
    └─ Save results
```

### Hybrid Fusion Logic

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Fusion Formula                     │
└─────────────────────────────────────────────────────────────┘

Final Score = α × Transformer_Prob + β × (1 - Factual_Score) + γ × (1 - Agentic_Score)

Where:
  - α = Weight for transformer model (default: 0.7)
  - β = Weight for entity verification (default: 0.2)
  - γ = Weight for agentic verification (default: 0.1)
  - All weights sum to 1.0

Classification:
  - If Final_Score >= Threshold (default: 0.5) → HALLUCINATION
  - If Final_Score < Threshold → CORRECT
```

## ✨ Features

### Core Capabilities

- ✅ **Transformer-based Classification**: Fine-tuned DistilBERT for binary hallucination detection
- ✅ **Entity Extraction & Verification**: NER with Wikipedia fact-checking
- ✅ **Agentic Verification**: LLM-based cross-verification (local or API)
- ✅ **Hybrid Fusion**: Weighted combination of multiple detection methods
- ✅ **Comprehensive Evaluation**: Metrics, confusion matrices, ROC curves
- ✅ **Automated Pipeline**: End-to-end automation with logging
- ✅ **Modular Design**: Reusable components for easy extension

### Technical Features

- **Dual NER Support**: spaCy or HuggingFace transformers
- **Flexible Verification**: Wikipedia API or knowledge graph integration
- **Batch Processing**: Efficient processing of multiple responses
- **Visualization**: Training curves, confusion matrices, ROC curves
- **Configuration-based**: JSON configuration for easy customization
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbbasShah01/Hallucination-Detector.git
cd Hallucination-Detector
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install spaCy Model (Optional, for Entity Verification)

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Step 4: Set Up API Keys (Optional, for Agentic Verification)

For OpenAI API:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Anthropic API:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 🏃 Quick Start

### 1. Preprocess Data

```bash
python src/preprocess_halueval.py
```

This will:
- Load HaluEval dataset (from HuggingFace or CSV)
- Extract prompt-response pairs
- Encode labels (1=hallucination, 0=correct)
- Tokenize for DistilBERT
- Save preprocessed data to `data/preprocessed/`

### 2. Run Master Pipeline

```bash
python src/master_pipeline.py --config config.json
```

This will:
- Load preprocessed data
- Train transformer model
- Run verification components
- Generate predictions
- Evaluate and create visualizations
- Save all results to `results/`

### 3. View Results

Check the `results/` directory for:
- Trained model
- Training curves
- Confusion matrix
- ROC curve
- Evaluation metrics
- Sample predictions

## 📁 Project Structure

```
Hallucination-Detector/
│
├── data/                          # Data directory
│   ├── preprocessed/              # Preprocessed datasets
│   │   ├── tokenized_data.json    # Tokenized data
│   │   └── tokenizer/             # Saved tokenizer
│   └── halueval.csv               # Raw dataset (if using CSV)
│
├── src/                           # Source code
│   ├── preprocess_halueval.py     # Data preprocessing
│   ├── train_model.py             # Model training
│   ├── entity_verification.py     # Entity extraction & verification
│   ├── hybrid_fusion.py           # Hybrid fusion logic
│   ├── agentic_verification.py    # LLM-based verification
│   ├── evaluate_model.py          # Evaluation & metrics
│   ├── master_pipeline.py         # Master orchestrator
│   ├── generate_placeholder_plots.py  # Visualization generator
│   └── test_entity_verification.py     # Unit tests
│
├── models/                        # Trained models
│   └── distilbert_halueval/       # Saved model checkpoints
│
├── results/                       # Output results
│   ├── trained_model/            # Saved trained model
│   ├── training_history.json      # Training metrics
│   ├── training_loss_accuracy.png # Training curves
│   ├── confusion_matrix.png       # Confusion matrix
│   ├── roc_curve.png              # ROC curve
│   ├── evaluation_metrics.json   # Evaluation metrics
│   └── sample_outputs.json        # Sample predictions
│
├── papers/                        # Research documentation
│   └── (IEEE LaTeX files)
│
├── config.json                    # Configuration file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── MASTER_PIPELINE_README.md      # Pipeline documentation
```

## 📖 Usage

### Basic Usage

Run the complete pipeline:

```bash
python src/master_pipeline.py
```

### With Custom Configuration

```bash
python src/master_pipeline.py --config config.json --output-dir results
```

### Individual Components

#### Preprocessing Only

```bash
python src/preprocess_halueval.py
```

#### Training Only

```bash
python src/train_model.py
```

#### Evaluation Only

```bash
python src/evaluate_model.py
```

### Configuration

Edit `config.json` to customize:

```json
{
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

## 📊 Results

### Example Output

The system generates comprehensive results including:

- **Training Metrics**: Loss and accuracy curves over epochs
- **Confusion Matrix**: Visual representation of classification performance
- **ROC Curve**: Receiver Operating Characteristic curve with AUC score
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Sample Predictions**: Examples of correctly and incorrectly classified responses

### Performance Metrics

Example results from test run:

- **Accuracy**: 0.923
- **Precision**: 0.841
- **Recall**: 0.812
- **F1-Score**: 0.826

### Visualization Examples

All visualizations are saved to the `results/` directory:
- Training/validation curves
- Confusion matrix heatmap
- ROC curve with AUC
- Metrics comparison bar chart
- Sample response tables

## 🔧 Advanced Usage

### Custom Entity Verification

```python
from src.entity_verification import EntityVerifier

verifier = EntityVerifier(
    extractor_method="spacy",  # or "transformers"
    use_wikipedia=True
)

result = verifier.verify_response("Your LLM response here")
print(f"Correctness score: {result.correctness_score}")
```

### Hybrid Fusion

```python
from src.hybrid_fusion import hybrid_predict

result = hybrid_predict(
    transformer_prob=0.3,
    factual_score=0.9,
    alpha=0.7,
    threshold=0.5
)

print(f"Fusion prob: {result.fusion_prob}")
print(f"Classification: {result.is_hallucination}")
```

### Agentic Verification

```python
from src.agentic_verification import AgenticVerifier

verifier = AgenticVerifier(
    method="local",  # or "api"
    provider="openai"  # if using API
)

result = verifier.verify("Your response here")
print(f"Verification score: {result.verification_score}")
```

## 🧪 Testing

Run unit tests:

```bash
python src/test_entity_verification.py
```

## 📝 Dependencies

Key dependencies (see `requirements.txt` for complete list):

- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.30.0` - HuggingFace transformers
- `datasets>=2.14.0` - Dataset handling
- `spacy>=3.5.0` - Named Entity Recognition
- `scikit-learn>=1.3.0` - Machine learning metrics
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Abbas Shah**

- GitHub: [@AbbasShah01](https://github.com/AbbasShah01)
- Repository: [Hallucination-Detector](https://github.com/AbbasShah01/Hallucination-Detector)

## 🙏 Acknowledgments

- HaluEval dataset for evaluation benchmarks
- HuggingFace for transformer models and tools
- spaCy for NLP capabilities
- The open-source community for inspiration and tools

## 📚 References

- HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
- DistilBERT: A distilled version of BERT
- Wikipedia API for entity verification

## 🔮 Future Work

- [ ] Support for more transformer models
- [ ] Integration with additional knowledge bases
- [ ] Real-time API endpoint
- [ ] Web interface for easy interaction
- [ ] Support for multiple languages
- [ ] Advanced ensemble methods

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**
