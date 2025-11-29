# System Architecture Documentation

## Overview

The Hybrid Hallucination Detection System is a multi-component architecture that combines transformer models, entity verification, and agentic verification to detect hallucinations in LLM outputs.

## Architecture Diagrams

### 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Hybrid Hallucination Detection System              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Input Layer                                │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  LLM Response Text                                      │  │  │
│  │  │  (Prompt + Generated Response)                         │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Detection Components Layer                       │  │
│  │                                                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │ Transformer  │  │   Entity     │  │   Agentic    │      │  │
│  │  │   Model      │  │ Verification │  │ Verification │      │  │
│  │  │              │  │              │  │              │      │  │
│  │  │ DistilBERT   │  │ NER + Wiki    │  │ LLM Cross-  │      │  │
│  │  │ Classifier   │  │ Fact-Check    │  │ Check        │      │  │
│  │  │              │  │              │  │              │      │  │
│  │  │ Output:      │  │ Output:       │  │ Output:      │      │  │
│  │  │ Prob [0-1]   │  │ Score [0-1]  │  │ Score [0-1] │      │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
│  │       │                  │                    │              │  │
│  └───────┼──────────────────┼────────────────────┼──────────────┘  │
│          │                  │                    │                 │
│          └──────────────────┼────────────────────┘                 │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Fusion Layer                                      │  │
│  │                                                               │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │  Hybrid Fusion Algorithm                              │   │  │
│  │  │                                                       │   │  │
│  │  │  Score = α×Trans + β×(1-Fact) + γ×(1-Agent)          │   │  │
│  │  │                                                       │   │  │
│  │  │  Where: α=0.7, β=0.2, γ=0.1                          │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Output Layer                                │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  Final Prediction:                                      │  │  │
│  │  │  • Hallucination Probability [0-1]                     │  │  │
│  │  │  • Binary Classification (Hallucination/Correct)       │  │  │
│  │  │  • Confidence Score                                     │  │  │
│  │  │  • Detailed Metrics                                     │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Data Flow Diagram

```
┌─────────────┐
│ Raw Dataset │
│ (HaluEval)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Preprocessing  │
│  • Extract pairs│
│  • Encode labels│
│  • Tokenize     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Training Data   │
│ (Train/Val/Test)│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────────┐
│ Model Training   │─────▶│ Trained Model   │
│ • DistilBERT     │      │ (Saved)         │
│ • Fine-tuning    │      └──────────────────┘
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Test Data      │
└──────┬──────────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Transformer  │  │   Entity     │  │   Agentic    │
│ Prediction   │  │ Verification  │  │ Verification  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┼──────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   Fusion     │
                  │   Algorithm  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Evaluation  │
                  │  & Metrics   │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Results     │
                  │  (JSON/PNG)  │
                  └──────────────┘
```

### 3. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Pipeline                           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Data Loader  │   │   Trainer    │   │  Evaluator   │
│              │   │              │   │              │
│ • Load JSON  │   │ • Initialize │   │ • Metrics    │
│ • Split Data │   │ • Train      │   │ • Plots      │
│ • Create DL  │   │ • Validate    │   │ • Reports    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Verification Layer  │
              │                      │
              │  ┌────────────────┐  │
              │  │ Entity Verifier│  │
              │  │ • NER Extract  │  │
              │  │ • Wiki Check   │  │
              │  └────────────────┘  │
              │                      │
              │  ┌────────────────┐  │
              │  │ Agentic Verif. │  │
              │  │ • LLM Check    │  │
              │  └────────────────┘  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌───────────────────────┐
              │   Fusion Engine       │
              │   • Weighted Sum      │
              │   • Classification    │
              └───────────────────────┘
```

### 4. Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
└─────────────────────────────────────────────────────────────┘

Epoch Loop (N epochs)
│
├─▶ Training Phase
│   │
│   ├─ For each batch:
│   │   ├─ Forward pass
│   │   ├─ Compute loss
│   │   ├─ Backward pass
│   │   └─ Update weights
│   │
│   └─ Calculate training metrics
│
├─▶ Validation Phase
│   │
│   ├─ For each batch:
│   │   ├─ Forward pass (no grad)
│   │   └─ Compute metrics
│   │
│   └─ Calculate validation metrics
│
└─▶ Save best model
    │
    └─▶ Generate plots
```

## Component Details

### Transformer Model Component

- **Model**: DistilBERT-base-uncased
- **Task**: Binary classification (Hallucination vs Correct)
- **Input**: Tokenized prompt-response pairs
- **Output**: Hallucination probability [0-1]
- **Training**: Fine-tuning with AdamW optimizer

### Entity Verification Component

- **Method 1**: spaCy NER (default)
- **Method 2**: HuggingFace Transformers NER
- **Verification**: Wikipedia API fact-checking
- **Output**: Factual correctness score [0-1]

### Agentic Verification Component

- **Method 1**: Local LLM (transformers library)
- **Method 2**: API-based (OpenAI/Anthropic)
- **Process**: LLM cross-checks the response
- **Output**: Verification score [0-1]

### Fusion Component

- **Algorithm**: Weighted linear combination
- **Weights**: Configurable (default: α=0.7, β=0.2, γ=0.1)
- **Threshold**: Configurable (default: 0.5)
- **Output**: Final hallucination probability + binary classification

## Data Formats

### Input Format

```json
{
  "response": "LLM generated response text",
  "prompt": "Original prompt (optional)",
  "label": 0 or 1  // 0=correct, 1=hallucination
}
```

### Output Format

```json
{
  "transformer_prob": 0.3,
  "factual_score": 0.9,
  "agentic_score": 0.85,
  "fusion_prob": 0.25,
  "is_hallucination": false,
  "confidence": 0.92
}
```

## Performance Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## Scalability Considerations

- Batch processing support
- GPU acceleration (if available)
- API rate limiting for external services
- Caching for repeated verifications
- Efficient data loading with DataLoaders

