# Sentence-Level (Span-Level) Hallucination Detector

## Overview

This module provides **fine-grained hallucination detection at the sentence level**, enabling precise localization of hallucinated content within LLM responses. Unlike response-level detection that classifies entire responses, this module identifies which specific sentences contain hallucinations.

## Key Features

- **Sentence-Level Splitting**: Robust sentence boundary detection using NLTK, spaCy, or regex fallback
- **Per-Sentence Classification**: Applies transformer-based classification to each sentence individually
- **Per-Sentence Entity Verification**: Extracts and verifies entities within each sentence
- **Per-Sentence Agent Verification**: Uses LLM-based verification for each sentence (optional)
- **Sentence-Level Fusion**: Combines multiple signals at the sentence level
- **Detailed JSON Output**: Returns per-sentence labels with scores and confidence

## Architecture

```
Input Text
    │
    ▼
Sentence Splitter (NLTK/spaCy/regex)
    │
    ▼
[For each sentence]
    ├─→ Span Classifier (Transformer)
    ├─→ Span Entity Verifier (NER + Wikipedia)
    ├─→ Span Agent Verifier (LLM critique) [optional]
    │
    ▼
Span Fusion (Weighted combination)
    │
    ▼
JSON Output (Per-sentence results)
```

## Installation

The module uses existing dependencies from the main project:

```bash
pip install -r requirements.txt
```

For sentence splitting:
```bash
# NLTK (recommended)
pip install nltk
python -c "import nltk; nltk.download('punkt')"

# spaCy (alternative)
pip install spacy
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
from modules.span_level_detector import SpanInferencePipeline

# Initialize pipeline
pipeline = SpanInferencePipeline(
    model_path="models/distilbert_halueval",  # Optional: path to fine-tuned model
    use_entity_verification=True,
    use_agent_verification=False
)

# Detect hallucinations
text = """
Large Language Models have achieved remarkable success.
However, they frequently generate hallucinations.
The moon is made of cheese.
"""

results = pipeline.detect(text, return_json=True)

# Print results
for result in results:
    print(f"Sentence: {result['sentence']}")
    print(f"Label: {result['label']}")
    print(f"Score: {result['final_hallucination_score']:.3f}")
    print()
```

### Advanced Usage

```python
# Custom fusion weights
pipeline = SpanInferencePipeline(
    fusion_alpha=0.6,  # Higher weight for classification
    fusion_beta=0.3,   # Weight for entity verification
    fusion_gamma=0.1,  # Weight for agent verification
    fusion_threshold=0.5,
    use_agent_verification=True
)

# Batch processing
texts = ["Text 1...", "Text 2...", "Text 3..."]
all_results = pipeline.detect_batch(texts)

# Save results
pipeline.save_results(results, "output/span_results.json")

# Get summary statistics
summary = pipeline.get_summary(results)
print(f"Hallucination rate: {summary['hallucination_rate']:.2%}")
```

### Individual Components

```python
from modules.span_level_detector import (
    SentenceSplitter,
    SpanClassifier,
    SpanEntityVerifier,
    SpanAgentVerifier,
    SpanFusion
)

# Sentence splitting
splitter = SentenceSplitter(method="nltk")
sentences = splitter.split(text)

# Per-sentence classification
classifier = SpanClassifier(model_path="models/distilbert_halueval")
classification_results = classifier.classify_sentences([s.text for s in sentences])

# Entity verification
entity_verifier = SpanEntityVerifier(extractor_method="transformers")
entity_results = entity_verifier.verify_sentences([s.text for s in sentences])

# Agent verification
agent_verifier = SpanAgentVerifier()
agent_results = agent_verifier.verify_sentences([s.text for s in sentences])

# Fusion
fusion = SpanFusion(alpha=0.5, beta=0.3, gamma=0.2)
fusion_results = fusion.fuse_batch(
    sentences=[s.text for s in sentences],
    classification_scores=[r.classification_score for r in classification_results],
    entity_verification_scores=[r.entity_verification_score for r in entity_results],
    agent_verification_scores=[r.agent_verification_score for r in agent_results]
)
```

## Output Format

Each sentence returns a JSON object:

```json
{
  "sentence": "The moon is made of cheese.",
  "classification_score": 0.8234,
  "entity_verification_score": 0.2500,
  "agent_verification_score": 0.1500,
  "final_hallucination_score": 0.7123,
  "label": "hallucinated",
  "confidence": 0.4246,
  "sentence_index": 2
}
```

### Fields

- `sentence`: The sentence text
- `classification_score`: Transformer-based hallucination probability (0-1)
- `entity_verification_score`: Entity verification correctness score (0-1, higher = more correct)
- `agent_verification_score`: Agent verification correctness score (0-1, optional)
- `final_hallucination_score`: Fused hallucination probability (0-1)
- `label`: "hallucinated" or "factual"
- `confidence`: Confidence in the prediction (0-1)
- `sentence_index`: Index of sentence in original text

## Integration with Main Pipeline

The module integrates seamlessly with the main pipeline:

```python
from src.master_pipeline import MasterPipeline

pipeline = MasterPipeline()
pipeline.run(mode="sentence_level", text="Your text here...")
```

## Evaluation

See `evaluation/span_level_evaluation.py` for evaluation scripts that compute:
- Precision/Recall/F1 at sentence level
- Confusion matrix for sentences
- Per-sentence error analysis
- Summary reports

## Research Contribution

This module represents a novel contribution to hallucination detection research:

1. **Fine-Grained Localization**: Identifies specific sentences containing hallucinations
2. **Multi-Signal Fusion**: Combines classification, entity verification, and agent verification at sentence level
3. **Context-Aware Processing**: Uses surrounding sentences as context for better classification
4. **Scalable Architecture**: Processes sentences in batches for efficiency

## Performance Considerations

- **Batch Processing**: Sentences are processed in batches for efficiency
- **Caching**: Entity verification results can be cached for repeated entities
- **Parallel Processing**: Components can be parallelized for faster processing
- **Model Loading**: Fine-tuned models are loaded once and reused

## Limitations

- Sentence splitting may not always be perfect (especially for complex punctuation)
- Entity verification depends on Wikipedia coverage
- Agent verification requires LLM API access (optional)
- Processing time scales with number of sentences

## Future Work

- [ ] Support for multi-lingual sentence splitting
- [ ] Cross-sentence consistency checking
- [ ] Adaptive fusion weights based on sentence characteristics
- [ ] Real-time streaming detection
- [ ] Integration with uncertainty-driven scoring

## Citation

If you use this module in your research, please cite:

```bibtex
@software{span_level_detector,
  title={Sentence-Level Hallucination Detector},
  author={Hallucination Detection System},
  year={2024},
  url={https://github.com/AbbasShah01/Hallucination-Detector}
}
```

