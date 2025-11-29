# HaluBench-Multi Dataset Generation

## Overview

This directory contains scripts for generating the HaluBench-Multi benchmark dataset, a comprehensive multi-dimensional dataset for hallucination detection.

## Files

- `generate_halubench.py` - Main dataset generation script
- `preprocess_halubench.py` - Preprocessing and conversion utilities
- `README.md` - This file

## Usage

### Generate Dataset

```bash
# Generate 1000 examples using templates
python data_generation/generate_halubench.py --num_examples 1000 --output data/halubench_multi.json

# Generate using LLM (requires API key)
export OPENAI_API_KEY="your-key"
python data_generation/generate_halubench.py --num_examples 1000 --use_llm --llm_provider openai
```

### Preprocess Dataset

```bash
# Convert to CSV
python data_generation/preprocess_halubench.py --input data/halubench_multi.json --output_csv data/halubench_multi.csv

# Split into train/val/test
python data_generation/preprocess_halubench.py --input data/halubench_multi.json --split --output_dir data/halubench_multi

# Generate statistics
python data_generation/preprocess_halubench.py --input data/halubench_multi.json --stats data/halubench_stats.json
```

## Dataset Schema

See `docs/NEW_BENCHMARK_DATASET.md` for complete schema documentation.

## Requirements

- Python 3.8+
- pandas
- openai (optional, for LLM generation)
- anthropic (optional, for Claude generation)

