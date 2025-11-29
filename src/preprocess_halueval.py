"""
Preprocessing script for HaluEval dataset
Loads the dataset, extracts prompt-response pairs, encodes labels, 
and tokenizes text for DistilBERT model.
"""

import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import json
from pathlib import Path


def load_halueval_dataset(dataset_path=None, use_huggingface=True):
    """
    Load HaluEval dataset from HuggingFace or local CSV file.
    
    Args:
        dataset_path: Path to local CSV file (if use_huggingface=False)
        use_huggingface: If True, load from HuggingFace datasets library
    
    Returns:
        Dataset object or pandas DataFrame
    """
    if use_huggingface:
        try:
            # Try loading from HuggingFace datasets
            print("Loading HaluEval dataset from HuggingFace...")
            dataset = load_dataset("HuggingFaceH4/HaluEval")
            print(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
            return dataset
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Falling back to CSV file...")
            use_huggingface = False
    
    if not use_huggingface:
        # Load from local CSV file
        if dataset_path is None:
            # Default path in data directory
            dataset_path = os.path.join("data", "halueval.csv")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found at {dataset_path}. "
                "Please download the HaluEval dataset and place it in the data/ directory."
            )
        
        print(f"Loading dataset from CSV: {dataset_path}")
        df = pd.read_csv(dataset_path)
        return df


def extract_prompt_response_pairs(data, source="huggingface"):
    """
    Extract prompt-response pairs from the dataset.
    
    Args:
        data: Dataset object or DataFrame
        source: "huggingface" or "csv" - indicates the data format
    
    Returns:
        List of dictionaries with 'prompt', 'response', and 'label' keys
    """
    pairs = []
    
    if source == "huggingface":
        # Handle HuggingFace dataset format
        # HaluEval typically has 'query' or 'prompt' and 'response' fields
        # and 'label' or 'is_hallucination' field
        
        # Process each split (train, validation, test)
        for split_name, split_data in data.items():
            print(f"Processing {split_name} split...")
            
            for item in split_data:
                # Extract prompt (may be 'query', 'prompt', or 'input')
                prompt = item.get('query') or item.get('prompt') or item.get('input', '')
                
                # Extract response (may be 'response', 'output', or 'answer')
                response = item.get('response') or item.get('output') or item.get('answer', '')
                
                # Extract label (may be 'label', 'is_hallucination', or 'hallucination')
                label = item.get('label') or item.get('is_hallucination') or item.get('hallucination')
                
                if prompt and response and label is not None:
                    pairs.append({
                        'prompt': prompt,
                        'response': response,
                        'label': label
                    })
    
    else:  # CSV format
        # Handle CSV format - adjust column names based on actual CSV structure
        print("Processing CSV data...")
        
        # Common column name variations
        prompt_col = None
        response_col = None
        label_col = None
        
        for col in data.columns:
            col_lower = col.lower()
            if 'prompt' in col_lower or 'query' in col_lower or 'input' in col_lower:
                prompt_col = col
            elif 'response' in col_lower or 'output' in col_lower or 'answer' in col_lower:
                response_col = col
            elif 'label' in col_lower or 'hallucination' in col_lower or 'is_hallucination' in col_lower:
                label_col = col
        
        if not all([prompt_col, response_col, label_col]):
            raise ValueError(
                f"Could not find required columns. Found: {list(data.columns)}\n"
                "Expected columns: prompt/query/input, response/output/answer, label/hallucination"
            )
        
        for _, row in data.iterrows():
            pairs.append({
                'prompt': str(row[prompt_col]),
                'response': str(row[response_col]),
                'label': row[label_col]
            })
    
    print(f"Extracted {len(pairs)} prompt-response pairs")
    return pairs


def encode_labels(pairs):
    """
    Encode labels: 1=hallucination, 0=correct
    
    Args:
        pairs: List of dictionaries with 'label' field
    
    Returns:
        List of dictionaries with encoded 'label' field (0 or 1)
    """
    encoded_pairs = []
    
    for pair in pairs:
        label = pair['label']
        
        # Handle different label formats
        if isinstance(label, bool):
            # True = hallucination (1), False = correct (0)
            encoded_label = 1 if label else 0
        elif isinstance(label, str):
            # String labels: "hallucination", "true", "1" -> 1; others -> 0
            label_lower = label.lower()
            if 'hallucination' in label_lower or label_lower in ['true', '1', 'yes']:
                encoded_label = 1
            else:
                encoded_label = 0
        elif isinstance(label, (int, float)):
            # Numeric labels: 1 or True-like -> 1, 0 or False-like -> 0
            encoded_label = 1 if int(label) == 1 else 0
        else:
            # Default: treat as correct (0)
            encoded_label = 0
        
        encoded_pairs.append({
            'prompt': pair['prompt'],
            'response': pair['response'],
            'label': encoded_label
        })
    
    # Print label distribution
    label_counts = {}
    for pair in encoded_pairs:
        label = pair['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    print(f"  Correct (0): {label_counts.get(0, 0)}")
    print(f"  Hallucination (1): {label_counts.get(1, 0)}")
    
    return encoded_pairs


def tokenize_text(pairs, model_name="distilbert-base-uncased", max_length=512):
    """
    Tokenize text for DistilBERT model.
    
    Args:
        pairs: List of dictionaries with 'prompt' and 'response' fields
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tokenized dataset ready for training
    """
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Combine prompt and response for tokenization
    # Format: [CLS] prompt [SEP] response [SEP]
    print("Tokenizing text...")
    
    tokenized_data = []
    for pair in pairs:
        # Combine prompt and response
        text = f"{pair['prompt']} [SEP] {pair['response']}"
        
        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None  # Return as lists, not tensors
        )
        
        tokenized_data.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': pair['label'],
            'prompt': pair['prompt'],  # Keep original for reference
            'response': pair['response']  # Keep original for reference
        })
    
    print(f"Tokenized {len(tokenized_data)} samples")
    print(f"Sequence length: {max_length}")
    
    return tokenized_data, tokenizer


def save_preprocessed_data(tokenized_data, tokenizer, output_dir="data/preprocessed"):
    """
    Save preprocessed data to disk.
    
    Args:
        tokenized_data: List of tokenized samples
        tokenizer: Tokenizer object to save
        output_dir: Directory to save preprocessed data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenized data as JSON
    output_file = os.path.join(output_dir, "tokenized_data.json")
    print(f"\nSaving tokenized data to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tokenized_data, f, indent=2, ensure_ascii=False)
    
    # Save tokenizer
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    print(f"Saving tokenizer to {tokenizer_dir}...")
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Save metadata
    metadata = {
        'num_samples': len(tokenized_data),
        'max_length': len(tokenized_data[0]['input_ids']) if tokenized_data else 0,
        'model_name': tokenizer.name_or_path,
        'vocab_size': tokenizer.vocab_size
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPreprocessed data saved successfully!")
    print(f"  - Tokenized data: {output_file}")
    print(f"  - Tokenizer: {tokenizer_dir}")
    print(f"  - Metadata: {metadata_file}")


def main():
    """
    Main preprocessing pipeline.
    """
    print("=" * 60)
    print("HaluEval Dataset Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Load dataset
    # Try HuggingFace first, fall back to CSV if needed
    try:
        data = load_halueval_dataset(use_huggingface=True)
        source = "huggingface"
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        data = load_halueval_dataset(use_huggingface=False)
        source = "csv"
    
    # Step 2: Extract prompt-response pairs
    pairs = extract_prompt_response_pairs(data, source=source)
    
    if len(pairs) == 0:
        raise ValueError("No prompt-response pairs extracted from dataset!")
    
    # Step 3: Encode labels
    encoded_pairs = encode_labels(pairs)
    
    # Step 4: Tokenize text for DistilBERT
    tokenized_data, tokenizer = tokenize_text(
        encoded_pairs,
        model_name="distilbert-base-uncased",
        max_length=512
    )
    
    # Step 5: Save preprocessed data
    save_preprocessed_data(tokenized_data, tokenizer)
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

