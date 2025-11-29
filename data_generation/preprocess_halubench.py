"""
Preprocessing script for HaluBench-Multi dataset.
Converts dataset to formats suitable for training and evaluation.
"""

import json
import pandas as pd
import argparse
from typing import List, Dict
from pathlib import Path


def load_halubench(json_path: str) -> Dict:
    """Load HaluBench-Multi dataset from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_csv(json_path: str, output_path: str):
    """Convert HaluBench-Multi JSON to CSV format."""
    data = load_halubench(json_path)
    examples = data["examples"]
    
    rows = []
    for example in examples:
        example_id = example["example_id"]
        conversation = example["conversation"]
        annotation = example["hallucination_annotation"]
        metadata = example["metadata"]
        
        # Process each turn
        for turn in conversation["turns"]:
            row = {
                "example_id": example_id,
                "conversation_id": f"conv_{example_id}",
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
                "has_hallucination": annotation["has_hallucination"],
                "binary_label": annotation["binary_label"],
                "hallucination_type": annotation["hallucination_type"],
                "hallucination_subtype": annotation["hallucination_subtype"],
                "severity": annotation["severity"],
                "confidence": annotation["confidence"],
                "detection_difficulty": annotation["detection_difficulty"],
                "requires_reasoning": annotation["requires_reasoning"],
                "requires_domain_knowledge": annotation["requires_domain_knowledge"],
                "domain": conversation["context"]["domain"],
                "topic": conversation["context"]["topic"],
                "generation_method": metadata["generation_method"],
                "quality_score": metadata["quality_score"]
            }
            
            # Add affected span info if this is the hallucinated turn
            if annotation["has_hallucination"] and turn["role"] == "assistant":
                if annotation["affected_spans"]:
                    span = annotation["affected_spans"][0]
                    row["affected_span_start"] = span["start_char"]
                    row["affected_span_end"] = span["end_char"]
                    row["error_text"] = span.get("error_text", "")
                    row["correct_text"] = span.get("correct_text", "")
                else:
                    row["affected_span_start"] = None
                    row["affected_span_end"] = None
                    row["error_text"] = ""
                    row["correct_text"] = ""
            else:
                row["affected_span_start"] = None
                row["affected_span_end"] = None
                row["error_text"] = ""
                row["correct_text"] = ""
            
            # Add root causes
            root_causes = annotation.get("root_cause", [])
            row["root_cause"] = "|".join(root_causes) if root_causes else ""
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"CSV saved to {output_path}")
    print(f"Total rows: {len(df)}")


def split_dataset(json_path: str, output_dir: str, train_ratio: float = 0.7,
                 val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Split dataset into train/val/test splits."""
    import random
    
    data = load_halubench(json_path)
    examples = data["examples"]
    
    # Shuffle
    random.seed(42)
    random.shuffle(examples)
    
    # Split
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]
    
    # Save splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for split_name, split_examples in [("train", train_examples),
                                       ("val", val_examples),
                                       ("test", test_examples)]:
        output_path = Path(output_dir) / f"halubench_multi_{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset_version": data["dataset_version"],
                "dataset_name": data["dataset_name"],
                "split": split_name,
                "num_examples": len(split_examples),
                "examples": split_examples
            }, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name.capitalize()} split: {len(split_examples)} examples")
        print(f"  Saved to: {output_path}")


def create_statistics(json_path: str, output_path: str):
    """Generate dataset statistics."""
    data = load_halubench(json_path)
    examples = data["examples"]
    
    stats = {
        "total_examples": len(examples),
        "by_type": {},
        "by_difficulty": {},
        "by_domain": {},
        "hallucination_rate": 0.0
    }
    
    hallucination_count = 0
    
    for example in examples:
        annotation = example["hallucination_annotation"]
        conversation = example["conversation"]
        
        # Count hallucinations
        if annotation["has_hallucination"]:
            hallucination_count += 1
        
        # Count by type
        h_type = annotation["hallucination_type"]
        stats["by_type"][h_type] = stats["by_type"].get(h_type, 0) + 1
        
        # Count by difficulty
        difficulty = annotation["detection_difficulty"]
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1
        
        # Count by domain
        domain = conversation["context"]["domain"]
        stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
    
    stats["hallucination_rate"] = hallucination_count / len(examples)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Hallucination rate: {stats['hallucination_rate']:.2%}")
    print(f"\nBy Type:")
    for h_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        print(f"  {h_type}: {count} ({count/stats['total_examples']:.1%})")
    print(f"\nBy Difficulty:")
    for diff, count in sorted(stats['by_difficulty'].items()):
        print(f"  {diff}: {count} ({count/stats['total_examples']:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HaluBench-Multi dataset")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV file (optional)")
    parser.add_argument("--split", action="store_true",
                       help="Split into train/val/test")
    parser.add_argument("--output_dir", type=str, default="data/halubench_multi",
                       help="Output directory for splits")
    parser.add_argument("--stats", type=str, default=None,
                       help="Generate statistics file")
    
    args = parser.parse_args()
    
    if args.output_csv:
        convert_to_csv(args.input, args.output_csv)
    
    if args.split:
        split_dataset(args.input, args.output_dir)
    
    if args.stats:
        create_statistics(args.input, args.stats)

