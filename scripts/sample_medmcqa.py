#!/usr/bin/env python3
"""
Script to sample a subset of MedMCQA training data for testing.

Usage:
    python scripts/sample_medmcqa.py --input data/raw/medmcqa/train.json \
                                      --output data/raw/medmcqa_sample/train_aug.json \
                                      --size 80000
"""

import json
import random
import os
import argparse
from pathlib import Path


def sample_medmcqa_data(input_file: str, output_file: str, sample_size: int = 80000, seed: int = 42):
    """
    Sample records uniformly at random from MedMCQA training data.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        sample_size: Number of records to sample (default: 80000)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {input_file}")
    
    # Read all training data
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Total records in training data: {len(all_data)}")
    
    # Sample records
    if len(all_data) < sample_size:
        print(f"Warning: Only {len(all_data)} records available, sampling all of them")
        sample_size = len(all_data)
    
    sampled_data = random.sample(all_data, sample_size)
    
    print(f"Sampled {len(sampled_data)} records")
    
    # Write sampled data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in sampled_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Sampled data saved to: {output_file}")
    
    # Verify the output
    with open(output_file, 'r', encoding='utf-8') as f:
        verification_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Verification: Output file contains {len(verification_data)} records")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Sample MedMCQA training data")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input MedMCQA train.json file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output sampled JSON file')
    parser.add_argument('--size', type=int, default=80000,
                        help='Number of samples to extract (default: 80000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    sample_medmcqa_data(
        input_file=args.input,
        output_file=args.output,
        sample_size=args.size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
