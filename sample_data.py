#!/usr/bin/env python3
"""
Script to sample 80k records uniformly at random from MedMcqa training data
and save as train_aug.json in the augmented_data/augmented_MedMcqa directory.
"""

import json
import random
import os
from pathlib import Path

def sample_medmcqa_data():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    input_file = "/home/bryan/Documents/code_repo/CSE_493s/Medistillation/data/MedMcqa_data/train.json"
    output_dir = "/home/bryan/Documents/code_repo/CSE_493s/Medistillation/augmented_data/augmented_MedMcqa"
    output_file = os.path.join(output_dir, "train_aug.json")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {input_file}")
    
    # Read all training data
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Total records in training data: {len(all_data)}")
    
    # Sample 80,000 records uniformly at random
    sample_size = 80000
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

if __name__ == "__main__":
    sample_medmcqa_data()
