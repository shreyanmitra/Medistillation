# Logit Knowledge Distillation (KD) - Simple Implementation Guide

**Project:** Medical LLM Distillation  
**Method:** Logit KD - Teaching a small model (Qwen2-1.5B) to copy a big model's (Meditron-7B) thinking patterns

---

## What is Logit KD? (Explain Like I'm 5)

### The Ice Cream Analogy 🍦

Imagine you're teaching a kid to pick ice cream flavors like an expert does:

**Normal Training (Bad Way):**
- Expert says: "Pick chocolate" (just the final answer)
- Kid learns: "Always pick chocolate"
- Kid never learns *why* or what other options were close

**Logit KD (Smart Way):**
- Expert says: "I'm 60% sure chocolate, 30% sure vanilla, 8% sure strawberry, 2% sure mint"
- Kid learns: "Chocolate is best, but vanilla is a close second, strawberry is okay too"
- Kid learns the *confidence levels* and can make better decisions in new situations

### What This Means for AI Models

Instead of just teaching the student model "this is the right answer," we teach it:
- How confident the teacher is about each possible word/token
- Which wrong answers are "less wrong" than others
- The teacher's full reasoning pattern, not just the final choice

---

## The Math (Simple Version)

### What Are Logits?

**Logits** are the raw scores a model gives to each possible next word before turning them into probabilities.

```
Example: "The patient has ___"

Teacher's raw scores (logits):
- "fever": 8.2
- "pain": 7.1
- "cancer": 3.4
- "pizza": -5.0

After softmax (turn into probabilities):
- "fever": 72%
- "pain": 25%
- "cancer": 2.8%
- "pizza": 0.0001%
```

The teacher isn't just saying "fever" - it's saying "fever is most likely, but pain is pretty close too."

### The Logit KD Loss Formula

```
Total Loss = α × KD_Loss + (1 - α) × Regular_Loss
```

**Where:**
- **KD_Loss** = How different the student's probabilities are from teacher's (measured with KL Divergence)
- **Regular_Loss** = How wrong the student's answer is (normal cross-entropy)
- **α (alpha)** = Mixing ratio - how much we care about copying teacher vs. being correct
  - α = 0.5 means "care equally about both"
  - α = 0.7 means "care more about copying teacher"
  - α = 0.3 means "care more about getting right answer"

**Temperature (T)** softens the probabilities:
- Higher T → softer (more even) probabilities → student learns more from teacher
- Lower T → sharper probabilities → more like normal training

---

## Implementation: Step-by-Step Code

### Step 1: Setup and Imports

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Step 2: Load Teacher and Student Models

```python
# Load the teacher model (Meditron-7B) - frozen, won't be trained
teacher_model = AutoModelForCausalLM.from_pretrained(
    "epfl-llm/meditron-7b",
    load_in_8bit=True,  # Use less memory
    device_map="auto",
    torch_dtype=torch.float16
)
teacher_model.eval()  # Put in evaluation mode - no training!

# Load the student model (Qwen2-1.5B) - this one we'll train
student_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load tokenizer (same for both models)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
```

### Step 3: Setup LoRA (Memory-Efficient Training)

LoRA = Low-Rank Adaptation - only train small "adapter" pieces instead of whole model

```python
# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank - how big the adapters are (16-32 is good)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to student model
student_model = prepare_model_for_kbit_training(student_model)
student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()  # Shows how many params we're training
```

### Step 4: The Core Logit KD Loss Function

This is the heart of the algorithm!

```python
def logit_kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=3.0):
    """
    Calculate the Logit Knowledge Distillation loss.
    
    Args:
        student_logits: Raw scores from student model [batch_size, seq_len, vocab_size]
        teacher_logits: Raw scores from teacher model [batch_size, seq_len, vocab_size]
        labels: Correct answers [batch_size, seq_len]
        alpha: Mix between KD loss and regular loss (0.0 to 1.0)
        temperature: Softening factor for probabilities (higher = softer)
    
    Returns:
        total_loss: Combined loss value
    """
    
    # --- Part 1: KD Loss (copying teacher) ---
    
    # Soften both student and teacher probabilities using temperature
    # Division by T makes probabilities more even
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Calculate KL Divergence (how different student is from teacher)
    # Lower KL = more similar to teacher
    kd_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)  # Multiply by T^2 to scale properly
    
    # --- Part 2: Regular Cross-Entropy Loss (getting right answer) ---
    
    # Shift logits and labels for next-token prediction
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate regular loss
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id  # Don't count padding tokens
    )
    
    # --- Part 3: Combine Both Losses ---
    
    total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
    
    return total_loss, kd_loss.item(), ce_loss.item()
```

### Step 5: Training Loop (Simplified)

```python
from torch.optim import AdamW
from tqdm import tqdm

# Hyperparameters
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
ALPHA = 0.5  # Try 0.3, 0.5, 0.7
TEMPERATURE = 3.0  # Try 2, 3, 4

# Setup optimizer
optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)

# Training loop
student_model.train()

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
    
    for batch in tqdm(train_dataloader):
        # Move batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # --- Forward pass through teacher (no gradients) ---
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # --- Forward pass through student (with gradients) ---
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits
        
        # --- Calculate Logit KD Loss ---
        loss, kd_loss_val, ce_loss_val = logit_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            alpha=ALPHA,
            temperature=TEMPERATURE
        )
        
        # --- Backward pass and optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print losses occasionally
        if batch_idx % 100 == 0:
            print(f"Loss: {loss.item():.4f} | KD: {kd_loss_val:.4f} | CE: {ce_loss_val:.4f}")
    
    # Save checkpoint after each epoch
    student_model.save_pretrained(f"./checkpoints/student_epoch_{epoch+1}")
```

### Step 6: Data Preparation (What Your Training Data Should Look Like)

```python
from datasets import load_dataset

# Load your medical QA dataset
dataset = load_dataset('json', data_files='data/processed/train.jsonl')

def format_medical_qa(example):
    """
    Format medical QA into prompt-response pairs.
    
    Example format:
    Question: A 45-year-old man presents with chest pain...
    Options: A) MI B) Pneumonia C) GERD D) Anxiety
    Answer: A) Myocardial infarction because...
    """
    
    prompt = f"Question: {example['question']}\n"
    prompt += f"Options: {example['options']}\n"
    prompt += "Answer: "
    
    # Teacher's response (you generated this earlier)
    response = example['teacher_response']
    
    # Combine
    full_text = prompt + response
    
    # Tokenize
    encoded = tokenizer(
        full_text,
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'][0],
        'attention_mask': encoded['attention_mask'][0],
        'labels': encoded['input_ids'][0]  # For causal LM, labels = input_ids
    }

# Apply formatting
train_dataset = dataset.map(format_medical_qa, remove_columns=dataset.column_names)

# Create DataLoader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
```

---

## Complete Script: `train_logit_kd.py`

Here's everything combined into one runnable script:

```python
#!/usr/bin/env python3
"""
Logit Knowledge Distillation Training Script
For Medical LLM Project - Teaching Qwen2-1.5B from Meditron-7B
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import wandb  # For experiment tracking (optional)

def logit_kd_loss(student_logits, teacher_logits, labels, alpha, temperature, pad_token_id):
    """Logit KD Loss Function"""
    
    # KD Loss
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    # CE Loss
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id
    )
    
    # Combined Loss
    total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
    
    return total_loss, kd_loss.item(), ce_loss.item()

def main(args):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Optional: Initialize wandb for tracking
    if args.use_wandb:
        wandb.init(project="medical-kd", name=f"logit_kd_T{args.temperature}_A{args.alpha}")
    
    # Load models
    print("Loading teacher model (Meditron-7B)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "epfl-llm/meditron-7b",
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    teacher_model.eval()
    
    print("Loading student model (Qwen2-1.5B)...")
    student_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B",
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    student_model = prepare_model_for_kbit_training(student_model)
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()
    
    # Load dataset
    print("Loading training data...")
    dataset = load_dataset('json', data_files=args.train_file)
    
    def format_example(example):
        prompt = f"Question: {example['question']}\nOptions: {example['options']}\nAnswer: "
        full_text = prompt + example['teacher_response']
        encoded = tokenizer(
            full_text,
            max_length=args.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'labels': encoded['input_ids'][0]
        }
    
    train_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Setup optimizer
    optimizer = AdamW(student_model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"\nStarting training with α={args.alpha}, T={args.temperature}")
    print(f"Total steps: {len(train_dataloader) * args.epochs}")
    
    student_model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        epoch_loss = 0
        epoch_kd_loss = 0
        epoch_ce_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # Student forward
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits
            
            # Calculate loss
            loss, kd_val, ce_val = logit_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                alpha=args.alpha,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_kd_loss += kd_val
            epoch_ce_loss += ce_val
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kd': f'{kd_val:.4f}',
                'ce': f'{ce_val:.4f}'
            })
            
            # Log to wandb
            if args.use_wandb and global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/kd_loss': kd_val,
                    'train/ce_loss': ce_val,
                    'train/step': global_step
                })
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = f"{args.output_dir}/checkpoint-{global_step}"
                student_model.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved to {checkpoint_dir}")
        
        # End of epoch summary
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kd = epoch_kd_loss / len(train_dataloader)
        avg_ce = epoch_ce_loss / len(train_dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg KD Loss: {avg_kd:.4f}")
        print(f"  Avg CE Loss: {avg_ce:.4f}")
        
        # Save epoch checkpoint
        epoch_dir = f"{args.output_dir}/epoch-{epoch+1}"
        student_model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Model saved to {epoch_dir}")
    
    # Final save
    print(f"\nTraining complete! Final model saved to {args.output_dir}/final")
    student_model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student model with Logit KD")
    
    # Model arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    
    # KD arguments
    parser.add_argument("--alpha", type=float, default=0.5, help="KD loss weight (0-1)")
    parser.add_argument("--temperature", type=float, default=3.0, help="Softmax temperature")
    
    # Training arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./models/logit_kd", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases tracking")
    
    args = parser.parse_args()
    main(args)
```

---

## How to Run

```bash
# Single run with default settings
python train_logit_kd.py \
    --train_file data/processed/train.jsonl \
    --output_dir models/students/logit_kd/T3_A0.5 \
    --alpha 0.5 \
    --temperature 3.0 \
    --epochs 3 \
    --batch_size 4 \
    --use_wandb

# Parameter sweep (try different combinations)
for temp in 2 3 4; do
    for alpha in 0.3 0.5 0.7; do
        python train_logit_kd.py \
            --train_file data/processed/train.jsonl \
            --output_dir models/students/logit_kd/T${temp}_A${alpha} \
            --alpha $alpha \
            --temperature $temp \
            --epochs 3
    done
done
```

---

## Key Hyperparameters to Tune

| Parameter | What it does | Recommended values | Notes |
|-----------|--------------|-------------------|--------|
| **alpha** | Balance KD vs regular loss | 0.3, 0.5, 0.7 | Higher = copy teacher more |
| **temperature** | Soften probabilities | 2, 3, 4 | Higher = learn more from teacher |
| **lora_rank** | Adapter size | 16, 32 | Higher = more capacity, more memory |
| **learning_rate** | How fast to learn | 1e-4 to 5e-4 | Too high = unstable, too low = slow |
| **batch_size** | Examples per step | 4, 8, 16 | Limited by GPU memory |

---

## Troubleshooting

### Problem: Out of Memory (OOM)
**Solutions:**
- Reduce batch size (4 → 2 → 1)
- Reduce max_length (1024 → 768 → 512)
- Reduce lora_rank (32 → 16 → 8)
- Use gradient checkpointing (add to model config)

### Problem: Loss not decreasing
**Solutions:**
- Lower learning rate (2e-4 → 1e-4)
- Check data format is correct
- Try different alpha values
- Make sure teacher logits are computed correctly

### Problem: Training too slow
**Solutions:**
- Increase batch size (use gradient accumulation)
- Reduce max_length
- Use mixed precision (already enabled with fp16)
- Profile code to find bottlenecks

---

## What Makes This Better Than Normal Training?

1. **Richer Signal**: Student learns from teacher's full probability distribution, not just hard labels
2. **Better Generalization**: Student learns which wrong answers are "less wrong"
3. **Softer Targets**: Temperature smooths probabilities, making them more informative
4. **Transfer Learning**: Student inherits teacher's knowledge more effectively

---

## Next Steps

After training:
1. Evaluate on MedQA/MedMCQA benchmarks
2. Compare against SFT baseline
3. Measure KL divergence to teacher
4. Run ablation studies (vary α and T)
5. Check if lower KL → better accuracy

Good luck with your implementation! 🚀
