#!/usr/bin/env python3
"""
(C) 2025. Bryan Zhao, Federico Baldan, Tim Avilov, and Shreyan Mitra
Written for CSE 493S: Advanced Topics in Machine Learning Course at the University of Washington, Seattle

Training Script for Medical LLM Distillation

This script orchestrates the complete distillation training pipeline including:
- Teacher/student model setup
- Dataset preparation
- Training with selected distillation method
- Evaluation on medical benchmarks
- Model persistence and checkpointing

Usage:
    python train_distillation.py --method sft --output_dir ./outputs/sft_run1
    python train_distillation.py --method logit_kd --alpha 0.5 --temperature 3.0
    python train_distillation.py --method cot --num_rationales 3
    python train_distillation.py --method dpo --beta 0.1
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# Import our custom modules
from DataLoader import create_train_val_dataloaders
from DistillationMethods import create_distillation_method

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class TrainingConfig:
    """Configuration class for training hyperparameters."""

    def __init__(self, args: argparse.Namespace):
        """Initialize config from command line arguments."""

        # Model paths
        self.teacher_model_name: str = args.teacher_model
        self.student_model_name: str = args.student_model

        # Data paths
        self.train_data_path: str = args.train_data
        self.val_data_path: str = args.val_data

        # Output paths
        self.output_dir: str = args.output_dir
        self.checkpoint_dir: str = os.path.join(self.output_dir, "checkpoints")
        self.results_dir: str = os.path.join(self.output_dir, "results")

        # Training hyperparameters
        self.distillation_method: str = args.method
        self.num_epochs: int = args.num_epochs
        self.batch_size: int = args.batch_size
        self.gradient_accumulation_steps: int = args.gradient_accumulation_steps
        self.learning_rate: float = args.learning_rate
        self.weight_decay: float = args.weight_decay
        self.warmup_steps: int = args.warmup_steps
        self.max_grad_norm: float = args.max_grad_norm
        self.max_length: int = args.max_length

        # LoRA/QLoRA hyperparameters
        self.use_lora: bool = args.use_lora
        self.lora_rank: int = args.lora_rank
        self.lora_alpha: int = args.lora_alpha
        self.lora_dropout: float = args.lora_dropout
        self.use_quantization: bool = args.use_quantization

        # Method-specific hyperparameters
        self.alpha: float = args.alpha  # For Logit-KD
        self.temperature: float = args.temperature  # For Logit-KD
        self.beta: float = args.beta  # For DPO
        self.num_rationales: int = args.num_rationales  # For CoT
        self.sampling_temperature: float = args.sampling_temperature  # For CoT
        self.cot_prompt: str = args.cot_prompt  # For CoT
        self.max_new_tokens: int = args.max_new_tokens

        # Training settings
        self.seed: int = args.seed
        self.save_steps: int = args.save_steps
        self.eval_steps: int = args.eval_steps
        self.logging_steps: int = args.logging_steps
        self.num_workers: int = args.num_workers

        # Device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ==============================================================================
# MODEL SETUP
# ==============================================================================

def setup_quantization_config() -> BitsAndBytesConfig:
    """
    Configure 8-bit quantization for memory-efficient model loading.

    :returns: Quantization configuration
    :rtype: BitsAndBytesConfig
    """
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )


def load_teacher_model(
    model_name: str,
    use_quantization: bool = True
) -> nn.Module:
    """
    Load teacher model with optional quantization.

    :param model_name: HuggingFace model identifier
    :type model_name: str
    :param use_quantization: Whether to use 8-bit quantization
    :type use_quantization: bool
    :returns: Loaded teacher model
    :rtype: nn.Module
    """
    logger.info(f"Loading teacher model: {model_name}")

    if use_quantization:
        quantization_config = setup_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    model.eval()
    logger.info("Teacher model loaded successfully")
    return model


def load_student_model(
    model_name: str,
    use_lora: bool = True,
    lora_config: Optional[Dict[str, Any]] = None,
    use_quantization: bool = True
) -> nn.Module:
    """
    Load student model with optional LoRA adapters.

    :param model_name: HuggingFace model identifier
    :type model_name: str
    :param use_lora: Whether to use LoRA fine-tuning
    :type use_lora: bool
    :param lora_config: LoRA configuration parameters
    :type lora_config: Optional[Dict[str, Any]]
    :param use_quantization: Whether to use quantization (QLoRA)
    :type use_quantization: bool
    :returns: Loaded student model
    :rtype: nn.Module
    """
    logger.info(f"Loading student model: {model_name}")

    if use_quantization:
        quantization_config = setup_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    if use_lora:
        logger.info("Applying LoRA adapters")
        if lora_config is None:
            lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'bias': 'none',
                'task_type': 'CAUSAL_LM',
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            }

        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    logger.info("Student model loaded successfully")
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load tokenizer and configure special tokens.

    :param model_name: HuggingFace model identifier
    :type model_name: str
    :returns: Configured tokenizer
    :rtype: AutoTokenizer
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Tokenizer loaded successfully")
    return tokenizer


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

class Trainer:
    """
    Main training class for distillation methods.

    Handles training loop, evaluation, checkpointing, and logging.
    """

    def __init__(
        self,
        config: TrainingConfig,
        distillation_method: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ):
        """Initialize trainer."""
        self.config = config
        self.distillation_method = distillation_method
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Method: {self.distillation_method.get_method_name()}")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(
            f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(
            f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*80}")

            # Training phase
            train_metrics = self.train_epoch(epoch)

            # Validation phase
            val_metrics = self.evaluate()

            # Log epoch summary
            self.log_epoch_summary(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, is_best=True)
                logger.info(
                    f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.save_checkpoint(epoch, is_best=False)

        # Save final model
        self.save_final_model()
        self.save_training_history()
        logger.info("Training completed!")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        :param epoch: Current epoch number
        :type epoch: int
        :returns: Dictionary of training metrics
        :rtype: Dict[str, float]
        """
        self.distillation_method.student_model.train()

        epoch_losses = []
        epoch_metrics = {}

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            leave=True
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Compute loss
            loss, metrics = self.distillation_method.compute_loss(batch)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # Accumulate metrics
            epoch_losses.append(
                loss.item() * self.config.gradient_accumulation_steps)
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.distillation_method.student_model.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{epoch_losses[-1]:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # Periodic logging
                if self.global_step % self.config.logging_steps == 0:
                    self.log_training_step(
                        epoch, step, epoch_losses[-1], metrics)

                # Periodic evaluation
                if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate()
                    logger.info(
                        f"Step {self.global_step} - Val Loss: {val_metrics['val_loss']:.4f}")
                    self.distillation_method.student_model.train()

        # Compute epoch averages
        avg_metrics = {
            'train_loss': np.mean(epoch_losses),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        for key, values in epoch_metrics.items():
            avg_metrics[f'train_{key}'] = np.mean(values)

        return avg_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        :returns: Dictionary of validation metrics
        :rtype: Dict[str, float]
        """
        self.distillation_method.student_model.eval()

        val_losses = []
        val_metrics = {}

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                loss, metrics = self.distillation_method.compute_loss(batch)

                val_losses.append(loss.item())
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)

        # Compute averages
        avg_metrics = {'val_loss': np.mean(val_losses)}
        for key, values in val_metrics.items():
            avg_metrics[f'val_{key}'] = np.mean(values)

        return avg_metrics

    def log_training_step(self, epoch: int, step: int, loss: float, metrics: Dict[str, float]):
        """Log training step metrics."""
        log_str = f"Epoch {epoch + 1}, Step {step + 1}, Global Step {self.global_step}: "
        log_str += f"Loss = {loss:.4f}, LR = {self.scheduler.get_last_lr()[0]:.2e}"
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f", {key} = {value:.4f}"
        logger.info(log_str)

    def log_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch summary."""
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")

        # Store in history
        self.training_history['train_loss'].append(train_metrics['train_loss'])
        self.training_history['val_loss'].append(val_metrics['val_loss'])
        self.training_history['learning_rate'].append(
            train_metrics['learning_rate'])

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        :param epoch: Current epoch number
        :type epoch: int
        :param is_best: Whether this is the best model so far
        :type is_best: bool
        """
        checkpoint_name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, checkpoint_name)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.distillation_method.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def save_final_model(self):
        """Save final trained model."""
        final_model_path = os.path.join(self.config.output_dir, "final_model")

        # Save student model
        self.distillation_method.student_model.save_pretrained(
            final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(
            self.config.results_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main training function."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train medical LLM using knowledge distillation")

    # Model arguments
    parser.add_argument('--teacher_model', type=str, default='epfl-llm/meditron-7b',
                        help='Teacher model name or path')
    parser.add_argument('--student_model', type=str, default='Qwen/Qwen2-1.5B',
                        help='Student model name or path')

    # Data arguments
    parser.add_argument('--train_data', type=str,
                        default='augmented_data/augmented_MedMcqa/train_aug.json',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str,
                        default='data/MedMcqa_data/dev.json',
                        help='Path to validation data')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/default_run',
                        help='Output directory for checkpoints and results')

    # Training arguments
    parser.add_argument('--method', type=str, default='sft',
                        choices=['sft', 'logit_kd', 'cot', 'dpo'],
                        help='Distillation method to use')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')

    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--use_quantization', action='store_true', default=True,
                        help='Use 8-bit quantization (QLoRA)')

    # Method-specific arguments
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for Logit-KD (weight for KD loss)')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for Logit-KD')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta for DPO')
    parser.add_argument('--num_rationales', type=int, default=1,
                        help='Number of rationales for CoT distillation')
    parser.add_argument('--sampling_temperature', type=float, default=0.7,
                        help='Sampling temperature for CoT diverse rationales')
    parser.add_argument('--cot_prompt', type=str, default="Let's think step by step:",
                        help='Chain-of-thought prompt')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum new tokens to generate')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps (0 to disable)')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Initialize config
    config = TrainingConfig(args)

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")

    # Save configuration
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Load tokenizer
    tokenizer = load_tokenizer(config.student_model_name)

    # Load models
    teacher_model = load_teacher_model(
        config.teacher_model_name,
        use_quantization=config.use_quantization
    )

    lora_config_dict = {
        'r': config.lora_rank,
        'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout,
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    } if config.use_lora else None

    student_model = load_student_model(
        config.student_model_name,
        use_lora=config.use_lora,
        lora_config=lora_config_dict,
        use_quantization=config.use_quantization
    )

    # Create distillation method configuration
    distillation_config = {
        'max_new_tokens': config.max_new_tokens,
    }

    if config.distillation_method in ['logit_kd', 'logit']:
        distillation_config['alpha'] = config.alpha
        distillation_config['temperature'] = config.temperature
    elif config.distillation_method in ['cot', 'chain_of_thought']:
        distillation_config['cot_prompt'] = config.cot_prompt
        distillation_config['num_rationales'] = config.num_rationales
        distillation_config['sampling_temperature'] = config.sampling_temperature
        distillation_config['max_new_tokens'] = 512  # Longer for reasoning
    elif config.distillation_method in ['dpo', 'preference']:
        distillation_config['beta'] = config.beta

    # Initialize distillation method
    distillation_method = create_distillation_method(
        method_name=config.distillation_method,
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        config=distillation_config
    )

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_train_val_dataloaders(
        train_path=config.train_data_path,
        val_path=config.val_data_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        distillation_method=config.distillation_method,
        num_workers=config.num_workers
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Setup scheduler
    total_steps = len(train_dataloader) * \
        config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Total training steps: {total_steps}")

    # Initialize trainer
    trainer = Trainer(
        config=config,
        distillation_method=distillation_method,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Start training
    trainer.train()

    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
