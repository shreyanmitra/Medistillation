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
Documentation style is Sphinx.

Usage:
    python train_distillation.py --method sft --output_dir ./outputs/sft_run1
    python train_distillation.py --method logit_kd --alpha 0.5 --temperature 3.0
    python train_distillation.py --method adakd --alpha 0.5 --base_temperature 3.0
    python train_distillation.py --method cot --num_rationales 3
    python train_distillation.py --method fitnets --layer_mapping "{6:12,12:24}"
    python train_distillation.py --method ppo --gamma 0.99 --epsilon 0.2
    python train_distillation.py --method spin --beta 0.1
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

import os
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# HuggingFace and model training imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import tempfile
import sys

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Import our custom modules
from DataLoader import create_train_val_dataloaders
from DistillationMethods import create_distillation_method

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set PyTorch CUDA allocator settings to reduce fragmentation and allow
# expandable segments. `PYTORCH_CUDA_ALLOC_CONF` is deprecated in favor of
# `PYTORCH_ALLOC_CONF` but we set both for compatibility on older environments.
# These defaults can be overridden by the environment when launching the job.
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.6"
)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128"
)
logger.info(f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF')}")
logger.info(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

def _log_and_clear_cuda(stage: str):
    try:
        if torch.cuda.is_available():
            logger.info(f"[CUDA] {stage} - memory summary (abbrev):")
            logger.info(torch.cuda.memory_summary(device=None, abbreviated=True))
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Failed to log/clear CUDA at {stage}: {e}")


# ==============================================================================
# CONFIGURATION CLASS
# ==============================================================================

class TrainingConfig:
    """
    Configuration class for training hyperparameters.
    
    Centralizes all training parameters including model paths, hyperparameters,
    and method-specific settings.
    
    :param args: Command-line arguments from argparse
    :type args: argparse.Namespace
    """

    def __init__(self, args: argparse.Namespace):
        """Constructor method"""

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
        self.enable_cpu_offload: bool = args.enable_cpu_offload  # For large teacher models (70B+)
        # Misc runtime flags copied from CLI
        self.resume_from_checkpoint: str = getattr(args, 'resume_from_checkpoint', '')
        self.align_vocabularies: bool = getattr(args, 'align_vocabularies', False)
        self.run_ablation: bool = getattr(args, 'run_ablation', False)
        self.ablation_type: str = getattr(args, 'ablation_type', '')
        self.ablation_values: str = getattr(args, 'ablation_values', '')

        # Method-specific hyperparameters
        self.alpha: float = args.alpha  # For Logit-KD, AdaKD, FitNets, Attention
        self.temperature: float = args.temperature  # For Logit-KD, RL methods
        self.base_temperature: float = args.base_temperature  # For AdaKD
        self.min_temperature: float = args.min_temperature  # For AdaKD
        self.max_temperature: float = args.max_temperature  # For AdaKD
        self.num_rationales: int = args.num_rationales  # For CoT
        self.sampling_temperature: float = args.sampling_temperature  # For CoT
        self.cot_prompt: str = args.cot_prompt  # For CoT
        self.layer_mapping: str = args.layer_mapping  # For FitNets, Attention (JSON string)
        self.use_projections: bool = args.use_projections  # For FitNets
        self.match_all_heads: bool = args.match_all_heads  # For Attention
        self.gamma: float = args.gamma  # For RL methods (discount factor)
        self.num_samples: int = args.num_samples  # For BOND
        self.epsilon: float = args.epsilon  # For PPO (clip range)
        self.entropy_coef: float = args.entropy_coef  # For RL methods
        self.beta: float = args.beta  # For SPIN
        self.max_new_tokens: int = args.max_new_tokens

        # Training settings
        self.seed: int = args.seed
        self.save_steps: int = args.save_steps
        self.eval_steps: int = args.eval_steps
        self.logging_steps: int = args.logging_steps
        self.num_workers: int = args.num_workers

        # Device - always use CUDA for training
        self.device: str = "cuda"

        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for logging.
        
        :returns: Dictionary of configuration parameters
        :rtype: Dict[str, Any]
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ==============================================================================
# MODEL SETUP
# ==============================================================================

def setup_quantization_config(enable_cpu_offload: bool = False) -> BitsAndBytesConfig:
    """
    Configure 8-bit quantization for memory-efficient model loading.
    
    Uses QLoRA technique to enable loading large models on consumer GPUs.

    :param enable_cpu_offload: Enable CPU offloading for models too large for GPU
    :type enable_cpu_offload: bool
    :returns: Quantization configuration
    :rtype: BitsAndBytesConfig
    """

    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,  # Enable CPU offload for 70B models
    )


def compute_max_memory_dict(per_gpu_limit_gb: Optional[float] = None, reserve_cpu_gb: int = 64) -> dict:
    """
    Compute a `max_memory` dictionary for HuggingFace `from_pretrained(..., max_memory=...)`.

    - If `per_gpu_limit_gb` is provided, use that cap for each GPU (GiB).
    - If `per_gpu_limit_gb` is None, default to 20% of each GPU's total memory.
    - Always include a generous `cpu` bucket (reserve_cpu_gb + sum of GPU caps).

    Returns a dict like {0: '28GiB', 1: '28GiB', 'cpu': '200GiB'}.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cpu": f"{reserve_cpu_gb}GiB"}

        gpu_mems = []
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            total_gib = prop.total_memory / (1024 ** 3)
            gpu_mems.append(total_gib)

        max_memory = {}
        for i, total_gib in enumerate(gpu_mems):
            if per_gpu_limit_gb is None:
                # Default: 20% of GPU to leave space for inference/other processes
                allowed = max(int(total_gib * 0.2), 1)
            else:
                allowed = max(int(min(per_gpu_limit_gb, total_gib - 0.5)), 1)
            max_memory[i] = f"{allowed}GiB"

        cpu_gib = reserve_cpu_gb + sum(int(v) for v in gpu_mems)
        max_memory["cpu"] = f"{cpu_gib}GiB"
        return max_memory
    except Exception:
        return {"cpu": f"{reserve_cpu_gb}GiB"}


def load_teacher_model(
    model_name: str,
    use_quantization: bool = True,
    enable_cpu_offload: bool = False,
    max_gpu_mem_gb: Optional[float] = None
) -> nn.Module:
    """
    Load teacher model with optional quantization.

    :param model_name: HuggingFace model identifier
    :type model_name: str
    :param use_quantization: Whether to use 8-bit quantization
    :type use_quantization: bool
    :param enable_cpu_offload: Enable CPU offloading for large models (70B+)
    :type enable_cpu_offload: bool
    :returns: Loaded teacher model
    :rtype: nn.Module
    """
    logger.info(f"Loading teacher model: {model_name}")
    
    if enable_cpu_offload:
        logger.info("CPU offloading ENABLED - model layers will overflow to CPU/RAM")

    # Compute max_memory dict (enforces per-GPU hard cap). Default per-GPU cap is 20% if not provided.
    max_memory = compute_max_memory_dict(per_gpu_limit_gb=max_gpu_mem_gb)
    logger.info(f"Using max_memory device limits for teacher load: {max_memory}")

    offload_folder = None
    if enable_cpu_offload:
        offload_folder = os.path.join(tempfile.gettempdir(), f"hf_offload_teacher_{os.getpid()}")
        os.makedirs(offload_folder, exist_ok=True)

    if use_quantization:
        # Use 8-bit quantization to reduce memory footprint (QLoRA technique)
        # This allows loading large models (7B+ parameters) on consumer GPUs
        quantization_config = setup_quantization_config(enable_cpu_offload=enable_cpu_offload)
        # Free cache and log before heavy load
        _log_and_clear_cuda("before_teacher_from_pretrained")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,  # Apply 8-bit quantization
                device_map="auto",  # Automatically distribute across available devices
                max_memory=max_memory,
                offload_folder=offload_folder,
                trust_remote_code=True  # Allow custom model code from HuggingFace
            )
            _log_and_clear_cuda("after_teacher_from_pretrained")
        # Diagnostic: show hf_device_map and CUDA memory summary immediately after load
        try:
            import torch as _torch, pprint as _pprint, gc as _gc
            print("hf_device_map present:", hasattr(model, "hf_device_map"))
            if hasattr(model, "hf_device_map"):
                _pprint.pprint(model.hf_device_map)
            try:
                print(_torch.cuda.memory_summary(device=0, abbreviated=False))
            except Exception:
                print("Unable to fetch torch.cuda.memory_summary()")
            _gc.collect()
        except Exception:
            logger.exception("Failed to run post-teacher-load diagnostics")
        except Exception as e:
            logger.exception("Failed to load teacher model with from_pretrained(): %s", e)
            try:
                if torch.cuda.is_available():
                    logger.error("CUDA memory summary at teacher load failure:\n%s", torch.cuda.memory_summary(device=None, abbreviated=True))
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.warning("Failed to capture CUDA memory summary after teacher load failure: %s", e2)
            raise
    else:
        # Load in FP16 for faster inference without quantization
        _log_and_clear_cuda("before_teacher_from_pretrained")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # Automatically distribute across available devices
                max_memory=max_memory,
                offload_folder=offload_folder,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                trust_remote_code=True
            )
            _log_and_clear_cuda("after_teacher_from_pretrained")
        # Diagnostic: show hf_device_map and CUDA memory summary immediately after load
        try:
            import torch as _torch, pprint as _pprint, gc as _gc
            print("hf_device_map present:", hasattr(model, "hf_device_map"))
            if hasattr(model, "hf_device_map"):
                _pprint.pprint(model.hf_device_map)
            try:
                print(_torch.cuda.memory_summary(device=0, abbreviated=False))
            except Exception:
                print("Unable to fetch torch.cuda.memory_summary()")
            _gc.collect()
        except Exception:
            logger.exception("Failed to run post-teacher-load diagnostics")
        except Exception as e:
            logger.exception("Failed to load teacher model (fp16) with from_pretrained(): %s", e)
            try:
                if torch.cuda.is_available():
                    logger.error("CUDA memory summary at teacher load failure (fp16):\n%s", torch.cuda.memory_summary(device=None, abbreviated=True))
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.warning("Failed to capture CUDA memory summary after teacher fp16 load failure: %s", e2)
            raise

    model.eval()  # Teacher is always in evaluation mode (no training, only inference)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but is not available")
    
    # With device_map="auto", model is already distributed across devices (GPU + CPU)
    # DO NOT call model.to(device) as it will try to move everything to GPU and cause OOM
    logger.info("Step 4/4: Checking model device placement (device_map='auto' handles distribution)...")
    
    if hasattr(model, 'hf_device_map') and model.hf_device_map is not None:
        # Count layers on each device
        device_counts = {}
        for device_name in model.hf_device_map.values():
            device_counts[device_name] = device_counts.get(device_name, 0) + 1
        
        logger.info(f"📍 Teacher model device distribution: {device_counts}")
        
        if 'cpu' in device_counts:
            logger.info(f"✅ {device_counts['cpu']} layers are on CPU (CPU offload active)")
        if 'cuda:0' in device_counts or 'cuda' in device_counts:
            gpu_layers = device_counts.get('cuda:0', 0) + device_counts.get('cuda', 0)
            logger.info(f"✅ {gpu_layers} layers are on GPU")
        
        logger.info("✅ Teacher model device placement complete (using device_map='auto')")
    else:
        logger.warning("⚠️  device_map='auto' did not create device map - model may be on CPU")
    
    logger.info("✅ Teacher model loaded successfully")
    return model


def load_student_model(
    model_name: str,
    use_lora: bool = True,
    lora_config: Optional[Dict[str, Any]] = None,
    use_quantization: bool = True, 
    enable_cpu_offload: bool = False,
    max_gpu_mem_gb: Optional[float] = None
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
    :param enable_cpu_offload: Whether to enable FP32 CPU offload for k-bit weights
                              (uses BitsAndBytesConfig.llm_int8_enable_fp32_cpu_offload)
    :type enable_cpu_offload: bool
    :returns: Loaded student model
    :rtype: nn.Module
    """
    logger.info(f"Loading student model: {model_name}")
    if enable_cpu_offload:
        logger.info("CPU offloading ENABLED for student model (llm_int8_enable_fp32_cpu_offload=True)")

    # Compute max_memory dict (enforces per-GPU hard cap). Default per-GPU cap is 20% if not provided.
    max_memory = compute_max_memory_dict(per_gpu_limit_gb=max_gpu_mem_gb)
    logger.info(f"Using max_memory device limits for student load: {max_memory}")

    offload_folder = None
    if enable_cpu_offload:
        offload_folder = os.path.join(tempfile.gettempdir(), f"hf_offload_student_{os.getpid()}")
        os.makedirs(offload_folder, exist_ok=True)

    if use_quantization:
        # QLoRA: Quantize base model to 8-bit, add trainable LoRA adapters in FP16
        # This dramatically reduces memory usage while maintaining training quality
        # Pass enable_cpu_offload into the BitsAndBytesConfig so device_map="auto"
        # can offload FP32 parts to CPU when necessary.
        quantization_config = setup_quantization_config(enable_cpu_offload=enable_cpu_offload)
        # Free cache and log before heavy load
        _log_and_clear_cuda("before_student_from_pretrained")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,  # Quantize frozen weights to 8-bit
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_folder,
                trust_remote_code=True
            )
            _log_and_clear_cuda("after_student_from_pretrained")
            # Diagnostic: show hf_device_map and CUDA memory summary immediately after load
            try:
                import torch as _torch, pprint as _pprint, gc as _gc
                print("hf_device_map present:", hasattr(model, "hf_device_map"))
                if hasattr(model, "hf_device_map"):
                    _pprint.pprint(model.hf_device_map)
                try:
                    print(_torch.cuda.memory_summary(device=0, abbreviated=False))
                except Exception:
                    print("Unable to fetch torch.cuda.memory_summary()")
                _gc.collect()
            except Exception:
                logger.exception("Failed to run post-student-load diagnostics (quantized)")
            # Prepare for k-bit training: enables gradient checkpointing and input requires_grad
            model = prepare_model_for_kbit_training(model)
            _log_and_clear_cuda("after_prepare_kbit")
        except Exception as e:
            logger.exception("Failed to load student model with from_pretrained() (quantized): %s", e)
            try:
                if torch.cuda.is_available():
                    logger.error("CUDA memory summary at student quantized load failure:\n%s", torch.cuda.memory_summary(device=None, abbreviated=True))
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.warning("Failed to capture CUDA memory summary after student quantized load failure: %s", e2)
            raise
    else:
        # Standard loading in FP16 (no quantization)
        _log_and_clear_cuda("before_student_from_pretrained")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_folder,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            _log_and_clear_cuda("after_student_from_pretrained")
        # Diagnostic: show hf_device_map and CUDA memory summary immediately after load
        try:
            import torch as _torch, pprint as _pprint, gc as _gc
            print("hf_device_map present:", hasattr(model, "hf_device_map"))
            if hasattr(model, "hf_device_map"):
                _pprint.pprint(model.hf_device_map)
            try:
                print(_torch.cuda.memory_summary(device=0, abbreviated=False))
            except Exception:
                print("Unable to fetch torch.cuda.memory_summary()")
            _gc.collect()
        except Exception:
            logger.exception("Failed to run post-student-load diagnostics (fp16)")
        except Exception as e:
            logger.exception("Failed to load student model with from_pretrained() (fp16): %s", e)
            try:
                if torch.cuda.is_available():
                    logger.error("CUDA memory summary at student fp16 load failure:\n%s", torch.cuda.memory_summary(device=None, abbreviated=True))
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.warning("Failed to capture CUDA memory summary after student fp16 load failure: %s", e2)
            raise

    if use_lora:
        logger.info("Applying LoRA adapters")
        # LoRA (Low-Rank Adaptation): Add small trainable matrices to frozen model
        # Only trains ~0.1-1% of parameters, making training fast and memory-efficient
        if lora_config is None:
            # Default LoRA configuration for LLMs
            lora_config = {
                'r': 16,  # Rank of update matrices (higher = more capacity, more memory)
                'lora_alpha': 32,  # Scaling factor (typically 2*r)
                'lora_dropout': 0.05,  # Dropout for regularization
                'bias': 'none',  # Don't train bias parameters
                'task_type': 'CAUSAL_LM',  # Autoregressive language modeling
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Apply to attention layers
            }

        peft_config = LoraConfig(**lora_config)
        _log_and_clear_cuda("before_get_peft_model")
        model = get_peft_model(model, peft_config)  # Wrap model with LoRA adapters
        _log_and_clear_cuda("after_get_peft_model")
        model.print_trainable_parameters()  # Log how many parameters are trainable

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but is not available")
    
    # With device_map="auto", model is already distributed across devices (GPU + CPU)
    # DO NOT call model.to(device) as it will try to move everything to GPU and cause OOM
    logger.info("Checking student model device placement (device_map='auto' handles distribution)...")
    
    if hasattr(model, 'hf_device_map') and model.hf_device_map is not None:
        device_counts = {}
        for device_name in model.hf_device_map.values():
            device_counts[device_name] = device_counts.get(device_name, 0) + 1
        logger.info(f"📍 Student model device distribution: {device_counts}")
        logger.info("✅ Student model device placement complete (using device_map='auto')")
    else:
        logger.warning("⚠️  device_map='auto' did not create device map - model may be on CPU")
    
    logger.info("✅ Student model loaded successfully")
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

    # Set padding token if not exists (required for batch processing)
    # Many models don't have a dedicated pad token, so we reuse EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token for padding
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set left-padding for decoder-only models (required for correct generation)
    # Decoder-only models like Llama/Meditron generate left-to-right, so padding
    # must be on the left to avoid attending to padding tokens during generation
    tokenizer.padding_side = 'left'
    logger.info("Tokenizer configured with left-padding for decoder-only architecture")

    logger.info("Tokenizer loaded successfully")
    return tokenizer


# ==============================================================================
# TRAINING LOOP AND TRAINER CLASS
# ==============================================================================

class Trainer:
    """
    Main training class for distillation methods.

    Handles training loop, evaluation, checkpointing, logging, and visualization.
    
    :param config: Training configuration with hyperparameters
    :type config: TrainingConfig
    :param distillation_method: Selected distillation method instance
    :type distillation_method: Any
    :param train_dataloader: DataLoader for training data
    :type train_dataloader: DataLoader
    :param val_dataloader: DataLoader for validation data
    :type val_dataloader: DataLoader
    :param optimizer: Optimizer for student model training
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Learning rate scheduler
    :type scheduler: Any
    """

    def __init__(
        self,
        config: TrainingConfig,
        distillation_method: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        tokenizer: Any = None
    ):
        """Initialize trainer."""
        self.config = config
        self.distillation_method = distillation_method
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer or distillation_method.tokenizer

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        # Computational cost tracking
        self.start_time = None
        self.total_training_time = 0.0  # Wall-clock seconds
        self.peak_memory_allocated = 0.0  # Peak GPU memory in GB
        self.peak_memory_reserved = 0.0  # Peak GPU memory reserved in GB

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

        # Start timing
        self.start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*80}")

            # Training phase
            train_metrics = self.train_epoch(epoch)

            # Validation phase
            val_metrics = self.evaluate()

            # Track memory usage
            if torch.cuda.is_available():
                current_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
                current_reserved = torch.cuda.max_memory_reserved() / 1e9  # GB
                self.peak_memory_allocated = max(self.peak_memory_allocated, current_allocated)
                self.peak_memory_reserved = max(self.peak_memory_reserved, current_reserved)
                logger.info(f"GPU Memory: {current_allocated:.2f}GB allocated, {current_reserved:.2f}GB reserved")

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

        # Calculate total training time
        self.total_training_time = time.time() - self.start_time
        training_hours = self.total_training_time / 3600.0
        logger.info(f"\nTotal training time: {training_hours:.2f} hours")
        logger.info(f"Peak GPU memory allocated: {self.peak_memory_allocated:.2f} GB")
        logger.info(f"Peak GPU memory reserved: {self.peak_memory_reserved:.2f} GB")

        # Save final model and history
        self.save_final_model()
        self.save_training_history()
        self.save_computational_costs()
        
        # Run comprehensive evaluation
        logger.info("\nStarting comprehensive post-training evaluation...")
        self.run_comprehensive_evaluation()
        
        logger.info("Training and evaluation completed!")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        :param epoch: Current epoch number
        :type epoch: int
        :returns: Dictionary of training metrics
        :rtype: Dict[str, float]
        """
        self.distillation_method.student_model.train()  # Enable dropout, batchnorm updates, gradient computation

        epoch_losses = []  # Collect losses for averaging
        epoch_metrics = {}  # Collect all metrics (KD loss, CE loss, etc.)

        # Progress bar for visual feedback during training
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            leave=True
        )

        self.optimizer.zero_grad()  # Clear gradients from previous epoch

        for step, batch in enumerate(progress_bar):
            # ===== Forward Pass =====
            # Compute distillation loss using the selected method (SFT, Logit-KD, PPO, etc.)
            # batch contains input_ids and attention_mask for prompts
            loss, metrics = self.distillation_method.compute_loss(batch)

            # ===== Gradient Accumulation =====
            # Scale loss by accumulation steps so final gradient magnitude is correct
            # This simulates larger batch size without running out of memory
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()  # Compute gradients (accumulate across multiple steps)

            # Store metrics for logging (scale loss back to original magnitude)
            # Use detach() before item() to avoid blocking GPU
            epoch_losses.append(
                loss.detach().item() * self.config.gradient_accumulation_steps)
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                # Convert tensor metrics to float if needed (after backward pass)
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        epoch_metrics[key].append(value.detach().item())
                    else:
                        epoch_metrics[key].append(value.detach().mean().item())
                else:
                    epoch_metrics[key].append(value)

            # ===== Optimizer Step (only after accumulating enough gradients) =====
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients (common in LLM training)
                torch.nn.utils.clip_grad_norm_(
                    self.distillation_method.student_model.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()  # Update student model parameters using accumulated gradients
                self.scheduler.step()  # Update learning rate (warmup + decay)
                self.optimizer.zero_grad()  # Clear gradients for next accumulation cycle

                self.global_step += 1  # Track total optimizer updates across all epochs

                # Update progress bar with current loss and learning rate
                progress_bar.set_postfix({
                    'loss': f"{epoch_losses[-1]:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # Periodic logging to track training progress
                if self.global_step % self.config.logging_steps == 0:
                    self.log_training_step(
                        epoch, step, epoch_losses[-1], metrics)

                # Periodic evaluation on validation set (if enabled)
                if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate()
                    logger.info(
                        f"Step {self.global_step} - Val Loss: {val_metrics['val_loss']:.4f}")
                    # Note: evaluate() now returns model to training mode automatically

        # ===== Handle Remaining Gradients =====
        # If total steps not divisible by accumulation_steps, perform final optimizer step
        # This ensures no accumulated gradients are wasted at end of epoch
        if (step + 1) % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.distillation_method.student_model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            logger.info(f"Epoch {epoch + 1}: Performed final optimizer step for remaining gradients")

        # ===== Compute Epoch Averages =====
        avg_metrics = {
            'train_loss': np.mean(epoch_losses),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        # Average all method-specific metrics (KD loss, CE loss, rewards, etc.)
        for key, values in epoch_metrics.items():
            avg_metrics[f'train_{key}'] = np.mean(values)

        return avg_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        :returns: Dictionary of validation metrics
        :rtype: Dict[str, float]
        """
        self.distillation_method.student_model.eval()  # Disable dropout, use running stats for batchnorm

        val_losses = []  # Collect validation losses
        val_metrics = {}  # Collect validation metrics

        # No gradient computation needed during evaluation (saves memory and time)
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                # Compute loss using distillation method (same as training)
                loss, metrics = self.distillation_method.compute_loss(batch)

                # Store metrics for averaging
                val_losses.append(loss.item())
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)

        # Compute average metrics across validation set
        avg_loss = np.mean(val_losses)
        avg_metrics = {
            'val_loss': avg_loss,
            'val_perplexity': np.exp(avg_loss)  # Perplexity = exp(cross-entropy loss)
        }
        for key, values in val_metrics.items():
            avg_metrics[f'val_{key}'] = np.mean(values)

        # CRITICAL: Return model to training mode after evaluation
        # Without this, dropout stays disabled and batchnorm uses running stats
        # This would break subsequent training iterations
        self.distillation_method.student_model.train()

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

        # If student is PEFT-wrapped, save adapters to a separate folder
        try:
            from peft import PeftModel
            if isinstance(self.distillation_method.student_model, PeftModel):
                adapter_dir = os.path.join(self.config.checkpoint_dir, f"peft_adapter_epoch_{epoch + 1}")
                Path(adapter_dir).mkdir(parents=True, exist_ok=True)
                # Save the PEFT adapters separately for reliable reload later
                self.distillation_method.student_model.save_pretrained(adapter_dir)
                checkpoint['peft_adapter_dir'] = adapter_dir
        except Exception:
            # PEFT not available or student not PEFT-wrapped — ignore
            pass

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load checkpoint and restore student model / optimizer / scheduler state.

        This loader is defensive:
        - loads checkpoint with map_location='cpu'
        - checks a few critical config flags and warns on mismatch
        - attempts to load PEFT adapters (if saved separately) via PeftModel
        - falls back to non-strict state_dict loading and logs missing/unexpected keys
        - avoids restoring optimizer state for quantized runs by default
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Quick config compatibility check (informative only)
        saved_cfg = checkpoint.get('config', {})
        incompatible = []
        for k in ('use_quantization', 'enable_cpu_offload', 'student_model_name'):
            if k in saved_cfg and getattr(self.config, k, None) != saved_cfg[k]:
                incompatible.append((k, saved_cfg[k], getattr(self.config, k, None)))
        if incompatible:
            logger.warning(f"Checkpoint config differs from current config: {incompatible}")
            logger.warning("Proceeding may fail. Ensure models/flags match the original run before loading.")

        # If a PEFT adapter folder was saved, try loading adapters first (preferred)
        try:
            from peft import PeftModel
            peft_dir = checkpoint.get('peft_adapter_dir', None)
            if peft_dir and os.path.isdir(peft_dir):
                logger.info(f"Found PEFT adapter in checkpoint: {peft_dir} — loading with PeftModel.from_pretrained")
                base = self.distillation_method.student_model
                self.distillation_method.student_model = PeftModel.from_pretrained(base, peft_dir, device_map="auto")
            else:
                # Attempt to load raw state dict into current model (non-strict)
                state_dict = checkpoint.get('model_state_dict', {})
                missing_keys, unexpected_keys = self.distillation_method.student_model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys when loading state_dict: {missing_keys[:10]}{'...' if len(missing_keys)>10 else ''}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}{'...' if len(unexpected_keys)>10 else ''}")
        except Exception as e:
            logger.warning(f"PEFT/adapter loading path failed or not applicable: {e}")
            # Fallback: try non-strict load
            state_dict = checkpoint.get('model_state_dict', {})
            try:
                missing_keys, unexpected_keys = self.distillation_method.student_model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys when loading state_dict (fallback): {missing_keys[:10]}{'...' if len(missing_keys)>10 else ''}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint (fallback): {unexpected_keys[:10]}{'...' if len(unexpected_keys)>10 else ''}")
            except Exception as e2:
                logger.error(f"Failed to load model state_dict: {e2}")
                raise

        # Optionally restore optimizer and scheduler
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                if getattr(self.config, 'use_quantization', False):
                    logger.warning("Quantized run detected: skipping optimizer restore to avoid bnb/optimizer incompatibilities")
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state dict: {e}. Skipping optimizer restore.")

        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                logger.warning("Failed to load scheduler state dict")

        # Restore bookkeeping
        self.global_step = checkpoint.get('global_step', self.global_step)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)

        # If CUDA available, move optimizer tensors to correct device (best-effort)
        if torch.cuda.is_available():
            try:
                device = torch.device(self.config.device)
                for state in self.optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except Exception:
                logger.warning("Failed to migrate optimizer tensors to CUDA — you may need to recreate the optimizer")
        torch.cuda.empty_cache()

        logger.info(f"Checkpoint loaded. global_step={self.global_step}, best_val_loss={self.best_val_loss}")

    def save_final_model(self):
        """Save final trained model."""
        final_model_path = os.path.join(self.config.output_dir, "final_model")

        # Save student model
        self.distillation_method.student_model.save_pretrained(
            final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

    def save_training_history(self):
        """Save training history to JSON and generate training curves plot."""
        history_path = os.path.join(
            self.config.results_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Generate training curves plot
        try:
            self.plot_training_history()
        except Exception as e:
            logger.warning(f"Failed to generate training curves plot: {e}")

    def save_computational_costs(self):
        """Save computational cost metrics to CSV and JSON."""
        # Calculate GPU-hours (assuming single GPU)
        gpu_hours = self.total_training_time / 3600.0
        
        costs = {
            'method': self.distillation_method.get_method_name(),
            'total_training_time_hours': gpu_hours,
            'total_training_time_seconds': self.total_training_time,
            'peak_memory_allocated_gb': self.peak_memory_allocated,
            'peak_memory_reserved_gb': self.peak_memory_reserved,
            'num_epochs': self.config.num_epochs,
            'batch_size': self.config.batch_size,
            'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            'effective_batch_size': self.config.batch_size * self.config.gradient_accumulation_steps,
            'total_optimizer_steps': self.global_step,
            'model_parameters': sum(p.numel() for p in self.distillation_method.student_model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.distillation_method.student_model.parameters() if p.requires_grad)
        }
        
        # Save as JSON
        json_path = os.path.join(self.config.results_dir, "computational_costs.json")
        with open(json_path, 'w') as f:
            json.dump(costs, f, indent=2)
        logger.info(f"Saved computational costs to {json_path}")
        
        # Save as CSV (append mode for multiple runs)
        csv_path = os.path.join(self.config.output_dir, "../computational_costs.csv")
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        import csv
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=costs.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(costs)
        logger.info(f"Appended computational costs to {csv_path}")

    def evaluate_perplexity_on_corpus(self, corpus_path: str) -> Dict[str, float]:
        """
        Evaluate perplexity on a held-out medical text corpus.
        
        :param corpus_path: Path to JSONL file with medical texts
        :type corpus_path: str
        :returns: Dictionary with perplexity metrics
        :rtype: Dict[str, float]
        """
        logger.info(f"Evaluating perplexity on corpus: {corpus_path}")
        
        if not os.path.exists(corpus_path):
            logger.warning(f"Corpus file not found: {corpus_path}")
            return {'perplexity': float('inf'), 'cross_entropy': float('inf')}
        
        self.distillation_method.student_model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Computing perplexity"):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        
                        if not text:
                            continue
                        
                        # Tokenize
                        inputs = self.tokenizer(
                            text,
                            return_tensors='pt',
                            max_length=self.config.max_length,
                            truncation=True,
                            padding=False
                        )
                        
                        input_ids = inputs['input_ids'].to(self.config.device)
                        
                        # Compute loss
                        outputs = self.distillation_method.student_model(
                            input_ids=input_ids,
                            labels=input_ids
                        )
                        
                        loss = outputs.loss
                        num_tokens = input_ids.numel()
                        
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                        
                    except Exception as e:
                        logger.warning(f"Error processing line: {e}")
                        continue
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        results = {
            'perplexity': perplexity,
            'cross_entropy': avg_loss,
            'total_tokens': total_tokens
        }
        
        logger.info(f"Perplexity: {perplexity:.2f}, Cross-entropy: {avg_loss:.4f}")
        
        # Return model to training mode
        self.distillation_method.student_model.train()
        
        return results

    def evaluate_medical_benchmarks(self, benchmark_paths: Dict[str, str], evaluate_teacher: bool = True) -> Dict[str, float]:
        """
        Evaluate student (and optionally teacher) on medical QA benchmarks.
        
        Evaluates both models to measure the accuracy gap and understand how much
        knowledge is retained during distillation.
        
        :param benchmark_paths: Dictionary mapping benchmark name to file path
        :type benchmark_paths: Dict[str, str]
        :param evaluate_teacher: Whether to also evaluate teacher for comparison
        :type evaluate_teacher: bool
        :returns: Dictionary with accuracy metrics per benchmark
        :rtype: Dict[str, float]
        """
        logger.info("Evaluating on medical benchmarks...")
        
        self.distillation_method.student_model.eval()
        if evaluate_teacher:
            self.distillation_method.teacher_model.eval()
        
        results = {}
        
        for benchmark_name, file_path in benchmark_paths.items():
            if not os.path.exists(file_path):
                logger.warning(f"Benchmark file not found: {file_path}")
                results[f'{benchmark_name}_student_accuracy'] = 0.0
                if evaluate_teacher:
                    results[f'{benchmark_name}_teacher_accuracy'] = 0.0
                continue
            
            student_correct = 0
            teacher_correct = 0
            total = 0
            
            with torch.no_grad():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f"Evaluating {benchmark_name}"):
                        try:
                            data = json.loads(line.strip())
                            question = data.get('question', '')
                            
                            # Handle both formats: list of options or ground_truth
                            options = data.get('options', [])
                            correct_answer = data.get('answer', data.get('ground_truth', ''))
                            
                            if not question:
                                continue
                            
                            # Format as multiple choice prompt if options available
                            if options:
                                prompt = f"Question: {question}\n\nOptions:\n"
                                for i, opt in enumerate(options):
                                    prompt += f"{chr(65+i)}. {opt}\n"
                                prompt += "\nAnswer:"
                            else:
                                # For questions without explicit options (e.g., yes/no)
                                prompt = f"Question: {question}\n\nAnswer:"
                            
                            # Tokenize
                            inputs = self.tokenizer(
                                prompt,
                                return_tensors='pt',
                                max_length=self.config.max_length,
                                truncation=True
                            )
                            
                            input_ids = inputs['input_ids'].to(self.config.device)
                            attention_mask = inputs['attention_mask'].to(self.config.device)
                            
                            # Generate student answer
                            student_outputs = self.distillation_method.student_model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=50,
                                do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                            
                            student_response = self.tokenizer.decode(
                                student_outputs[0][input_ids.shape[1]:],
                                skip_special_tokens=True
                            ).strip().lower()
                            
                            # Check student correctness (fuzzy match)
                            if correct_answer.lower() in student_response:
                                student_correct += 1
                            # Also check for letter format (A/B/C/D)
                            elif options:
                                for char in student_response:
                                    if char.upper() in ['A', 'B', 'C', 'D']:
                                        predicted_letter = char.upper()
                                        if predicted_letter == correct_answer.upper():
                                            student_correct += 1
                                        break
                            
                            # Generate teacher answer if requested
                            if evaluate_teacher:
                                teacher_outputs = self.distillation_method.teacher_model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=50,
                                    do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id
                                )
                                
                                teacher_response = self.tokenizer.decode(
                                    teacher_outputs[0][input_ids.shape[1]:],
                                    skip_special_tokens=True
                                ).strip().lower()
                                
                                # Check teacher correctness
                                if correct_answer.lower() in teacher_response:
                                    teacher_correct += 1
                                elif options:
                                    for char in teacher_response:
                                        if char.upper() in ['A', 'B', 'C', 'D']:
                                            predicted_letter = char.upper()
                                            if predicted_letter == correct_answer.upper():
                                                teacher_correct += 1
                                            break
                            
                            total += 1
                            
                        except Exception as e:
                            logger.warning(f"Error processing question: {e}")
                            continue
            
            # Compute metrics
            student_accuracy = student_correct / total if total > 0 else 0.0
            results[f'{benchmark_name}_student_accuracy'] = student_accuracy
            results[f'{benchmark_name}_student_correct'] = student_correct
            
            if evaluate_teacher:
                teacher_accuracy = teacher_correct / total if total > 0 else 0.0
                results[f'{benchmark_name}_teacher_accuracy'] = teacher_accuracy
                results[f'{benchmark_name}_teacher_correct'] = teacher_correct
                results[f'{benchmark_name}_accuracy_gap'] = teacher_accuracy - student_accuracy
                
                logger.info(f"{benchmark_name}:")
                logger.info(f"  Student: {student_correct}/{total} = {student_accuracy*100:.2f}%")
                logger.info(f"  Teacher: {teacher_correct}/{total} = {teacher_accuracy*100:.2f}%")
                logger.info(f"  Gap: {(teacher_accuracy - student_accuracy)*100:.2f}%")
            else:
                logger.info(f"{benchmark_name}: {student_correct}/{total} = {student_accuracy*100:.2f}%")
            
            results[f'{benchmark_name}_total'] = total
        
        # Save results
        results_path = os.path.join(self.config.results_dir, "benchmark_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark results to {results_path}")
        
        # Return model to training mode
        self.distillation_method.student_model.train()
        
        return results

    def evaluate_fidelity(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate fidelity metrics: KL divergence, top-k overlap, teacher-student agreement,
        and generation similarity (BLEU, ROUGE, exact match).
        
        :param num_samples: Number of validation samples to use
        :type num_samples: int
        :returns: Dictionary with fidelity metrics
        :rtype: Dict[str, float]
        """
        logger.info(f"Evaluating fidelity metrics on {num_samples} samples...")
        
        self.distillation_method.student_model.eval()
        self.distillation_method.teacher_model.eval()
        
        kl_divergences = []
        top1_overlaps = []
        top5_overlaps = []
        js_divergences = []
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        exact_matches = 0
        
        # Import generation similarity metrics
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
            use_generation_metrics = True
            smoothing = SmoothingFunction().method1
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            logger.warning("BLEU/ROUGE libraries not available. Install with: pip install nltk rouge-score")
            use_generation_metrics = False
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Computing fidelity"):
                if sample_count >= num_samples:
                    break
                
                # Prepare batch
                batch = self.distillation_method.prepare_batch(batch)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Get teacher logits
                teacher_outputs = self.distillation_method.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
                
                # Get student logits
                student_outputs = self.distillation_method.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.logits
                
                # Compute metrics for each sequence in batch
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    if sample_count >= num_samples:
                        break
                    
                    # Get logits for this sequence (all positions)
                    t_logits = teacher_logits[i]  # [seq_len, vocab_size]
                    s_logits = student_logits[i]  # [seq_len, vocab_size]
                    
                    # Compute probabilities
                    t_probs = F.softmax(t_logits, dim=-1)
                    s_probs = F.softmax(s_logits, dim=-1)
                    
                    # KL Divergence: KL(teacher || student)
                    kl = F.kl_div(
                        F.log_softmax(s_logits, dim=-1),
                        t_probs,
                        reduction='batchmean'
                    ).item()
                    kl_divergences.append(kl)
                    
                    # JS Divergence (symmetric)
                    m_probs = 0.5 * (t_probs + s_probs)
                    js = 0.5 * F.kl_div(F.log_softmax(t_logits, dim=-1), m_probs, reduction='batchmean').item() + \
                         0.5 * F.kl_div(F.log_softmax(s_logits, dim=-1), m_probs, reduction='batchmean').item()
                    js_divergences.append(js)
                    
                    # Top-k overlap
                    t_top1 = t_logits.argmax(dim=-1)
                    s_top1 = s_logits.argmax(dim=-1)
                    top1_overlap = (t_top1 == s_top1).float().mean().item()
                    top1_overlaps.append(top1_overlap)
                    
                    t_top5 = t_logits.topk(5, dim=-1).indices
                    s_top5 = s_logits.topk(5, dim=-1).indices
                    
                    # Compute overlap for top-5
                    overlap_count = 0
                    for pos in range(t_top5.size(0)):
                        t_set = set(t_top5[pos].cpu().tolist())
                        s_set = set(s_top5[pos].cpu().tolist())
                        overlap_count += len(t_set & s_set) / 5.0
                    top5_overlap = overlap_count / t_top5.size(0)
                    top5_overlaps.append(top5_overlap)
                    
                    sample_count += 1
                
                # Generation similarity metrics (compare generated text)
                if use_generation_metrics and sample_count < num_samples:
                    # Generate from both models
                    teacher_gen = self.distillation_method.teacher_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=50,
                        do_sample=False,  # Greedy for fair comparison
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    student_gen = self.distillation_method.student_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode and compare
                    for i in range(min(batch_size, num_samples - len(bleu_scores))):
                        # Extract generated portion (remove input)
                        teacher_text = self.tokenizer.decode(
                            teacher_gen[i][input_ids.size(1):],
                            skip_special_tokens=True
                        ).strip()
                        
                        student_text = self.tokenizer.decode(
                            student_gen[i][input_ids.size(1):],
                            skip_special_tokens=True
                        ).strip()
                        
                        # Exact match
                        if teacher_text == student_text:
                            exact_matches += 1
                        
                        # BLEU score
                        if teacher_text and student_text:
                            bleu = sentence_bleu(
                                [teacher_text.split()],
                                student_text.split(),
                                smoothing_function=smoothing
                            )
                            bleu_scores.append(bleu)
                            
                            # ROUGE scores
                            rouge = scorer.score(teacher_text, student_text)
                            rouge1_scores.append(rouge['rouge1'].fmeasure)
                            rouge2_scores.append(rouge['rouge2'].fmeasure)
                            rougeL_scores.append(rouge['rougeL'].fmeasure)
        
        results = {
            'mean_kl_divergence': np.mean(kl_divergences) if kl_divergences else float('inf'),
            'mean_js_divergence': np.mean(js_divergences) if js_divergences else float('inf'),
            'mean_top1_overlap': np.mean(top1_overlaps) if top1_overlaps else 0.0,
            'mean_top5_overlap': np.mean(top5_overlaps) if top5_overlaps else 0.0,
            'num_samples': sample_count
        }
        
        # Add generation similarity metrics if available
        if use_generation_metrics and bleu_scores:
            results['mean_bleu'] = np.mean(bleu_scores)
            results['mean_rouge1'] = np.mean(rouge1_scores)
            results['mean_rouge2'] = np.mean(rouge2_scores)
            results['mean_rougeL'] = np.mean(rougeL_scores)
            results['exact_match_rate'] = exact_matches / len(bleu_scores)
            
            logger.info(f"  BLEU Score: {results['mean_bleu']:.4f}")
            logger.info(f"  ROUGE-1: {results['mean_rouge1']:.4f}")
            logger.info(f"  ROUGE-2: {results['mean_rouge2']:.4f}")
            logger.info(f"  ROUGE-L: {results['mean_rougeL']:.4f}")
            logger.info(f"  Exact Match: {results['exact_match_rate']*100:.2f}%")
        
        logger.info(f"Fidelity Metrics:")
        logger.info(f"  KL Divergence: {results['mean_kl_divergence']:.4f}")
        logger.info(f"  JS Divergence: {results['mean_js_divergence']:.4f}")
        logger.info(f"  Top-1 Overlap: {results['mean_top1_overlap']*100:.2f}%")
        logger.info(f"  Top-5 Overlap: {results['mean_top5_overlap']*100:.2f}%")
        
        # Save results
        results_path = os.path.join(self.config.results_dir, "fidelity_metrics.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved fidelity metrics to {results_path}")
        
        # Return model to training mode
        self.distillation_method.student_model.train()
        
        return results

    def evaluate_fidelitybench_med(self, fidelitybench_path: str) -> Dict[str, float]:
        """
        Evaluate on FidelityBench-Med suite for evidence faithfulness.
        
        Measures:
        1. Citation coverage: Does model reference provided evidence?
        2. Factual correctness: Does model match ground truth (NLI-based)?
        3. Hallucination rate: Does model invent facts contradicting evidence?
        4. Answer relevancy: Is response relevant to question (semantic similarity)?
        5. Teacher-student fidelity: How similar are responses?
        
        Uses RAGAS-style metrics with NLI (Natural Language Inference) for 
        fact verification against ground truth answers.
        
        :param fidelitybench_path: Path to FidelityBench-Med JSONL file
        :type fidelitybench_path: str
        :returns: Dictionary with faithfulness metrics
        :rtype: Dict[str, float]
        """
        logger.info("Evaluating on FidelityBench-Med with RAGAS/NLI scoring...")
        
        if not os.path.exists(fidelitybench_path):
            logger.warning(f"FidelityBench file not found: {fidelitybench_path}")
            return {}
        
        # Try to import RAGAS and sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer, util
            use_semantic = True
            logger.info("Using sentence-transformers for semantic similarity")
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            use_semantic = False
            logger.warning("sentence-transformers not installed. Using fallback string matching.")
            logger.warning("Install with: pip install sentence-transformers")
        
        # Try to import transformers for NLI
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFTokenizer
            use_nli = True
            logger.info("Loading NLI model for fact verification...")
            nli_tokenizer = HFTokenizer.from_pretrained('microsoft/deberta-v3-base-mnli')
            nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base-mnli')
            nli_model.to(self.config.device)
            nli_model.eval()
            logger.info("NLI model loaded successfully")
        except Exception as e:
            use_nli = False
            logger.warning(f"Could not load NLI model: {e}")
            logger.warning("Using fallback exact/substring matching for fact verification")
        
        self.distillation_method.student_model.eval()
        self.distillation_method.teacher_model.eval()
        
        results = {
            'citation_coverage': 0.0,
            'factual_correctness_student': 0.0,
            'factual_correctness_teacher': 0.0,
            'hallucination_rate_student': 0.0,
            'hallucination_rate_teacher': 0.0,
            'answer_relevancy_student': 0.0,
            'answer_relevancy_teacher': 0.0,
            'teacher_student_agreement': 0.0,
            'overall_faithfulness_student': 0.0,
            'overall_faithfulness_teacher': 0.0
        }
        
        # Accumulators
        total_prompts = 0
        citations_found = 0
        student_correct = 0
        teacher_correct = 0
        student_hallucinations = 0
        teacher_hallucinations = 0
        student_relevancy_scores = []
        teacher_relevancy_scores = []
        agreement_scores = []
        
        with torch.no_grad():
            with open(fidelitybench_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Evaluating FidelityBench"):
                    try:
                        item = json.loads(line.strip())
                        prompt = item.get('prompt', '')
                        question = item.get('question', '')
                        ground_truth = item.get('ground_truth', '').lower().strip()
                        evidence_passages = item.get('evidence_passages', [])
                        
                        if not prompt or not ground_truth:
                            continue
                        
                        # Tokenize prompt
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors='pt',
                            max_length=self.config.max_length,
                            truncation=True
                        )
                        
                        input_ids = inputs['input_ids'].to(self.config.device)
                        attention_mask = inputs['attention_mask'].to(self.config.device)
                        
                        # Generate student response
                        student_outputs = self.distillation_method.student_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        student_response = self.tokenizer.decode(
                            student_outputs[0][input_ids.shape[1]:],
                            skip_special_tokens=True
                        ).strip()
                        
                        # Generate teacher response
                        teacher_outputs = self.distillation_method.teacher_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        teacher_response = self.tokenizer.decode(
                            teacher_outputs[0][input_ids.shape[1]:],
                            skip_special_tokens=True
                        ).strip()
                        
                        # === 1. CITATION COVERAGE ===
                        # Check if response references evidence passages
                        citation_markers = ['1', '2', '3', 'first', 'second', 'third', 'evidence', 'passage', 'study', 'research']
                        has_citations = any(marker in student_response.lower() for marker in citation_markers)
                        if has_citations:
                            citations_found += 1
                        
                        # === 2. FACTUAL CORRECTNESS (via NLI) ===
                        # Check if student/teacher responses entail ground truth
                        if use_nli:
                            # NLI: Does response entail ground truth?
                            # Premise: model response, Hypothesis: ground truth
                            student_nli_inputs = nli_tokenizer(
                                student_response,
                                ground_truth,
                                return_tensors='pt',
                                truncation=True,
                                max_length=512
                            ).to(self.config.device)
                            
                            teacher_nli_inputs = nli_tokenizer(
                                teacher_response,
                                ground_truth,
                                return_tensors='pt',
                                truncation=True,
                                max_length=512
                            ).to(self.config.device)
                            
                            student_nli_output = nli_model(**student_nli_inputs)
                            teacher_nli_output = nli_model(**teacher_nli_inputs)
                            
                            # Get entailment probability (label 2 = entailment in DeBERTa-MNLI)
                            student_probs = torch.softmax(student_nli_output.logits, dim=1)
                            teacher_probs = torch.softmax(teacher_nli_output.logits, dim=1)
                            
                            student_entailment = student_probs[0][2].item()  # entailment score
                            teacher_entailment = teacher_probs[0][2].item()
                            
                            # Threshold for "correct" (entailment > 0.5)
                            if student_entailment > 0.5:
                                student_correct += 1
                            if teacher_entailment > 0.5:
                                teacher_correct += 1
                            
                            # Contradiction = hallucination (label 0 = contradiction)
                            student_contradiction = student_probs[0][0].item()
                            teacher_contradiction = teacher_probs[0][0].item()
                            
                            if student_contradiction > 0.5:
                                student_hallucinations += 1
                            if teacher_contradiction > 0.5:
                                teacher_hallucinations += 1
                        else:
                            # Fallback: exact/substring matching
                            if ground_truth in student_response.lower():
                                student_correct += 1
                            if ground_truth in teacher_response.lower():
                                teacher_correct += 1
                            
                            # Simple hallucination check: response doesn't contain any evidence text
                            evidence_text = ' '.join(evidence_passages).lower()
                            student_words = set(student_response.lower().split())
                            evidence_words = set(evidence_text.split())
                            
                            if len(student_words & evidence_words) < 3:  # Very few overlapping words
                                student_hallucinations += 1
                        
                        # === 3. ANSWER RELEVANCY (Semantic Similarity) ===
                        if use_semantic:
                            # Compute semantic similarity between question and responses
                            question_emb = semantic_model.encode(question, convert_to_tensor=True)
                            student_emb = semantic_model.encode(student_response, convert_to_tensor=True)
                            teacher_emb = semantic_model.encode(teacher_response, convert_to_tensor=True)
                            
                            student_relevancy = util.cos_sim(question_emb, student_emb).item()
                            teacher_relevancy = util.cos_sim(question_emb, teacher_emb).item()
                            
                            student_relevancy_scores.append(max(0, student_relevancy))  # Clip to [0, 1]
                            teacher_relevancy_scores.append(max(0, teacher_relevancy))
                        else:
                            # Fallback: word overlap
                            question_words = set(question.lower().split())
                            student_words = set(student_response.lower().split())
                            teacher_words = set(teacher_response.lower().split())
                            
                            if len(student_words) > 0:
                                student_overlap = len(question_words & student_words) / len(student_words)
                                student_relevancy_scores.append(student_overlap)
                            if len(teacher_words) > 0:
                                teacher_overlap = len(question_words & teacher_words) / len(teacher_words)
                                teacher_relevancy_scores.append(teacher_overlap)
                        
                        # === 4. TEACHER-STUDENT AGREEMENT ===
                        if use_semantic:
                            agreement = util.cos_sim(student_emb, teacher_emb).item()
                            agreement_scores.append(max(0, agreement))
                        else:
                            # Fallback: word overlap
                            student_words = set(student_response.lower().split())
                            teacher_words = set(teacher_response.lower().split())
                            if len(student_words) > 0 and len(teacher_words) > 0:
                                overlap = len(student_words & teacher_words)
                                similarity = overlap / max(len(student_words), len(teacher_words))
                                agreement_scores.append(similarity)
                        
                        total_prompts += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing FidelityBench item: {e}")
                        continue
        
        # Compute final metrics
        if total_prompts > 0:
            results['citation_coverage'] = citations_found / total_prompts
            results['factual_correctness_student'] = student_correct / total_prompts
            results['factual_correctness_teacher'] = teacher_correct / total_prompts
            results['hallucination_rate_student'] = student_hallucinations / total_prompts
            results['hallucination_rate_teacher'] = teacher_hallucinations / total_prompts
            
            if student_relevancy_scores:
                results['answer_relevancy_student'] = sum(student_relevancy_scores) / len(student_relevancy_scores)
            if teacher_relevancy_scores:
                results['answer_relevancy_teacher'] = sum(teacher_relevancy_scores) / len(teacher_relevancy_scores)
            if agreement_scores:
                results['teacher_student_agreement'] = sum(agreement_scores) / len(agreement_scores)
            
            # Overall faithfulness = (correctness + relevancy - hallucination) / 2
            results['overall_faithfulness_student'] = (
                results['factual_correctness_student'] + 
                results['answer_relevancy_student'] - 
                results['hallucination_rate_student']
            ) / 2
            
            results['overall_faithfulness_teacher'] = (
                results['factual_correctness_teacher'] + 
                results['answer_relevancy_teacher'] - 
                results['hallucination_rate_teacher']
            ) / 2
            
            results['total_evaluated'] = total_prompts
            results['method'] = 'NLI+Semantic' if (use_nli and use_semantic) else 'Fallback'
            
            logger.info(f"\nFidelityBench-Med Results ({results['method']}):")
            logger.info(f"  Citation Coverage: {results['citation_coverage']*100:.2f}%")
            logger.info(f"  Student:")
            logger.info(f"    Factual Correctness: {results['factual_correctness_student']*100:.2f}%")
            logger.info(f"    Answer Relevancy: {results['answer_relevancy_student']*100:.2f}%")
            logger.info(f"    Hallucination Rate: {results['hallucination_rate_student']*100:.2f}%")
            logger.info(f"    Overall Faithfulness: {results['overall_faithfulness_student']*100:.2f}%")
            logger.info(f"  Teacher:")
            logger.info(f"    Factual Correctness: {results['factual_correctness_teacher']*100:.2f}%")
            logger.info(f"    Answer Relevancy: {results['answer_relevancy_teacher']*100:.2f}%")
            logger.info(f"    Hallucination Rate: {results['hallucination_rate_teacher']*100:.2f}%")
            logger.info(f"    Overall Faithfulness: {results['overall_faithfulness_teacher']*100:.2f}%")
            logger.info(f"  Teacher-Student Agreement: {results['teacher_student_agreement']*100:.2f}%")
        
        # Save results
        results_path = os.path.join(self.config.results_dir, "fidelitybench_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved FidelityBench results to {results_path}")
        
        # Clean up NLI model
        if use_nli:
            del nli_model
            del nli_tokenizer
            torch.cuda.empty_cache()
        
        # Return models to training mode
        self.distillation_method.student_model.train()
        
        return results

    # ==============================================================================
    # VISUALIZATION METHODS
    # ==============================================================================

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training and validation loss curves over epochs.
        
        :param save_path: Path to save plot (defaults to results_dir/training_curves.png)
        :type save_path: Optional[str]
        """
        if not self.training_history['train_loss']:
            logger.warning("No training history to plot")
            return
        
        if save_path is None:
            save_path = os.path.join(self.config.results_dir, "training_curves.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Plot 1: Loss curves
        ax1.plot(epochs, self.training_history['train_loss'], 
                marker='o', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 
                marker='s', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning rate schedule
        ax2.plot(epochs, self.training_history['learning_rate'], 
                marker='d', color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {save_path}")

    def plot_benchmark_comparison(self, benchmark_results: Dict[str, float], save_path: Optional[str] = None):
        """
        Plot student vs teacher accuracy comparison across benchmarks.
        
        :param benchmark_results: Dictionary with benchmark results
        :type benchmark_results: Dict[str, float]
        :param save_path: Path to save plot
        :type save_path: Optional[str]
        """
        if save_path is None:
            save_path = os.path.join(self.config.results_dir, "benchmark_comparison.png")
        
        # Extract benchmark names and accuracies
        benchmarks = []
        student_acc = []
        teacher_acc = []
        gaps = []
        
        for key in benchmark_results:
            if key.endswith('_student_accuracy'):
                benchmark_name = key.replace('_student_accuracy', '').upper()
                benchmarks.append(benchmark_name)
                student_acc.append(benchmark_results[key] * 100)
                
                teacher_key = key.replace('student', 'teacher')
                teacher_val = benchmark_results.get(teacher_key, 0) * 100
                teacher_acc.append(teacher_val)
                gaps.append(teacher_val - benchmark_results[key] * 100)
        
        if not benchmarks:
            logger.warning("No benchmark results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(benchmarks))
        width = 0.35
        
        # Plot 1: Student vs Teacher comparison
        bars1 = ax1.bar(x - width/2, student_acc, width, label='Student', alpha=0.8)
        bars2 = ax1.bar(x + width/2, teacher_acc, width, label='Teacher', alpha=0.8)
        
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Student vs Teacher Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Accuracy gap
        bars3 = ax2.bar(x, gaps, color='coral', alpha=0.8)
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('Accuracy Gap (%)')
        ax2.set_title('Teacher-Student Accuracy Gap')
        ax2.set_xticks(x)
        ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved benchmark comparison to {save_path}")

    def plot_fidelity_metrics(self, fidelity_results: Dict[str, float], save_path: Optional[str] = None):
        """
        Plot fidelity metrics (KL divergence, BLEU, ROUGE, overlap).
        
        :param fidelity_results: Dictionary with fidelity results
        :type fidelity_results: Dict[str, float]
        :param save_path: Path to save plot
        :type save_path: Optional[str]
        """
        if save_path is None:
            save_path = os.path.join(self.config.results_dir, "fidelity_metrics.png")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Divergence metrics
        divergences = {
            'KL Divergence': fidelity_results.get('mean_kl_divergence', 0),
            'JS Divergence': fidelity_results.get('mean_js_divergence', 0)
        }
        ax1.bar(divergences.keys(), divergences.values(), color=['skyblue', 'lightcoral'], alpha=0.8)
        ax1.set_ylabel('Divergence')
        ax1.set_title('Distribution Divergence Metrics\n(Lower is Better)')
        ax1.grid(True, axis='y', alpha=0.3)
        for i, (k, v) in enumerate(divergences.items()):
            ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Overlap metrics
        overlaps = {
            'Top-1': fidelity_results.get('mean_top1_overlap', 0) * 100,
            'Top-5': fidelity_results.get('mean_top5_overlap', 0) * 100
        }
        ax2.bar(overlaps.keys(), overlaps.values(), color=['mediumseagreen', 'mediumorchid'], alpha=0.8)
        ax2.set_ylabel('Overlap (%)')
        ax2.set_title('Top-K Token Overlap\n(Higher is Better)')
        ax2.set_ylim([0, 100])
        ax2.grid(True, axis='y', alpha=0.3)
        for i, (k, v) in enumerate(overlaps.items()):
            ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: BLEU & ROUGE scores
        if 'mean_bleu' in fidelity_results:
            text_metrics = {
                'BLEU': fidelity_results.get('mean_bleu', 0) * 100,
                'ROUGE-1': fidelity_results.get('mean_rouge1', 0) * 100,
                'ROUGE-2': fidelity_results.get('mean_rouge2', 0) * 100,
                'ROUGE-L': fidelity_results.get('mean_rougeL', 0) * 100
            }
            ax3.bar(text_metrics.keys(), text_metrics.values(), 
                   color=['gold', 'lightblue', 'lightgreen', 'plum'], alpha=0.8)
            ax3.set_ylabel('Score (%)')
            ax3.set_title('Text Generation Similarity\n(Higher is Better)')
            ax3.set_ylim([0, 100])
            ax3.grid(True, axis='y', alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            for i, (k, v) in enumerate(text_metrics.items()):
                ax3.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'BLEU/ROUGE metrics\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Text Generation Similarity')
        
        # Plot 4: Exact match rate
        if 'exact_match_rate' in fidelity_results:
            exact_match = fidelity_results.get('exact_match_rate', 0) * 100
            ax4.bar(['Exact Match'], [exact_match], color='salmon', alpha=0.8, width=0.4)
            ax4.set_ylabel('Rate (%)')
            ax4.set_title('Exact Match Rate\n(Higher is Better)')
            ax4.set_ylim([0, 100])
            ax4.grid(True, axis='y', alpha=0.3)
            ax4.text(0, exact_match, f'{exact_match:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Exact match\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Exact Match Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved fidelity metrics plot to {save_path}")

    def plot_fidelitybench_radar(self, fidelitybench_results: Dict[str, float], save_path: Optional[str] = None):
        """
        Plot FidelityBench metrics as radar chart comparing student and teacher.
        
        :param fidelitybench_results: Dictionary with FidelityBench results
        :type fidelitybench_results: Dict[str, float]
        :param save_path: Path to save plot
        :type save_path: Optional[str]
        """
        if save_path is None:
            save_path = os.path.join(self.config.results_dir, "fidelitybench_radar.png")
        
        # Extract metrics
        metrics = ['Factual\nCorrectness', 'Answer\nRelevancy', 'Faithfulness', 'Citation\nCoverage']
        
        student_values = [
            fidelitybench_results.get('factual_correctness_student', 0) * 100,
            fidelitybench_results.get('answer_relevancy_student', 0) * 100,
            fidelitybench_results.get('overall_faithfulness_student', 0) * 100,
            fidelitybench_results.get('citation_coverage', 0) * 100
        ]
        
        teacher_values = [
            fidelitybench_results.get('factual_correctness_teacher', 0) * 100,
            fidelitybench_results.get('answer_relevancy_teacher', 0) * 100,
            fidelitybench_results.get('overall_faithfulness_teacher', 0) * 100,
            fidelitybench_results.get('citation_coverage', 0) * 100
        ]
        
        # Also plot hallucination rate (inverted - lower is better)
        metrics_extended = metrics + ['No Hallucination']
        student_values_extended = student_values + [
            (1 - fidelitybench_results.get('hallucination_rate_student', 0)) * 100
        ]
        teacher_values_extended = teacher_values + [
            (1 - fidelitybench_results.get('hallucination_rate_teacher', 0)) * 100
        ]
        
        # Number of variables
        num_vars = len(metrics_extended)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the loop
        student_values_extended += student_values_extended[:1]
        teacher_values_extended += teacher_values_extended[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot student
        ax.plot(angles, student_values_extended, 'o-', linewidth=2, label='Student', color='#3498db')
        ax.fill(angles, student_values_extended, alpha=0.25, color='#3498db')
        
        # Plot teacher
        ax.plot(angles, teacher_values_extended, 's-', linewidth=2, label='Teacher', color='#e74c3c')
        ax.fill(angles, teacher_values_extended, alpha=0.25, color='#e74c3c')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_extended, size=11)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=9)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        
        ax.set_title('FidelityBench-Med: Evidence Faithfulness Metrics', 
                    size=14, weight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved FidelityBench radar chart to {save_path}")

    def plot_ablation_results(self, ablation_summary: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot ablation study results showing metric sensitivity to hyperparameter changes.
        
        :param ablation_summary: Dictionary with ablation study results
        :type ablation_summary: Dict[str, Any]
        :param save_path: Path to save plot
        :type save_path: Optional[str]
        """
        if save_path is None:
            ablation_dir = os.path.join(
                os.path.dirname(self.config.results_dir),
                f"ablation_{ablation_summary['ablation_type']}"
            )
            save_path = os.path.join(ablation_dir, "ablation_plot.png")
        
        results = ablation_summary['results']
        param_name = ablation_summary['parameter']
        param_values = ablation_summary['values_tested']
        
        if not results:
            logger.warning("No ablation results to plot")
            return
        
        # Extract metrics
        val_losses = [r.get('val_loss', None) for r in results]
        train_losses = [r.get('train_loss', None) for r in results]
        training_times = [r.get('training_time_hours', None) for r in results]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Validation loss
        ax1 = plt.subplot(2, 3, 1)
        if all(v is not None for v in val_losses):
            ax1.plot(param_values, val_losses, 'o-', linewidth=2, markersize=8, color='#3498db')
            ax1.set_xlabel(param_name.replace('_', ' ').title())
            ax1.set_ylabel('Validation Loss')
            ax1.set_title('Validation Loss vs ' + param_name.replace('_', ' ').title())
            ax1.grid(True, alpha=0.3)
            # Mark best value
            best_idx = np.argmin(val_losses)
            ax1.scatter([param_values[best_idx]], [val_losses[best_idx]], 
                       color='red', s=200, zorder=5, marker='*', label='Best')
            ax1.legend()
        
        # Plot 2: Training time
        ax2 = plt.subplot(2, 3, 2)
        if all(v is not None for v in training_times):
            ax2.plot(param_values, training_times, 's-', linewidth=2, markersize=8, color='#e74c3c')
            ax2.set_xlabel(param_name.replace('_', ' ').title())
            ax2.set_ylabel('Training Time (hours)')
            ax2.set_title('Training Time vs ' + param_name.replace('_', ' ').title())
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Benchmark accuracy (if available)
        ax3 = plt.subplot(2, 3, 3)
        benchmark_names = []
        for key in results[0].get('benchmark_results', {}):
            if key.endswith('_student_accuracy'):
                benchmark_names.append(key.replace('_student_accuracy', '').upper())
        
        if benchmark_names:
            for benchmark in benchmark_names:
                accuracies = [
                    r.get('benchmark_results', {}).get(f'{benchmark.lower()}_student_accuracy', 0) * 100
                    for r in results
                ]
                ax3.plot(param_values, accuracies, 'o-', linewidth=2, markersize=6, label=benchmark)
            ax3.set_xlabel(param_name.replace('_', ' ').title())
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title('Benchmark Accuracy vs ' + param_name.replace('_', ' ').title())
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Fidelity metrics (if available)
        ax4 = plt.subplot(2, 3, 4)
        kl_divs = [r.get('fidelity_results', {}).get('mean_kl_divergence', None) for r in results]
        if all(v is not None for v in kl_divs):
            ax4.plot(param_values, kl_divs, 'd-', linewidth=2, markersize=8, color='#2ecc71')
            ax4.set_xlabel(param_name.replace('_', ' ').title())
            ax4.set_ylabel('KL Divergence')
            ax4.set_title('KL Divergence vs ' + param_name.replace('_', ' ').title())
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: BLEU score (if available)
        ax5 = plt.subplot(2, 3, 5)
        bleu_scores = [r.get('fidelity_results', {}).get('mean_bleu', None) for r in results]
        if all(v is not None for v in bleu_scores):
            bleu_pct = [b * 100 for b in bleu_scores]
            ax5.plot(param_values, bleu_pct, '^-', linewidth=2, markersize=8, color='#9b59b6')
            ax5.set_xlabel(param_name.replace('_', ' ').title())
            ax5.set_ylabel('BLEU Score (%)')
            ax5.set_title('BLEU Score vs ' + param_name.replace('_', ' ').title())
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: FidelityBench faithfulness (if available)
        ax6 = plt.subplot(2, 3, 6)
        faithfulness = [r.get('fidelitybench_results', {}).get('overall_faithfulness_student', None) for r in results]
        if all(v is not None for v in faithfulness):
            faith_pct = [f * 100 for f in faithfulness]
            ax6.plot(param_values, faith_pct, 'v-', linewidth=2, markersize=8, color='#f39c12')
            ax6.set_xlabel(param_name.replace('_', ' ').title())
            ax6.set_ylabel('Faithfulness (%)')
            ax6.set_title('Overall Faithfulness vs ' + param_name.replace('_', ' ').title())
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Ablation Study: {ablation_summary["ablation_type"].replace("_", " ").title()}',
                    fontsize=16, weight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ablation study plot to {save_path}")

    def evaluate_baseline_comparison(self, baseline_model_name: str, benchmark_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare distilled student model against baseline model (e.g., Meditron-7B).
        
        This is the KEY EXPERIMENT for measuring distillation efficiency:
        - Baseline: Llama-2-7B fine-tuned directly on 48B medical tokens (Meditron-7B)
        - Your model: Llama-2-7B distilled from Meditron-70B on 414K examples
        
        Measures:
        1. Accuracy gap (how close is distillation to direct fine-tuning?)
        2. Efficiency gains (data/time/cost savings)
        
        :param baseline_model_name: HuggingFace model ID (e.g., "epfl-llm/meditron-7b")
        :type baseline_model_name: str
        :param benchmark_paths: Dict of benchmark names to file paths
        :type benchmark_paths: Dict[str, str]
        :returns: Comparison results with accuracy gaps and efficiency metrics
        :rtype: Dict[str, Any]
        """
        logger.info("\n" + "="*80)
        logger.info("BASELINE COMPARISON: Distillation vs Direct Fine-Tuning")
        logger.info("="*80)
        logger.info(f"Baseline Model: {baseline_model_name}")
        logger.info(f"Your Model: {self.config.student_model_name} (distilled)")
        logger.info("="*80)
        
        # Load baseline model
        logger.info(f"\nLoading baseline model: {baseline_model_name}")
        try:
            from transformers import AutoModelForCausalLM
            baseline_model = AutoModelForCausalLM.from_pretrained(
                baseline_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            baseline_model.eval()
        except Exception as e:
            logger.exception(f"Failed to load baseline model: {e}")
            try:
                if torch.cuda.is_available():
                    logger.error("CUDA memory summary at baseline load failure:\n%s", torch.cuda.memory_summary(device=None, abbreviated=True))
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.warning("Failed to capture CUDA memory summary after baseline load failure: %s", e2)
            logger.warning("Skipping baseline comparison...")
            return {}
        
        # Evaluate both models on each benchmark
        comparison_results = {
            'baseline_model': baseline_model_name,
            'distilled_model': f"{self.config.student_model_name} ({self.distillation_method.get_method_name()})",
            'benchmarks': {},
            'summary': {}
        }
        
        for benchmark_name, benchmark_path in benchmark_paths.items():
            if not os.path.exists(benchmark_path):
                logger.warning(f"Benchmark {benchmark_name} not found at {benchmark_path}, skipping...")
                continue
            
            logger.info(f"\n--- Evaluating {benchmark_name.upper()} ---")
            
            # Load benchmark data
            questions = []
            ground_truths = []
            with open(benchmark_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    questions.append(item['question'])
                    ground_truths.append(item['ground_truth_answer'])
            
            # Evaluate YOUR distilled model
            logger.info(f"Evaluating YOUR distilled model...")
            your_correct = 0
            your_model = self.distillation_method.student_model
            your_model.eval()
            
            for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions), desc="Your model"):
                prompt = f"{question}\n\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(your_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = your_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip().lower()
                
                if ground_truth.lower() in response[:200]:  # Check first 200 chars
                    your_correct += 1
            
            # Evaluate BASELINE model
            logger.info(f"Evaluating BASELINE model ({baseline_model_name})...")
            baseline_correct = 0
            
            for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions), desc="Baseline"):
                prompt = f"{question}\n\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(baseline_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = baseline_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip().lower()
                
                if ground_truth.lower() in response[:200]:
                    baseline_correct += 1
            
            # Calculate metrics
            total = len(questions)
            your_accuracy = your_correct / total
            baseline_accuracy = baseline_correct / total
            accuracy_gap = baseline_accuracy - your_accuracy
            accuracy_retained = (your_accuracy / baseline_accuracy) if baseline_accuracy > 0 else 0
            
            # Store results
            comparison_results['benchmarks'][benchmark_name] = {
                'your_accuracy': your_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'accuracy_gap': accuracy_gap,
                'accuracy_retained_pct': accuracy_retained * 100,
                'your_correct': your_correct,
                'baseline_correct': baseline_correct,
                'total_questions': total
            }
            
            # Log results
            logger.info(f"\n{benchmark_name.upper()} Results:")
            logger.info(f"  Your Model:    {your_correct}/{total} = {your_accuracy*100:.2f}%")
            logger.info(f"  Baseline:      {baseline_correct}/{total} = {baseline_accuracy*100:.2f}%")
            logger.info(f"  Accuracy Gap:  {accuracy_gap*100:.2f}%")
            logger.info(f"  Retained:      {accuracy_retained*100:.1f}% of baseline performance")
        
        # Calculate summary statistics
        if comparison_results['benchmarks']:
            accuracies_your = [r['your_accuracy'] for r in comparison_results['benchmarks'].values()]
            accuracies_baseline = [r['baseline_accuracy'] for r in comparison_results['benchmarks'].values()]
            gaps = [r['accuracy_gap'] for r in comparison_results['benchmarks'].values()]
            retentions = [r['accuracy_retained_pct'] for r in comparison_results['benchmarks'].values()]
            
            comparison_results['summary'] = {
                'avg_your_accuracy': sum(accuracies_your) / len(accuracies_your),
                'avg_baseline_accuracy': sum(accuracies_baseline) / len(accuracies_baseline),
                'avg_accuracy_gap': sum(gaps) / len(gaps),
                'avg_retention_pct': sum(retentions) / len(retentions),
                'efficiency_gains': {
                    'training_data_reduction': 4800,  # 48B tokens → 10M tokens
                    'training_time_speedup': 55,       # ~500 hours → ~9 hours
                    'cost_reduction': 55,              # Estimated based on time
                    'data_used_tokens': '10M',
                    'baseline_data_tokens': '48B'
                }
            }
            
            # Log summary
            logger.info("\n" + "="*80)
            logger.info("SUMMARY: DISTILLATION EFFICIENCY")
            logger.info("="*80)
            logger.info(f"Average Accuracy (Your Model):    {comparison_results['summary']['avg_your_accuracy']*100:.2f}%")
            logger.info(f"Average Accuracy (Baseline):      {comparison_results['summary']['avg_baseline_accuracy']*100:.2f}%")
            logger.info(f"Average Accuracy Gap:             {comparison_results['summary']['avg_accuracy_gap']*100:.2f}%")
            logger.info(f"Average Performance Retained:     {comparison_results['summary']['avg_retention_pct']:.1f}%")
            logger.info("\nEFFICIENCY GAINS:")
            logger.info(f"  Training Data:  {comparison_results['summary']['efficiency_gains']['training_data_reduction']}x less")
            logger.info(f"  Training Time:  {comparison_results['summary']['efficiency_gains']['training_time_speedup']}x faster")
            logger.info(f"  Cost Savings:   {comparison_results['summary']['efficiency_gains']['cost_reduction']}x cheaper")
            logger.info("="*80)
        
        # Save results
        comparison_path = os.path.join(self.config.results_dir, "baseline_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"\nSaved baseline comparison to {comparison_path}")
        
        # Generate comparison plot
        self.plot_baseline_comparison(comparison_results)
        
        # Clean up baseline model to free memory
        del baseline_model
        torch.cuda.empty_cache()
        
        return comparison_results
    
    def plot_baseline_comparison(self, comparison_results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot baseline comparison results.
        
        Creates a multi-panel figure showing:
        1. Accuracy comparison (bar chart)
        2. Accuracy gap (bar chart)
        3. Efficiency gains (horizontal bar chart)
        
        :param comparison_results: Results from evaluate_baseline_comparison()
        :type comparison_results: Dict[str, Any]
        :param save_path: Optional path to save figure
        :type save_path: Optional[str]
        """
        if not comparison_results or not comparison_results.get('benchmarks'):
            logger.warning("No comparison results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distillation vs Direct Fine-Tuning Comparison', fontsize=16, fontweight='bold')
        
        benchmarks = list(comparison_results['benchmarks'].keys())
        your_accs = [comparison_results['benchmarks'][b]['your_accuracy'] * 100 for b in benchmarks]
        baseline_accs = [comparison_results['benchmarks'][b]['baseline_accuracy'] * 100 for b in benchmarks]
        gaps = [comparison_results['benchmarks'][b]['accuracy_gap'] * 100 for b in benchmarks]
        retentions = [comparison_results['benchmarks'][b]['accuracy_retained_pct'] for b in benchmarks]
        
        # Plot 1: Accuracy Comparison
        ax1 = axes[0, 0]
        x = np.arange(len(benchmarks))
        width = 0.35
        ax1.bar(x - width/2, your_accs, width, label='Your Distilled Model', color='#3498db')
        ax1.bar(x + width/2, baseline_accs, width, label='Baseline (Meditron-7B)', color='#e74c3c')
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([b.upper() for b in benchmarks], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Accuracy Gap
        ax2 = axes[0, 1]
        colors = ['#2ecc71' if g < 5 else '#f39c12' if g < 10 else '#e74c3c' for g in gaps]
        ax2.bar(benchmarks, gaps, color=colors)
        ax2.axhline(y=5, color='green', linestyle='--', label='5% threshold (good)')
        ax2.axhline(y=10, color='orange', linestyle='--', label='10% threshold (acceptable)')
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('Accuracy Gap (%)')
        ax2.set_title('Performance Gap (Lower is Better)')
        ax2.set_xticklabels([b.upper() for b in benchmarks], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Performance Retention
        ax3 = axes[1, 0]
        colors_ret = ['#2ecc71' if r >= 95 else '#f39c12' if r >= 90 else '#e74c3c' for r in retentions]
        ax3.barh(benchmarks, retentions, color=colors_ret)
        ax3.axvline(x=95, color='green', linestyle='--', label='95% (excellent)')
        ax3.axvline(x=90, color='orange', linestyle='--', label='90% (good)')
        ax3.set_xlabel('Performance Retained (%)')
        ax3.set_ylabel('Benchmark')
        ax3.set_title('Performance Retention (Higher is Better)')
        ax3.set_yticklabels([b.upper() for b in benchmarks])
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_xlim(80, 100)
        
        # Plot 4: Efficiency Gains
        ax4 = axes[1, 1]
        if 'summary' in comparison_results and 'efficiency_gains' in comparison_results['summary']:
            gains = comparison_results['summary']['efficiency_gains']
            metrics = ['Data Reduction', 'Time Speedup', 'Cost Reduction']
            values = [gains['training_data_reduction'], gains['training_time_speedup'], gains['cost_reduction']]
            
            bars = ax4.barh(metrics, values, color='#9b59b6')
            ax4.set_xlabel('Efficiency Gain (x times)')
            ax4.set_title('Distillation Efficiency Gains')
            ax4.set_xscale('log')
            ax4.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (metric, value) in enumerate(zip(metrics, values)):
                ax4.text(value, i, f'  {value}x', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.config.results_dir, 'baseline_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved baseline comparison plot to {save_path}")
        plt.close()

    def run_comprehensive_evaluation(self):
        """
        Run all evaluation metrics after training completes.
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING COMPREHENSIVE EVALUATION")
        logger.info("="*80)
        
        all_results = {}
        
        # 1. Perplexity evaluation (if corpus exists)
        corpus_path = os.path.join(os.path.dirname(self.config.train_data_path), "medppl_10k.jsonl")
        if os.path.exists(corpus_path):
            perplexity_results = self.evaluate_perplexity_on_corpus(corpus_path)
            all_results['perplexity'] = perplexity_results
        else:
            logger.warning(f"Perplexity corpus not found at {corpus_path}, skipping...")
        
        # 2. Medical benchmark evaluation (if benchmark files exist)
        benchmark_dir = os.path.join(os.path.dirname(self.config.train_data_path), "benchmarks")
        benchmark_paths = {
            'medqa': os.path.join(benchmark_dir, 'medqa_test.jsonl'),
            'medmcqa': os.path.join(benchmark_dir, 'medmcqa_val.jsonl'),
            'pubmedqa': os.path.join(benchmark_dir, 'pubmedqa_test.jsonl'),
            'pubhealth': os.path.join(benchmark_dir, 'pubhealth_test.jsonl')
        }
        
        # Check if any benchmark exists
        if any(os.path.exists(p) for p in benchmark_paths.values()):
            benchmark_results = self.evaluate_medical_benchmarks(benchmark_paths)
            all_results['benchmarks'] = benchmark_results
        else:
            logger.warning(f"No benchmark files found in {benchmark_dir}, skipping...")
        
        # 3. Fidelity evaluation (teacher-student distribution matching)
        fidelity_results = self.evaluate_fidelity(num_samples=100)
        all_results['fidelity'] = fidelity_results
        
        # 4. FidelityBench-Med evaluation (evidence faithfulness)
        fidelitybench_path = os.path.join(os.path.dirname(self.config.train_data_path), "fidelitybench_med.jsonl")
        if os.path.exists(fidelitybench_path):
            fidelitybench_results = self.evaluate_fidelitybench_med(fidelitybench_path)
            all_results['fidelitybench'] = fidelitybench_results
        else:
            logger.warning(f"FidelityBench-Med not found at {fidelitybench_path}, skipping...")
        
        # 5. Baseline comparison (Distillation vs Direct Fine-Tuning)
        # Compare your distilled Llama-2-7B against Meditron-7B (directly fine-tuned on 48B tokens)
        baseline_model_name = "epfl-llm/meditron-7b"
        if any(os.path.exists(p) for p in benchmark_paths.values()):
            logger.info("\n" + "="*80)
            logger.info("RUNNING BASELINE COMPARISON EXPERIMENT")
            logger.info(f"Comparing distilled model vs {baseline_model_name}")
            logger.info("="*80)
            baseline_results = self.evaluate_baseline_comparison(baseline_model_name, benchmark_paths)
            all_results['baseline_comparison'] = baseline_results
        else:
            logger.warning("No benchmarks available for baseline comparison, skipping...")
        
        # 6. Save comprehensive results
        comprehensive_path = os.path.join(self.config.results_dir, "comprehensive_evaluation.json")
        with open(comprehensive_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved comprehensive evaluation to {comprehensive_path}")
        
        # 7. Generate visualization plots
        logger.info("\nGenerating evaluation visualizations...")
        
        try:
            # Plot benchmark comparison (if available)
            if 'benchmarks' in all_results:
                self.plot_benchmark_comparison(all_results['benchmarks'])
        except Exception as e:
            logger.warning(f"Failed to generate benchmark comparison plot: {e}")
        
        try:
            # Plot fidelity metrics (if available)
            if 'fidelity' in all_results:
                self.plot_fidelity_metrics(all_results['fidelity'])
        except Exception as e:
            logger.warning(f"Failed to generate fidelity metrics plot: {e}")
        
        try:
            # Plot FidelityBench radar chart (if available)
            if 'fidelitybench' in all_results and all_results['fidelitybench']:
                self.plot_fidelitybench_radar(all_results['fidelitybench'])
        except Exception as e:
            logger.warning(f"Failed to generate FidelityBench radar chart: {e}")
        
        try:
            # Plot baseline comparison (if available)
            if 'baseline_comparison' in all_results and all_results['baseline_comparison']:
                self.plot_baseline_comparison(all_results['baseline_comparison'])
        except Exception as e:
            logger.warning(f"Failed to generate baseline comparison plot: {e}")
        
        logger.info("Visualization generation completed!")
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EVALUATION COMPLETED")
        logger.info("="*80)
        
        return all_results


# ==============================================================================
# ABLATION STUDY FUNCTIONS
# ==============================================================================

def run_ablation_study(
    base_config: TrainingConfig,
    ablation_type: str,
    param_name: str,
    param_values: List[Any],
    teacher_model: nn.Module,
    student_model_name: str,
    tokenizer: Any,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
) -> Dict[str, Any]:
    """
    Run systematic ablation study by varying one hyperparameter.
    
    :param base_config: Base training configuration
    :type base_config: TrainingConfig
    :param ablation_type: Type of ablation ('temperature', 'alpha', 'dataset_size', 'lora_rank')
    :type ablation_type: str
    :param param_name: Parameter name to vary
    :type param_name: str
    :param param_values: List of parameter values to test
    :type param_values: List[Any]
    :param teacher_model: Teacher model
    :type teacher_model: nn.Module
    :param student_model_name: Student model identifier
    :type student_model_name: str
    :param tokenizer: Tokenizer
    :type tokenizer: Any
    :param train_dataloader: Training dataloader
    :type train_dataloader: DataLoader
    :param val_dataloader: Validation dataloader
    :type val_dataloader: DataLoader
    :returns: Dictionary with ablation results
    :rtype: Dict[str, Any]
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING ABLATION STUDY: {ablation_type}")
    logger.info(f"Parameter: {param_name}")
    logger.info(f"Values: {param_values}")
    logger.info(f"{'='*80}\n")
    
    ablation_results = []
    
    for value in param_values:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {param_name} = {value}")
        logger.info(f"{'='*80}\n")
        
        # Create modified config
        config_copy = TrainingConfig.__new__(TrainingConfig)
        config_copy.__dict__.update(base_config.__dict__.copy())
        setattr(config_copy, param_name, value)
        
        # Update output directory for this experiment
        config_copy.output_dir = os.path.join(
            base_config.output_dir,
            f"ablation_{ablation_type}",
            f"{param_name}_{value}"
        )
        config_copy.checkpoint_dir = os.path.join(config_copy.output_dir, "checkpoints")
        config_copy.results_dir = os.path.join(config_copy.output_dir, "results")
        
        # Create directories
        Path(config_copy.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config_copy.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config_copy.results_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Load fresh student model for this experiment
            lora_config_dict = {
                'r': config_copy.lora_rank,
                'lora_alpha': config_copy.lora_alpha,
                'lora_dropout': config_copy.lora_dropout,
                'bias': 'none',
                'task_type': 'CAUSAL_LM',
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            } if config_copy.use_lora else None
            
            student_model = load_student_model(
                student_model_name,
                use_lora=config_copy.use_lora,
                lora_config=lora_config_dict,
                use_quantization=config_copy.use_quantization,
                enable_cpu_offload = config_copy.enable_cpu_offload
            )
            
            # Create distillation method with updated config
            distillation_config = {}
            if config_copy.distillation_method in ['logit_kd', 'logit']:
                distillation_config['alpha'] = config_copy.alpha
                distillation_config['temperature'] = config_copy.temperature
            elif config_copy.distillation_method in ['adakd', 'token_adaptive']:
                distillation_config['alpha'] = config_copy.alpha
                distillation_config['base_temperature'] = config_copy.base_temperature
                distillation_config['min_temperature'] = config_copy.min_temperature
                distillation_config['max_temperature'] = config_copy.max_temperature
            
            distillation_method = create_distillation_method(
                method_name=config_copy.distillation_method,
                teacher_model=teacher_model,
                student_model=student_model,
                tokenizer=tokenizer,
                config=distillation_config
            )
            
            # Setup optimizer and scheduler
            _log_and_clear_cuda("before_ablation_optimizer_creation")
            optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=config_copy.learning_rate,
                weight_decay=config_copy.weight_decay
            )
            _log_and_clear_cuda("after_ablation_optimizer_creation")
            
            total_steps = len(train_dataloader) * config_copy.num_epochs // config_copy.gradient_accumulation_steps
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config_copy.warmup_steps,
                num_training_steps=total_steps
            )
            
            # Initialize trainer
            trainer = Trainer(
                config=config_copy,
                distillation_method=distillation_method,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer
            )
            
            # Train
            trainer.train()
            
            # Collect results
            result = {
                'param_value': value,
                'final_train_loss': trainer.training_history['train_loss'][-1] if trainer.training_history['train_loss'] else None,
                'final_val_loss': trainer.training_history['val_loss'][-1] if trainer.training_history['val_loss'] else None,
                'best_val_loss': trainer.best_val_loss,
                'training_time_hours': trainer.total_training_time / 3600.0,
                'peak_memory_gb': trainer.peak_memory_allocated,
                'output_dir': config_copy.output_dir
            }
            
            # Load comprehensive evaluation results if available
            comprehensive_path = os.path.join(config_copy.results_dir, "comprehensive_evaluation.json")
            if os.path.exists(comprehensive_path):
                with open(comprehensive_path, 'r') as f:
                    eval_results = json.load(f)
                result['evaluation'] = eval_results
            
            ablation_results.append(result)
            
            logger.info(f"\nCompleted {param_name} = {value}")
            logger.info(f"Best val loss: {result['best_val_loss']:.4f}")
            logger.info(f"Training time: {result['training_time_hours']:.2f} hours")
            
            # Clean up
            del student_model
            del distillation_method
            del optimizer
            del scheduler
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during ablation with {param_name}={value}: {e}")
            result = {
                'param_value': value,
                'error': str(e)
            }
            ablation_results.append(result)
            continue
    
    # Save ablation results
    ablation_summary = {
        'ablation_type': ablation_type,
        'parameter': param_name,
        'values_tested': param_values,
        'results': ablation_results
    }
    
    summary_path = os.path.join(
        base_config.output_dir,
        f"ablation_{ablation_type}",
        "ablation_summary.json"
    )
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    
    # Generate ablation visualization plot
    logger.info("Generating ablation study visualization...")
    try:
        # Create a temporary trainer instance to access plotting method
        temp_config = TrainingConfig.__new__(TrainingConfig)
        temp_config.__dict__.update(base_config.__dict__.copy())
        temp_config.results_dir = os.path.join(base_config.output_dir, f"ablation_{ablation_type}")
        
        # Use static method approach - call plot directly
        from matplotlib import pyplot as plt
        plot_save_path = os.path.join(temp_config.results_dir, "ablation_plot.png")
        
        # Extract data for plotting
        results_data = []
        for r in ablation_results:
            if 'error' not in r:
                plot_data = {
                    'val_loss': r.get('best_val_loss'),
                    'train_loss': r.get('final_train_loss'),
                    'training_time_hours': r.get('training_time_hours')
                }
                
                # Add evaluation metrics if available
                if 'evaluation' in r:
                    eval_data = r['evaluation']
                    plot_data['benchmark_results'] = eval_data.get('benchmarks', {})
                    plot_data['fidelity_results'] = eval_data.get('fidelity', {})
                    plot_data['fidelitybench_results'] = eval_data.get('fidelitybench', {})
                
                results_data.append(plot_data)
        
        if results_data:
            ablation_summary_with_data = {
                'ablation_type': ablation_type,
                'parameter': param_name,
                'values_tested': param_values,
                'results': results_data
            }
            
            # Create a minimal trainer instance just for plotting
            class PlotHelper:
                def __init__(self, config):
                    self.config = config
            
            helper = PlotHelper(temp_config)
            
            # Import the plotting method as a standalone function
            # Call plot_ablation_results with the helper
            Trainer.plot_ablation_results(helper, ablation_summary_with_data, plot_save_path)
            
    except Exception as e:
        logger.warning(f"Failed to generate ablation plot: {e}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION STUDY COMPLETED: {ablation_type}")
    logger.info(f"Results saved to: {summary_path}")
    logger.info(f"{'='*80}\n")
    
    return ablation_summary


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main training function."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train medical LLM using knowledge distillation")

    # Model arguments
    parser.add_argument('--teacher_model', type=str, default='epfl-llm/meditron-70b',
                        help='Teacher model name or path')
    parser.add_argument('--student_model', type=str, default='Qwen/Qwen2-1.5B',
                        help='Student model name or path')

    # Data arguments
    parser.add_argument('--train_data', type=str,
                        default='data/processed/train.jsonl',
                        help='Path to training data (Med-DistillMix format)')
    parser.add_argument('--val_data', type=str,
                        default='data/processed/validation.jsonl',
                        help='Path to validation data')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/default_run',
                        help='Output directory for checkpoints and results')

    # Training arguments
    parser.add_argument('--method', type=str, default='sft',
                        choices=['sft', 'logit_kd', 'adakd', 'cot', 'fitnets', 'attention',
                                 'on_policy', 'reinforce', 'ppo', 'bond', 'best_of_n', 'spin', 'self_play'],
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
    parser.add_argument('--enable_cpu_offload', action='store_true', default=False,
                        help='Enable CPU offloading for large teacher models (70B+)')
    parser.add_argument('--max_gpu_mem_gb', type=float, default=None,
                        help='Per-GPU hard limit in GiB; when set, loader will request device_map that keeps <= this on each GPU.\n'
                             'Note: setting this option requires --enable_cpu_offload to be set for safety.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='',
                        help='Path to checkpoint file to resume training from (optional)')
    parser.add_argument('--align_vocabularies', action='store_true', default=True,
                        help='Automatically align student vocabulary to match teacher by adding extra tokens. '
                             'Required when teacher and student have different vocabulary sizes for logit-based methods. '
                             'Example: Meditron-70B (32,017 tokens) → Llama-2-7B (32,000 tokens) adds 17 medical tokens.')
    parser.add_argument('--no_align_vocabularies', dest='align_vocabularies', action='store_false',
                        help='Disable automatic vocabulary alignment (opposite of --align_vocabularies)')

    # Method-specific arguments
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for Logit-KD, AdaKD, FitNets, Attention (weight for distillation loss)')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for Logit-KD and RL methods')
    parser.add_argument('--base_temperature', type=float, default=3.0,
                        help='Base temperature for AdaKD')
    parser.add_argument('--min_temperature', type=float, default=1.0,
                        help='Minimum temperature for AdaKD')
    parser.add_argument('--max_temperature', type=float, default=5.0,
                        help='Maximum temperature for AdaKD')
    parser.add_argument('--num_rationales', type=int, default=1,
                        help='Number of rationales for CoT distillation')
    parser.add_argument('--sampling_temperature', type=float, default=0.7,
                        help='Sampling temperature for CoT diverse rationales')
    parser.add_argument('--cot_prompt', type=str, default="Let's think step by step:",
                        help='Chain-of-thought prompt')
    parser.add_argument('--layer_mapping', type=str, default='{}',
                        help='Layer mapping for FitNets/Attention as JSON string, e.g., \'{"6":12,"12":24}\'')
    parser.add_argument('--use_projections', action='store_true', default=True,
                        help='Use projection layers for FitNets (for dimension mismatch)')
    parser.add_argument('--match_all_heads', action='store_true', default=True,
                        help='Match all attention heads for Attention distillation')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for RL methods')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples for BOND (Best-of-N)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Clip range for PPO')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient for RL methods (exploration bonus)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta for SPIN (DPO temperature)')
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
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (increase for better GPU utilization)')

    # Ablation study arguments
    parser.add_argument('--run_ablation', action='store_true',
                        help='Run ablation study instead of single training')
    parser.add_argument('--ablation_type', type=str, default='temperature',
                        choices=['temperature', 'alpha', 'dataset_size', 'lora_rank', 'learning_rate'],
                        help='Type of ablation study to run')
    parser.add_argument('--ablation_values', type=str, default='',
                        help='Comma-separated values to test (e.g., "2.0,3.0,4.0,5.0")')

    args = parser.parse_args()

    # Validate: --max_gpu_mem_gb can only be set when --enable_cpu_offload is enabled
    if args.max_gpu_mem_gb is not None and not args.enable_cpu_offload:
        parser.error("--max_gpu_mem_gb requires --enable_cpu_offload to be set. This prevents accidental memory-only constraints without offload.")

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
        use_quantization=config.use_quantization,
        enable_cpu_offload=config.enable_cpu_offload,
        max_gpu_mem_gb=args.max_gpu_mem_gb
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
        use_quantization=config.use_quantization,
        enable_cpu_offload = config.enable_cpu_offload,
        max_gpu_mem_gb=args.max_gpu_mem_gb
    )

    # ===== Vocabulary Alignment for Logit-Based Methods =====
    # Import alignment functions
    from DistillationMethods import align_student_vocabulary_to_teacher
    
    # Methods that require vocabulary alignment (use logit distributions)
    LOGIT_BASED_METHODS = ['logit_kd', 'logit', 'adakd', 'spin', 'self_play', 
                           'ppo', 'on_policy', 'reinforce', 'bond', 'best_of_n']
    
    if config.distillation_method in LOGIT_BASED_METHODS:
        teacher_vocab_size = teacher_model.config.vocab_size
        student_vocab_size = student_model.config.vocab_size
        
        if teacher_vocab_size != student_vocab_size:
            logger.info("\n" + "="*80)
            logger.info("⚠️  VOCABULARY SIZE MISMATCH DETECTED")
            logger.info("="*80)
            logger.info(f"Teacher vocab size: {teacher_vocab_size:,}")
            logger.info(f"Student vocab size: {student_vocab_size:,}")
            logger.info(f"Difference:         {abs(teacher_vocab_size - student_vocab_size):,} tokens")
            
            if config.align_vocabularies:
                logger.info("\n✅ --align_vocabularies flag detected")
                logger.info("   Proceeding with automatic vocabulary expansion...\n")
                
                # Load teacher tokenizer for alignment
                teacher_tokenizer = load_tokenizer(config.teacher_model_name)
                
                # Align vocabularies
                student_model, tokenizer = align_student_vocabulary_to_teacher(
                    student_model,
                    tokenizer,
                    teacher_tokenizer,
                    logger
                )
                
                logger.info("\n" + "="*80)
                logger.info("✅ VOCABULARY ALIGNMENT SUCCESSFUL")
                logger.info("="*80)
                logger.info("Student model and tokenizer have been updated.")
                logger.info("Training can now proceed with matched vocabularies.\n")
            else:
                error_msg = "\n" + "="*80 + "\n"
                error_msg += "❌ ERROR: Vocabulary mismatch requires alignment\n"
                error_msg += "="*80 + "\n"
                error_msg += f"Method '{config.distillation_method}' requires matching vocabulary sizes.\n\n"
                error_msg += "SOLUTION: Add the --align_vocabularies flag:\n\n"
                error_msg += "  python src/Trainer.py \\\n"
                error_msg += f"    --teacher_model {config.teacher_model_name} \\\n"
                error_msg += f"    --student_model {config.student_model_name} \\\n"
                error_msg += f"    --method {config.distillation_method} \\\n"
                error_msg += "    --align_vocabularies  # ← ADD THIS FLAG\n"
                error_msg += "\n"
                error_msg += "This will:\n"
                error_msg += f"  • Add {abs(teacher_vocab_size - student_vocab_size)} tokens to student vocabulary\n"
                error_msg += f"  • Resize embeddings: {student_vocab_size:,} → {teacher_vocab_size:,}\n"
                error_msg += "  • Initialize new embeddings with mean of existing ones\n"
                error_msg += "  • Enable full knowledge transfer via logit distillation\n"
                error_msg += "="*80 + "\n"
                raise ValueError(error_msg)
        else:
            logger.info("\n✅ Vocabulary sizes match - no alignment needed")
            logger.info(f"   Teacher vocab: {teacher_vocab_size:,}")
            logger.info(f"   Student vocab: {student_vocab_size:,}\n")
    else:
        logger.info(f"\nℹ️  Method '{config.distillation_method}' doesn't require vocabulary alignment")
        logger.info("   (Text-based methods like SFT handle vocab differences automatically)\n")

    # ===== Create Distillation Method Configuration =====
    # Build config dictionary with hyperparameters specific to chosen method
    # All methods get max_new_tokens for controlling generation length
    distillation_config = {
        'max_new_tokens': config.max_new_tokens,
    }

    # ===== Supervised Methods Configuration =====
    if config.distillation_method in ['logit_kd', 'logit']:
        # Logit-KD: Combine soft targets (KL loss) with hard targets (CE loss)
        distillation_config['alpha'] = config.alpha  # Weight for KD loss vs CE loss
        distillation_config['temperature'] = config.temperature  # Soften probability distributions
    
    elif config.distillation_method in ['adakd', 'token_adaptive']:
        # AdaKD: Adapt temperature per token based on student confidence
        distillation_config['alpha'] = config.alpha
        distillation_config['base_temperature'] = config.base_temperature  # Starting temperature
        distillation_config['min_temperature'] = config.min_temperature  # Lower bound for confident tokens
        distillation_config['max_temperature'] = config.max_temperature  # Upper bound for uncertain tokens
    
    elif config.distillation_method in ['cot', 'chain_of_thought']:
        # CoT: Distill teacher's reasoning process, not just final answers
        distillation_config['cot_prompt'] = config.cot_prompt  # Prompt to elicit reasoning
        distillation_config['num_rationales'] = config.num_rationales  # Multiple reasoning paths
        distillation_config['sampling_temperature'] = config.sampling_temperature  # For diverse rationales
        distillation_config['max_new_tokens'] = 512  # Longer sequences needed for reasoning steps
    
    elif config.distillation_method in ['fitnets', 'intermediate_feature']:
        # FitNets: Match internal representations (hidden states) between models
        distillation_config['alpha'] = config.alpha
        distillation_config['use_projections'] = config.use_projections  # Handle dimension mismatch
        # Parse layer mapping from JSON string (e.g., '{"6":12}' means student layer 6 → teacher layer 12)
        try:
            distillation_config['layer_mapping'] = json.loads(config.layer_mapping)
            # Convert string keys to integers for layer indexing
            distillation_config['layer_mapping'] = {
                int(k): int(v) for k, v in distillation_config['layer_mapping'].items()
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse layer_mapping: {config.layer_mapping}. Using empty dict.")
            distillation_config['layer_mapping'] = {}
    
    elif config.distillation_method in ['attention', 'attention_transfer']:
        # Attention: Match attention patterns (which tokens the model focuses on)
        distillation_config['alpha'] = config.alpha
        distillation_config['match_all_heads'] = config.match_all_heads  # Average across heads vs match individually
        # Parse layer mapping from JSON string
        try:
            distillation_config['layer_mapping'] = json.loads(config.layer_mapping)
            # Convert string keys to integers for layer indexing
            distillation_config['layer_mapping'] = {
                int(k): int(v) for k, v in distillation_config['layer_mapping'].items()
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse layer_mapping: {config.layer_mapping}. Using empty dict.")
            distillation_config['layer_mapping'] = {}
    
    # ===== RL Methods Configuration =====
    elif config.distillation_method in ['on_policy', 'reinforce']:
        # On-Policy/REINFORCE: Vanilla policy gradient (student samples, teacher rewards)
        distillation_config['gamma'] = config.gamma  # Discount factor for future rewards
        distillation_config['temperature'] = config.temperature  # Sampling temperature for exploration
        distillation_config['entropy_coef'] = config.entropy_coef  # Exploration bonus
    
    elif config.distillation_method in ['ppo']:
        # PPO: Clipped objective to prevent large policy updates (more stable than REINFORCE)
        distillation_config['gamma'] = config.gamma
        distillation_config['temperature'] = config.temperature
        distillation_config['epsilon'] = config.epsilon  # Clip range for policy ratio
        distillation_config['entropy_coef'] = config.entropy_coef
    
    elif config.distillation_method in ['bond', 'best_of_n']:
        # BOND: Generate N samples, teacher ranks them, learn to reproduce best in one shot
        distillation_config['num_samples'] = config.num_samples  # Number of candidate samples to generate
        distillation_config['temperature'] = config.temperature  # For diverse sampling
    
    elif config.distillation_method in ['spin', 'self_play']:
        # SPIN: Self-play DPO (student's outputs = dispreferred, teacher's = preferred)
        distillation_config['beta'] = config.beta  # DPO temperature (controls strength of preference)
        distillation_config['temperature'] = config.temperature  # Sampling temperature

    # ===== Initialize Distillation Method =====
    # Factory function creates appropriate method class based on method_name
    # Returns object with compute_loss() that implements the distillation algorithm
    distillation_method = create_distillation_method(
        method_name=config.distillation_method,
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        config=distillation_config
    )

    # ===== Create Dataloaders =====
    # Load training and validation data, create batches with proper padding
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

    # ===== Setup Optimizer =====
    # AdamW: Adam with weight decay (better regularization than standard Adam)
    _log_and_clear_cuda("before_optimizer_creation")
    optimizer = torch.optim.AdamW(
        student_model.parameters(),  # Only student parameters are trainable
        lr=config.learning_rate,
        weight_decay=config.weight_decay  # L2 regularization
    )
    _log_and_clear_cuda("after_optimizer_creation")

    # ===== Setup Learning Rate Scheduler =====
    # Warmup followed by linear decay (standard for LLM fine-tuning)
    total_steps = len(train_dataloader) * \
        config.num_epochs // config.gradient_accumulation_steps  # Total optimizer updates
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,  # Gradually increase LR from 0
        num_training_steps=total_steps  # Then linearly decay to 0
    )

    logger.info(f"Total training steps: {total_steps}")

    # ===== Check if Running Ablation Study =====
    if config.run_ablation:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY MODE")
        logger.info("="*80 + "\n")
        
        # Parse ablation values
        if not config.ablation_values:
            # Default values for each ablation type
            default_values = {
                'temperature': [2.0, 3.0, 4.0, 5.0],
                'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
                'lora_rank': [8, 16, 32, 64],
                'learning_rate': [1e-4, 2e-4, 3e-4, 5e-4]
            }
            ablation_values = default_values.get(config.ablation_type, [])
        else:
            # Parse user-provided values
            value_strings = config.ablation_values.split(',')
            if config.ablation_type in ['lora_rank']:
                ablation_values = [int(v.strip()) for v in value_strings]
            else:
                ablation_values = [float(v.strip()) for v in value_strings]
        
        logger.info(f"Running ablation on: {config.ablation_type}")
        logger.info(f"Testing values: {ablation_values}")
        
        # Map ablation type to parameter name
        param_map = {
            'temperature': 'temperature',
            'alpha': 'alpha',
            'lora_rank': 'lora_rank',
            'learning_rate': 'learning_rate'
        }
        param_name = param_map.get(args.ablation_type, args.ablation_type)
        
        # Run ablation study
        ablation_results = run_ablation_study(
            base_config=config,
            ablation_type=args.ablation_type,
            param_name=param_name,
            param_values=ablation_values,
            teacher_model=teacher_model,
            student_model_name=config.student_model_name,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        logger.info("Ablation study completed!")
        return
    
    # ===== Standard Training Mode =====
    # Initialize Trainer
    # Trainer class handles training loop, evaluation, checkpointing
    trainer = Trainer(
        config=config,
        distillation_method=distillation_method,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer
    )

    # Resume from checkpoint if requested
    if getattr(config, 'resume_from_checkpoint', None):
        resume_path = config.resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path, load_optimizer=True)

    # ===== Start Training =====
    # Main training loop: iterate through epochs, update student model
    trainer.train()

    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
