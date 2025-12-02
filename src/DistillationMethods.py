"""
(C) 2025. Bryan Zhao, Federico Baldan, Tim Avilov, and Shreyan Mitra
Written for CSE 493S: Advanced Topics in Machine Learning Course at the University of Washington, Seattle

Distillation Methods for Medical LLM Training

This file contains an abstract base class representing a distillation method, followed by specific distillation method classes which implement the base class.
Documentation style is Sphinx.
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

from abc import ABC, abstractmethod #For the abstract base class

#Needed to interact with and train models
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Import HuggingFace transformers for tokenizer types and models
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from typing import Dict, Tuple, Any, Optional, final, TypedDict, Union, List #For type-checking

# Type alias for cleaner type hints
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


# ==============================================================================
# DEBUG HELPER - Token ID Validation
# ==============================================================================

def check_token_ids(input_ids: torch.Tensor, model: nn.Module, name: str = "input_ids", verbose: bool = True):
    """
    Validate that token IDs are within valid range for model's embedding layer.
    
    This catches the exact cause of CUDA "vectorized_gather_kernel" index out-of-bounds errors
    by checking BEFORE the forward pass instead of inside CUDA kernels.
    
    :param input_ids: Token IDs tensor to validate
    :type input_ids: torch.Tensor
    :param model: Model to check embedding size against
    :type model: nn.Module
    :param name: Name for debug output (e.g., "student_input_ids")
    :type name: str
    :param verbose: Whether to print debug info
    :type verbose: bool
    :raises AssertionError: If token IDs are out of valid range
    """
    emb = model.get_input_embeddings()
    vocab_size = emb.num_embeddings
    min_id = input_ids.min().item()
    max_id = input_ids.max().item()
    
    if verbose:
        print(f"[DEBUG CHECK] {name}: shape={tuple(input_ids.shape)}, min={min_id}, max={max_id}, vocab_size={vocab_size}")
    
    # Strong guards - will fire BEFORE CUDA error with clear message
    assert min_id >= 0, (
        f"Found negative token ID in {name}: min={min_id}. "
        f"Negative IDs (like -1 or -100) should only be in labels, never in input_ids."
    )
    assert max_id < vocab_size, (
        f"Found token ID >= vocab_size in {name}: max={max_id}, vocab_size={vocab_size}. "
        f"This means some token IDs exceed the model's embedding table size. "
        f"Check: (1) tokenizer used matches vocab alignment, (2) vocab alignment completed successfully."
    )


# ==============================================================================
# ABSTRACT BASE CLASS - The Template for All Distillation Methods
# ==============================================================================

class BaseDistillationMethod(ABC):
    """
    Abstract base class for all knowledge distillation methods.
    
    ⚠️  CRITICAL TOKENIZER REQUIREMENT ⚠️
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The tokenizer parameter MUST be compatible with BOTH teacher and student models!
    
    This implementation uses a SINGLE SHARED TOKENIZER for both models. This means:
    - Token ID 12345 must mean the same thing to both teacher and student
    - Vocabulary sizes should match (or be very similar)
    - Special tokens (pad, EOS, BOS) must align
    
    ✅ VALID COMBINATIONS (same tokenizer family):
        • Teacher: epfl-llm/meditron-70b + Student: epfl-llm/meditron-7b (both Llama)
        • Teacher: Qwen/Qwen2-72B + Student: Qwen/Qwen2-1.5B (both Qwen)
        • Teacher: meta-llama/Llama-3.1-70B + Student: TinyLlama/TinyLlama-1.1B (both Llama)
    
    ❌ INVALID COMBINATIONS (different tokenizers):
        • Teacher: epfl-llm/meditron-70b (Llama, 50K vocab) + Student: Qwen/Qwen2-1.5B (Qwen, 152K vocab)
        • Teacher: any-llama-model + Student: microsoft/phi-2 (different tokenizers)
        • Teacher: any-model + Student: different-family-model (always risky!)
    
    WHY THIS MATTERS:
    - Teacher receives student's token IDs as input → must understand them
    - LogitKD/AdaKD compare probability distributions → vocab sizes must match
    - Feature/attention matching uses same token positions → must align
    
    Use validate_tokenizer_compatibility() to check before training!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Every distillation method MUST implement:
    1. compute_loss() - Calculate how wrong the student is
    2. get_method_name() - Return a name for logging/results

    :param teacher_model: The large model to be distilled
    :type teacher_model: nn.Module
    :param student_model: The small model to train
    :type student_model: nn.Module
    :param tokenizer: Tokenizer used by both ``teacher_model`` and ``student_model`` (MUST be compatible with both!)
    :type tokenizer: TokenizerType
    :param config: Hyperparameters (learning rate, etc.)
    :type config: Dict[str, Any]
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ):
        """Constructor method
        """

        # Set class variables
        self.teacher_model: nn.Module = teacher_model
        self.student_model: nn.Module = student_model
        self.tokenizer: TokenizerType = tokenizer
        self.config: Dict[str, Any] = config
        
        # Set teacher model in eval mode since we do not want to train it
        self.teacher_model.eval()
        
        # Don't use memory for gradients to save GPU memory
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    class BatchDict(TypedDict, total=False):
        """
        A dictionary representing a batch

        :ivar prompts: Raw text prompts as a list of strings [batch_size]
        :ivar input_ids: Optional pre-tokenized prompt tokens [batch_size, prompt_length] 
        :ivar attention_mask: Optional attention mask for pre-tokenized prompts [batch_size, prompt_length]
        :ivar labels: Teacher-generated sequences (prompt + response) with -100 masking prompt tokens [batch_size, full_length]. Can be INPUT (pre-generated for offline distillation) or OUTPUT (populated by compute_loss() in online distillation for inspection/logging).
        
        Note: Either 'prompts' (raw text) OR ('input_ids' + 'attention_mask') must be provided as input.
        """
        prompts: list[str]  # Raw text prompts - primary input method
        input_ids: torch.Tensor  # Optional - for pre-tokenized inputs
        attention_mask: torch.Tensor  # Optional - for pre-tokenized inputs
        labels: torch.Tensor  # Optional INPUT or OUTPUT for inspection/logging

    
    @abstractmethod
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the loss (error) for one training batch.
        Different distillation methods will calculate loss differently.
        
        :param batch: Input batch containing prompts (and optionally pre-generated labels)
        :type batch: class:`BatchDict`
        :return: loss (a single number representing how wrong the student is, where lower loss is better) and metrics (a dictionary of metrics for logging)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        #Do something in child
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """
        Return a human-readable name for this distillation method.
        
        :returns: String representation of distillation method (e.g., "Logit-KD (α=0.5, T=3.0)" or "Standard-SFT")
        :rtype: str
        """
        #Do something in child
        pass
    
    def get_device(self) -> torch.device:
        """
        Helper method: Get the device (CPU or GPU) that the student model is on.

        :returns: The device the model is on
        :rtype: class:`torch.device`
        """
        return next(self.student_model.parameters()).device
    
    def tokenize_prompts(self, prompts: list[str]) -> Dict[str, torch.Tensor]:
        """
        Helper method: Tokenize raw text prompts into input_ids and attention_mask.
        
        :param prompts: List of raw text prompt strings
        :type prompts: list[str]
        :returns: Dictionary with 'input_ids' and 'attention_mask' tensors
        :rtype: Dict[str, torch.Tensor]
        """
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    def prepare_batch(self, batch: 'BaseDistillationMethod.BatchDict') -> 'BaseDistillationMethod.BatchDict':
        """
        Helper method: Tokenize prompts (if needed) and move batch data to correct device (GPU/CPU).

        :param batch: The batch (either raw prompts or pre-tokenized)
        :type batch: BatchDict
        :returns: Batch with tokenized tensors on the correct device
        :rtype: BatchDict
        """
        # If raw prompts are provided, tokenize them first
        if 'prompts' in batch and 'input_ids' not in batch:
            tokenized = self.tokenize_prompts(batch['prompts'])
            batch['input_ids'] = tokenized['input_ids']
            batch['attention_mask'] = tokenized['attention_mask']
        
        # Move all tensors to the correct device
        device = self.get_device()
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}


# ==============================================================================
# METHOD 1: STANDARD SUPERVISED FINE-TUNING (SFT)
# ==============================================================================

class StandardSFT(BaseDistillationMethod):
    """
    Standard Supervised Fine-Tuning (SFT) using cross-entropy loss.
    
    Trains the student model to reproduce teacher-generated text without learning 
    probability distributions. Operates in online mode: teacher generates responses 
    during training for each batch. Serves as baseline for comparison with other 
    distillation methods.
    
    **Reference**: Standard supervised learning approach, widely used baseline.
    See general NLP textbooks or:
    https://huggingface.co/docs/transformers/en/tasks/language_modeling

    :param teacher_model: Pre-trained teacher model (generates labels during training)
    :type teacher_model: class: `nn.Module`
    :param student_model: Student model to be trained
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with optional 'max_new_tokens' (default 256)
    :type config: Dict[str, Any]
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ) -> None:
        """Constructor
        """
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.max_new_tokens: int = config.get('max_new_tokens', 256) #Set max tokens for response with default value 256
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute standard cross-entropy loss for supervised fine-tuning.
        
        Generates teacher responses on-the-fly for each batch, then trains student 
        to reproduce those responses using cross-entropy loss.
        
        :param batch: Input batch containing tokenized input sequences (prompts only)
        :type batch: BatchDict
        :returns: Tuple of (loss tensor for backpropagation, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch) #Bring everything to the same device
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Generate teacher responses on-the-fly (online distillation)
        # Use inference_mode instead of no_grad for better performance
        with torch.inference_mode():
            # Disable KV cache to reduce peak GPU memory during teacher generation.
            # This prevents large KV-cache allocations + temporary int32 buffers
            # (e.g., from bitsandbytes int8 kernels) from causing OOMs when
            # the trainer already consumes most GPU memory.
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache to lower peak GPU memory
                num_beams=1  # Greedy decoding (no beam search overhead)
            )
            
            # Calculate where prompt ends and response begins
            prompt_length = input_ids.size(1)
            response_length = full_sequence.size(1) - prompt_length
            
            # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for response
            labels = torch.full_like(full_sequence, -100)
            labels[:, prompt_length:] = full_sequence[:, prompt_length:]
            
            # Store labels in batch for external inspection/logging
            batch['labels'] = labels
            
            # Create attention mask for full sequence (preserve original padding info)
            full_attention_mask = torch.cat([
                attention_mask,  # Preserve original prompt mask (including padding)
                torch.ones(
                    (attention_mask.size(0), response_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )  # All 1s for generated tokens (no padding in generation)
            ], dim=1)
        
        # DEBUG: Validate token IDs before forward pass to catch CUDA errors early
        check_token_ids(full_sequence, self.student_model, name="student_input_ids", verbose=True)
        
        # Student forward pass with teacher-generated labels gives us logits and loss
        student_outputs = self.student_model(
            input_ids=full_sequence,
            attention_mask=full_attention_mask,
            labels=labels
        )
        
        loss = student_outputs.loss #Get cross-entropy loss, which in the case of SFT is the same as total loss
        
        # Store metrics - convert to float immediately but use detach() first
        # This avoids blocking but still provides float values for logging
        metrics = {
            'ce_loss': loss.detach().item(),
            'total_loss': loss.detach().item(),
            'avg_response_length': float(response_length)
        }
        
        return loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier for logging.
        
        :returns: Method name string
        :rtype: str
        """
        return "Standard-SFT"


# ==============================================================================
# METHOD 2: LOGIT KNOWLEDGE DISTILLATION (Logit-KD)
# ==============================================================================

class LogitKD(BaseDistillationMethod):
    """
    Logit Knowledge Distillation using temperature-scaled KL divergence.
    
    Distills teacher's probability distributions (soft targets) in addition to hard labels,
    allowing student to learn relative confidences and relationships between output classes.
    
    Operates in online mode: teacher generates responses during training for each batch,
    then computes logits for the generated sequences.
    
    Combines KL divergence loss (for matching teacher distributions) with cross-entropy loss
    (for correctness), weighted by alpha parameter. Temperature parameter controls distribution softness.
    
    **Reference**: 
    Hinton, G., Vinyals, O., & Dean, J. (2015). 
    "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531

    :param teacher_model: Pre-trained teacher model to distill from
    :type teacher_model: class: `nn.Module`
    :param student_model: Student model to be trained
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with 'alpha' (KD weight, default 0.5), 'temperature' (softening factor, default 3.0), and optional 'max_new_tokens' (default 256)
    :type config: Dict[str, Any]
    :raises ValueError: If alpha not in [0, 1] or temperature <= 0
    """
    
    alpha: float
    temperature: float
    max_new_tokens: int
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ) -> None:
        """Constructor
        """
        super().__init__(teacher_model, student_model, tokenizer, config)
        
        self.alpha: float = config.get('alpha', 0.5) #Get alpha or use default
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")
        
        self.temperature: float = config.get('temperature', 3.0) #Get temperature or use default
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {self.temperature}")
        
        self.max_new_tokens: int = config.get('max_new_tokens', 256)
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Alpha (α) = {self.alpha}")
        print(f"  → Temperature (T) = {self.temperature}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined KL divergence and cross-entropy loss.
        
        Generates teacher responses on-the-fly for each batch, then computes both
        teacher and student logits on the full sequences to calculate KL divergence
        and cross-entropy losses.
        
        :param batch: Input batch containing tokenized input sequences (prompts only)
        :type batch: BatchDict
        :returns: Tuple of (combined loss tensor, metrics dictionary with kd_loss, ce_loss, and weights)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch) #Bring everything to same device
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Generate teacher responses on-the-fly (online distillation)
        # Use inference_mode instead of no_grad for better performance
        with torch.inference_mode():
            # Disable KV cache to reduce peak GPU memory during teacher generation.
            # See comment above for rationale (prevents OOM with int8 kernels).
            # Use optimized generation settings for faster inference
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache to lower peak GPU memory
                num_beams=1  # Greedy decoding (no beam search overhead)
            )
            
            # Calculate where prompt ends and response begins
            prompt_length = input_ids.size(1)
            response_length = full_sequence.size(1) - prompt_length
            
            # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for response
            labels = torch.full_like(full_sequence, -100)
            labels[:, prompt_length:] = full_sequence[:, prompt_length:]
            
            # Store labels in batch for external inspection/logging
            batch['labels'] = labels
            
            # Create attention mask for full sequence
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), response_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ], dim=1)
            
            # Teacher forward pass on full sequence to get logits (this is the first difference from SFT's compute_loss)
            teacher_outputs = self.teacher_model(
                input_ids=full_sequence,
                attention_mask=full_attention_mask,
                use_cache=True  # Enable KV cache
            )
            teacher_logits = teacher_outputs.logits
        
        # Student forward pass on full sequence. This returns logits only because we did not pass in labels. This is unlike in the SFT implementation, where we passed in labels and so both logits and labels were returned. We do this because it more efficient here to just calculate the cross-entropy loss manually. 
        student_outputs = self.student_model(
            input_ids=full_sequence,
            attention_mask=full_attention_mask
        )
        student_logits = student_outputs.logits
        
        # Apply temperature scaling for soft targets - softer distributions help student understand relationships between tokens better instead of just learning to select the top token correctly
        student_soft_logits = student_logits / self.temperature
        teacher_soft_logits = teacher_logits / self.temperature
        
        # Compute KL divergence loss (only on response tokens)
        # Mask to only compute KD loss on generated response, not prompt
        response_mask = (labels != -100)  # [batch, seq]
        
        student_log_probs = F.log_softmax(student_soft_logits, dim=-1)
        teacher_probs = F.softmax(teacher_soft_logits, dim=-1)
        
        # KL divergence per token
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)  # [batch, seq]
        
        # Apply mask and average only over response tokens
        kd_loss = (kl_per_token * response_mask).sum() / response_mask.sum()
        kd_loss = kd_loss * (self.temperature ** 2) #Multiplying by T^2 restores gradient magnitude back to normal so that it is not dominated by cross entropy loss in the the total loss expression
        
        # Compute cross-entropy loss for hard targets (only on response tokens). Need to shift due to autoregressive off-by-one.
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100  # Ignore prompt tokens
        )
        
        # Combine losses
        total_loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_weight': self.alpha,
            'ce_weight': 1.0 - self.alpha,
            'avg_response_length': float(response_length)
        }
        
        return total_loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier with hyperparameters for logging.
        
        :returns: Method name string including alpha and temperature values
        :rtype: str
        """
        return f"Logit-KD (α={self.alpha}, T={self.temperature})"


# ==============================================================================
# METHOD 3: TOKEN-ADAPTIVE KNOWLEDGE DISTILLATION (AdaKD)
# ==============================================================================

class TokenAdaptiveKD(BaseDistillationMethod):
    """
    Token-Adaptive Knowledge Distillation with per-token temperature scaling.
    
    Extends standard Logit-KD by adaptively adjusting the temperature parameter for each 
    token based on the student's prediction confidence. Tokens where the student is less 
    confident receive higher temperatures (softer distributions) to provide more learning 
    signal, while confident predictions use lower temperatures.
    
    **Reference**:
    Li, X., et al. (2025)
    "Token-Level Adaptive Knowledge Distillation for Large Language Models"
    (Related concept from adaptive distillation literature)

    :param teacher_model: Pre-trained teacher model to distill from
    :type teacher_model: class: `nn.Module`
    :param student_model: Student model to be trained
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with 'alpha' (KD weight, default 0.5), 'base_temperature' (base T, default 3.0), 'min_temperature' (min T, default 1.0), 'max_temperature' (max T, default 5.0), and optional 'max_new_tokens' (default 256)
    :type config: Dict[str, Any]
    :raises ValueError: If alpha not in [0, 1] or temperatures invalid
    """
    
    alpha: float
    base_temperature: float
    min_temperature: float
    max_temperature: float
    max_new_tokens: int
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ) -> None:
        """Constructor
        """
        super().__init__(teacher_model, student_model, tokenizer, config)
        
        self.alpha: float = config.get('alpha', 0.5)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")
        
        self.base_temperature: float = config.get('base_temperature', 3.0)
        self.min_temperature: float = config.get('min_temperature', 1.0)
        self.max_temperature: float = config.get('max_temperature', 5.0)
        
        if not (0 < self.min_temperature <= self.base_temperature <= self.max_temperature):
            raise ValueError(
                f"Temperature constraints violated: 0 < min_T <= base_T <= max_T, "
                f"got min={self.min_temperature}, base={self.base_temperature}, max={self.max_temperature}"
            )
        
        self.max_new_tokens: int = config.get('max_new_tokens', 256)
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Alpha (α) = {self.alpha}")
        print(f"  → Base Temperature = {self.base_temperature}")
        print(f"  → Temperature Range: [{self.min_temperature}, {self.max_temperature}]")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute token-adaptive KL divergence and cross-entropy loss.
        
        Adjusts temperature per token based on student confidence (entropy of predictions).
        Higher entropy (uncertainty) → higher temperature → softer targets → more learning signal.
        
        :param batch: Input batch containing tokenized input sequences (prompts only)
        :type batch: BatchDict
        :returns: Tuple of (combined loss tensor, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch)
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Generate teacher responses on-the-fly
        with torch.no_grad():
            # Disable KV cache to reduce peak GPU memory during teacher generation.
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disable KV cache to lower peak GPU memory
            )
            
            prompt_length = input_ids.size(1)
            response_length = full_sequence.size(1) - prompt_length
            
            labels = torch.full_like(full_sequence, -100)
            labels[:, prompt_length:] = full_sequence[:, prompt_length:]
            batch['labels'] = labels
            
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), response_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ], dim=1)
            
            # Teacher forward pass
            teacher_outputs = self.teacher_model(
                input_ids=full_sequence,
                attention_mask=full_attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=full_sequence,
            attention_mask=full_attention_mask
        )
        student_logits = student_outputs.logits
        
        # Calculate per-token uncertainty (entropy) for adaptive temperature
        student_probs = F.softmax(student_logits, dim=-1)
        entropy = -(student_probs * torch.log(student_probs + 1e-10)).sum(dim=-1)  # [batch, seq]
        
        # Normalize entropy to [0, 1] range
        max_entropy = torch.log(torch.tensor(student_logits.size(-1), dtype=torch.float, device=entropy.device))
        normalized_entropy = entropy / max_entropy
        
        # Map entropy to temperature: high entropy → high temperature
        adaptive_temp = self.min_temperature + (self.max_temperature - self.min_temperature) * normalized_entropy
        adaptive_temp = adaptive_temp.unsqueeze(-1)  # [batch, seq, 1] for broadcasting
        
        # Apply adaptive temperature scaling
        student_soft_logits = student_logits / adaptive_temp
        teacher_soft_logits = teacher_logits / adaptive_temp
        
        # Compute KL divergence loss (only on response tokens)
        response_mask = (labels != -100)  # [batch, seq]
        
        student_log_probs = F.log_softmax(student_soft_logits, dim=-1)
        teacher_probs = F.softmax(teacher_soft_logits, dim=-1)
        
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)  # [batch, seq]
        
        # Scale by temperature squared (per token)
        kl_per_token = kl_per_token * (adaptive_temp.squeeze(-1) ** 2)
        
        # Apply mask and average
        kd_loss = (kl_per_token * response_mask).sum() / response_mask.sum()
        
        # Compute cross-entropy loss for hard targets
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Combine losses
        total_loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        
        # Calculate average temperature used (for logging)
        avg_temp = (adaptive_temp.squeeze(-1) * response_mask).sum() / response_mask.sum()
        
        metrics = {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'avg_temperature': avg_temp.item(),
            'avg_entropy': (entropy * response_mask).sum().item() / response_mask.sum().item(),
            'avg_response_length': float(response_length)
        }
        
        return total_loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier with hyperparameters for logging.
        
        :returns: Method name string
        :rtype: str
        """
        return f"AdaKD (α={self.alpha}, T∈[{self.min_temperature},{self.max_temperature}])"


# ==============================================================================
# METHOD 4: CHAIN-OF-THOUGHT DISTILLATION (CoT)
# ==============================================================================

class ChainOfThoughtDistillation(BaseDistillationMethod):
    """
    Chain-of-Thought (CoT) Knowledge Distillation using rationale generation.
    
    Distills both the teacher's reasoning process (chain-of-thought) and final answers.
    Teacher generates responses with explicit reasoning steps, and student learns to 
    replicate both the reasoning process and final conclusions.
    
    Operates in online mode: teacher generates CoT responses during training for each batch.
    Uses cross-entropy loss on the full reasoning chain, encouraging the student to 
    develop similar step-by-step reasoning capabilities.
    
    **Reference**:
    Ho, N., Schmid, L., & Yun, S. (2023).
    "Large Language Models Are Reasoning Teachers"
    https://arxiv.org/abs/2212.10071
    
    Wei, J., Wang, X., Schuurmans, D., et al. (2022).
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    https://arxiv.org/abs/2201.11903

    :param teacher_model: Pre-trained teacher model (generates CoT responses during training)
    :type teacher_model: class: `nn.Module`
    :param student_model: Student model to be trained
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with optional 'max_new_tokens' (default 512), optional 'cot_prompt' (default "Let's think step by step:"), optional 'num_rationales' (default 1, number of diverse reasoning chains per problem), and optional 'sampling_temperature' (default 0.7, for diverse generation when num_rationales > 1)
    :type config: Dict[str, Any]
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ) -> None:
        """Constructor
        """
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.max_new_tokens: int = config.get('max_new_tokens', 512) #Longer default for reasoning chains
        self.cot_prompt: str = config.get('cot_prompt', "Let's think step by step:") #Prompt to elicit reasoning
        self.num_rationales: int = config.get('num_rationales', 1) #Number of diverse reasoning chains to generate per problem
        self.sampling_temperature: float = config.get('sampling_temperature', 0.7) #Temperature for diverse rationale generation
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
        print(f"  → CoT prompt: '{self.cot_prompt}'")
        print(f"  → Num rationales: {self.num_rationales}")
        if self.num_rationales > 1:
            print(f"  → Sampling temperature: {self.sampling_temperature}")
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cross-entropy loss on teacher's chain-of-thought reasoning.
        
        Generates teacher CoT responses on-the-fly for each batch. If num_rationales > 1,
        generates multiple diverse reasoning chains per problem to enrich training data,
        as described in Ho et al. (2023).
        
        :param batch: Input batch containing tokenized input sequences (prompts only)
        :type batch: BatchDict
        :returns: Tuple of (loss tensor for backpropagation, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch) #Bring everything to the same device
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.size(0)
        
        # Augment prompts with CoT instruction to elicit reasoning from teacher
        if 'prompts' in batch:
            cot_prompts = [f"{prompt}\n{self.cot_prompt}" for prompt in batch['prompts']]
            cot_tokenized = self.tokenize_prompts(cot_prompts)
            input_ids = cot_tokenized['input_ids'].to(self.get_device())
            attention_mask = cot_tokenized['attention_mask'].to(self.get_device())
        
        # Generate multiple diverse CoT rationales per problem (Fine-tune-CoT approach)
        all_losses = []
        all_response_lengths = []
        
        for rationale_idx in range(self.num_rationales):
            # Use sampling for diverse rationales (if num_rationales > 1), greedy otherwise
            do_sample = self.num_rationales > 1
            temperature = self.sampling_temperature if do_sample else 1.0
            
            # Generate teacher CoT responses on-the-fly (online distillation)
            with torch.no_grad():
                # Disable KV cache to reduce peak GPU memory during teacher generation.
                full_sequence = self.teacher_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False  # Disable KV cache to lower peak GPU memory
                )
                
                # Calculate where prompt ends and response begins
                prompt_length = input_ids.size(1)
                response_length = full_sequence.size(1) - prompt_length
                
                # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for CoT response
                labels = torch.full_like(full_sequence, -100)
                labels[:, prompt_length:] = full_sequence[:, prompt_length:]
                
                # Store labels from first rationale in batch for external inspection/logging
                if rationale_idx == 0:
                    batch['labels'] = labels
                
                # Create attention mask for full sequence (preserve original padding info)
                full_attention_mask = torch.cat([
                    attention_mask,  # Preserve original prompt mask (including padding)
                    torch.ones(
                        (attention_mask.size(0), response_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )  # All 1s for generated tokens (no padding in generation)
                ], dim=1)
            
            # Student forward pass with teacher-generated CoT labels gives us logits and loss
            student_outputs = self.student_model(
                input_ids=full_sequence,
                attention_mask=full_attention_mask,
                labels=labels
            )
            
            loss = student_outputs.loss #Get cross-entropy loss on reasoning chain
            all_losses.append(loss)
            all_response_lengths.append(float(response_length))
        
        # Average loss across all rationales
        total_loss = torch.stack(all_losses).mean()
        avg_response_length = sum(all_response_lengths) / len(all_response_lengths)
        
        metrics = {
            'ce_loss': total_loss.item(),
            'total_loss': total_loss.item(),
            'avg_response_length': avg_response_length,
            'num_rationales': self.num_rationales
        }
        
        return total_loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier for logging.
        
        :returns: Method name string
        :rtype: str
        """
        return "Chain-of-Thought"


# ==============================================================================
# METHOD 5: INTERMEDIATE FEATURE MATCHING / FITNETS
# ==============================================================================

class IntermediateFeatureMatching(BaseDistillationMethod):
    """
    Intermediate Feature Matching (FitNets-style) for hidden state distillation.
    
    Transfers knowledge by matching the student's internal representations (hidden states)
    to the teacher's at corresponding layers. Uses MSE loss between teacher and student
    hidden states, optionally with projection layers to handle dimension mismatches.
    
    Operates in online mode: teacher generates responses and computes hidden states
    during training. Combines feature matching loss with standard cross-entropy.
    
    **Reference**:
    Romero, A., et al. (2014).
    "FitNets: Hints for Thin Deep Nets"
    https://arxiv.org/abs/1412.6550

    :param teacher_model: Pre-trained teacher model
    :type teacher_model: class: `nn.Module`
    :param student_model: Student model to be trained
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with 'alpha' (feature weight, default 0.5), 
        'layer_mapping' (dict mapping student layers to teacher layers, e.g., {6: 12}),
        optional 'max_new_tokens' (default 256), and optional 'use_projections' (default True)
    :type config: Dict[str, Any]
    :raises ValueError: If alpha not in [0, 1] or layer_mapping invalid
    """
    
    alpha: float
    layer_mapping: Dict[int, int]
    use_projections: bool
    max_new_tokens: int
    projections: nn.ModuleDict
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ) -> None:
        """Constructor
        """
        super().__init__(teacher_model, student_model, tokenizer, config)
        
        self.alpha: float = config.get('alpha', 0.5)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")
        
        self.layer_mapping: Dict[int, int] = config.get('layer_mapping', {})
        if not self.layer_mapping:
            raise ValueError("layer_mapping must be provided (e.g., {6: 12} maps student layer 6 to teacher layer 12)")
        
        self.use_projections: bool = config.get('use_projections', True)
        self.max_new_tokens: int = config.get('max_new_tokens', 256)
        
        # Create projection layers if dimensions don't match
        self.projections = nn.ModuleDict()
        if self.use_projections:
            student_hidden_size = student_model.config.hidden_size
            teacher_hidden_size = teacher_model.config.hidden_size
            
            if student_hidden_size != teacher_hidden_size:
                for student_layer in self.layer_mapping.keys():
                    self.projections[str(student_layer)] = nn.Linear(
                        student_hidden_size,
                        teacher_hidden_size,
                        bias=False
                    ).to(self.get_device())
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Alpha (α) = {self.alpha}")
        print(f"  → Layer mapping: {self.layer_mapping}")
        print(f"  → Use projections: {self.use_projections}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined feature matching and cross-entropy loss.
        
        Matches student's intermediate hidden states to teacher's using MSE loss,
        combined with standard cross-entropy loss on outputs.
        
        :param batch: Input batch containing tokenized input sequences (prompts only)
        :type batch: BatchDict
        :returns: Tuple of (combined loss tensor, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch)
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Generate teacher responses
        with torch.no_grad():
            # Disable KV cache to reduce peak GPU memory during teacher generation.
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disable KV cache to lower peak GPU memory
            )
            
            prompt_length = input_ids.size(1)
            response_length = full_sequence.size(1) - prompt_length
            
            labels = torch.full_like(full_sequence, -100)
            labels[:, prompt_length:] = full_sequence[:, prompt_length:]
            batch['labels'] = labels
            
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), response_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ], dim=1)
            
            # Teacher forward pass with hidden states
            teacher_outputs = self.teacher_model(
                input_ids=full_sequence,
                attention_mask=full_attention_mask,
                output_hidden_states=True
            )
            teacher_hidden_states = teacher_outputs.hidden_states  # Tuple of [batch, seq, hidden]
        
        # Student forward pass with hidden states and labels
        student_outputs = self.student_model(
            input_ids=full_sequence,
            attention_mask=full_attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        student_hidden_states = student_outputs.hidden_states  # Tuple of [batch, seq, hidden]
        
        # Cross-entropy loss
        ce_loss = student_outputs.loss
        
        # Feature matching loss
        feature_loss = 0.0
        response_mask = (labels != -100).unsqueeze(-1)  # [batch, seq, 1]
        
        for student_layer, teacher_layer in self.layer_mapping.items():
            # Get hidden states (add 1 because layer 0 is embedding, layer 1 is first transformer layer)
            student_hidden = student_hidden_states[student_layer + 1]  # [batch, seq, hidden_s]
            teacher_hidden = teacher_hidden_states[teacher_layer + 1]  # [batch, seq, hidden_t]
            
            # Apply projection if needed
            if str(student_layer) in self.projections:
                student_hidden = self.projections[str(student_layer)](student_hidden)
            
            # MSE loss only on response tokens
            mse_per_token = F.mse_loss(student_hidden, teacher_hidden, reduction='none')  # [batch, seq, hidden]
            mse_per_token = mse_per_token.mean(dim=-1, keepdim=True)  # [batch, seq, 1]
            
            # Mask and average
            masked_mse = (mse_per_token * response_mask).sum() / response_mask.sum()
            feature_loss += masked_mse
        
        # Average across layers
        feature_loss = feature_loss / len(self.layer_mapping)
        
        # Combine losses
        total_loss = self.alpha * feature_loss + (1.0 - self.alpha) * ce_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'feature_loss': feature_loss.item(),
            'ce_loss': ce_loss.item(),
            'feature_weight': self.alpha,
            'ce_weight': 1.0 - self.alpha,
            'avg_response_length': float(response_length)
        }
        
        return total_loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier with hyperparameters for logging.
        
        :returns: Method name string
        :rtype: str
        """
        return f"FitNets (α={self.alpha}, layers={len(self.layer_mapping)})"


# ==============================================================================
# METHOD 6: ATTENTION DISTILLATION
# ==============================================================================

class AttentionDistillation(BaseDistillationMethod):
    """
    Attention Transfer for distilling attention patterns.
    
    Transfers knowledge by matching the student's self-attention maps to the teacher's
    at specified layers. Teaches the student which parts of the input to focus on,
    beyond just matching outputs or hidden states.
    
    Uses MSE loss between attention weight matrices. Can specify which attention heads
    and layers to match.
    
    **Reference**:
    Zagoruyko, S., & Komodakis, N. (2016).
    "Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    (Originally for vision; adapted for LLMs by matching self-attention patterns)

    :param teacher_model: Pre-trained teacher model
    :type teacher_model: class: `nn.Module`
    :param student_model: Student model to be trained
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with 'alpha' (attention weight, default 0.5),
        'layer_mapping' (dict mapping student layers to teacher layers),
        optional 'max_new_tokens' (default 256), and optional 'match_all_heads' (default True)
    :type config: Dict[str, Any]
    :raises ValueError: If alpha not in [0, 1] or layer_mapping invalid
    """
    
    alpha: float
    layer_mapping: Dict[int, int]
    match_all_heads: bool
    max_new_tokens: int
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ) -> None:
        """Constructor
        """
        super().__init__(teacher_model, student_model, tokenizer, config)
        
        self.alpha: float = config.get('alpha', 0.5)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")
        
        self.layer_mapping: Dict[int, int] = config.get('layer_mapping', {})
        if not self.layer_mapping:
            raise ValueError("layer_mapping must be provided (e.g., {6: 12})")
        
        self.match_all_heads: bool = config.get('match_all_heads', True)
        self.max_new_tokens: int = config.get('max_new_tokens', 256)
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Alpha (α) = {self.alpha}")
        print(f"  → Layer mapping: {self.layer_mapping}")
        print(f"  → Match all heads: {self.match_all_heads}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined attention matching and cross-entropy loss.
        
        Matches student's attention patterns to teacher's using MSE loss on attention weights.
        
        :param batch: Input batch containing tokenized input sequences (prompts only)
        :type batch: BatchDict
        :returns: Tuple of (combined loss tensor, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch)
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Generate teacher responses
        with torch.no_grad():
            # Disable KV cache to reduce peak GPU memory during teacher generation.
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disable KV cache to lower peak GPU memory
            )
            
            prompt_length = input_ids.size(1)
            response_length = full_sequence.size(1) - prompt_length
            
            labels = torch.full_like(full_sequence, -100)
            labels[:, prompt_length:] = full_sequence[:, prompt_length:]
            batch['labels'] = labels
            
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), response_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ], dim=1)
            
            # Teacher forward pass with attentions
            teacher_outputs = self.teacher_model(
                input_ids=full_sequence,
                attention_mask=full_attention_mask,
                output_attentions=True
            )
            teacher_attentions = teacher_outputs.attentions  # Tuple of [batch, heads, seq, seq]
        
        # Student forward pass with attentions and labels
        student_outputs = self.student_model(
            input_ids=full_sequence,
            attention_mask=full_attention_mask,
            labels=labels,
            output_attentions=True
        )
        student_attentions = student_outputs.attentions  # Tuple of [batch, heads, seq, seq]
        
        # Cross-entropy loss
        ce_loss = student_outputs.loss
        
        # Attention matching loss
        attention_loss = 0.0
        response_mask_2d = (labels != -100)  # [batch, seq]
        
        for student_layer, teacher_layer in self.layer_mapping.items():
            student_attn = student_attentions[student_layer]  # [batch, heads, seq, seq]
            teacher_attn = teacher_attentions[teacher_layer]  # [batch, heads, seq, seq]
            
            if self.match_all_heads:
                # Average over heads for simpler matching
                student_attn = student_attn.mean(dim=1)  # [batch, seq, seq]
                teacher_attn = teacher_attn.mean(dim=1)  # [batch, seq, seq]
                
                # MSE loss on attention matrices, masked to response tokens
                # Mask both query and key dimensions
                mask_query = response_mask_2d.unsqueeze(-1).float()  # [batch, seq, 1]
                mask_key = response_mask_2d.unsqueeze(1).float()  # [batch, 1, seq]
                mask_2d = mask_query * mask_key  # [batch, seq, seq]
                
                mse_per_position = F.mse_loss(student_attn, teacher_attn, reduction='none')  # [batch, seq, seq]
                masked_mse = (mse_per_position * mask_2d).sum() / mask_2d.sum()
                attention_loss += masked_mse
            else:
                # Match each head separately
                num_heads = student_attn.size(1)
                for head in range(num_heads):
                    student_head = student_attn[:, head, :, :]  # [batch, seq, seq]
                    teacher_head = teacher_attn[:, head, :, :]  # [batch, seq, seq]
                    
                    mask_query = response_mask_2d.unsqueeze(-1).float()
                    mask_key = response_mask_2d.unsqueeze(1).float()
                    mask_2d = mask_query * mask_key
                    
                    mse_per_position = F.mse_loss(student_head, teacher_head, reduction='none')
                    masked_mse = (mse_per_position * mask_2d).sum() / mask_2d.sum()
                    attention_loss += masked_mse / num_heads
        
        # Average across layers
        attention_loss = attention_loss / len(self.layer_mapping)
        
        # Combine losses
        total_loss = self.alpha * attention_loss + (1.0 - self.alpha) * ce_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'attention_loss': attention_loss.item(),
            'ce_loss': ce_loss.item(),
            'attention_weight': self.alpha,
            'ce_weight': 1.0 - self.alpha,
            'avg_response_length': float(response_length)
        }
        
        return total_loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier with hyperparameters for logging.
        
        :returns: Method name string
        :rtype: str
        """
        return f"Attention-Distill (α={self.alpha}, layers={len(self.layer_mapping)})"


# ==============================================================================
# REINFORCEMENT LEARNING BASE CLASS
# ==============================================================================

class BaseRLDistillationMethod(BaseDistillationMethod):
    """
    Base class for Reinforcement Learning-based distillation methods.
    
    Extends BaseDistillationMethod with RL-specific capabilities:
    - Trajectory generation (student exploration via sampling)
    - Reward computation (teacher evaluation of student's actions)
    - Policy updates (RL algorithms like REINFORCE, PPO, DPO)
    
    Unlike supervised methods where teacher generates fixed responses and student
    imitates them, RL methods have the student generate responses (exploration)
    and receive feedback (rewards) from the teacher to improve its policy.
    
    :param teacher_model: Pre-trained teacher model (acts as evaluator/reward source)
    :type teacher_model: nn.Module
    :param student_model: Student model to be trained (the policy being optimized)
    :type student_model: nn.Module
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with RL-specific parameters:
        - 'gamma' (float): Discount factor for future rewards (default 0.99)
        - 'max_new_tokens' (int): Maximum tokens to generate (default 256)
        - 'temperature' (float): Sampling temperature for exploration (default 1.0)
    :type config: Dict[str, Any]
    """
    
    gamma: float
    max_new_tokens: int
    temperature: float
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ):
        """Constructor"""
        super().__init__(teacher_model, student_model, tokenizer, config)
        
        # RL-specific hyperparameters
        self.gamma: float = config.get('gamma', 0.99)  # Discount factor
        self.max_new_tokens: int = config.get('max_new_tokens', 256)
        self.temperature: float = config.get('temperature', 1.0)  # Exploration temperature
    
    @abstractmethod
    def generate_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate trajectory via student model's current policy with sampling.
        
        Student explores by sampling from its probability distribution rather than
        greedy decoding. This enables diverse responses needed for RL training.
        
        :param prompts: List of text prompts to generate from
        :type prompts: List[str]
        :returns: Dictionary containing:
            - 'sequences': Generated token sequences [batch, seq_len]
            - 'log_probs': Log probabilities of sampled actions [batch, seq_len]
            - 'full_sequences': Complete sequences including prompts [batch, full_len]
            - 'prompt_lengths': Length of each prompt [batch]
        :rtype: Dict[str, torch.Tensor]
        """
        pass
    
    @abstractmethod
    def compute_rewards(
        self, 
        rollout: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute rewards for generated trajectory using teacher as evaluator.
        
        Different RL methods compute rewards differently:
        - On-Policy: Teacher's log-prob of student's actions
        - BOND: Ranking among multiple samples
        - SPIN: Implicit preference-based rewards
        
        :param rollout: Output from generate_rollout()
        :type rollout: Dict[str, torch.Tensor]
        :returns: Reward values [batch, seq_len] or [batch]
        :rtype: torch.Tensor
        """
        pass
    
    @abstractmethod
    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Update student policy using RL algorithm (REINFORCE, PPO, DPO, etc.).
        
        Implements the policy gradient or preference optimization update rule
        to make the student more likely to take high-reward actions.
        
        :param rollout: Generated trajectory with log probabilities
        :type rollout: Dict[str, torch.Tensor]
        :param rewards: Reward signal from compute_rewards()
        :type rewards: torch.Tensor
        :returns: Tuple of (policy loss for backprop, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        pass
    
    def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        RL-compatible loss computation. Overrides BaseDistillationMethod.compute_loss().
        
        Instead of supervised loss on fixed labels, this:
        1. Generates rollouts (student exploration)
        2. Computes rewards (teacher evaluation)
        3. Updates policy (RL algorithm)
        
        :param batch: Input batch with prompts or input_ids
        :type batch: BatchDict
        :returns: Tuple of (policy loss, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        # Extract prompts from batch
        if 'prompts' in batch:
            prompts = batch['prompts']
        else:
            # Decode from input_ids if needed
            prompts = self.tokenizer.batch_decode(
                batch['input_ids'],
                skip_special_tokens=True
            )
        
        # Step 1: Generate trajectory from current policy
        rollout = self.generate_rollout(prompts)
        
        # Step 2: Compute rewards
        rewards = self.compute_rewards(rollout)
        
        # Step 3: Update policy via RL algorithm
        policy_loss, metrics = self.update_policy(rollout, rewards)
        
        return policy_loss, metrics
    
    def compute_discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted cumulative rewards for variance reduction.
        
        R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        
        :param rewards: Per-timestep rewards [batch, seq_len]
        :type rewards: torch.Tensor
        :returns: Discounted cumulative rewards [batch, seq_len]
        :rtype: torch.Tensor
        """
        batch_size, seq_len = rewards.shape
        discounted = torch.zeros_like(rewards)
        
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                discounted[:, t] = rewards[:, t]
            else:
                discounted[:, t] = rewards[:, t] + self.gamma * discounted[:, t + 1]
        
        return discounted


# ==============================================================================
# RL METHOD 1: ON-POLICY DISTILLATION (REINFORCE)
# ==============================================================================

class OnPolicyDistillation(BaseRLDistillationMethod):
    """
    On-Policy Distillation using REINFORCE algorithm.
    
    Student generates responses token-by-token via sampling, and teacher evaluates
    each token choice in real-time. Uses vanilla policy gradient (REINFORCE) to
    increase probability of actions that received high rewards from teacher.
    
    **Reference**:
    Agarwal et al. (2024) - On-Policy Distillation (related concept)
    Williams, R. J. (1992) - Simple Statistical Gradient-Following Algorithms (REINFORCE)
    
    :param teacher_model: Teacher model (acts as reward function)
    :type teacher_model: nn.Module
    :param student_model: Student model (policy to optimize)
    :type student_model: nn.Module
    :param tokenizer: Shared tokenizer
    :type tokenizer: TokenizerType
    :param config: Configuration with optional 'gamma' (default 0.99), 'temperature' (default 1.0),
        'max_new_tokens' (default 256), 'entropy_coef' (default 0.01 for exploration bonus)
    :type config: Dict[str, Any]
    """
    
    entropy_coef: float
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ):
        """Constructor"""
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.entropy_coef: float = config.get('entropy_coef', 0.01)  # Entropy bonus for exploration
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Gamma (γ) = {self.gamma}")
        print(f"  → Temperature = {self.temperature}")
        print(f"  → Entropy coefficient = {self.entropy_coef}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def generate_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate trajectory by sampling from student's policy.
        
        Uses temperature-controlled sampling for exploration.
        """
        batch_size = len(prompts)
        
        # Tokenize prompts into input tensors
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.get_device())
        attention_mask = inputs['attention_mask'].to(self.get_device())
        prompt_lengths = attention_mask.sum(dim=1)  # Track where prompts end for reward computation later
        
        # Storage for trajectory - we collect each sampled token and its log probability
        all_actions = []  # List of [batch, 1] tensors, one per generation step
        all_log_probs = []  # List of [batch, 1] tensors, log π(a_t|s_t)
        
        # Generate token-by-token with sampling (NOT greedy decoding like supervised methods)
        for step in range(self.max_new_tokens):
            # Forward pass with gradient tracking (needed for policy gradient update later)
            # Unlike supervised methods, we need gradients during generation
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            next_token_logits = outputs.logits[:, -1, :]  # Extract logits for next token [batch, vocab_size]
            
            # Apply temperature for exploration - higher T = more random, lower T = more greedy
            # Temperature scaling flattens/sharpens the distribution without changing argmax
            scaled_logits = next_token_logits / self.temperature
            
            # Sample action from policy distribution (RL requires exploration!)
            probs = F.softmax(scaled_logits, dim=-1)  # Convert logits to probability distribution
            action = torch.multinomial(probs, num_samples=1)  # Sample one token [batch, 1]
            
            # Compute log probability of the chosen action for REINFORCE update
            log_probs = F.log_softmax(scaled_logits, dim=-1)  # log π(·|s_t)
            action_log_prob = log_probs.gather(1, action)  # Extract log π(a_t|s_t) for selected action
            
            # Store trajectory data for later policy update
            all_actions.append(action)
            all_log_probs.append(action_log_prob)
            
            # Update context for next generation step (autoregressive)
            input_ids = torch.cat([input_ids, action], dim=1)  # Append generated token
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.get_device())
            ], dim=1)  # Extend attention mask to include new token
            
            # Early stopping if all sequences generate EOS
            if (action == self.tokenizer.eos_token_id).all():
                break
        
        # Concatenate all timesteps into single tensors
        sequences = torch.cat(all_actions, dim=1)  # [batch, seq_len] - generated tokens only
        log_probs = torch.cat(all_log_probs, dim=1)  # [batch, seq_len] - log π(a|s) for each token
        
        return {
            'sequences': sequences,  # Student's generated tokens
            'log_probs': log_probs,  # Log probabilities under student policy
            'full_sequences': input_ids,  # Prompt + generated tokens
            'prompt_lengths': prompt_lengths  # Where prompt ends (for masking)
        }
    
    def compute_rewards(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute rewards as teacher's log probability of student's actions.
        
        High reward if teacher would have made the same choice.
        """
        full_sequences = rollout['full_sequences']  # Prompt + student generation
        student_actions = rollout['sequences']  # Just the generated tokens
        
        with torch.no_grad():  # Don't backprop through reward computation
            # Teacher evaluates the full sequence including student's choices
            # Teacher sees what student generated and assigns quality score
            teacher_outputs = self.teacher_model(
                input_ids=full_sequences,
                attention_mask=torch.ones_like(full_sequences)  # All tokens are valid (no padding)
            )
            teacher_logits = teacher_outputs.logits  # [batch, full_seq_len, vocab_size]
            
            # Get teacher's probability distribution over vocabulary at each position
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            
            # Extract teacher's log-prob for student's actual actions
            # Due to autoregressive off-by-one: teacher_logits[:, t] predicts token at position t+1
            # So teacher_logits[:, prompt_len-1:-1] aligns with student_actions
            prompt_len = full_sequences.size(1) - student_actions.size(1)
            relevant_teacher_logits = teacher_logits[:, prompt_len-1:-1, :]  # Slice to match student actions
            
            # Reward = log P_teacher(student's token | context)
            # High reward when teacher would have made same choice as student
            rewards = relevant_teacher_logits.log_softmax(dim=-1).gather(
                dim=2,
                index=student_actions.unsqueeze(-1)  # Gather teacher's log-prob for student's action
            ).squeeze(-1)  # [batch, seq_len] - one reward per generated token
        
        return rewards
    
    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        REINFORCE policy gradient update.
        
        Loss = -E[log π(a|s) * (R - baseline)]
        """
        log_probs = rollout['log_probs']  # [batch, seq_len] - log π(a_t|s_t) for each action
        
        # Compute discounted rewards: R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        # This gives credit to early actions for future rewards
        discounted_rewards = self.compute_discounted_rewards(rewards)
        
        # Use discounted rewards as advantages (simple baseline: no value function)
        # Normalize advantages for training stability (prevents explosion/vanishing gradients)
        advantages = discounted_rewards
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Z-score normalization
        
        # REINFORCE policy gradient: ∇J = E[∇log π(a|s) * A(s,a)]
        # We minimize negative expected return: -E[log π(a|s) * advantage]
        # Detach advantages so we don't backprop through reward computation
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Optional: Add entropy bonus for exploration to prevent premature convergence
        # Entropy = -Σ p(a) log p(a), higher entropy = more exploration
        if self.entropy_coef > 0:
            probs = torch.exp(log_probs)  # Convert log probabilities back to probabilities
            entropy = -(probs * log_probs).sum(dim=-1).mean()  # Average entropy over batch and sequence
            policy_loss = policy_loss - self.entropy_coef * entropy  # Subtract because we want to maximize entropy
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'total_loss': policy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'avg_response_length': float(rollout['sequences'].size(1))
        }
        
        return policy_loss, metrics
    
    def get_method_name(self) -> str:
        """Return method identifier for logging."""
        return f"On-Policy-RL (γ={self.gamma}, T={self.temperature})"


# ==============================================================================
# RL METHOD 2: PPO DISTILLATION
# ==============================================================================

class PPODistillation(BaseRLDistillationMethod):
    """
    Proximal Policy Optimization (PPO) for distillation.
    
    Extends on-policy distillation with clipped objective to prevent destructively
    large policy updates. More stable than vanilla REINFORCE.
    
    **Reference**:
    Schulman et al. (2017) - Proximal Policy Optimization Algorithms
    https://arxiv.org/abs/1707.06347
    
    :param teacher_model: Teacher model (reward source)
    :type teacher_model: nn.Module
    :param student_model: Student model (policy)
    :type student_model: nn.Module
    :param tokenizer: Shared tokenizer
    :type tokenizer: TokenizerType
    :param config: Configuration with 'epsilon' (clip range, default 0.2), 'gamma',
        'temperature', 'max_new_tokens', 'value_coef' (default 0.5), 'entropy_coef' (default 0.01)
    :type config: Dict[str, Any]
    """
    
    epsilon: float
    value_coef: float
    entropy_coef: float
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ):
        """Constructor"""
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.epsilon: float = config.get('epsilon', 0.2)
        self.value_coef: float = config.get('value_coef', 0.5)
        self.entropy_coef: float = config.get('entropy_coef', 0.01)
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Epsilon (ε) = {self.epsilon}")
        print(f"  → Gamma (γ) = {self.gamma}")
        print(f"  → Temperature = {self.temperature}")
        print(f"  → Value coef = {self.value_coef}")
        print(f"  → Entropy coef = {self.entropy_coef}")
    
    def generate_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Generate trajectory with sampling (same as OnPolicyDistillation)"""
        batch_size = len(prompts)
        
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.get_device())
        attention_mask = inputs['attention_mask'].to(self.get_device())
        prompt_lengths = attention_mask.sum(dim=1)
        
        all_actions = []
        all_log_probs = []
        
        for step in range(self.max_new_tokens):
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            scaled_logits = next_token_logits / self.temperature
            
            probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            action_log_prob = log_probs.gather(1, action)
            
            all_actions.append(action)
            all_log_probs.append(action_log_prob)
            
            input_ids = torch.cat([input_ids, action], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.get_device())
            ], dim=1)
            
            if (action == self.tokenizer.eos_token_id).all():
                break
        
        sequences = torch.cat(all_actions, dim=1)
        log_probs = torch.cat(all_log_probs, dim=1)
        
        return {
            'sequences': sequences,
            'log_probs': log_probs,
            'full_sequences': input_ids,
            'prompt_lengths': prompt_lengths,
            'old_log_probs': log_probs.detach()  # Store old policy's log-probs for computing PPO ratio
        }
    
    def compute_rewards(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute rewards (same as OnPolicyDistillation)"""
        full_sequences = rollout['full_sequences']  # Prompt + student generation
        student_actions = rollout['sequences']  # Just the generated tokens
        
        with torch.no_grad():  # No gradients needed for reward computation
            # Teacher evaluates student's trajectory
            teacher_outputs = self.teacher_model(
                input_ids=full_sequences,
                attention_mask=torch.ones_like(full_sequences)
            )
            teacher_logits = teacher_outputs.logits  # [batch, full_seq_len, vocab_size]
            
            # Extract teacher log-probs for student's actions (handle autoregressive offset)
            prompt_len = full_sequences.size(1) - student_actions.size(1)
            relevant_teacher_logits = teacher_logits[:, prompt_len-1:-1, :]
            
            # Reward = log P_teacher(student's token | context)
            rewards = relevant_teacher_logits.log_softmax(dim=-1).gather(
                dim=2,
                index=student_actions.unsqueeze(-1)
            ).squeeze(-1)  # [batch, seq_len]
        
        return rewards
    
    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        PPO clipped objective update.
        
        Prevents large policy updates via ratio clipping.
        """
        old_log_probs = rollout['old_log_probs']  # Log-probs from policy that generated rollout
        sequences = rollout['sequences']  # Generated tokens
        full_sequences = rollout['full_sequences']  # Prompt + generation
        
        # Re-evaluate the same actions with the CURRENT policy (after gradient updates)
        # This is the key difference from REINFORCE - we compare old and new policies
        current_outputs = self.student_model(
            input_ids=full_sequences,
            attention_mask=torch.ones_like(full_sequences)
        )
        current_logits = current_outputs.logits  # Current policy's logits [batch, full_seq, vocab]
        
        # Extract logits for generated tokens (handle autoregressive offset)
        prompt_len = full_sequences.size(1) - sequences.size(1)
        relevant_logits = current_logits[:, prompt_len-1:-1, :]  # Align with sequences
        
        # Compute log-probs under CURRENT policy for the actions we took
        current_log_probs = F.log_softmax(relevant_logits / self.temperature, dim=-1).gather(
            dim=2,
            index=sequences.unsqueeze(-1)  # Extract log π_new(a|s) for each action
        ).squeeze(-1)  # [batch, seq_len]
        
        # Compute advantages from rewards (same as REINFORCE)
        discounted_rewards = self.compute_discounted_rewards(rewards)
        advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # PPO clipped objective: prevents policy from changing too fast
        # Ratio = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
        ratio = torch.exp(current_log_probs - old_log_probs)  # How much policy changed
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)  # Limit to [1-ε, 1+ε]
        
        # Take minimum of clipped and unclipped objectives (conservative update)
        # If advantage > 0 (good action), we want to increase π(a|s), but not too much (ratio < 1+ε)
        # If advantage < 0 (bad action), we want to decrease π(a|s), but not too much (ratio > 1-ε)
        policy_loss_unclipped = ratio * advantages.detach()  # Standard policy gradient
        policy_loss_clipped = clipped_ratio * advantages.detach()  # Clipped version
        policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()  # Pessimistic bound
        
        # Entropy bonus for exploration (prevents policy from collapsing)
        probs = torch.exp(current_log_probs)
        entropy = -(probs * current_log_probs).sum(dim=-1).mean()
        
        # Total loss: policy loss - entropy bonus (subtract because we maximize entropy)
        total_loss = policy_loss - self.entropy_coef * entropy
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_ratio': ratio.mean().item(),  # Track how much policy is changing
            'avg_response_length': float(sequences.size(1))
        }
        
        return total_loss, metrics
    
    def get_method_name(self) -> str:
        """Return method identifier for logging."""
        return f"PPO (ε={self.epsilon}, γ={self.gamma})"


# ==============================================================================
# RL METHOD 3: BEST-OF-N DISTILLATION (BOND)
# ==============================================================================

class BestOfNDistillation(BaseRLDistillationMethod):
    """
    Best-of-N Distillation (BOND).
    
    Student generates N diverse samples per prompt, teacher ranks them, and student
    learns to reproduce the best sample in a single pass. Distills the implicit
    reward of "best-of-N sampling" into the student's policy.
    
    **Reference**:
    Yang et al. (2024) - BOND: Aligning LLMs with Best-of-N Distillation
    
    :param teacher_model: Teacher model (scores/ranks samples)
    :type teacher_model: nn.Module
    :param student_model: Student model (generates N samples)
    :type student_model: nn.Module
    :param tokenizer: Shared tokenizer
    :type tokenizer: TokenizerType
    :param config: Configuration with 'num_samples' (N, default 16), 'temperature',
        'max_new_tokens', 'use_ranking' (bool, default False for binary rewards)
    :type config: Dict[str, Any]
    """
    
    num_samples: int
    use_ranking: bool
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ):
        """Constructor"""
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.num_samples: int = config.get('num_samples', 16)  # N candidate samples to generate per prompt
        self.use_ranking: bool = config.get('use_ranking', False)  # Optional: use full ranking instead of just best
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Num samples (N) = {self.num_samples}")
        print(f"  → Temperature = {self.temperature}")
        print(f"  → Use ranking: {self.use_ranking}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def generate_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate N diverse samples per prompt.
        """
        batch_size = len(prompts)
        
        # Storage for all N samples - we'll collect them separately and teacher will rank
        all_samples = []  # Will be List of N tensors, each [batch, seq_len_i]
        all_log_probs = []  # List of N tensors, each [batch, seq_len_i]
        all_full_sequences = []  # List of N tensors, each [batch, full_len_i]
        
        # Generate N independent samples per prompt (memory intensive!)
        for n in range(self.num_samples):
            # Tokenize fresh each time to reset generation state
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(self.get_device())
            attention_mask = inputs['attention_mask'].to(self.get_device())
            prompt_lengths = attention_mask.sum(dim=1)  # Track prompt boundaries
            
            actions = []  # Tokens for this sample
            log_probs_list = []  # Log-probs for this sample
            
            # Generate one complete sample via sampling
            for step in range(self.max_new_tokens):
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
                scaled_logits = next_token_logits / self.temperature  # Temperature for diversity
                
                # Sample action (each of N samples will be different due to randomness)
                probs = F.softmax(scaled_logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                
                # Track log-prob for this choice
                log_probs = F.log_softmax(scaled_logits, dim=-1)
                action_log_prob = log_probs.gather(1, action)
                
                actions.append(action)
                log_probs_list.append(action_log_prob)
                
                # Update context for next token
                input_ids = torch.cat([input_ids, action], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.get_device())
                ], dim=1)
                
                # Early stopping if all sequences hit EOS
                if (action == self.tokenizer.eos_token_id).all():
                    break
            
            # Concatenate this sample's tokens and log-probs
            sample_seq = torch.cat(actions, dim=1) if actions else torch.zeros((batch_size, 0), device=self.get_device())
            sample_log_probs = torch.cat(log_probs_list, dim=1) if log_probs_list else torch.zeros((batch_size, 0), device=self.get_device())
            
            all_samples.append(sample_seq)  # Store this sample
            all_log_probs.append(sample_log_probs)  # Store its log-probs
            all_full_sequences.append(input_ids)  # Store full sequence (prompt + sample)
        
        return {
            'samples': all_samples,  # List of N tensors [batch, seq_len] - generated tokens for each sample
            'log_probs': all_log_probs,  # List of N tensors [batch, seq_len] - log π(a|s) for each sample
            'full_sequences': all_full_sequences,  # List of N tensors [batch, full_len] - prompt + generation
            'num_samples': self.num_samples,
            'prompt_lengths': prompt_lengths
        }
    
    def compute_rewards(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Teacher scores all N samples and returns best indices.
        """
        samples = rollout['samples']  # List of N sample tensors
        full_sequences = rollout['full_sequences']  # List of N full sequence tensors
        batch_size = full_sequences[0].size(0)
        
        scores_per_sample = []  # Will collect teacher's score for each of N samples
        
        with torch.no_grad():  # Teacher evaluation doesn't need gradients
            # Evaluate each of the N samples
            for full_seq, sample_seq in zip(full_sequences, samples):
                if sample_seq.size(1) == 0:
                    # Empty sequence (shouldn't happen but handle gracefully)
                    scores_per_sample.append(torch.full((batch_size,), float('-inf'), device=self.get_device()))
                    continue
                
                # Teacher evaluates this sample by computing log-likelihood
                teacher_outputs = self.teacher_model(
                    input_ids=full_seq,
                    attention_mask=torch.ones_like(full_seq)
                )
                teacher_logits = teacher_outputs.logits  # [batch, full_seq_len, vocab]
                
                # Compute score = average log probability under teacher (quality metric)
                # Higher score = teacher thinks this is a better response
                prompt_len = full_seq.size(1) - sample_seq.size(1)
                relevant_logits = teacher_logits[:, prompt_len-1:-1, :]  # Align with sample tokens
                
                # Get teacher's log-prob for each token in this sample
                token_log_probs = relevant_logits.log_softmax(dim=-1).gather(
                    dim=2,
                    index=sample_seq.unsqueeze(-1)
                ).squeeze(-1)  # [batch, seq_len]
                
                # Average over sequence length to get overall quality score
                avg_score = token_log_probs.mean(dim=1)  # [batch] - one score per batch element
                scores_per_sample.append(avg_score)
        
        # Stack scores: [num_samples, batch] - each row is scores for one sample across batch
        all_scores = torch.stack(scores_per_sample, dim=0)
        
        # Find best sample index for each batch element (argmax over samples dimension)
        # best_indices[i] tells us which of the N samples was best for batch element i
        best_indices = all_scores.argmax(dim=0)  # [batch]
        
        return {
            'best_indices': best_indices,  # Which sample won for each batch element
            'all_scores': all_scores  # Full score matrix for analysis
        }
    
    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        rewards: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Learn to reproduce the best sample.
        """
        best_indices = rewards['best_indices']  # [batch] - which of N samples was best
        all_scores = rewards['all_scores']  # [num_samples, batch] - all scores
        
        samples = rollout['samples']  # List of N sample tensors
        log_probs_list = rollout['log_probs']  # List of N log-prob tensors
        batch_size = best_indices.size(0)
        
        # Extract best sample for each batch element and train via supervised loss
        # Unlike REINFORCE/PPO, BOND uses supervised learning on the winner
        loss = 0.0
        total_tokens = 0
        
        for b in range(batch_size):
            # Find which sample won for this batch element
            best_idx = best_indices[b].item()  # Integer in [0, N-1]
            best_log_probs = log_probs_list[best_idx][b:b+1]  # Extract this batch element's log-probs [1, seq_len]
            
            if best_log_probs.size(1) > 0:  # Only if non-empty sequence
                # Maximize log probability of the winning trajectory (supervised learning)
                # We already have log π(a|s) from generation, so just maximize it
                loss += -best_log_probs.mean()  # Negative because we minimize loss
                total_tokens += best_log_probs.size(1)
        
        # Average loss over batch
        loss = loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.get_device())
        
        metrics = {
            'policy_loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'total_loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'avg_best_score': all_scores.max(dim=0)[0].mean().item(),  # Average score of winners
            'avg_worst_score': all_scores.min(dim=0)[0].mean().item(),  # Average score of losers
            'score_range': (all_scores.max(dim=0)[0] - all_scores.min(dim=0)[0]).mean().item(),  # Diversity metric
            'avg_response_length': total_tokens / batch_size if batch_size > 0 else 0.0
        }
        
        return loss, metrics
    
    def get_method_name(self) -> str:
        """Return method identifier for logging."""
        return f"BOND (N={self.num_samples}, T={self.temperature})"


# ==============================================================================
# RL METHOD 4: SPIN (SELF-PLAY DISTILLATION)
# ==============================================================================

class SPINDistillation(BaseRLDistillationMethod):
    """
    Self-Play based INstruction optimization (SPIN).
    
    Iterative self-play where student's own generations become "dispreferred" examples
    and teacher's generations are "preferred". Uses DPO-style preference optimization
    without explicit reward model.
    
    **Reference**:
    Chen et al. (2024a) - Self-Play Fine-Tuning Converts Weak Language Models to Strong
    
    :param teacher_model: Teacher model (generates preferred responses)
    :type teacher_model: nn.Module
    :param student_model: Student model (generates dispreferred responses, gets optimized)
    :type student_model: nn.Module
    :param tokenizer: Shared tokenizer
    :type tokenizer: TokenizerType
    :param config: Configuration with 'beta' (DPO temperature, default 0.1), 'temperature',
        'max_new_tokens'
    :type config: Dict[str, Any]
    """
    
    beta: float
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: TokenizerType,
        config: Dict[str, Any]
    ):
        """Constructor"""
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.beta: float = config.get('beta', 0.1)  # DPO temperature for preference strength
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Beta (β) = {self.beta}")
        print(f"  → Temperature = {self.temperature}")
        print(f"  → Max new tokens: {self.max_new_tokens}")
    
    def generate_rollout(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate both student (dispreferred) and teacher (preferred) responses.
        """
        batch_size = len(prompts)
        
        # ===== STUDENT GENERATION (Dispreferred) =====
        # Student samples from its current policy - these become the "losing" examples
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.get_device())
        attention_mask = inputs['attention_mask'].to(self.get_device())
        prompt_lengths = attention_mask.sum(dim=1)
        
        student_actions = []  # Student's sampled tokens
        student_log_probs = []  # Log-probs for tracking
        
        # Generate student response token-by-token with sampling
        for step in range(self.max_new_tokens):
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
            scaled_logits = next_token_logits / self.temperature  # Exploration temperature
            
            # Sample from student's policy (diverse responses)
            probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Track log-prob for this action
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            action_log_prob = log_probs.gather(1, action)
            
            student_actions.append(action)
            student_log_probs.append(action_log_prob)
            
            # Update context
            input_ids = torch.cat([input_ids, action], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.get_device())
            ], dim=1)
            
            # Early stopping
            if (action == self.tokenizer.eos_token_id).all():
                break
        
        # Concatenate student's generation
        student_sequences = torch.cat(student_actions, dim=1) if student_actions else torch.zeros((batch_size, 0), device=self.get_device())
        student_full = input_ids  # Prompt + student generation
        
        # ===== TEACHER GENERATION (Preferred) =====
        # Teacher generates high-quality response - these become the "winning" examples
        inputs_teacher = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids_teacher = inputs_teacher['input_ids'].to(self.get_device())
        attention_mask_teacher = inputs_teacher['attention_mask'].to(self.get_device())
        
        with torch.no_grad():  # Teacher doesn't get trained
            teacher_full = self.teacher_model.generate(
                input_ids=input_ids_teacher,
                attention_mask=attention_mask_teacher,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Teacher uses greedy decoding (high quality)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract teacher's generated portion (remove prompt)
        teacher_sequences = teacher_full[:, input_ids_teacher.size(1):]
        
        return {
            'student_sequences': student_sequences,  # Student's generated tokens (dispreferred)
            'student_full': student_full,  # Prompt + student generation
            'teacher_sequences': teacher_sequences,  # Teacher's generated tokens (preferred)
            'teacher_full': teacher_full,  # Prompt + teacher generation
            'prompt_lengths': prompt_lengths
        }
    
    def compute_rewards(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        For SPIN, rewards are implicit in DPO loss. Return rollout for update_policy.
        """
        # SPIN doesn't use explicit rewards - preference is implicit in DPO loss
        # Just pass through the rollout data to update_policy
        return rollout
    
    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        rewards: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        DPO-style preference optimization.
        
        Maximize margin between preferred (teacher) and dispreferred (student) responses.
        """
        student_full = rollout['student_full']  # Prompt + student's response
        teacher_full = rollout['teacher_full']  # Prompt + teacher's response
        student_sequences = rollout['student_sequences']  # Just student tokens
        teacher_sequences = rollout['teacher_sequences']  # Just teacher tokens
        
        # Handle edge case of empty sequences
        if student_sequences.size(1) == 0 or teacher_sequences.size(1) == 0:
            return torch.tensor(0.0, device=self.get_device()), {
                'dpo_loss': 0.0,
                'total_loss': 0.0,
                'margin': 0.0,
                'accuracy': 0.0
            }
        
        # ===== Compute log probs under CURRENT student policy =====
        # We need student's probability for both its own response (dispreferred) and teacher's response (preferred)
        
        # Student evaluates its own generated response
        student_outputs = self.student_model(
            input_ids=student_full,  # Full sequence including prompt
            attention_mask=torch.ones_like(student_full)
        )
        student_logits = student_outputs.logits  # [batch, student_full_len, vocab]
        
        # Student evaluates teacher's response (what teacher did)
        teacher_outputs = self.student_model(  # Note: still student model evaluating
            input_ids=teacher_full,  # Full sequence with teacher's response
            attention_mask=torch.ones_like(teacher_full)
        )
        teacher_logits = teacher_outputs.logits  # [batch, teacher_full_len, vocab]
        
        # Compute sequence-level log probabilities (sum over tokens)
        # For student's own response: log π_student(student_response | prompt)
        prompt_len_student = student_full.size(1) - student_sequences.size(1)
        student_log_prob = student_logits[:, prompt_len_student-1:-1, :].log_softmax(dim=-1).gather(
            dim=2,
            index=student_sequences.unsqueeze(-1)  # Extract log-prob for each actual token
        ).squeeze(-1).sum(dim=1)  # Sum over sequence length → [batch]
        
        # For teacher's response: log π_student(teacher_response | prompt)
        prompt_len_teacher = teacher_full.size(1) - teacher_sequences.size(1)
        teacher_log_prob = teacher_logits[:, prompt_len_teacher-1:-1, :].log_softmax(dim=-1).gather(
            dim=2,
            index=teacher_sequences.unsqueeze(-1)  # Extract log-prob for each teacher token
        ).squeeze(-1).sum(dim=1)  # Sum over sequence length → [batch]
        
        # ===== DPO Loss Computation =====
        # DPO maximizes: log σ(β * (log π(y_w) - log π(y_l)))
        # where y_w = preferred (teacher), y_l = dispreferred (student)
        # We minimize negative of this
        log_ratio = teacher_log_prob - student_log_prob  # How much better is teacher response?
        dpo_loss = -F.logsigmoid(self.beta * log_ratio).mean()  # Maximize preference margin
        
        # ===== Metrics =====
        margin = log_ratio.mean().item()  # Average log-prob difference (should increase over training)
        accuracy = (log_ratio > 0).float().mean().item()  # Fraction where teacher > student
        
        metrics = {
            'dpo_loss': dpo_loss.item(),
            'total_loss': dpo_loss.item(),
            'margin': margin,  # Positive = student prefers teacher response
            'accuracy': accuracy,  # Fraction of examples where teacher is preferred
            'avg_student_logprob': student_log_prob.mean().item(),
            'avg_teacher_logprob': teacher_log_prob.mean().item()
        }
        
        return dpo_loss, metrics
    
    def get_method_name(self) -> str:
        """Return method identifier for logging."""
        return f"SPIN (β={self.beta})"


# ==============================================================================
# TOKENIZER COMPATIBILITY VALIDATION
# ==============================================================================

def validate_tokenizer_compatibility(
    teacher_model: nn.Module,
    student_model: nn.Module,
    tokenizer: TokenizerType,
    method_name: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Validate tokenizer compatibility and return alignment requirements.
    
    Instead of raising errors on vocabulary mismatch, this function now returns
    information about what alignment is needed. This allows the trainer to
    automatically expand student vocabulary when --align_vocabularies flag is used.
    
    :param teacher_model: The teacher model
    :type teacher_model: nn.Module
    :param student_model: The student model
    :type student_model: nn.Module
    :param tokenizer: The tokenizer to validate
    :type tokenizer: TokenizerType
    :param method_name: Name of distillation method (for error messages)
    :type method_name: str
    :param strict: If True, raises error on mismatch without --align_vocabularies flag
    :type strict: bool
    :returns: Dictionary with alignment requirements
    :rtype: Dict[str, Any]
    :raises ValueError: If tokenizer is incompatible and strict=True
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Robustly get vocabulary sizes from model config or tokenizer.
    def _safe_get_vocab_size(model, tk=None):
        cfg = getattr(model, 'config', None)
        if cfg is not None:
            for key in ('vocab_size', 'n_vocab'):
                if hasattr(cfg, key):
                    try:
                        val = getattr(cfg, key)
                        if isinstance(val, int) and val > 0:
                            return val
                    except Exception:
                        pass
        # Fall back to tokenizer if provided
        if tk is not None:
            try:
                if hasattr(tk, 'vocab_size') and tk.vocab_size:
                    return tk.vocab_size
                return len(tk)
            except Exception:
                pass
        return None

    tokenizer_vocab_size = len(tokenizer)
    teacher_vocab_size = _safe_get_vocab_size(teacher_model, tokenizer) or tokenizer_vocab_size
    student_vocab_size = _safe_get_vocab_size(student_model, tokenizer) or tokenizer_vocab_size
    
    # Check if vocab sizes match
    vocab_mismatch = teacher_vocab_size != student_vocab_size
    
    if not vocab_mismatch:
        logger.info(f"✅ Tokenizer compatibility validated for method '{method_name}'")
        logger.info(f"   Teacher vocab: {teacher_vocab_size:,}")
        logger.info(f"   Student vocab: {student_vocab_size:,}")
        logger.info(f"   Tokenizer vocab: {tokenizer_vocab_size:,}")
        return {
            'requires_alignment': False,
            'teacher_vocab_size': teacher_vocab_size,
            'student_vocab_size': student_vocab_size,
            'num_extra_tokens': 0
        }
    
    # Vocabulary mismatch detected
    diff = abs(teacher_vocab_size - student_vocab_size)
    
    error_msg = f"\n{'='*80}\n"
    error_msg += "⚠️  VOCABULARY SIZE MISMATCH DETECTED ⚠️\n"
    error_msg += f"{'='*80}\n"
    error_msg += f"Method: {method_name}\n\n"
    error_msg += f"  Teacher vocab size: {teacher_vocab_size:,}\n"
    error_msg += f"  Student vocab size: {student_vocab_size:,}\n"
    error_msg += f"  Difference:         {diff:,} tokens\n"
    error_msg += "\n"
    error_msg += "This will cause training to FAIL because:\n"
    error_msg += "  • Logit-based methods require matching vocabulary dimensions\n"
    error_msg += "  • KL divergence cannot be computed between different vocab sizes\n"
    error_msg += "\n"
    error_msg += "SOLUTION:\n"
    error_msg += "  Add the --align_vocabularies flag to automatically expand student vocabulary:\n\n"
    error_msg += "  python src/Trainer.py \\\n"
    error_msg += f"    --teacher_model {teacher_model.config._name_or_path} \\\n"
    error_msg += f"    --student_model {student_model.config._name_or_path} \\\n"
    error_msg += f"    --method {method_name} \\\n"
    error_msg += "    --align_vocabularies  # ← ADD THIS FLAG\n"
    error_msg += "\n"
    error_msg += "This will:\n"
    error_msg += f"  1. Add {diff} tokens to student tokenizer\n"
    error_msg += f"  2. Resize student embeddings: {student_vocab_size:,} → {teacher_vocab_size:,}\n"
    error_msg += "  3. Initialize new embeddings with mean of existing embeddings\n"
    error_msg += "  4. Enable full knowledge transfer via logit distillation\n"
    error_msg += f"{'='*80}\n"
    
    if strict:
        raise ValueError(error_msg)
    else:
        logger.warning(error_msg)
        return {
            'requires_alignment': True,
            'teacher_vocab_size': teacher_vocab_size,
            'student_vocab_size': student_vocab_size,
            'num_extra_tokens': diff
        }


def align_student_vocabulary_to_teacher(
    student_model: nn.Module,
    student_tokenizer: TokenizerType,
    teacher_tokenizer: TokenizerType,
    logger: Any = None
) -> Tuple[nn.Module, TokenizerType]:
    """
    Add teacher's extra tokens to student vocabulary for perfect alignment.
    
    This is the PREFERRED method for handling vocabulary mismatches in cross-tokenizer
    distillation (e.g., Meditron-70B → Llama-2-7B with 17 extra medical tokens).
    
    Process:
        1. Find tokens in teacher but not in student
        2. Add those tokens to student tokenizer
        3. Resize student model embeddings
        4. Initialize new embeddings with mean of existing embeddings
    
    :param student_model: Student model to modify
    :type student_model: nn.Module
    :param student_tokenizer: Student tokenizer to expand
    :type student_tokenizer: TokenizerType
    :param teacher_tokenizer: Teacher tokenizer with extra tokens
    :type teacher_tokenizer: TokenizerType
    :param logger: Logger for status messages
    :type logger: Any
    :returns: Tuple of (updated_student_model, updated_student_tokenizer)
    :rtype: Tuple[nn.Module, TokenizerType]
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("🔧 VOCABULARY ALIGNMENT: Expanding Student Vocabulary")
    logger.info("="*60)
    
    # Find extra tokens
    teacher_vocab = set(teacher_tokenizer.get_vocab().keys())
    student_vocab = set(student_tokenizer.get_vocab().keys())
    extra_tokens = sorted(list(teacher_vocab - student_vocab))
    
    if not extra_tokens:
        logger.info("✅ Vocabularies already match - no expansion needed")
        logger.info("="*60)
        return student_model, student_tokenizer
    
    original_student_vocab_size = len(student_tokenizer)
    
    logger.info(f"📊 Found {len(extra_tokens)} extra tokens in teacher vocabulary")
    logger.info(f"\n🔍 Extra tokens (showing first 10):")
    for i, token in enumerate(extra_tokens[:10], 1):
        token_id = teacher_tokenizer.get_vocab()[token]
        try:
            decoded = teacher_tokenizer.decode([token_id]).strip()
            if decoded:
                logger.info(f"   {i:2d}. Token ID {token_id:>5d}: '{token}' → \"{decoded}\"")
            else:
                logger.info(f"   {i:2d}. Token ID {token_id:>5d}: '{token}' (special/control token)")
        except:
            logger.info(f"   {i:2d}. Token ID {token_id:>5d}: '{token}'")
    
    if len(extra_tokens) > 10:
        logger.info(f"   ... and {len(extra_tokens) - 10} more tokens")
    
    # Add tokens to student tokenizer
    logger.info(f"\n🔧 Adding {len(extra_tokens)} tokens to student tokenizer...")
    num_added = student_tokenizer.add_tokens(extra_tokens)
    new_student_vocab_size = len(student_tokenizer)
    
    logger.info(f"✅ Added {num_added} tokens")
    logger.info(f"   Original student vocab: {original_student_vocab_size:,}")
    logger.info(f"   New student vocab:      {new_student_vocab_size:,}")
    logger.info(f"   Teacher vocab:          {len(teacher_tokenizer):,}")
    
    # Resize student model embeddings
    logger.info(f"\n🔧 Resizing student model embeddings...")
    student_model.resize_token_embeddings(new_student_vocab_size)
    
    # Initialize new embeddings with mean of existing embeddings
    logger.info(f"🔧 Initializing new token embeddings...")
    with torch.no_grad():
        # Get embedding layer
        embed_layer = student_model.get_input_embeddings()
        
        # Calculate mean of existing embeddings
        existing_embeddings = embed_layer.weight[:original_student_vocab_size]
        mean_embedding = existing_embeddings.mean(dim=0)
        
        # Initialize new token embeddings with mean
        embed_layer.weight[original_student_vocab_size:] = mean_embedding
        
        logger.info(f"✅ Initialized {num_added} new embeddings")
        logger.info(f"   Method: Mean of existing {original_student_vocab_size:,} embeddings")
        logger.info(f"   Embedding dimension: {mean_embedding.shape[0]}")
    
    # Verify alignment
    logger.info(f"\n✅ VOCABULARY ALIGNMENT COMPLETE!")
    logger.info(f"   Teacher vocab: {len(teacher_tokenizer):,}")
    logger.info(f"   Student vocab: {len(student_tokenizer):,}")
    match = len(teacher_tokenizer) == len(student_tokenizer)
    logger.info(f"   Match: {'✅ YES' if match else '❌ NO'}")
    
    # Post-alignment sanity checks (catch issues BEFORE training)
    logger.info(f"\n🔍 POST-ALIGNMENT SANITY CHECKS:")
    teacher_ids = list(teacher_tokenizer.get_vocab().values())
    student_ids = list(student_tokenizer.get_vocab().values())
    
    logger.info(f"   Teacher max token ID: {max(teacher_ids):,}")
    logger.info(f"   Student max token ID: {max(student_ids):,}")
    logger.info(f"   Teacher len(tokenizer): {len(teacher_tokenizer):,}")
    logger.info(f"   Student len(tokenizer): {len(student_tokenizer):,}")
    
    student_emb_size = student_model.get_input_embeddings().num_embeddings
    logger.info(f"   Student embedding size: {student_emb_size:,}")
    
    # Critical check: embedding size must match tokenizer vocab size
    if student_emb_size != len(student_tokenizer):
        logger.warning(f"   ⚠️  WARNING: Embedding size ({student_emb_size:,}) != tokenizer vocab ({len(student_tokenizer):,})")
        logger.warning(f"   This may cause index out-of-bounds errors during training!")
    else:
        logger.info(f"   ✅ Embedding size matches tokenizer vocab")
    
    # Check that max token IDs are within bounds
    if max(student_ids) >= student_emb_size:
        logger.error(f"   ❌ ERROR: Max student token ID ({max(student_ids):,}) >= embedding size ({student_emb_size:,})")
        logger.error(f"   This WILL cause CUDA errors during training!")
    else:
        logger.info(f"   ✅ All token IDs within embedding range")
    
    logger.info("="*60)
    
    return student_model, student_tokenizer


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_custom_distillation_method(
    method_name: str,
    compute_loss_fn: callable,
    teacher_model: nn.Module,
    student_model: nn.Module,
    tokenizer: TokenizerType,
    config: Dict[str, Any]
) -> BaseDistillationMethod:
    """
    Factory function to create a custom distillation method dynamically.
    
    Allows users to define their own compute_loss logic without creating a new class.
    The compute_loss function will have access to self (the distillation method instance).
    
    :param method_name: Name for the custom distillation method (for logging)
    :type method_name: str
    :param compute_loss_fn: User-defined function with signature:
        compute_loss_fn(self, batch: BatchDict) -> Tuple[torch.Tensor, Dict[str, float]]
        The function receives the method instance (self) and batch, returns (loss, metrics).
    :type compute_loss_fn: callable
    :param teacher_model: Pre-trained teacher model
    :type teacher_model: nn.Module
    :param student_model: Student model to train
    :type student_model: nn.Module
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Method-specific configuration dictionary (accessible via self.config)
    :type config: Dict[str, Any]
    :returns: Custom distillation method instance
    :rtype: BaseDistillationMethod
    
    Example::
    
        # Define custom loss function
        def my_custom_loss(self, batch):
            batch = self.prepare_batch(batch)
            
            # Custom logic here
            with torch.no_grad():
                teacher_outputs = self.teacher_model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=self.config.get('max_new_tokens', 128)
                )
            
            # Your custom loss computation
            student_outputs = self.student_model(...)
            loss = my_custom_loss_calculation(student_outputs, teacher_outputs)
            
            metrics = {'total_loss': loss.item()}
            return loss, metrics
        
        # Create method
        custom_method = create_custom_distillation_method(
            method_name='My-Custom-Method',
            compute_loss_fn=my_custom_loss,
            teacher_model=teacher,
            student_model=student,
            tokenizer=tokenizer,
            config={'max_new_tokens': 128, 'custom_param': 0.5}
        )
        
        # Use like any other method
        loss, metrics = custom_method.compute_loss(batch)
    """
    
    # Dynamically create a class that inherits from BaseDistillationMethod
    class CustomDistillationMethod(BaseDistillationMethod):
        def __init__(
            self,
            teacher_model: nn.Module,
            student_model: nn.Module,
            tokenizer: TokenizerType,
            config: Dict[str, Any],
            name: str,
            loss_fn: callable
        ):
            super().__init__(teacher_model, student_model, tokenizer, config)
            self._method_name = name
            self._compute_loss_fn = loss_fn
            print(f"Initialized custom method: {self.get_method_name()}")
        
        def compute_loss(self, batch: 'BaseDistillationMethod.BatchDict') -> Tuple[torch.Tensor, Dict[str, float]]:
            """Calls user-defined compute_loss function"""
            return self._compute_loss_fn(self, batch)
        
        def get_method_name(self) -> str:
            """Returns user-defined method name"""
            return self._method_name
    
    # Instantiate and return the custom class
    return CustomDistillationMethod(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        config=config,
        name=method_name,
        loss_fn=compute_loss_fn
    )

def create_distillation_method(
    method_name: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    tokenizer: TokenizerType,
    config: Dict[str, Any]
) -> BaseDistillationMethod:
    """
    Factory function to instantiate distillation methods by name.
    
    :param method_name: Method identifier. Supported methods:
        - Supervised: 'sft', 'logit_kd', 'adakd', 'cot', 'fitnets', 'attention'
        - RL-based: 'on_policy' (or 'reinforce'), 'ppo', 'bond' (or 'best_of_n'), 'spin' (or 'self_play')
    :type method_name: str
    :param teacher_model: Pre-trained teacher model
    :type teacher_model: nn.Module
    :param student_model: Student model to train
    :type student_model: nn.Module
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Method-specific configuration dictionary. Required keys vary by method:
        - SFT: optional 'max_new_tokens' (default 256)
        - Logit-KD: 'alpha' (default 0.5), 'temperature' (default 3.0), optional 'max_new_tokens' (default 256)
        - AdaKD: 'alpha' (default 0.5), 'base_temperature' (default 3.0), 'min_temperature' (default 1.0),
                 'max_temperature' (default 5.0), optional 'max_new_tokens' (default 256)
        - CoT: optional 'max_new_tokens' (default 512), optional 'cot_prompt', 
               optional 'num_rationales' (default 1), optional 'sampling_temperature' (default 0.7)
        - FitNets: 'alpha' (default 0.5), 'layer_mapping' (required), optional 'use_projections' (default True),
                   optional 'max_new_tokens' (default 256)
        - Attention: 'alpha' (default 0.5), 'layer_mapping' (required), optional 'match_all_heads' (default True),
                     optional 'max_new_tokens' (default 256)
        - On-Policy: 'max_new_tokens' (default 256), optional 'temperature' (default 1.0), 
                     optional 'gamma' (default 0.99)
        - PPO: 'max_new_tokens' (default 256), optional 'temperature' (default 1.0), optional 'gamma' (default 0.99),
               optional 'clip_epsilon' (default 0.2), optional 'kl_penalty' (default 0.0)
        - BOND: 'num_samples' (default 4), 'max_new_tokens' (default 256), optional 'temperature' (default 0.8)
        - SPIN: 'max_new_tokens' (default 256), optional 'temperature' (default 1.0), optional 'beta' (default 0.1)
    :type config: Dict[str, Any]
    :returns: Instantiated distillation method
    :rtype: BaseDistillationMethod
    :raises ValueError: If method_name is not recognized
    
    Example::
    
        # Standard SFT
        sft = create_distillation_method('sft', teacher, student, tokenizer, {})
        
        # Logit-KD with custom hyperparameters
        logit_kd = create_distillation_method(
            'logit_kd',
            teacher,
            student,
            tokenizer,
            {'alpha': 0.5, 'temperature': 3.0}
        )
        
        # Token-Adaptive KD
        adakd = create_distillation_method(
            'adakd',
            teacher,
            student,
            tokenizer,
            {'alpha': 0.5, 'base_temperature': 3.0, 'min_temperature': 1.0, 'max_temperature': 5.0}
        )
        
        # Chain-of-Thought
        cot = create_distillation_method('cot', teacher, student, tokenizer, {})
        
        # FitNets with layer mapping
        fitnets = create_distillation_method(
            'fitnets',
            teacher,
            student,
            tokenizer,
            {'alpha': 0.5, 'layer_mapping': {6: 12, 12: 24}}
        )
        
        # Attention Distillation
        attention = create_distillation_method(
            'attention',
            teacher,
            student,
            tokenizer,
            {'alpha': 0.5, 'layer_mapping': {6: 12}}
        )
        
        # RL Methods
        
        # On-Policy Distillation (REINFORCE)
        on_policy = create_distillation_method(
            'on_policy',
            teacher,
            student,
            tokenizer,
            {'gamma': 0.99, 'temperature': 1.0}
        )
        
        # PPO Distillation
        ppo = create_distillation_method(
            'ppo',
            teacher,
            student,
            tokenizer,
            {'epsilon': 0.2, 'gamma': 0.99}
        )
        
        # BOND (Best-of-N)
        bond = create_distillation_method(
            'bond',
            teacher,
            student,
            tokenizer,
            {'num_samples': 16, 'temperature': 1.0}
        )
        
        # SPIN (Self-Play)
        spin = create_distillation_method(
            'spin',
            teacher,
            student,
            tokenizer,
            {'beta': 0.1}
        )
        
        # Compute loss
        loss, metrics = method.compute_loss(batch)
    """
    method_name = method_name.lower()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⚠️  VALIDATE TOKENIZER COMPATIBILITY UPFRONT ⚠️
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # This check logs vocabulary size info but doesn't raise errors.
    # The Trainer.py handles alignment before calling this function.
    validate_tokenizer_compatibility(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        method_name=method_name,
        strict=False  # Don't raise error - Trainer.py handles alignment
    )
    
    # ===== SUPERVISED DISTILLATION METHODS =====
    
    if method_name in ('sft', 'standard_sft'):
        return StandardSFT(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('logit_kd', 'logit'):
        return LogitKD(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('adakd', 'token_adaptive_kd', 'token_adaptive'):
        return TokenAdaptiveKD(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('cot', 'chain_of_thought'):
        return ChainOfThoughtDistillation(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('fitnets', 'intermediate_feature_matching', 'feature_matching'):
        return IntermediateFeatureMatching(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('attention', 'attention_distillation', 'attention_transfer'):
        return AttentionDistillation(teacher_model, student_model, tokenizer, config)
    
    # ===== RL-BASED DISTILLATION METHODS =====
    
    elif method_name in ('on_policy', 'on_policy_distillation', 'reinforce'):
        return OnPolicyDistillation(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('ppo', 'ppo_distillation'):
        return PPODistillation(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('bond', 'best_of_n', 'best_of_n_distillation'):
        return BestOfNDistillation(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('spin', 'self_play', 'spin_distillation'):
        return SPINDistillation(teacher_model, student_model, tokenizer, config)
    
    else:
        raise ValueError(
            f"Unknown distillation method: {method_name}. "
            f"Supervised: ['sft', 'logit_kd', 'adakd', 'cot', 'fitnets', 'attention']. "
            f"RL-based: ['on_policy', 'ppo', 'bond', 'spin']"
        )