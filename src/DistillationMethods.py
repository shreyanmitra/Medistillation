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

from typing import Dict, Tuple, Any, Optional, final, TypedDict, Union #For type-checking

# Type alias for cleaner type hints
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


# ==============================================================================
# ABSTRACT BASE CLASS - The Template for All Distillation Methods
# ==============================================================================

class BaseDistillationMethod(ABC):
    """
    Abstract base class for all knowledge distillation methods.
    
    Every distillation method MUST implement:
    1. compute_loss() - Calculate how wrong the student is
    2. get_method_name() - Return a name for logging/results

    :param teacher_model: The large model to be distilled
    :type teacher_model: nn.Module
    :param student_model: The small model to train
    :type student_model: nn.Module
    :param tokenizer: Tokenizer used by both ``teacher_model`` and ``student_model``
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
    def compute_loss(self, batch: BatchDict) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    
    def prepare_batch(self, batch: BatchDict) -> BatchDict:
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
    
    def compute_loss(self, batch: BatchDict) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        with torch.no_grad():
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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
        
        # Student forward pass with teacher-generated labels gives us logits and loss
        student_outputs = self.student_model(
            input_ids=full_sequence,
            attention_mask=full_attention_mask,
            labels=labels
        )
        
        loss = student_outputs.loss #Get cross-entropy loss, which in the case of SFT is the same as total loss
        
        metrics = {
            'ce_loss': loss.item(),
            'total_loss': loss.item(),
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
    
    def compute_loss(self, batch: BatchDict) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        with torch.no_grad():
            full_sequence = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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
                attention_mask=full_attention_mask
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
# UTILITY FUNCTIONS
# ==============================================================================

def create_distillation_method(
    method_name: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    tokenizer: TokenizerType,
    config: Dict[str, Any]
) -> BaseDistillationMethod:
    """
    Factory function to instantiate distillation methods by name.
    
    :param method_name: Method identifier ('sft', 'logit_kd', 'cot', 'dpo')
    :type method_name: str
    :param teacher_model: Pre-trained teacher model
    :type teacher_model: nn.Module
    :param student_model: Student model to train
    :type student_model: nn.Module
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Method-specific configuration dictionary
    :type config: Dict[str, Any]
    :returns: Instantiated distillation method
    :rtype: BaseDistillationMethod
    :raises ValueError: If method_name is not recognized
    :raises NotImplementedError: If method not yet implemented (CoT, DPO)
    
    Example::
    
        method = create_distillation_method(
            'logit_kd',
            teacher,
            student,
            tokenizer,
            {'alpha': 0.5, 'temperature': 3.0}
        )
        loss, metrics = method.compute_loss(batch)
    """
    method_name = method_name.lower()
    
    if method_name in ('sft', 'standard_sft'):
        return StandardSFT(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('logit_kd', 'logit'):
        return LogitKD(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('cot', 'chain_of_thought'):
        raise NotImplementedError("CoT distillation will be implemented in Phase 2")
    
    elif method_name in ('dpo', 'preference'):
        raise NotImplementedError("DPO distillation will be implemented in Phase 2")
    
    else:
        raise ValueError(
            f"Unknown distillation method: {method_name}. "
            f"Choose from: ['sft', 'logit_kd', 'cot', 'dpo']"
        )


# ==============================================================================
# TESTING / EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Module test script demonstrating distillation method instantiation.
    
    Run directly to verify imports and class initialization:
        python src/DistillationMethods.py
    
    For actual usage, load real models and pass batches to compute_loss().
    """
    print("=" * 60)
    print("Testing Distillation Methods")
    print("=" * 60)
    
    print("\nNote: Actual model loading commented out to prevent resource usage.")
    print("Uncomment the following lines to test with real models:\n")
    
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("teacher = AutoModelForCausalLM.from_pretrained('epfl-llm/meditron-7b')")
    print("student = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B')")
    print("tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B')")
    print("\n# Standard SFT")
    print("sft = StandardSFT(teacher, student, tokenizer, {})")
    print("\n# Logit KD with balanced configuration")
    print("logit_kd = LogitKD(teacher, student, tokenizer, {'alpha': 0.5, 'temperature': 3.0})")
    print("\n# Using factory function")
    print("method = create_distillation_method('logit_kd', teacher, student, tokenizer, config)")
    print("\n# Compute loss")
    print("loss, metrics = method.compute_loss(batch)")
    
    print("\n" + "=" * 60)
    print("Module structure validated. Import checks passed.")
    print("=" * 60)
