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
# METHOD 3: CHAIN-OF-THOUGHT DISTILLATION (CoT)
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
    
    def compute_loss(self, batch: BatchDict) -> Tuple[torch.Tensor, Dict[str, float]]:
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
                full_sequence = self.teacher_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
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
# METHOD 4: DIRECT PREFERENCE OPTIMIZATION (DPO)
# ==============================================================================

class DirectPreferenceOptimization(BaseDistillationMethod):
    """
    Direct Preference Optimization (DPO) for preference-based alignment.
    
    Trains student model using pre-collected preference pairs (preferred vs dispreferred responses)
    without requiring explicit reward models. Uses a simple classification-style loss that directly
    optimizes the policy to prefer better responses.
    
    Unlike RLHF, DPO does NOT use reinforcement learning or sample during training. Instead, it
    operates on fixed preference pairs and uses a closed-form solution derived from the
    Bradley-Terry preference model.
    
    **Reference**:
    Rafailov, R., Sharma, A., Mitchell, E., et al. (2023).
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    https://arxiv.org/abs/2305.18290

    :param teacher_model: Used as reference model (typically the initial/SFT model, often same as student initially)
    :type teacher_model: class: `nn.Module`
    :param student_model: Policy model to be trained/aligned
    :type student_model: class: `nn.Module`
    :param tokenizer: Tokenizer compatible with both models
    :type tokenizer: TokenizerType
    :param config: Configuration dictionary with 'beta' (DPO temperature, default 0.1), controls strength of KL regularization
    :type config: Dict[str, Any]
    :raises ValueError: If beta <= 0
    
    Note: This implementation expects batches to contain 'preferred_ids', 'preferred_mask', 
    'dispreferred_ids', 'dispreferred_mask' instead of generating responses on-the-fly.
    """
    
    beta: float
    
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
        
        self.beta: float = config.get('beta', 0.1) #DPO temperature parameter (KL regularization strength)
        if self.beta <= 0:
            raise ValueError(f"Beta must be > 0, got {self.beta}")
        
        print(f"Initialized {self.get_method_name()}")
        print(f"  → Beta (β) = {self.beta}")
        print(f"  → Note: Requires preference pairs in batch data")
        print(f"  → Batch must contain: preferred_ids, preferred_mask, dispreferred_ids, dispreferred_mask")
    
    def compute_loss(self, batch: BatchDict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss using pre-collected preference pairs.
        
        Expects batch to contain both preferred and dispreferred response pairs.
        Computes log probabilities under policy (student) and reference (teacher) models,
        then optimizes the policy to increase probability of preferred responses relative
        to dispreferred ones using the DPO objective.
        
        DPO Loss: -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
        where y_w is preferred (chosen), y_l is dispreferred (rejected)
        
        :param batch: Input batch containing preference pairs. Must have keys:
            - 'preferred_ids': Tokenized preferred completions (prompt + response) [batch, seq_len_w]
            - 'preferred_mask': Attention mask for preferred [batch, seq_len_w]
            - 'dispreferred_ids': Tokenized dispreferred completions [batch, seq_len_l]
            - 'dispreferred_mask': Attention mask for dispreferred [batch, seq_len_l]
            - 'prompt_lengths': Length of prompt for each example [batch] (to mask prompt in loss)
        :type batch: BatchDict (extended with preference-specific keys)
        :returns: Tuple of (DPO loss tensor, metrics dictionary)
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        batch = self.prepare_batch(batch) #Bring everything to same device
        
        # Extract preference pair data from batch
        if 'preferred_ids' not in batch or 'dispreferred_ids' not in batch:
            raise ValueError(
                "DPO requires preference pairs in batch. "
                "Batch must contain: 'preferred_ids', 'preferred_mask', 'dispreferred_ids', 'dispreferred_mask', 'prompt_lengths'"
            )
        
        preferred_ids = batch['preferred_ids']  # [batch, seq_len_w]
        preferred_mask = batch['preferred_mask']  # [batch, seq_len_w]
        dispreferred_ids = batch['dispreferred_ids']  # [batch, seq_len_l]
        dispreferred_mask = batch['dispreferred_mask']  # [batch, seq_len_l]
        prompt_lengths = batch.get('prompt_lengths', None)  # [batch]
        
        # Helper function to compute per-token log probabilities
        def get_log_probs(model, input_ids, attention_mask, labels):
            """Compute log probabilities for the response tokens only"""
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq, vocab]
            
            # Shift for autoregressive: predict next token
            shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq-1, vocab]
            shift_labels = labels[..., 1:].contiguous()  # [batch, seq-1]
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq-1, vocab]
            
            # Gather log probs of actual tokens
            per_token_logps = torch.gather(
                log_probs, 
                dim=2, 
                index=shift_labels.unsqueeze(2)
            ).squeeze(2)  # [batch, seq-1]
            
            # Mask out prompt tokens (only compute on response)
            response_mask = (shift_labels != -100).float()  # [batch, seq-1]
            
            # Average log prob over response tokens
            sequence_logp = (per_token_logps * response_mask).sum(dim=1) / response_mask.sum(dim=1)
            return sequence_logp
        
        # Create labels (mask prompt with -100)
        preferred_labels = preferred_ids.clone()
        dispreferred_labels = dispreferred_ids.clone()
        
        if prompt_lengths is not None:
            for i, prompt_len in enumerate(prompt_lengths):
                preferred_labels[i, :prompt_len] = -100
                dispreferred_labels[i, :prompt_len] = -100
        
        # Store first preferred sequence in batch for inspection
        batch['labels'] = preferred_labels
        
        # Compute policy (student) log probs
        policy_preferred_logps = get_log_probs(self.student_model, preferred_ids, preferred_mask, preferred_labels)
        policy_dispreferred_logps = get_log_probs(self.student_model, dispreferred_ids, dispreferred_mask, dispreferred_labels)
        
        # Compute reference (teacher) log probs
        with torch.no_grad():
            ref_preferred_logps = get_log_probs(self.teacher_model, preferred_ids, preferred_mask, preferred_labels)
            ref_dispreferred_logps = get_log_probs(self.teacher_model, dispreferred_ids, dispreferred_mask, dispreferred_labels)
        
        # DPO loss computation
        # Compute log ratios: log(π_θ/π_ref) for preferred and dispreferred
        preferred_log_ratio = policy_preferred_logps - ref_preferred_logps
        dispreferred_log_ratio = policy_dispreferred_logps - ref_dispreferred_logps
        
        # DPO objective: -log(σ(β * (log_ratio_preferred - log_ratio_dispreferred)))
        logits_diff = preferred_log_ratio - dispreferred_log_ratio
        dpo_loss = -F.logsigmoid(self.beta * logits_diff).mean()
        
        # Compute implicit reward (for logging)
        implicit_reward_preferred = self.beta * preferred_log_ratio
        implicit_reward_dispreferred = self.beta * dispreferred_log_ratio
        reward_margin = (implicit_reward_preferred - implicit_reward_dispreferred).mean().item()
        
        # Compute accuracy (how often preferred is ranked higher)
        accuracy = (logits_diff > 0).float().mean().item()
        
        metrics = {
            'dpo_loss': dpo_loss.item(),
            'total_loss': dpo_loss.item(),
            'reward_margin': reward_margin,
            'accuracy': accuracy,
            'policy_preferred_logp': policy_preferred_logps.mean().item(),
            'policy_dispreferred_logp': policy_dispreferred_logps.mean().item(),
            'ref_preferred_logp': ref_preferred_logps.mean().item(),
            'ref_dispreferred_logp': ref_dispreferred_logps.mean().item()
        }
        
        return dpo_loss, metrics
    
    def get_method_name(self) -> str:
        """
        Return method identifier with hyperparameters for logging.
        
        :returns: Method name string including beta value
        :rtype: str
        """
        return f"DPO (β={self.beta})"


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
    
    :param method_name: Method identifier ('sft', 'logit_kd', 'cot', 'dpo')
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
        - CoT: optional 'max_new_tokens' (default 512), optional 'cot_prompt' (default "Let's think step by step:"),
               optional 'num_rationales' (default 1), optional 'sampling_temperature' (default 0.7)
        - DPO: 'beta' (default 0.1). NOTE: DPO requires pre-collected preference pairs in batch
               (see DirectPreferenceOptimization docstring for batch format)
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
        
        # Chain-of-Thought
        cot = create_distillation_method('cot', teacher, student, tokenizer, {})
        
        # Direct Preference Optimization
        dpo = create_distillation_method('dpo', teacher, student, tokenizer, {'beta': 0.1})
        
        # Compute loss
        loss, metrics = method.compute_loss(batch)
    """
    method_name = method_name.lower()
    
    if method_name in ('sft', 'standard_sft'):
        return StandardSFT(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('logit_kd', 'logit'):
        return LogitKD(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('cot', 'chain_of_thought'):
        return ChainOfThoughtDistillation(teacher_model, student_model, tokenizer, config)
    
    elif method_name in ('dpo', 'preference', 'direct_preference_optimization'):
        return DirectPreferenceOptimization(teacher_model, student_model, tokenizer, config)
    
    else:
        raise ValueError(
            f"Unknown distillation method: {method_name}. "
            f"Choose from: ['sft', 'logit_kd', 'cot', 'dpo']"
        )