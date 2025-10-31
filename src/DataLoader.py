"""
(C) 2025. Bryan Zhao, Federico Baldan, Tim Avilov, and Shreyan Mitra
Written for CSE 493S: Advanced Topics in Machine Learning Course at the University of Washington, Seattle

Data Loader for Medical LLM Distillation

This file contains data loading utilities for medical question-answering datasets,
specifically designed to work with the distillation methods in DistillationMethods.py.
Supports various data formats and distillation approaches including SFT, Logit-KD, CoT, and DPO.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Union, Tuple, TYPE_CHECKING
import random
import logging

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedMcqaDataset(Dataset):
    """
    Dataset class for MedMCQA medical question-answering data.
    
    Supports multiple data formats and can be configured for different distillation methods.
    Handles both single-choice and multi-choice questions from the MedMCQA dataset.
    
    :param data_path: Path to the JSON data file
    :type data_path: str
    :param tokenizer: HuggingFace tokenizer for text processing
    :type tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    :param max_length: Maximum sequence length for tokenization
    :type max_length: int
    :param distillation_method: Type of distillation method ('sft', 'logit_kd', 'cot', 'dpo')
    :type distillation_method: str
    :param cot_prompt: Chain-of-thought prompt for CoT distillation
    :type cot_prompt: str
    :param num_rationales: Number of diverse rationales for CoT (if > 1)
    :type num_rationales: int
    :param sampling_temperature: Temperature for diverse generation in CoT
    :type sampling_temperature: float
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast'],
        max_length: int = 512,
        distillation_method: str = 'sft',
        cot_prompt: str = "Let's think step by step:",
        num_rationales: int = 1,
        sampling_temperature: float = 0.7,
        **kwargs  # noqa: F841
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.distillation_method = distillation_method.lower()
        self.cot_prompt = cot_prompt
        self.num_rationales = num_rationales
        self.sampling_temperature = sampling_temperature
        
        # Load data
        self.data = self._load_data()
        logger.info("Loaded %d examples from %s", len(self.data), data_path)
        logger.info("Distillation method: %s", self.distillation_method)
        
        # Validate distillation method
        valid_methods = ['sft', 'logit_kd', 'cot', 'dpo']
        if self.distillation_method not in valid_methods:
            raise ValueError(f"Invalid distillation method: {self.distillation_method}. "
                           f"Choose from: {valid_methods}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
                return data
        except Exception as e:
            logger.error("Error loading data from %s: %s", self.data_path, e)
            raise
    
    def _format_question(self, item: Dict[str, Any]) -> str:
        """
        Format a single question with options.
        
        :param item: Single data item from the dataset
        :type item: Dict[str, Any]
        :returns: Formatted question string
        :rtype: str
        """
        question = item['question']
        options = [
            f"A) {item['opa']}",
            f"B) {item['opb']}",
            f"C) {item['opc']}",
            f"D) {item['opd']}"
        ]
        
        formatted_question = f"Question: {question}\n\nOptions:\n" + "\n".join(options)
        return formatted_question
    
    def _get_correct_answer(self, item: Dict[str, Any]) -> str:
        """
        Get the correct answer text for a question.
        
        :param item: Single data item from the dataset
        :type item: Dict[str, Any]
        :returns: Correct answer text
        :rtype: str
        """
        cop = item['cop']
        option_map = {1: 'opa', 2: 'opb', 3: 'opc', 4: 'opd'}
        return item[option_map[cop]]
    
    def _create_response(self, item: Dict[str, Any], include_explanation: bool = True) -> str:
        """
        Create a response for the question.
        
        :param item: Single data item from the dataset
        :type item: Dict[str, Any]
        :param include_explanation: Whether to include the explanation
        :type include_explanation: bool
        :returns: Formatted response string
        :rtype: str
        """
        correct_answer = self._get_correct_answer(item)
        response = f"The correct answer is: {correct_answer}"
        
        if include_explanation and item.get('exp') and item['exp'] != 'null':
            response += f"\n\nExplanation: {item['exp']}"
        
        return response
    
    def _create_cot_response(self, item: Dict[str, Any]) -> str:
        """
        Create a chain-of-thought response for CoT distillation.
        
        :param item: Single data item from the dataset
        :type item: Dict[str, Any]
        :returns: CoT formatted response
        :rtype: str
        """
        correct_answer = self._get_correct_answer(item)
        
        # Create reasoning steps
        reasoning = "Let me analyze each option:\n\n"
        
        options = [
            ("A", item['opa']),
            ("B", item['opb']),
            ("C", item['opc']),
            ("D", item['opd'])
        ]
        
        for letter, option in options:
            reasoning += f"Option {letter}: {option}\n"
            # Add some basic reasoning (this could be enhanced with more sophisticated logic)
            if letter == chr(ord('A') + item['cop'] - 1):
                reasoning += "  → This appears to be the correct answer based on medical knowledge.\n\n"
            else:
                reasoning += "  → This option seems less likely to be correct.\n\n"
        
        reasoning += f"After careful consideration, the correct answer is: {correct_answer}"
        
        if item.get('exp') and item['exp'] != 'null':
            reasoning += f"\n\nDetailed explanation: {item['exp']}"
        
        return reasoning
    
    def _create_preference_pairs(self, item: Dict[str, Any]) -> Tuple[str, str]:
        """
        Create preference pairs for DPO training.
        
        :param item: Single data item from the dataset
        :type item: Dict[str, Any]
        :returns: Tuple of (preferred_response, dispreferred_response)
        :rtype: Tuple[str, str]
        """
        correct_answer = self._get_correct_answer(item)
        correct_letter = chr(ord('A') + item['cop'] - 1)
        
        # Create preferred response (correct answer with good explanation)
        preferred = f"The correct answer is {correct_letter}) {correct_answer}.\n\n"
        if item.get('exp') and item['exp'] != 'null':
            preferred += f"Explanation: {item['exp']}"
        else:
            preferred += "This is the most appropriate choice based on medical knowledge."
        
        # Create dispreferred response (wrong answer with poor explanation)
        wrong_options = [i for i in range(1, 5) if i != item['cop']]
        wrong_choice = random.choice(wrong_options)
        wrong_letter = chr(ord('A') + wrong_choice - 1)
        wrong_answer = item[['opa', 'opb', 'opc', 'opd'][wrong_choice - 1]]
        
        dispreferred = f"I think the answer is {wrong_letter}) {wrong_answer}.\n\n"
        dispreferred += "This seems like a reasonable choice, though I'm not entirely certain."
        
        return preferred, dispreferred
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        :param idx: Index of the example
        :type idx: int
        :returns: Dictionary containing the example data
        :rtype: Dict[str, Any]
        """
        item = self.data[idx]
        
        # Format the question
        question = self._format_question(item)
        
        if self.distillation_method == 'dpo':
            # For DPO, create preference pairs
            preferred, dispreferred = self._create_preference_pairs(item)
            
            return {
                'question': question,
                'preferred_response': preferred,
                'dispreferred_response': dispreferred,
                'correct_answer': self._get_correct_answer(item),
                'explanation': item.get('exp', ''),
                'subject': item.get('subject_name', ''),
                'topic': item.get('topic_name', ''),
                'id': item.get('id', '')
            }
        
        elif self.distillation_method == 'cot':
            # For CoT, create reasoning-based responses
            response = self._create_cot_response(item)
            
            return {
                'question': question,
                'response': response,
                'correct_answer': self._get_correct_answer(item),
                'explanation': item.get('exp', ''),
                'subject': item.get('subject_name', ''),
                'topic': item.get('topic_name', ''),
                'id': item.get('id', '')
            }
        
        else:
            # For SFT and Logit-KD, create standard responses
            response = self._create_response(item)
            
            return {
                'question': question,
                'response': response,
                'correct_answer': self._get_correct_answer(item),
                'explanation': item.get('exp', ''),
                'subject': item.get('subject_name', ''),
                'topic': item.get('topic_name', ''),
                'id': item.get('id', '')
            }


class MedMcqaCollator:
    """
    Collator class for batching MedMCQA data.
    
    Handles tokenization and batching for different distillation methods.
    Converts raw text data into tensors suitable for model training.
    
    :param tokenizer: HuggingFace tokenizer
    :type tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    :param max_length: Maximum sequence length
    :type max_length: int
    :param distillation_method: Type of distillation method
    :type distillation_method: str
    :param cot_prompt: Chain-of-thought prompt for CoT
    :type cot_prompt: str
    """
    
    def __init__(
        self,
        tokenizer: Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast'],
        max_length: int = 512,
        distillation_method: str = 'sft',
        cot_prompt: str = "Let's think step by step:",
        **kwargs  # noqa: F841
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.distillation_method = distillation_method.lower()
        self.cot_prompt = cot_prompt
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of examples.
        
        :param batch: List of examples from the dataset
        :type batch: List[Dict[str, Any]]
        :returns: Batched and tokenized data
        :rtype: Dict[str, Any]
        """
        if self.distillation_method == 'dpo':
            return self._collate_dpo_batch(batch)
        elif self.distillation_method == 'cot':
            return self._collate_cot_batch(batch)
        else:
            return self._collate_standard_batch(batch)
    
    def _collate_standard_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for SFT and Logit-KD methods."""
        questions = [item['question'] for item in batch]
        responses = [item['response'] for item in batch]
        
        # Create prompts (questions + responses)
        prompts = [f"{q}\n\nAnswer: {r}" for q, r in zip(questions, responses)]
        
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'prompts': prompts,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'questions': questions,
            'responses': responses,
            'correct_answers': [item['correct_answer'] for item in batch],
            'explanations': [item['explanation'] for item in batch],
            'subjects': [item['subject'] for item in batch],
            'topics': [item['topic'] for item in batch],
            'ids': [item['id'] for item in batch]
        }
    
    def _collate_cot_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for Chain-of-Thought distillation."""
        questions = [item['question'] for item in batch]
        responses = [item['response'] for item in batch]
        
        # Add CoT prompt to questions
        cot_questions = [f"{q}\n\n{self.cot_prompt}" for q in questions]
        
        # Create prompts (CoT questions + reasoning responses)
        prompts = [f"{q}\n\n{r}" for q, r in zip(cot_questions, responses)]
        
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'prompts': prompts,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'questions': questions,
            'cot_questions': cot_questions,
            'responses': responses,
            'correct_answers': [item['correct_answer'] for item in batch],
            'explanations': [item['explanation'] for item in batch],
            'subjects': [item['subject'] for item in batch],
            'topics': [item['topic'] for item in batch],
            'ids': [item['id'] for item in batch]
        }
    
    def _collate_dpo_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for DPO training."""
        questions = [item['question'] for item in batch]
        preferred_responses = [item['preferred_response'] for item in batch]
        dispreferred_responses = [item['dispreferred_response'] for item in batch]
        
        # Create prompts for preferred and dispreferred responses
        preferred_prompts = [f"{q}\n\nAnswer: {r}" for q, r in zip(questions, preferred_responses)]
        dispreferred_prompts = [f"{q}\n\nAnswer: {r}" for q, r in zip(questions, dispreferred_responses)]
        
        # Tokenize both preferred and dispreferred sequences
        preferred_tokenized = self.tokenizer(
            preferred_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        dispreferred_tokenized = self.tokenizer(
            dispreferred_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Calculate prompt lengths for masking
        prompt_lengths = []
        for question in questions:
            # Tokenize just the question to get its length
            q_tokenized = self.tokenizer(
                question,
                add_special_tokens=False,
                return_tensors="pt"
            )
            prompt_lengths.append(q_tokenized['input_ids'].size(1))
        
        return {
            'questions': questions,
            'preferred_prompts': preferred_prompts,
            'dispreferred_prompts': dispreferred_prompts,
            'preferred_ids': preferred_tokenized['input_ids'],
            'preferred_mask': preferred_tokenized['attention_mask'],
            'dispreferred_ids': dispreferred_tokenized['input_ids'],
            'dispreferred_mask': dispreferred_tokenized['attention_mask'],
            'prompt_lengths': torch.tensor(prompt_lengths),
            'correct_answers': [item['correct_answer'] for item in batch],
            'explanations': [item['explanation'] for item in batch],
            'subjects': [item['subject'] for item in batch],
            'topics': [item['topic'] for item in batch],
            'ids': [item['id'] for item in batch]
        }


def create_dataloader(
        data_path: str,
        tokenizer: Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast'],
        batch_size: int = 8,
        max_length: int = 512,
        distillation_method: str = 'sft',
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
    """
    Create a DataLoader for medical QA distillation.
    
    :param data_path: Path to the JSON data file
    :type data_path: str
    :param tokenizer: HuggingFace tokenizer
    :type tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    :param batch_size: Batch size for training
    :type batch_size: int
    :param max_length: Maximum sequence length
    :type max_length: int
    :param distillation_method: Type of distillation method
    :type distillation_method: str
    :param shuffle: Whether to shuffle the data
    :type shuffle: bool
    :param num_workers: Number of worker processes for data loading
    :type num_workers: int
    :returns: Configured DataLoader
    :rtype: DataLoader
    """
    # Create dataset
    dataset = MedMcqaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        distillation_method=distillation_method,
        **kwargs
    )
    
    # Create collator
    collator = MedMcqaCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        distillation_method=distillation_method,
        **kwargs
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info("Created DataLoader with %d examples, batch_size=%d", len(dataset), batch_size)
    return dataloader


def create_train_val_dataloaders(
        train_path: str,
        val_path: str,
        tokenizer: Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast'],
        batch_size: int = 8,
        max_length: int = 512,
        distillation_method: str = 'sft',
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Create both training and validation DataLoaders for model training.
    
    This function is useful when you have separate training and validation datasets
    and want to create DataLoaders for both. The validation dataloader is typically
    used for:
    - Model evaluation during training
    - Early stopping based on validation loss
    - Monitoring overfitting
    - Computing validation metrics
    
    :param train_path: Path to training data (e.g., train_aug.json)
    :type train_path: str
    :param val_path: Path to validation data (e.g., dev.json)
    :type val_path: str
    :param tokenizer: HuggingFace tokenizer
    :type tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    :param batch_size: Batch size for both dataloaders
    :type batch_size: int
    :param max_length: Maximum sequence length
    :type max_length: int
    :param distillation_method: Type of distillation method
    :type distillation_method: str
    :param num_workers: Number of worker processes for data loading
    :type num_workers: int
    :returns: Tuple of (train_dataloader, val_dataloader)
    :rtype: Tuple[DataLoader, DataLoader]
    
    Example::
        # Create train/val dataloaders for SFT training
        train_dl, val_dl = create_train_val_dataloaders(
            train_path="augmented_data/augmented_MedMcqa/train_aug.json",
            val_path="data/MedMcqa_data/dev.json",
            tokenizer=tokenizer,
            batch_size=8,
            distillation_method="sft"
        )
        
        # Use in training loop
        for epoch in range(num_epochs):
            # Training
            for batch in train_dl:
                loss, metrics = model.compute_loss(batch)
                # ... training step
            
            # Validation
            val_losses = []
            for batch in val_dl:
                with torch.no_grad():
                    loss, metrics = model.compute_loss(batch)
                    val_losses.append(loss.item())
            print(f"Epoch {epoch}, Val Loss: {np.mean(val_losses)}")
    """
    # Create training dataloader
    train_dataloader = create_dataloader(
        data_path=train_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        distillation_method=distillation_method,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )
    
    # Create validation dataloader
    val_dataloader = create_dataloader(
        data_path=val_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        distillation_method=distillation_method,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    return train_dataloader, val_dataloader
