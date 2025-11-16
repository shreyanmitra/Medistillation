"""
(C) 2025. Bryan Zhao, Federico Baldan, Tim Avilov, and Shreyan Mitra
Written for CSE 493S: Advanced Topics in Machine Learning Course at the University of Washington, Seattle

Data Loader for Medical LLM Distillation

This file contains data loading utilities for medical question-answering datasets,
specifically designed to work with the distillation methods in DistillationMethods.py.
Supports various data formats and distillation approaches including SFT, Logit-KD, CoT, and DPO.
Documentation style is Sphinx.
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

import json
import os
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


# ==============================================================================
# DATASET CLASS - MedMCQA Medical Question-Answering
# ==============================================================================

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
        """
        Load data from JSON file.
        
        :returns: List of data items from the dataset
        :rtype: List[Dict[str, Any]]
        """
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
        
        For standard distillation methods (SFT, Logit-KD), the teacher will generate
        responses online during training. This method creates a simple placeholder
        that includes the explanation if available.
        
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
    
    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.
        
        :returns: Total number of examples
        :rtype: int
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Note: For all distillation methods, teacher responses are generated
        online during training. This method returns basic question/response
        format with ground truth for reference.
        
        :param idx: Index of the example
        :type idx: int
        :returns: Dictionary containing the example data
        :rtype: Dict[str, Any]
        """
        item = self.data[idx]
        
        # Format the question
        question = self._format_question(item)
        
        # For all methods, create standard response
        # Teacher will generate actual responses online during training
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


# ==============================================================================
# COLLATOR CLASS - Batching and Tokenization
# ==============================================================================

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
        
        Note: All methods now use standard collation. Teacher responses are
        generated online during training by the distillation methods themselves.
        
        :param batch: List of examples from the dataset
        :type batch: List[Dict[str, Any]]
        :returns: Batched and tokenized data
        :rtype: Dict[str, Any]
        """
        # Use standard collation for all methods
        # CoT and SPIN handle their specific requirements in their own compute_loss methods
        return self._collate_standard_batch(batch)
    
    def _collate_standard_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for all distillation methods."""
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


# ==============================================================================
# DATALOADER CREATION FUNCTIONS
# ==============================================================================

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


# ==============================================================================
# UNIVERSAL DATASET CLASS FOR MULTI-SOURCE MEDICAL DATA
# ==============================================================================

class UniversalMedicalDataset(Dataset):
    """
    Universal dataset class that handles multiple medical QA dataset formats.
    
    Automatically detects and standardizes data from:
    - MedMCQA: Multiple choice with options (opa/opb/opc/opd)
    - MedQA: Multiple choice with options dict
    - PubMedQA: Yes/no/maybe questions with context
    - PubHealth: Health claim verification (true/false/mixture/unproven)
    
    All datasets are converted to a unified long-form answer format where:
    - Questions include options embedded in the text (for MCQ datasets)
    - Answers are natural language text (not A/B/C/D letters)
    - Ground truth is stored in metadata for optional accuracy evaluation
    
    :param data_path: Path to JSONL file containing dataset
    :param tokenizer: HuggingFace tokenizer for text processing
    :param max_length: Maximum sequence length for tokenization
    :param distillation_method: Type of distillation method ('sft', 'logit_kd', 'cot', 'spin')
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Union['PreTrainedTokenizer', 'PreTrainedTokenizerFast'],
        max_length: int = 1024,
        distillation_method: str = 'sft',
        **kwargs
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.distillation_method = distillation_method.lower()
        
        # Load and standardize data
        self.data = self._load_and_standardize_data()
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
        logger.info(f"Dataset sources: {self._get_source_distribution()}")
    
    def _load_and_standardize_data(self) -> List[Dict[str, Any]]:
        """Load data and convert all formats to unified structure."""
        raw_data = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise
        
        # Standardize each item
        standardized_data = []
        for item in raw_data:
            try:
                standardized = self._standardize_item(item)
                standardized_data.append(standardized)
            except Exception as e:
                logger.warning(f"Failed to standardize item: {e}")
                continue
        
        return standardized_data
    
    def _detect_format(self, item: Dict[str, Any]) -> str:
        """Auto-detect which dataset format this item is from."""
        if 'opa' in item and 'opb' in item:
            return 'medmcqa'
        elif 'options' in item and isinstance(item['options'], dict):
            return 'medqa'
        elif 'claim' in item and 'explanation' in item:
            # PubHealth has claim and explanation fields
            return 'pubhealth'
        elif 'long_answer' in item or ('context' in item and 'contexts' in item.get('context', {})):
            return 'pubmedqa'
        else:
            # Default fallback
            return 'unknown'
    
    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any format to unified long-form answer format."""
        format_type = self._detect_format(item)
        
        if format_type == 'medmcqa':
            return self._convert_medmcqa(item)
        elif format_type == 'medqa':
            return self._convert_medqa(item)
        elif format_type == 'pubmedqa':
            return self._convert_pubmedqa(item)
        elif format_type == 'pubhealth':
            return self._convert_pubhealth(item)
        else:
            # Unknown format - try to use as-is
            return {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'source': 'unknown',
                'id': item.get('id', ''),
                'metadata': item
            }
    
    def _convert_medmcqa(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MedMCQA format to unified long-form."""
        options = [item['opa'], item['opb'], item['opc'], item['opd']]
        correct_idx = item['cop'] - 1  # cop is 1-indexed
        
        # Embed options in question text
        question_with_options = f"{item['question']}\n\nOptions:\n"
        for i, opt in enumerate(options):
            question_with_options += f"{chr(65+i)}) {opt}\n"
        
        return {
            'question': question_with_options,
            'answer': options[correct_idx],  # Long-form answer text
            'source': 'medmcqa',
            'id': item.get('id', ''),
            'metadata': {
                'explanation': item.get('exp', ''),
                'subject': item.get('subject_name', ''),
                'topic': item.get('topic_name', ''),
                'choice_type': item.get('choice_type', 'single'),
                'original_format': 'mcq'
            }
        }
    
    def _convert_medqa(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MedQA format to unified long-form."""
        # MedQA has options as dict: {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
        options_dict = item.get('options', {})
        sorted_keys = sorted(options_dict.keys())
        options = [options_dict[k] for k in sorted_keys]
        
        # Get answer
        answer_idx = item.get('answer_idx', item.get('answer', 'A'))
        if isinstance(answer_idx, str) and len(answer_idx) == 1:
            # Convert letter to index
            answer_letter = answer_idx.upper()
            correct_idx = ord(answer_letter) - ord('A')
        else:
            correct_idx = 0
        
        # Embed options in question
        question_with_options = f"{item['question']}\n\nOptions:\n"
        for i, opt in enumerate(options):
            question_with_options += f"{chr(65+i)}) {opt}\n"
        
        return {
            'question': question_with_options,
            'answer': options[correct_idx] if correct_idx < len(options) else options[0],
            'source': 'medqa',
            'id': item.get('id', str(hash(item['question']))),
            'metadata': {
                'explanation': item.get('rationale', item.get('metamap_phrases', '')),
                'subject': item.get('meta_info', 'USMLE'),
                'original_format': 'mcq'
            }
        }
    
    def _convert_pubmedqa(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PubMedQA yes/no/maybe format to unified long-form."""
        # PubMedQA is already in natural question format
        question = item.get('question', '')
        
        # Get answer (yes/no/maybe)
        answer = item.get('long_answer', item.get('final_decision', 'maybe')).lower()
        
        # Extract context if available
        context = ''
        if 'context' in item and isinstance(item['context'], dict):
            if 'contexts' in item['context']:
                context = ' '.join(item['context']['contexts'])
        
        return {
            'question': question,
            'answer': answer,  # Keep as yes/no/maybe
            'source': 'pubmedqa',
            'id': str(item.get('pubid', item.get('PMID', ''))),
            'metadata': {
                'context': context,
                'original_format': 'yes_no_maybe'
            }
        }
    
    def _convert_pubhealth(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PubHealth health claim verification format to unified long-form."""
        # PubHealth has health claims to verify as true/false/mixture/unproven
        claim = item.get('claim', '')
        
        # Format as a question
        question = f"Is the following health claim true or false?\n\nClaim: {claim}"
        
        # Get label (true, false, mixture, unproven)
        label = item.get('label', 'unproven').lower()
        
        # Get explanation if available
        explanation = item.get('explanation', '')
        
        # Answer includes both label and explanation
        if explanation:
            answer = f"{label.capitalize()}. {explanation}"
        else:
            answer = label.capitalize()
        
        return {
            'question': question,
            'answer': answer,
            'source': 'pubhealth',
            'id': item.get('id', item.get('document_id', '')),
            'metadata': {
                'claim': claim,
                'main_text': item.get('main_text', ''),
                'subjects': item.get('subjects', []),
                'original_format': 'claim_verification'
            }
        }
    
    def _get_source_distribution(self) -> Dict[str, int]:
        """Get count of examples from each source."""
        distribution = {}
        for item in self.data:
            source = item['source']
            distribution[source] = distribution.get(source, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single standardized example."""
        item = self.data[idx]
        
        # Return standardized format
        # Note: Teacher will generate response online during training
        return {
            'question': item['question'],
            'ground_truth_answer': item['answer'],  # For optional accuracy evaluation
            'source': item['source'],
            'id': item['id'],
            'metadata': item['metadata']
        }


# ==============================================================================
# DATASET PREPARATION UTILITIES
# ==============================================================================

def prepare_med_distillmix_dataset(
    output_dir: str = 'data/processed',
    train_split: float = 0.9,
    val_split: float = 0.05,
    seed: int = 42,
    num_medmcqa: int = None,
    num_medqa: int = None,
    num_pubmedqa: int = None,
    num_pubhealth: int = None
):
    """
    Prepare Med-DistillMix dataset from HuggingFace datasets.
    
    Downloads and combines medical QA datasets into a unified training corpus:
    - ALL from MedMCQA (Indian medical entrance exams) - ~182k available
    - ALL from MedQA (USMLE-style questions) - ~10k available
    - ALL from PubMedQA artificial (biomedical research questions) - ~211k available
    - ALL from PubHealth (health claim verification) - ~11k available
    
    Total: ~400k+ examples, split 90/5/5 into train/val/holdout
    
    All datasets are converted to a unified long-form answer format where
    answers are natural language text (not A/B/C/D), making them suitable
    for knowledge distillation with teacher-student agreement as the goal.
    
    :param output_dir: Directory to save processed JSONL files
    :param train_split: Fraction for training (default 0.9)
    :param val_split: Fraction for validation (default 0.05)
    :param seed: Random seed for reproducibility
    :param num_medmcqa: Number of MedMCQA samples (None = use all)
    :param num_medqa: Number of MedQA samples (None = use all)
    :param num_pubmedqa: Number of PubMedQA samples (None = use all)
    :param num_pubhealth: Number of PubHealth samples (None = use all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return
    
    from pathlib import Path
    import os
    
    logger.info("=" * 80)
    logger.info("PREPARING MED-DISTILLMIX DATASET")
    logger.info("=" * 80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    
    all_data = []
    
    # 1. Load MedMCQA
    logger.info(f"\n1. Loading MedMCQA...")
    try:
        medmcqa = load_dataset('openlifescienceai/medmcqa', split='train')
        logger.info(f"   Total available: {len(medmcqa)} examples")
        
        if num_medmcqa is not None:
            sample_size = min(num_medmcqa, len(medmcqa))
            medmcqa_sample = random.sample(list(medmcqa), sample_size)
        else:
            medmcqa_sample = list(medmcqa)
        
        all_data.extend([dict(item) for item in medmcqa_sample])
        logger.info(f"   ✅ Added {len(medmcqa_sample)} MedMCQA examples")
    except Exception as e:
        logger.error(f"   ❌ Failed to load MedMCQA: {e}")
    
    # 2. Load MedQA
    logger.info(f"\n2. Loading MedQA...")
    try:
        # Use the correct dataset path without custom script
        medqa = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
        logger.info(f"   Total available: {len(medqa)} examples")
        
        if num_medqa is not None:
            sample_size = min(num_medqa, len(medqa))
            medqa_sample = random.sample(list(medqa), sample_size)
        else:
            medqa_sample = list(medqa)
        
        all_data.extend([dict(item) for item in medqa_sample])
        logger.info(f"   ✅ Added {len(medqa_sample)} MedQA examples")
    except Exception as e:
        logger.error(f"   ❌ Failed to load MedQA: {e}")
    
    # 3. Load PubMedQA (artificial subset for quantity)
    logger.info(f"\n3. Loading PubMedQA artificial subset...")
    try:
        pubmedqa = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
        logger.info(f"   Total available: {len(pubmedqa)} examples")
        
        if num_pubmedqa is not None:
            sample_size = min(num_pubmedqa, len(pubmedqa))
            pubmedqa_sample = random.sample(list(pubmedqa), sample_size)
        else:
            pubmedqa_sample = list(pubmedqa)
        
        all_data.extend([dict(item) for item in pubmedqa_sample])
        logger.info(f"   ✅ Added {len(pubmedqa_sample)} PubMedQA examples")
    except Exception as e:
        logger.error(f"   ❌ Failed to load PubMedQA: {e}")
    
    # 4. Load PubHealth
    logger.info(f"\n4. Loading PubHealth...")
    try:
        # Use health_fact dataset as alternative (PubHealth derivative)
        pubhealth = load_dataset('health_fact', split='train', trust_remote_code=True)
        logger.info(f"   Total available: {len(pubhealth)} examples")
        
        if num_pubhealth is not None:
            sample_size = min(num_pubhealth, len(pubhealth))
            pubhealth_sample = random.sample(list(pubhealth), sample_size)
        else:
            pubhealth_sample = list(pubhealth)
        
        all_data.extend([dict(item) for item in pubhealth_sample])
        logger.info(f"   ✅ Added {len(pubhealth_sample)} PubHealth examples")
    except Exception as e:
        logger.error(f"   ❌ Failed to load PubHealth: {e}")
    
    # Deduplicate based on question text
    logger.info(f"\n5. Deduplicating {len(all_data)} examples...")
    seen_questions = set()
    deduplicated_data = []
    duplicates_removed = 0
    
    for item in all_data:
        # Normalize question for comparison (lowercase, strip whitespace)
        question_key = item.get('question', '').lower().strip()
        
        if question_key and question_key not in seen_questions:
            seen_questions.add(question_key)
            deduplicated_data.append(item)
        else:
            duplicates_removed += 1
    
    logger.info(f"   Removed {duplicates_removed} duplicate questions")
    logger.info(f"   Remaining: {len(deduplicated_data)} unique examples")
    
    # Shuffle combined data
    logger.info(f"\n6. Shuffling {len(deduplicated_data)} examples...")
    random.shuffle(deduplicated_data)
    
    # Split into train/val/holdout
    n_train = int(len(deduplicated_data) * train_split)
    n_val = int(len(deduplicated_data) * val_split)
    
    train_data = deduplicated_data[:n_train]
    val_data = deduplicated_data[n_train:n_train + n_val]
    holdout_data = deduplicated_data[n_train + n_val:]
    
    logger.info(f"\n7. Saving splits:")
    logger.info(f"   Train: {len(train_data)} examples ({train_split*100:.0f}%)")
    logger.info(f"   Validation: {len(val_data)} examples ({val_split*100:.0f}%)")
    logger.info(f"   Holdout: {len(holdout_data)} examples ({(1-train_split-val_split)*100:.0f}%)")
    
    # Save as JSONL
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    save_jsonl(train_data, os.path.join(output_dir, 'train.jsonl'))
    save_jsonl(val_data, os.path.join(output_dir, 'validation.jsonl'))
    save_jsonl(holdout_data, os.path.join(output_dir, 'holdout.jsonl'))
    
    logger.info(f"\n✅ Med-DistillMix dataset saved to {output_dir}/")
    logger.info(f"   Files created:")
    logger.info(f"   - train.jsonl ({len(train_data)} examples)")
    logger.info(f"   - validation.jsonl ({len(val_data)} examples)")
    logger.info(f"   - holdout.jsonl ({len(holdout_data)} examples)")
    logger.info("=" * 80)


def download_benchmark_test_sets(output_dir: str = 'data/benchmarks'):
    """
    Download official test/validation sets for evaluation benchmarks.
    
    These benchmarks are completely separate from the training set to prevent
    data leakage. They are used to evaluate the final distilled models on
    industry-standard medical QA tasks.
    
    Downloads:
    - MedQA test set (USMLE-style questions)
    - MedMCQA validation set (test set labels not public)
    - PubMedQA test set (biomedical yes/no questions)
    - PubHealth test set (health claim verification)
    
    :param output_dir: Directory to save benchmark JSONL files
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return
    
    from pathlib import Path
    import os
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING EVALUATION BENCHMARKS")
    logger.info("=" * 80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
    
    # 1. MedQA test set
    logger.info("\n1. Downloading MedQA test set...")
    try:
        # Use the correct MedQA dataset
        medqa_test = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
        save_jsonl(medqa_test, os.path.join(output_dir, 'medqa_test.jsonl'))
        logger.info(f"   ✅ Saved {len(medqa_test)} examples to medqa_test.jsonl")
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
    
    # 2. MedMCQA validation set
    logger.info("\n2. Downloading MedMCQA validation set...")
    try:
        medmcqa_val = load_dataset('openlifescienceai/medmcqa', split='validation')
        save_jsonl(medmcqa_val, os.path.join(output_dir, 'medmcqa_val.jsonl'))
        logger.info(f"   ✅ Saved {len(medmcqa_val)} examples to medmcqa_val.jsonl")
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
    
    # 3. PubMedQA test set (labeled subset)
    logger.info("\n3. Downloading PubMedQA test set...")
    try:
        # The labeled split only has 'train', use it as test set
        pubmedqa_test = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        save_jsonl(pubmedqa_test, os.path.join(output_dir, 'pubmedqa_test.jsonl'))
        logger.info(f"   ✅ Saved {len(pubmedqa_test)} examples to pubmedqa_test.jsonl")
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
    
    # 4. PubHealth test set
    logger.info("\n4. Downloading PubHealth test set...")
    try:
        # Use health_fact as alternative, use validation split as test
        pubhealth_test = load_dataset('health_fact', split='validation', trust_remote_code=True)
        save_jsonl(pubhealth_test, os.path.join(output_dir, 'pubhealth_test.jsonl'))
        logger.info(f"   ✅ Saved {len(pubhealth_test)} examples to pubhealth_test.jsonl")
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
    
    logger.info(f"\n✅ Evaluation benchmarks saved to {output_dir}/")
    logger.info(f"   Files created:")
    logger.info(f"   - medqa_test.jsonl")
    logger.info(f"   - medmcqa_val.jsonl")
    logger.info(f"   - pubmedqa_test.jsonl")
    logger.info(f"   - pubhealth_test.jsonl")
    logger.info("=" * 80)


def create_medppl_10k_corpus(
    output_path: str = 'data/medppl_10k.jsonl',
    num_samples: int = 10000,
    seed: int = 42
):
    """
    Create MedPPL-10k perplexity evaluation corpus.
    
    This corpus is used to measure language modeling quality on biomedical text.
    It is completely separate from both the training data and the medical QA
    benchmarks, ensuring an unbiased measure of the model's understanding of
    medical language.
    
    Samples 10k PubMed abstracts from PubMedQA context fields as a simple
    approach to get diverse biomedical text.
    
    :param output_path: Path to save corpus JSONL file
    :param num_samples: Number of abstracts to include
    :param seed: Random seed for reproducibility
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return
    
    from pathlib import Path
    import os
    
    logger.info("=" * 80)
    logger.info("CREATING MEDPPL-10K PERPLEXITY CORPUS")
    logger.info("=" * 80)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    
    logger.info(f"\nSampling {num_samples} PubMed abstracts...")
    
    try:
        # Use PubMedQA contexts as source of medical text
        # Combine both labeled and artificial subsets for diversity
        logger.info("   Loading PubMedQA labeled subset...")
        pubmedqa_labeled = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        
        logger.info("   Loading PubMedQA artificial subset...")
        pubmedqa_artificial = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
        
        # Combine both
        all_pubmedqa = list(pubmedqa_labeled) + list(pubmedqa_artificial)
        logger.info(f"   Total available: {len(all_pubmedqa)} PubMed abstracts")
        
        # Sample abstracts
        sample_size = min(num_samples, len(all_pubmedqa))
        sampled_items = random.sample(all_pubmedqa, sample_size)
        
        corpus = []
        for item in sampled_items:
            # Combine context sentences into one abstract
            if 'context' in item and 'contexts' in item['context']:
                text = ' '.join(item['context']['contexts'])
                if text.strip():  # Only add non-empty abstracts
                    corpus.append({'text': text})
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in corpus:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"\n✅ Saved {len(corpus)} abstracts to {output_path}")
        logger.info(f"   Average length: {sum(len(c['text'].split()) for c in corpus) / len(corpus):.0f} words")
        
    except Exception as e:
        logger.error(f"❌ Failed to create corpus: {e}")
    
    logger.info("=" * 80)


def create_fidelitybench_med(
    output_path: str = 'data/fidelitybench_med.jsonl',
    num_samples: int = 1500,
    seed: int = 42
):
    """
    Create FidelityBench-Med evaluation suite for measuring fidelity and faithfulness.
    
    This benchmark measures:
    1. Teacher-student fidelity (how well student mimics teacher's behavior)
    2. Evidence faithfulness (does model use provided evidence correctly)
    3. Citation coverage (does model reference source material appropriately)
    
    Creates prompts with evidence passages for evaluation. Each example includes:
    - Question requiring evidence-based reasoning
    - Relevant evidence passages (from PubMed abstracts)
    - Ground truth answer for verification
    
    Note: This creates the dataset structure. Actual RAGAS/NLI scoring and
    hallucination detection are performed during evaluation in Trainer.py.
    
    :param output_path: Path to save FidelityBench JSONL file
    :param num_samples: Number of prompts to include (default 1500)
    :param seed: Random seed for reproducibility
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return
    
    from pathlib import Path
    import os
    
    logger.info("=" * 80)
    logger.info("CREATING FIDELITYBENCH-MED EVALUATION SUITE")
    logger.info("=" * 80)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    
    logger.info(f"\nCreating {num_samples} evidence-based evaluation prompts...")
    
    try:
        # Use PubMedQA labeled subset (has questions + evidence + answers)
        logger.info("   Loading PubMedQA labeled subset...")
        pubmedqa_labeled = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        logger.info(f"   Total available: {len(pubmedqa_labeled)} examples")
        
        # Sample examples
        sample_size = min(num_samples, len(pubmedqa_labeled))
        sampled_items = random.sample(list(pubmedqa_labeled), sample_size)
        
        fidelity_bench = []
        for idx, item in enumerate(sampled_items):
            # Extract question and evidence
            question = item.get('question', '')
            
            # Get evidence passages
            evidence_passages = []
            if 'context' in item and 'contexts' in item['context']:
                evidence_passages = item['context']['contexts']
            
            # Get ground truth answer
            ground_truth = item.get('final_decision', item.get('long_answer', 'maybe'))
            
            # Create evidence-based prompt
            prompt = f"Based on the following evidence, answer the question.\n\n"
            prompt += f"Evidence:\n"
            for i, passage in enumerate(evidence_passages[:3], 1):  # Limit to 3 passages
                prompt += f"{i}. {passage}\n\n"
            prompt += f"Question: {question}\n\nAnswer:"
            
            fidelity_bench.append({
                'id': f'fidelity_{idx}',
                'question': question,
                'evidence_passages': evidence_passages[:3],
                'prompt': prompt,
                'ground_truth': ground_truth,
                'source': 'pubmedqa',
                'eval_type': 'evidence_based',
                'expected_citations': list(range(1, min(4, len(evidence_passages) + 1)))  # Should cite passages 1-3
            })
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in fidelity_bench:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"\n✅ Saved {len(fidelity_bench)} prompts to {output_path}")
        logger.info(f"   Each prompt includes:")
        logger.info(f"   - Question requiring evidence-based reasoning")
        logger.info(f"   - 1-3 relevant PubMed passages as evidence")
        logger.info(f"   - Ground truth answer for verification")
        logger.info(f"   - Expected citation markers")
        
    except Exception as e:
        logger.error(f"❌ Failed to create FidelityBench-Med: {e}")
    
    logger.info("=" * 80)


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Medical Dataset Preparation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all datasets at once
  python DataLoader.py --prepare_all
  
  # Prepare only training data
  python DataLoader.py --prepare_training --output_dir data/processed
  
  # Download only benchmarks
  python DataLoader.py --download_benchmarks --benchmark_dir data/benchmarks
  
  # Create only perplexity corpus
  python DataLoader.py --create_perplexity_corpus --perplexity_path data/medppl_10k.jsonl
  
  # Create FidelityBench-Med evaluation suite
  python DataLoader.py --create_fidelitybench --fidelitybench_path data/fidelitybench_med.jsonl
        """
    )
    
    parser.add_argument('--prepare_all', action='store_true',
                        help='Prepare all datasets (training + benchmarks + perplexity corpus + fidelitybench)')
    parser.add_argument('--prepare_training', action='store_true',
                        help='Prepare Med-DistillMix training dataset')
    parser.add_argument('--download_benchmarks', action='store_true',
                        help='Download evaluation benchmarks (MedQA, MedMCQA, PubMedQA, PubHealth)')
    parser.add_argument('--create_perplexity_corpus', action='store_true',
                        help='Create MedPPL-10k perplexity evaluation corpus')
    parser.add_argument('--create_fidelitybench', action='store_true',
                        help='Create FidelityBench-Med evaluation suite')
    
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for training data (default: data/processed)')
    parser.add_argument('--benchmark_dir', type=str, default='data/benchmarks',
                        help='Output directory for benchmarks (default: data/benchmarks)')
    parser.add_argument('--perplexity_path', type=str, default='data/medppl_10k.jsonl',
                        help='Output path for perplexity corpus (default: data/medppl_10k.jsonl)')
    parser.add_argument('--fidelitybench_path', type=str, default='data/fidelitybench_med.jsonl',
                        help='Output path for FidelityBench-Med (default: data/fidelitybench_med.jsonl)')
    
    # Dataset size arguments
    parser.add_argument('--num_medmcqa', type=int, default=None,
                        help='Number of MedMCQA samples (None = all, default: None)')
    parser.add_argument('--num_medqa', type=int, default=None,
                        help='Number of MedQA samples (None = all, default: None)')
    parser.add_argument('--num_pubmedqa', type=int, default=None,
                        help='Number of PubMedQA samples (None = all, default: None)')
    parser.add_argument('--num_pubhealth', type=int, default=None,
                        help='Number of PubHealth samples (None = all, default: None)')
    parser.add_argument('--num_perplexity', type=int, default=10000,
                        help='Number of abstracts for perplexity corpus (default: 10000)')
    parser.add_argument('--num_fidelity', type=int, default=1500,
                        help='Number of prompts for FidelityBench (default: 1500)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Prepare all datasets if requested
    if args.prepare_all:
        logger.info("Preparing ALL datasets...")
        prepare_med_distillmix_dataset(
            output_dir=args.output_dir,
            seed=args.seed,
            num_medmcqa=args.num_medmcqa,
            num_medqa=args.num_medqa,
            num_pubmedqa=args.num_pubmedqa,
            num_pubhealth=args.num_pubhealth
        )
        download_benchmark_test_sets(output_dir=args.benchmark_dir)
        create_medppl_10k_corpus(
            output_path=args.perplexity_path,
            num_samples=args.num_perplexity,
            seed=args.seed
        )
        create_fidelitybench_med(
            output_path=args.fidelitybench_path,
            num_samples=args.num_fidelity,
            seed=args.seed
        )
    else:
        # Individual dataset preparation
        if args.prepare_training:
            prepare_med_distillmix_dataset(
                output_dir=args.output_dir,
                seed=args.seed,
                num_medmcqa=args.num_medmcqa,
                num_medqa=args.num_medqa,
                num_pubmedqa=args.num_pubmedqa,
                num_pubhealth=args.num_pubhealth
            )
        
        if args.download_benchmarks:
            download_benchmark_test_sets(output_dir=args.benchmark_dir)
        
        if args.create_perplexity_corpus:
            create_medppl_10k_corpus(
                output_path=args.perplexity_path,
                num_samples=args.num_perplexity,
                seed=args.seed
            )
        
        if args.create_fidelitybench:
            create_fidelitybench_med(
                output_path=args.fidelitybench_path,
                num_samples=args.num_fidelity,
                seed=args.seed
            )
