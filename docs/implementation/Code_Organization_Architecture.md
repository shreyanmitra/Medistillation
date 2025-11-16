# Project Code Organization and Architecture

**Project:** Medical LLM Distillation  
**Last Updated:** October 28, 2025

---

## Overview

This document describes the modular architecture for implementing and comparing four knowledge distillation methods for medical question answering. The codebase is organized into three main components with clear separation of concerns.

---

## Architecture Diagram

```
Medistillation/
│
├── src/
│   ├── DatasetBuilder.py          # Data preparation and loading
│   ├── DistillationMethods.py     # Four distillation implementations
│   ├── Trainer.py                 # Training orchestration
│   └── Evaluator.py               # Evaluation and analysis (optional separate module)
│
├── data/
│   ├── raw/                       # Downloaded datasets
│   ├── processed/                 # Cleaned training data
│   └── teacher_outputs/           # Pre-generated teacher responses
│
├── models/
│   ├── teacher/                   # Meditron-7B
│   └── students/                  # Trained student models
│       ├── sft/
│       ├── logit_kd/
│       ├── cot/
│       └── dpo/
│
├── results/
│   ├── benchmarks/                # Evaluation results
│   ├── ablations/                 # Ablation study data
│   └── plots/                     # Generated visualizations
│
└── config/
    ├── sft_config.yaml
    ├── logit_kd_config.yaml
    ├── cot_config.yaml
    └── dpo_config.yaml
```

---

## Module 1: DatasetBuilder.py

**Purpose:** Centralized data preparation, loading, and preprocessing for all experiments.

### Responsibilities

1. **Download and Cache Datasets**
   - MedQA, MedMCQA, PubMedQA, PubHealth
   - Handle authentication and API access
   - Cache locally to avoid re-downloading

2. **Build Med-DistillMix-120k**
   - Combine 80k MedQA/MedMCQA + 20k PubMedQA + 20k PubHealth
   - Deduplicate questions
   - Split into train/validation/holdout (90%/5%/5%)

3. **Create Specialized Datasets**
   - MedPPL-10k (perplexity evaluation corpus)
   - FidelityBench-Med (hallucination detection)
   - DPO preference pairs (chosen/rejected responses)

4. **Format Data for Training**
   - Tokenize questions and responses
   - Create prompts in consistent format
   - Handle padding and truncation
   - Generate PyTorch DataLoaders

### Key Classes

```python
class DatasetBuilder:
    """Main class for all data operations."""
    
    def __init__(self, config):
        """Initialize with data config (paths, sizes, splits)."""
        pass
    
    def download_raw_datasets(self):
        """Download MedQA, MedMCQA, PubMedQA, PubHealth from HuggingFace."""
        pass
    
    def build_training_dataset(self, target_size=120000):
        """Create Med-DistillMix-120k from raw datasets."""
        pass
    
    def create_perplexity_corpus(self, size=10000):
        """Sample 10k PubMed abstracts for perplexity evaluation."""
        pass
    
    def create_fidelity_bench(self, size=1500):
        """Build FidelityBench-Med with evidence passages."""
        pass
    
    def load_teacher_responses(self, response_type='standard'):
        """Load pre-generated teacher outputs (standard, CoT, or logits)."""
        pass
    
    def get_dataloader(self, split='train', batch_size=8, method='sft'):
        """Return PyTorch DataLoader for specific split and distillation method."""
        pass
    
    def get_benchmark_dataloader(self, benchmark_name):
        """Return DataLoader for evaluation benchmarks (MedQA, MedMCQA, etc.)."""
        pass
```

### Example Usage

```python
from src.DatasetBuilder import DatasetBuilder

# Initialize
builder = DatasetBuilder(config='config/data_config.yaml')

# Build training data (run once)
builder.download_raw_datasets()
builder.build_training_dataset(target_size=120000)

# Get DataLoaders for training
train_loader = builder.get_dataloader(split='train', batch_size=8, method='logit_kd')
val_loader = builder.get_dataloader(split='validation', batch_size=8, method='logit_kd')

# Get benchmark DataLoaders for evaluation
medqa_loader = builder.get_benchmark_dataloader('medqa')
medmcqa_loader = builder.get_benchmark_dataloader('medmcqa')
```

---

## Module 2: DistillationMethods.py

**Purpose:** Implement all four knowledge distillation methods with a consistent, unified API.

### Design Pattern: Abstract Base Class

Using an abstract base class ensures all methods implement the same interface, making them interchangeable in the training pipeline.

### Class Hierarchy

```python
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class BaseDistillationMethod(ABC):
    """Abstract base class for all distillation methods."""
    
    def __init__(self, teacher_model, student_model, tokenizer, config):
        """
        Initialize distillation method.
        
        Args:
            teacher_model: Pre-trained teacher (Meditron-7B)
            student_model: Student to train (Qwen2-1.5B)
            tokenizer: Shared tokenizer
            config: Method-specific hyperparameters
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Put teacher in eval mode (never train)
        self.teacher_model.eval()
    
    @abstractmethod
    def compute_loss(self, batch):
        """
        Compute loss for one training batch.
        
        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', 'labels'
        
        Returns:
            loss: Scalar tensor for backpropagation
            metrics: Dict with detailed loss components for logging
        """
        pass
    
    @abstractmethod
    def get_method_name(self):
        """Return human-readable method name (e.g., 'Logit-KD')."""
        pass
    
    def prepare_student_for_training(self, lora_config):
        """Setup LoRA adapters for efficient training."""
        pass
```

### Concrete Implementations

#### 1. Sequence-Level SFT

```python
class SequenceSFT(BaseDistillationMethod):
    """Standard supervised fine-tuning on teacher's text outputs."""
    
    def compute_loss(self, batch):
        """
        Simple cross-entropy loss on teacher's generated text.
        Student learns to reproduce teacher's text output.
        """
        outputs = self.student_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        return loss, {'ce_loss': loss.item()}
    
    def get_method_name(self):
        return "Sequence-SFT"
```

#### 2. Logit Knowledge Distillation

```python
class LogitKD(BaseDistillationMethod):
    """Token-level distillation using KL divergence on softened logits."""
    
    def __init__(self, teacher_model, student_model, tokenizer, config):
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.alpha = config.get('alpha', 0.5)  # KD vs CE mixing
        self.temperature = config.get('temperature', 3.0)  # Softening factor
    
    def compute_loss(self, batch):
        """
        Combines KL divergence (teacher similarity) with cross-entropy (correctness).
        """
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            teacher_logits = teacher_outputs.logits
        
        # Student forward pass (with gradients)
        student_outputs = self.student_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        student_logits = student_outputs.logits
        
        # KD Loss (teacher similarity)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        # CE Loss (correctness)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }
    
    def get_method_name(self):
        return f"Logit-KD (α={self.alpha}, T={self.temperature})"
```

#### 3. Chain-of-Thought Distillation

```python
class CoTDistillation(BaseDistillationMethod):
    """Teach student to generate reasoning chains like the teacher."""
    
    def compute_loss(self, batch):
        """
        Train on teacher's reasoning chains (question + reasoning + answer).
        Uses longer sequences to accommodate step-by-step explanations.
        """
        # Batch contains full reasoning chains from teacher
        outputs = self.student_model(
            input_ids=batch['input_ids'],  # Longer sequences (up to 1536 tokens)
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        return loss, {'cot_loss': loss.item()}
    
    def get_method_name(self):
        return "CoT-Distillation"
```

#### 4. Preference-Based KD (DPO)

```python
class PreferenceKD(BaseDistillationMethod):
    """Direct Preference Optimization using chosen/rejected pairs."""
    
    def __init__(self, teacher_model, student_model, tokenizer, config):
        super().__init__(teacher_model, student_model, tokenizer, config)
        self.beta = config.get('beta', 0.1)  # DPO temperature parameter
    
    def compute_loss(self, batch):
        """
        DPO loss: maximize likelihood of chosen response over rejected.
        Batch contains both chosen (teacher) and rejected (baseline) responses.
        """
        # Get logits for chosen response
        chosen_outputs = self.student_model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask']
        )
        
        # Get logits for rejected response
        rejected_outputs = self.student_model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask']
        )
        
        # Compute DPO loss (simplified)
        # Real implementation would use log probabilities and reference model
        chosen_logps = self._get_log_probs(chosen_outputs.logits, batch['chosen_labels'])
        rejected_logps = self._get_log_probs(rejected_outputs.logits, batch['rejected_labels'])
        
        dpo_loss = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps)).mean()
        
        return dpo_loss, {'dpo_loss': dpo_loss.item()}
    
    def _get_log_probs(self, logits, labels):
        """Helper to compute log probabilities."""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).mean()
    
    def get_method_name(self):
        return f"DPO (β={self.beta})"
```

### Example Usage

```python
from src.DistillationMethods import SequenceSFT, LogitKD, CoTDistillation, PreferenceKD

# Load models
teacher_model = load_teacher_model()
student_model = load_student_model()
tokenizer = load_tokenizer()

# Initialize any method with same API
method = LogitKD(
    teacher_model=teacher_model,
    student_model=student_model,
    tokenizer=tokenizer,
    config={'alpha': 0.5, 'temperature': 3.0}
)

# Use in training loop
for batch in train_loader:
    loss, metrics = method.compute_loss(batch)
    loss.backward()
    optimizer.step()
```

---

## Module 3: Trainer.py

**Purpose:** Orchestrate training, evaluation, ablation studies, and result collection for all distillation methods.

### Responsibilities

1. **Training Orchestration**
   - Train any distillation method using the unified API
   - Handle checkpointing and early stopping
   - Log metrics (loss, learning rate, GPU usage)
   - Support multi-GPU training (optional)

2. **Evaluation**
   - Run models on benchmark datasets (MedQA, MedMCQA, PubMedQA, PubHealth)
   - Calculate accuracy, F1, other metrics
   - Compute perplexity on MedPPL-10k
   - Measure fidelity metrics (KL divergence, top-k overlap)

3. **Ablation Studies**
   - Systematically vary hyperparameters
   - Track computational cost (GPU-hours, memory)
   - Compare results across configurations

4. **Result Management**
   - Save all results to structured format (CSV, JSON)
   - Generate comparison tables
   - Create plots and visualizations

### Key Classes

```python
class Trainer:
    """Main training orchestrator for distillation experiments."""
    
    def __init__(self, method, dataloader_builder, config):
        """
        Initialize trainer.
        
        Args:
            method: Instance of BaseDistillationMethod
            dataloader_builder: DatasetBuilder instance
            config: Training configuration (epochs, lr, etc.)
        """
        self.method = method
        self.dataloader_builder = dataloader_builder
        self.config = config
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
    
    def train(self, num_epochs):
        """
        Main training loop.
        
        Returns:
            training_history: Dict with loss curves and metrics
        """
        pass
    
    def evaluate_benchmarks(self):
        """
        Evaluate on all four medical benchmarks.
        
        Returns:
            results: Dict with accuracy/F1 for each benchmark
        """
        pass
    
    def compute_perplexity(self):
        """
        Measure perplexity on MedPPL-10k corpus.
        
        Returns:
            perplexity: Float value
        """
        pass
    
    def measure_fidelity(self):
        """
        Compute teacher-student fidelity metrics.
        
        Returns:
            fidelity_metrics: Dict with KL divergence, top-k overlap, etc.
        """
        pass
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint and training state."""
        pass
    
    def load_checkpoint(self, path):
        """Resume training from checkpoint."""
        pass


class Evaluator:
    """Separate class for evaluation and analysis (optional)."""
    
    def __init__(self, model, tokenizer, dataloader_builder):
        """Initialize evaluator with trained model."""
        pass
    
    def run_benchmark(self, benchmark_name):
        """Run single benchmark and return results."""
        pass
    
    def compute_perplexity(self, corpus):
        """Compute perplexity on text corpus."""
        pass
    
    def measure_hallucination_rate(self):
        """Use FidelityBench-Med to measure hallucinations."""
        pass
    
    def compare_to_teacher(self):
        """Measure KL divergence and agreement with teacher."""
        pass


class AblationRunner:
    """Helper class to run ablation studies (optional)."""
    
    def __init__(self, base_config):
        """Initialize with baseline configuration."""
        pass
    
    def run_ablation(self, param_name, param_values):
        """
        Run multiple experiments varying one parameter.
        
        Args:
            param_name: e.g., 'alpha', 'temperature', 'lora_rank'
            param_values: List of values to try
        
        Returns:
            results: DataFrame with results for each configuration
        """
        pass
    
    def generate_ablation_plots(self, results):
        """Create plots showing parameter effects."""
        pass
```

### Example Usage

```python
from src.Trainer import Trainer
from src.DatasetBuilder import DatasetBuilder
from src.DistillationMethods import LogitKD

# Setup
dataloader_builder = DatasetBuilder(config='config/data_config.yaml')
method = LogitKD(teacher, student, tokenizer, config={'alpha': 0.5, 'temperature': 3.0})

# Initialize trainer
trainer = Trainer(
    method=method,
    dataloader_builder=dataloader_builder,
    config={'epochs': 3, 'learning_rate': 2e-4, 'batch_size': 8}
)

# Train
training_history = trainer.train(num_epochs=3)

# Evaluate
benchmark_results = trainer.evaluate_benchmarks()
perplexity = trainer.compute_perplexity()
fidelity_metrics = trainer.measure_fidelity()

# Print results
print(f"MedQA Accuracy: {benchmark_results['medqa']['accuracy']:.2f}%")
print(f"Perplexity: {perplexity:.2f}")
print(f"KL Divergence to Teacher: {fidelity_metrics['kl_divergence']:.4f}")
```

---

## Complete Workflow Example

Here's how all three modules work together:

```python
#!/usr/bin/env python3
"""
Complete training pipeline for medical LLM distillation.
"""

from src.DatasetBuilder import DatasetBuilder
from src.DistillationMethods import SequenceSFT, LogitKD, CoTDistillation, PreferenceKD
from src.Trainer import Trainer
import yaml

# 1. Setup data
print("Building datasets...")
data_builder = DatasetBuilder(config='config/data_config.yaml')
data_builder.download_raw_datasets()
data_builder.build_training_dataset(target_size=120000)

# 2. Load models
teacher_model = load_model("epfl-llm/meditron-7b")
student_model = load_model("Qwen/Qwen2-1.5B")
tokenizer = load_tokenizer("Qwen/Qwen2-1.5B")

# 3. Define methods to compare
methods_to_train = [
    SequenceSFT(teacher_model, student_model, tokenizer, config={}),
    LogitKD(teacher_model, student_model, tokenizer, config={'alpha': 0.5, 'temperature': 3.0}),
    CoTDistillation(teacher_model, student_model, tokenizer, config={}),
    PreferenceKD(teacher_model, student_model, tokenizer, config={'beta': 0.1})
]

# 4. Train and evaluate each method
results_summary = []

for method in methods_to_train:
    print(f"\n{'='*60}")
    print(f"Training: {method.get_method_name()}")
    print(f"{'='*60}")
    
    # Train
    trainer = Trainer(method, data_builder, config='config/train_config.yaml')
    history = trainer.train(num_epochs=3)
    
    # Evaluate
    benchmark_results = trainer.evaluate_benchmarks()
    perplexity = trainer.compute_perplexity()
    fidelity = trainer.measure_fidelity()
    
    # Save results
    results_summary.append({
        'method': method.get_method_name(),
        'medqa_acc': benchmark_results['medqa']['accuracy'],
        'medmcqa_acc': benchmark_results['medmcqa']['accuracy'],
        'perplexity': perplexity,
        'kl_divergence': fidelity['kl_divergence'],
        'gpu_hours': trainer.get_gpu_hours()
    })

# 5. Generate comparison report
import pandas as pd
df = pd.DataFrame(results_summary)
df.to_csv('results/comparison_table.csv', index=False)
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(df.to_string(index=False))
```

---

## Configuration Files

Each distillation method can have its own YAML config:

### config/logit_kd_config.yaml
```yaml
method: LogitKD
model:
  teacher: epfl-llm/meditron-7b
  student: Qwen/Qwen2-1.5B
  
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  
hyperparameters:
  alpha: 0.5
  temperature: 3.0
  learning_rate: 2e-4
  batch_size: 8
  epochs: 3
  max_length: 1024
  
training:
  save_steps: 500
  eval_steps: 1000
  logging_steps: 10
  gradient_accumulation_steps: 16
  warmup_steps: 100
  
output:
  checkpoint_dir: models/students/logit_kd
  log_dir: logs/logit_kd
```

---

## Benefits of This Architecture

### 1. **Modularity**
- Each component has a single, well-defined responsibility
- Easy to modify one part without affecting others

### 2. **Consistency**
- All distillation methods use the same API (BaseDistillationMethod)
- Easy to add new methods without changing training code

### 3. **Reusability**
- DatasetBuilder can be used across all experiments
- Trainer can work with any distillation method

### 4. **Testability**
- Each module can be tested independently
- Mock objects can be used for unit tests

### 5. **Scalability**
- Easy to add new distillation methods
- Easy to add new evaluation benchmarks
- Easy to run parameter sweeps and ablations

### 6. **Maintainability**
- Clear separation of concerns
- Well-documented interfaces
- Easy for team members to work on different parts

---

## Development Workflow

### Phase 1: Data Setup
```bash
python -m src.DatasetBuilder --download
python -m src.DatasetBuilder --build-training-set
python -m src.DatasetBuilder --create-perplexity-corpus
```

### Phase 2: Implement Methods
```bash
# Test each method individually
python test_distillation_methods.py --method sft
python test_distillation_methods.py --method logit_kd
python test_distillation_methods.py --method cot
python test_distillation_methods.py --method dpo
```

### Phase 3: Train
```bash
# Train single method
python train.py --method logit_kd --config config/logit_kd_config.yaml

# Train all methods
python train_all.py
```

### Phase 4: Evaluate
```bash
# Evaluate single model
python evaluate.py --model models/students/logit_kd/final

# Compare all methods
python compare_methods.py --output results/comparison_report.pdf
```

### Phase 5: Ablations
```bash
# Run ablation study
python run_ablation.py --method logit_kd --param temperature --values 2,3,4
python run_ablation.py --method logit_kd --param alpha --values 0.3,0.5,0.7
```

---

## Testing Strategy

### Unit Tests
```python
# test_dataset_builder.py
def test_download_datasets():
    """Test dataset downloading."""
    pass

def test_data_splitting():
    """Test 90/5/5 split."""
    pass

# test_distillation_methods.py
def test_logit_kd_loss():
    """Test LogitKD loss computation."""
    pass

def test_method_consistency():
    """Ensure all methods implement required interface."""
    pass

# test_trainer.py
def test_training_loop():
    """Test training completes without errors."""
    pass
```

### Integration Tests
```python
# test_end_to_end.py
def test_full_pipeline():
    """Test complete workflow from data to evaluation."""
    pass
```

---

## Summary

This architecture provides:
- **DatasetBuilder**: Single source of truth for all data operations
- **DistillationMethods**: Unified API for all four distillation approaches
- **Trainer**: Centralized training, evaluation, and analysis

The modular design makes it easy to:
- Add new distillation methods
- Run fair comparisons
- Perform systematic ablations
- Scale to more methods or benchmarks

Each component can be developed and tested independently while working together seamlessly through well-defined interfaces.
