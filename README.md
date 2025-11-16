# Med-Distillation 🏥🤖

> **Comparative Analysis of Knowledge Distillation Methods for Medical Question Answering**

A comprehensive framework for distilling large medical language models (70B parameters) into efficient smaller models (1.5B parameters) using multiple state-of-the-art distillation techniques.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Distillation Methods](#distillation-methods)
- [Documentation](#documentation)
- [Usage Examples](#usage-examples)
- [Citation](#citation)

---

## 🔬 Overview

This project implements and compares **10 different knowledge distillation methods** for medical language models, focusing on transferring knowledge from a large teacher model (Meditron-70B) to a compact student model (Qwen2-1.5B).

### Key Research Questions:
1. Which distillation method best preserves medical reasoning capabilities?
2. How does distillation affect factual accuracy in medical QA?
3. What is the trade-off between model size, inference speed, and accuracy?
4. Can we measure evidence faithfulness in distilled models?

---

## ✨ Features

### 🎯 Distillation Methods
- **Supervised Fine-Tuning (SFT)**: Baseline imitation learning
- **Logit-KD**: Soft target distribution matching
- **Adaptive-KD (AdaKD)**: Token-adaptive temperature adjustment
- **Chain-of-Thought (CoT)**: Reasoning process distillation
- **FitNets**: Intermediate layer representation matching
- **Attention Transfer**: Attention pattern mimicry
- **On-Policy RL**: Policy gradient with teacher rewards
- **PPO**: Proximal policy optimization
- **BOND**: Best-of-N distillation
- **SPIN**: Self-play iterative refinement

### 📊 Comprehensive Evaluation
- **4 Medical Benchmarks**: MedQA, MedMCQA, PubMedQA, PubHealth
- **Perplexity Analysis**: Language modeling quality on MedPPL-10k
- **Fidelity Metrics**: KL divergence, BLEU, ROUGE, top-k overlap
- **FidelityBench-Med**: NLI-based fact verification and hallucination detection
- **Automatic Visualization**: Training curves, benchmark comparisons, radar charts

### ⚙️ Technical Features
- **QLoRA**: 8-bit quantization + LoRA for memory efficiency
- **Gradient Accumulation**: Effective large batch training
- **Automatic Mixed Precision**: Faster training with FP16
- **Ablation Studies**: Hyperparameter sensitivity analysis
- **Comparative Teacher Evaluation**: Measure knowledge retention gap

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 24GB+ VRAM (RTX 4090 or A100 recommended)
- 32GB+ System RAM

### Option 1: Using `uv` (Recommended)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/shreyanmitra/Medistillation.git
cd Medistillation

# Sync dependencies
uv sync
```

### Option 2: Using `pip`

```bash
# Clone repository
git clone https://github.com/shreyanmitra/Medistillation.git
cd Medistillation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Step 1: Prepare Datasets

```bash
# Download and prepare all datasets (Med-DistillMix, benchmarks, etc.)
python src/DataLoader.py --prepare_all
```

This creates:
- `data/processed/train.jsonl` (~414k examples)
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `data/benchmarks/` (MedQA, MedMCQA, PubMedQA, PubHealth)
- `data/medppl_10k.jsonl` (perplexity corpus)
- `data/fidelitybench_med.jsonl` (faithfulness evaluation)

### Step 2: Train a Student Model

```bash
# Supervised Fine-Tuning (SFT) - Baseline
python src/Trainer.py \
    --method sft \
    --teacher_model epfl-llm/meditron-70b \
    --student_model Qwen/Qwen2-1.5B \
    --num_epochs 3 \
    --batch_size 8 \
    --output_dir outputs/sft_run1
```

### Step 3: View Results

```bash
# Training curves and evaluation plots
ls outputs/sft_run1/results/*.png

# Evaluation metrics (JSON)
cat outputs/sft_run1/results/comprehensive_evaluation.json
```

**Generated Plots:**
- `training_curves.png` - Loss and learning rate over epochs
- `benchmark_comparison.png` - Student vs teacher accuracy
- `fidelity_metrics.png` - KL/BLEU/ROUGE scores
- `fidelitybench_radar.png` - Evidence faithfulness radar chart

---

## 📁 Project Structure

```
Medistillation/
├── data/                          # Datasets (gitignored)
│   ├── raw/                       # Original downloaded data
│   ├── processed/                 # Train/val/test splits
│   ├── benchmarks/                # Evaluation benchmarks
│   ├── medppl_10k.jsonl          # Perplexity corpus
│   └── fidelitybench_med.jsonl   # Faithfulness eval
│
├── src/                           # Source code
│   ├── DataLoader.py             # Dataset preparation and loading
│   ├── DistillationMethods.py    # Distillation algorithm implementations
│   └── Trainer.py                # Training loop and evaluation
│
├── docs/                          # Documentation
│   ├── guides/                    # User guides
│   │   ├── EXPERIMENT_PROCEDURE.md
│   │   ├── FIDELITYBENCH_GUIDE.md
│   │   └── VISUALIZATION_GUIDE.md
│   ├── implementation/            # Technical details
│   └── research/                  # Research materials
│
├── scripts/                       # Utility scripts
│   └── sample_medmcqa.py         # Data sampling utility
│
├── outputs/                       # Training results (gitignored)
│   └── {method}_{run_name}/
│       ├── checkpoints/
│       ├── results/
│       └── final_model/
│
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧪 Distillation Methods

### 1. Supervised Fine-Tuning (SFT)
```bash
python src/Trainer.py --method sft --num_epochs 3
```

### 2. Logit-KD
```bash
python src/Trainer.py --method logit_kd --alpha 0.5 --temperature 3.0
```

### 3. Adaptive-KD (AdaKD)
```bash
python src/Trainer.py --method adakd --base_temperature 3.0 --min_temperature 1.0 --max_temperature 5.0
```

### 4. Chain-of-Thought (CoT)
```bash
python src/Trainer.py --method cot --num_rationales 3 --sampling_temperature 0.7
```

### 5. FitNets
```bash
python src/Trainer.py --method fitnets --layer_mapping '{"6":12,"12":24}' --use_projections
```

### 6. Attention Transfer
```bash
python src/Trainer.py --method attention --layer_mapping '{"6":12,"12":24}' --match_all_heads
```

### 7. PPO
```bash
python src/Trainer.py --method ppo --epsilon 0.2 --gamma 0.99
```

### 8. BOND (Best-of-N)
```bash
python src/Trainer.py --method bond --num_samples 16
```

### 9. SPIN (Self-Play)
```bash
python src/Trainer.py --method spin --beta 0.1
```

---

## 📖 Documentation

Comprehensive guides available in `docs/`:

### User Guides (`docs/guides/`)
- **[EXPERIMENT_PROCEDURE.md](docs/guides/EXPERIMENT_PROCEDURE.md)** - Complete experimental protocol
- **[FIDELITYBENCH_GUIDE.md](docs/guides/FIDELITYBENCH_GUIDE.md)** - Evidence faithfulness evaluation
- **[VISUALIZATION_GUIDE.md](docs/guides/VISUALIZATION_GUIDE.md)** - Plotting and analysis

### Implementation Details (`docs/implementation/`)
- **[Code_Organization_Architecture.md](docs/implementation/Code_Organization_Architecture.md)** - Codebase structure
- **[DATASET_STRATEGY.md](docs/implementation/DATASET_STRATEGY.md)** - Data preparation strategy
- **[VISUALIZATION_FEATURES.md](docs/implementation/VISUALIZATION_FEATURES.md)** - Plotting features

### Research Materials (`docs/research/`)
- **[project_details_latex.tex](docs/research/project_details_latex.tex)** - LaTeX project description
- **[references.bib](docs/research/references.bib)** - BibTeX references

---

## 💡 Usage Examples

### Example 1: Hyperparameter Ablation

```bash
# Test different temperatures for Logit-KD
python src/Trainer.py \
    --run_ablation \
    --ablation_type temperature \
    --ablation_values "2.0,3.0,4.0,5.0" \
    --method logit_kd

# View sensitivity analysis plot
start outputs/ablation_temperature/ablation_plot.png  # Windows
# open outputs/ablation_temperature/ablation_plot.png  # Mac/Linux
```

### Example 2: Compare Multiple Methods

```bash
# Train different methods
python src/Trainer.py --method sft --output_dir outputs/sft
python src/Trainer.py --method logit_kd --alpha 0.5 --temperature 3.0 --output_dir outputs/logit_kd
python src/Trainer.py --method adakd --output_dir outputs/adakd

# Compare results
python scripts/compare_results.py outputs/sft outputs/logit_kd outputs/adakd
```

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{medistillation2025,
  title={Comparative Analysis of Knowledge Distillation Methods for Medical Question Answering},
  author={CSE 493S Team},
  year={2025},
  publisher={University of Washington},
  url={https://github.com/shreyanmitra/Medistillation}
}
```

---

## 🙏 Acknowledgments

- **Teacher Model**: [Meditron-70B](https://huggingface.co/epfl-llm/meditron-70b) by EPFL LLM Team
- **Student Model**: [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) by Alibaba Cloud
- **Datasets**: MedQA, MedMCQA, PubMedQA, PubHealth teams
- **Course**: CSE 493S - Advanced Topics in Machine Learning, University of Washington

---

**Happy Distilling! 🧪✨**