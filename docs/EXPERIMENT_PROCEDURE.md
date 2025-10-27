# Medical LLM Distillation Experiment: Complete Step-by-Step Procedure

**Project:** Comparison of Performance of Large Language Model Distillation Methodologies for Medical QA

**Goal:** Compare four different methods of teaching a small AI model (student) to imitate a larger, more capable AI model (teacher) for answering medical questions.

---

## Table of Contents
1. [Prerequisites and Setup](#prerequisites-and-setup)
2. [Phase 1: Environment Preparation](#phase-1-environment-preparation)
3. [Phase 2: Data Collection and Preparation](#phase-2-data-collection-and-preparation)
4. [Phase 3: Teacher Model Setup and Response Generation](#phase-3-teacher-model-setup-and-response-generation)
5. [Phase 4: Student Model Distillation (4 Methods)](#phase-4-student-model-distillation-4-methods)
6. [Phase 5: Evaluation and Benchmarking](#phase-5-evaluation-and-benchmarking)
7. [Phase 6: Analysis and Reporting](#phase-6-analysis-and-reporting)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Prerequisites and Setup

### Hardware Requirements
- **Minimum:** 1× NVIDIA RTX 4090 GPU with 24GB VRAM
- **Recommended:** 1-2× NVIDIA A100 GPU with 40GB VRAM each
- **RAM:** At least 32GB system memory
- **Storage:** 500GB+ free disk space (for models, datasets, and checkpoints)

### Software Requirements
- **Operating System:** Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **Python:** Version 3.9 or higher
- **CUDA:** Version 11.8 or higher (for GPU support)
- **Git:** For version control and downloading repositories

### Account Setup
1. **Hugging Face Account:**
   - Go to https://huggingface.co/join
   - Create a free account
   - Go to Settings → Access Tokens
   - Create a new token with "Read" permissions
   - Save this token securely (you'll need it later)

2. **GitHub Account:**
   - If you don't have one, create an account at https://github.com

---

## Phase 1: Environment Preparation

### Step 1.1: Create Project Directory Structure
```
Medistillation/
├── data/
│   ├── raw/
│   ├── processed/
│   └── teacher_outputs/
├── models/
│   ├── teacher/
│   └── students/
│       ├── sft/
│       ├── logit_kd/
│       ├── cot/
│       └── dpo/
├── scripts/
│   ├── data_preparation/
│   ├── training/
│   └── evaluation/
├── results/
│   ├── benchmarks/
│   ├── ablations/
│   └── plots/
├── logs/
└── config/
```

**Action:** Create these folders manually or use the provided setup script.

### Step 1.2: Install Python Dependencies

Create a file named `requirements.txt` with the following content:
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0
lm-eval>=0.4.0
pyterrier>=0.10.0
ragas>=0.1.0
```

**Commands to run:**
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login
# (Enter your token from Prerequisites)
```

### Step 1.3: Configure Experiment Tracking (Optional but Recommended)

**Using Weights & Biases:**
1. Create account at https://wandb.ai
2. Get your API key from https://wandb.ai/authorize
3. Run: `wandb login` and enter your API key

---

## Phase 2: Data Collection and Preparation

### Step 2.1: Download Raw Datasets

Create a script `scripts/data_preparation/download_datasets.py`:

**Purpose:** Download the four medical QA datasets from Hugging Face.

**What this does:**
- Downloads MedQA-USMLE (medical licensing exam questions)
- Downloads MedMCQA (Indian medical entrance exam questions)
- Downloads PubMedQA (biomedical research questions)
- Downloads PubHealth (health claim verification data)

**Run command:**
```bash
python scripts/data_preparation/download_datasets.py
```

**Expected output:** Raw datasets saved in `data/raw/` folder (approximately 50-100GB).

### Step 2.2: Create Med-DistillMix-120k Training Dataset

Create a script `scripts/data_preparation/create_training_dataset.py`:

**Purpose:** Combine and prepare 120,000 training examples.

**What this script should do:**
1. Load all four datasets
2. Extract 80,000 examples from MedQA and MedMCQA (combined)
3. Extract 20,000 examples from PubMedQA
4. Extract 20,000 examples from PubHealth
5. Remove duplicates based on question text similarity
6. Standardize format (question, options, correct_answer)
7. Split data:
   - 90% training (108,000 examples)
   - 5% validation (6,000 examples)
   - 5% holdout (6,000 examples)
8. Ensure no overlap with official test sets

**Run command:**
```bash
python scripts/data_preparation/create_training_dataset.py
```

**Expected output:**
- `data/processed/train.jsonl` (108,000 examples)
- `data/processed/validation.jsonl` (6,000 examples)
- `data/processed/holdout.jsonl` (6,000 examples)

**Validation check:**
- Verify file sizes match expected counts
- Check for duplicate questions: should be minimal
- Inspect first 10 examples to ensure proper formatting

### Step 2.3: Prepare Perplexity Evaluation Corpus

Create a script `scripts/data_preparation/create_perplexity_corpus.py`:

**Purpose:** Create a 10,000 passage corpus from PubMed abstracts for measuring language quality.

**What this script should do:**
1. Download PubMed abstracts (use PubMed API or existing dataset)
2. Randomly sample 10,000 abstracts
3. Clean and preprocess text
4. Save as `data/processed/medppl_10k.jsonl`

**Run command:**
```bash
python scripts/data_preparation/create_perplexity_corpus.py
```

### Step 2.4: Prepare FidelityBench-Med Suite

Create a script `scripts/data_preparation/create_fidelity_bench.py`:

**Purpose:** Create 1,000-2,000 prompts with evidence passages for measuring hallucination.

**What this script should do:**
1. Sample 1,500 questions requiring evidence-based answers
2. For each question, retrieve 3-5 relevant passages using BM25 search
3. Annotate expected citations and key facts
4. Save as `data/processed/fidelitybench_med.jsonl`

**Run command:**
```bash
python scripts/data_preparation/create_fidelity_bench.py
```

---

## Phase 3: Teacher Model Setup and Response Generation

### Step 3.1: Download and Test Teacher Model

Create a script `scripts/model_setup/download_teacher.py`:

**Purpose:** Download Meditron-7B model and verify it works.

**What this script should do:**
1. Download Meditron-7B from Hugging Face: `epfl-llm/meditron-7b`
2. Load model in 8-bit precision to save memory
3. Run a test inference on 5 sample questions
4. Save model to `models/teacher/`

**Run command:**
```bash
python scripts/model_setup/download_teacher.py
```

**Expected duration:** 30-60 minutes (downloading ~14GB model)

**Validation:** Check that test questions produce reasonable medical answers.

### Step 3.2: Generate Teacher Responses for All Training Data

Create a script `scripts/data_preparation/generate_teacher_responses.py`:

**Purpose:** Get the teacher model's answers for all 120,000 training examples.

**What this script should do:**
1. Load Meditron-7B teacher model
2. For each question in training/validation/holdout:
   - Generate standard answer (for SFT)
   - Generate answer with reasoning (for CoT distillation)
   - Save top-k token probabilities (for Logit KD)
3. Process in batches of 8-16 to maximize GPU usage
4. Save outputs to `data/teacher_outputs/`

**Configuration:**
- Temperature: 0.7 for standard answers
- Max tokens: 512 for standard, 1024 for CoT
- Batch size: 8-16 depending on GPU memory

**Run command:**
```bash
python scripts/data_preparation/generate_teacher_responses.py
```

**Expected duration:** 20-40 GPU-hours (1-2 days on single A100)

**Output files:**
- `data/teacher_outputs/standard_responses.jsonl` (all examples with teacher answers)
- `data/teacher_outputs/cot_responses.jsonl` (40k examples with reasoning chains)
- `data/teacher_outputs/logits_cache/` (optional: saved probability distributions)

**Progress tracking:** Use tqdm progress bars and save checkpoints every 10,000 examples.

### Step 3.3: Generate DPO Preference Pairs

Create a script `scripts/data_preparation/generate_dpo_pairs.py`:

**Purpose:** Create chosen/rejected pairs for preference optimization.

**What this script should do:**
1. For each training example:
   - **Chosen response:** Teacher's answer
   - **Rejected response:** Either:
     - Student baseline answer (before training)
     - Perturbed teacher answer (with introduced errors)
     - Random incorrect answer from options
2. Save pairs to `data/teacher_outputs/dpo_pairs.jsonl`

**Run command:**
```bash
python scripts/data_preparation/generate_dpo_pairs.py
```

---

## Phase 4: Student Model Distillation (4 Methods)

### Overview
You will train the Qwen2-1.5B student model using four different methods. Each method will take 25-55 GPU-hours.

### Step 4.1: Download Student Base Model

Create a script `scripts/model_setup/download_student.py`:

**Purpose:** Download Qwen2-1.5B model.

**Run command:**
```bash
python scripts/model_setup/download_student.py
```

### Step 4.2: Method 1 - Sequence-Level SFT (Supervised Fine-Tuning)

Create a script `scripts/training/train_sft.py`:

**What this method does:** Trains the student to directly copy the teacher's text outputs.

**Training configuration:**
```python
{
    "model": "Qwen/Qwen2-1.5B",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "batch_size": 128 (via gradient accumulation),
    "max_steps": 10000,
    "sequence_length": 1024,
    "precision": "bf16"
}
```

**Run command:**
```bash
python scripts/training/train_sft.py --config config/sft_config.yaml
```

**Expected duration:** 25-40 GPU-hours

**Output:** Trained model adapters saved to `models/students/sft/`

**Monitoring:**
- Watch training loss (should decrease steadily)
- Check validation loss every 500 steps
- Save checkpoint every 2,000 steps

### Step 4.3: Method 2 - Logit KD (Knowledge Distillation with Token Probabilities)

Create a script `scripts/training/train_logit_kd.py`:

**What this method does:** Trains student to match teacher's probability distributions over tokens, not just the final answer.

**Training configuration:**
```python
{
    "model": "Qwen/Qwen2-1.5B",
    "lora_rank": 16,
    "temperature": [2, 3, 4],  # Will try all three
    "alpha": [0.3, 0.5, 0.7],  # Mix between KD loss and CE loss
    "learning_rate": 2e-4,
    "batch_size": 128,
    "max_steps": 15000,
    "sequence_length": 1024
}
```

**Run command (parameter sweep):**
```bash
# Try 9 combinations (3 temperatures × 3 alphas)
for temp in 2 3 4; do
    for alpha in 0.3 0.5 0.7; do
        python scripts/training/train_logit_kd.py \
            --temperature $temp \
            --alpha $alpha \
            --output_dir models/students/logit_kd/T${temp}_A${alpha}
    done
done
```

**Expected duration:** 35-55 GPU-hours per configuration (315-495 total for all 9)

**Recommendation:** Start with temperature=3, alpha=0.5, then sweep if time permits.

**Output:** Models saved with naming `logit_kd/T3_A0.5/` etc.

### Step 4.4: Method 3 - Chain-of-Thought (CoT) Distillation

Create a script `scripts/training/train_cot.py`:

**What this method does:** Trains student to generate reasoning steps before answering, like the teacher.

**Training configuration:**
```python
{
    "model": "Qwen/Qwen2-1.5B",
    "lora_rank": 32,  # Higher rank for complex reasoning
    "learning_rate": 2e-4,
    "batch_size": 64,  # Lower due to longer sequences
    "max_steps": 12000,
    "sequence_length": 1536,  # Longer for reasoning chains
    "use_cot_data": True  # 40k examples with reasoning
}
```

**Run command:**
```bash
python scripts/training/train_cot.py --config config/cot_config.yaml
```

**Expected duration:** 30-50 GPU-hours

**Output:** Model saved to `models/students/cot/`

**Note:** This training uses longer sequences, so memory usage is higher.

### Step 4.5: Method 4 - Preference-Based KD (DPO)

Create a script `scripts/training/train_dpo.py`:

**What this method does:** Trains student using Direct Preference Optimization, learning from chosen/rejected answer pairs.

**Training configuration:**
```python
{
    "model": "Qwen/Qwen2-1.5B",
    "lora_rank": 16,
    "learning_rate": 5e-5,  # Lower for DPO
    "batch_size": 128,
    "max_steps": 10000,
    "beta": 0.1,  # DPO temperature parameter
    "sequence_length": 1024
}
```

**Run command:**
```bash
python scripts/training/train_dpo.py --config config/dpo_config.yaml
```

**Expected duration:** 30-45 GPU-hours

**Output:** Model saved to `models/students/dpo/`

---

## Phase 5: Evaluation and Benchmarking

### Step 5.1: Download Official Test Sets

Create a script `scripts/evaluation/download_benchmarks.py`:

**Purpose:** Download official test sets that were NOT used in training.

**Datasets:**
- MedQA test set
- MedMCQA validation set
- PubMedQA test set
- PubHealth test set

**Run command:**
```bash
python scripts/evaluation/download_benchmarks.py
```

### Step 5.2: Benchmark Evaluation - Medical QA Accuracy

Create a script `scripts/evaluation/evaluate_medical_benchmarks.py`:

**Purpose:** Test all four student models on medical question benchmarks.

**What this script should do:**
1. Load each trained student model (SFT, Logit KD, CoT, DPO)
2. For each model, run inference on:
   - MedQA test set → measure accuracy
   - MedMCQA validation set → measure accuracy
   - PubMedQA test set → measure accuracy and macro-F1
   - PubHealth test set → measure accuracy and macro-F1
3. Also evaluate the teacher model as baseline
4. Save results to CSV

**Run command:**
```bash
python scripts/evaluation/evaluate_medical_benchmarks.py --all-models
```

**Expected duration:** 4-8 hours per model

**Output:** `results/benchmarks/medical_qa_results.csv`

**Format:**
```
Model,MedQA_Acc,MedMCQA_Acc,PubMedQA_Acc,PubMedQA_F1,PubHealth_Acc,PubHealth_F1
Teacher,85.2,78.3,76.4,74.1,79.8,77.2
SFT,72.1,68.5,71.2,69.3,73.4,71.8
LogitKD_T3_A05,74.3,70.2,72.8,70.9,74.9,73.1
CoT,73.8,69.7,73.1,71.2,74.2,72.5
DPO,75.1,71.4,73.5,71.8,75.3,73.7
```

### Step 5.3: Perplexity Evaluation

Create a script `scripts/evaluation/evaluate_perplexity.py`:

**Purpose:** Measure how well models understand biomedical language.

**What this script should do:**
1. Load MedPPL-10k corpus (PubMed abstracts)
2. For each model, compute perplexity on this corpus
3. Lower perplexity = better language modeling

**Run command:**
```bash
python scripts/evaluation/evaluate_perplexity.py
```

**Output:** `results/benchmarks/perplexity_results.csv`

### Step 5.4: FidelityBench-Med Evaluation

Create a script `scripts/evaluation/evaluate_fidelity.py`:

**Purpose:** Measure whether models hallucinate or faithfully cite evidence.

**What this script should do:**
1. Load FidelityBench-Med prompts
2. For each model:
   - Generate answers with evidence retrieval
   - Check citation coverage (did it cite provided evidence?)
   - Check hallucination rate (did it invent facts?)
   - Measure teacher-student agreement (token-level KL, top-k overlap)
3. Use RAGAS or NLI models to score faithfulness

**Run command:**
```bash
python scripts/evaluation/evaluate_fidelity.py
```

**Output:** `results/benchmarks/fidelity_results.csv`

### Step 5.5: Computational Cost Tracking

**What to record for each method:**
- Total GPU-hours used
- Peak GPU memory usage
- Training time (wall-clock hours)
- Inference speed (tokens/second)

Create a summary in `results/computational_costs.csv`:
```
Method,GPU_Hours,Peak_Memory_GB,Training_Time_Hours,Inference_Speed
SFT,32,38,28,145
LogitKD,48,40,42,138
CoT,41,39,36,122
DPO,38,37,34,141
```

---

## Phase 6: Analysis and Reporting

### Step 6.1: Ablation Studies

Create a script `scripts/evaluation/run_ablations.py`:

**Purpose:** Understand which hyperparameters matter most.

**Ablation experiments to run:**

1. **Dataset Size Ablation:**
   - Train SFT with 30k, 60k, 90k, 120k examples
   - Plot accuracy vs. dataset size

2. **Temperature Ablation (Logit KD):**
   - Already done if you ran all T={2,3,4}
   - Plot accuracy vs. temperature

3. **Alpha Ablation (Logit KD):**
   - Already done if you ran all α={0.3,0.5,0.7}
   - Plot accuracy vs. alpha mixing ratio

4. **LoRA Rank Ablation:**
   - Train SFT with rank={8, 16, 32, 64}
   - Measure accuracy and training time

**Run command:**
```bash
python scripts/evaluation/run_ablations.py --ablation dataset_size
python scripts/evaluation/run_ablations.py --ablation temperature
python scripts/evaluation/run_ablations.py --ablation alpha
python scripts/evaluation/run_ablations.py --ablation lora_rank
```

**Output:** Save results to `results/ablations/` with plots.

### Step 6.2: Theoretical Analysis - Fidelity vs. Performance

Create a script `scripts/analysis/fidelity_correlation.py`:

**Purpose:** Answer: "Does being closer to the teacher actually help?"

**What this script should do:**
1. Load all models' fidelity metrics (KL divergence, top-k overlap)
2. Load all models' benchmark accuracies
3. Compute correlations:
   - Correlation between KL divergence and accuracy
   - Correlation between top-k overlap and accuracy
   - Correlation between KL divergence and faithfulness
4. Fit regression models with confidence intervals
5. Create scatter plots with trend lines

**Run command:**
```bash
python scripts/analysis/fidelity_correlation.py
```

**Output:**
- `results/analysis/fidelity_vs_accuracy.png`
- `results/analysis/correlation_report.txt`

### Step 6.3: Create Visualization Plots

Create a script `scripts/analysis/create_plots.py`:

**Plots to generate:**

1. **Main Results Bar Chart:**
   - X-axis: Four distillation methods
   - Y-axis: Accuracy on each benchmark
   - Grouped bars for MedQA, MedMCQA, PubMedQA, PubHealth

2. **Perplexity Comparison:**
   - Bar chart of perplexity scores (lower is better)

3. **Computational Efficiency:**
   - Scatter plot: GPU-hours vs. accuracy
   - Shows which methods are most efficient

4. **Ablation Curves:**
   - Line plots for each ablation study

5. **Fidelity Analysis:**
   - Scatter plots from Step 6.2

**Run command:**
```bash
python scripts/analysis/create_plots.py
```

**Output:** All plots saved to `results/plots/`

### Step 6.4: Generate Summary Tables

Create a script `scripts/analysis/generate_tables.py`:

**Tables to create:**

1. **Main Results Table:**
   - Rows: Teacher, SFT, Logit KD, CoT, DPO
   - Columns: MedQA, MedMCQA, PubMedQA, PubHealth, Perplexity, GPU-hours

2. **Ablation Tables:**
   - One table per ablation study

3. **Fidelity Table:**
   - Rows: Each model
   - Columns: KL divergence, Top-1 agreement, Citation coverage, Hallucination rate

**Run command:**
```bash
python scripts/analysis/generate_tables.py
```

**Output:** LaTeX and Markdown tables saved to `results/tables/`

### Step 6.5: Write Final Report

Create a document `FINAL_REPORT.md` with the following sections:

**Structure:**
```markdown
# Medical LLM Distillation: Comparative Study Results

## Executive Summary
- Best performing method
- Key findings
- Answers to research questions

## Methods
- Brief description of four distillation approaches
- Experimental setup

## Results
### Main Findings
- Insert main results table
- Insert plots

### Benchmark Performance
- Detailed analysis of each benchmark

### Computational Efficiency
- GPU-hour costs
- Speed comparisons

### Ablation Studies
- Dataset size effects
- Temperature effects
- Alpha mixing effects
- LoRA rank effects

### Fidelity Analysis
- Correlation between teacher-student similarity and performance
- Evidence faithfulness results

## Discussion
### Answers to Research Questions
1. **Which method works best?**
2. **What are the trade-offs?**
3. **When does CoT help vs. hurt?**
4. **Does lower KL divergence help?**

## Limitations and Future Work

## Conclusion

## Appendix
- Hyperparameter details
- Computational logs
- Sample outputs
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Out of GPU Memory (OOM)
**Symptoms:** Training crashes with "CUDA out of memory" error

**Solutions:**
1. Reduce batch size in half
2. Reduce sequence length (1024 → 768)
3. Use gradient checkpointing (add to training config)
4. Reduce LoRA rank (32 → 16)
5. Use 8-bit quantization for base model

#### Issue: Training Loss Not Decreasing
**Symptoms:** Loss stays flat or increases

**Solutions:**
1. Check learning rate (try 1e-4 instead of 2e-4)
2. Verify data loading (print first batch to inspect)
3. Check for gradient clipping (add max_grad_norm=1.0)
4. Increase training steps
5. Verify labels are correct in dataset

#### Issue: Model Outputs Gibberish
**Symptoms:** Generated text is nonsensical

**Solutions:**
1. Check tokenizer is correctly loaded
2. Verify training data format is correct
3. Check if model merged with adapters properly
4. Try lower temperature during inference (0.7 → 0.3)
5. Ensure padding tokens are handled correctly

#### Issue: Slow Training Speed
**Symptoms:** Training taking much longer than estimated

**Solutions:**
1. Enable Flash Attention 2 if available
2. Increase batch size with gradient accumulation
3. Use bf16 instead of fp32 precision
4. Check if CPU bottlenecks (DataLoader num_workers)
5. Profile code to find bottlenecks

#### Issue: Poor Benchmark Performance
**Symptoms:** Models score worse than expected

**Solutions:**
1. Verify test set has no overlap with training
2. Check prompt formatting matches training format
3. Try different decoding strategies (greedy vs. sampling)
4. Ensure model is in eval mode (model.eval())
5. Verify benchmark implementation is correct

#### Issue: Inconsistent Results Across Runs
**Symptoms:** Running same experiment gives different results

**Solutions:**
1. Set random seeds (torch, numpy, random)
2. Use deterministic algorithms (torch.use_deterministic_algorithms(True))
3. Fix CUDA randomness (set CUBLAS environment variables)
4. Check for data shuffling randomness
5. Use same batch ordering

---

## Appendix A: Example Commands Summary

### Complete Workflow Commands

```bash
# Step 1: Setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
huggingface-cli login

# Step 2: Data Preparation
python scripts/data_preparation/download_datasets.py
python scripts/data_preparation/create_training_dataset.py
python scripts/data_preparation/create_perplexity_corpus.py
python scripts/data_preparation/create_fidelity_bench.py

# Step 3: Teacher Setup
python scripts/model_setup/download_teacher.py
python scripts/data_preparation/generate_teacher_responses.py
python scripts/data_preparation/generate_dpo_pairs.py

# Step 4: Student Training
python scripts/model_setup/download_student.py
python scripts/training/train_sft.py
python scripts/training/train_logit_kd.py --temperature 3 --alpha 0.5
python scripts/training/train_cot.py
python scripts/training/train_dpo.py

# Step 5: Evaluation
python scripts/evaluation/download_benchmarks.py
python scripts/evaluation/evaluate_medical_benchmarks.py --all-models
python scripts/evaluation/evaluate_perplexity.py
python scripts/evaluation/evaluate_fidelity.py

# Step 6: Analysis
python scripts/evaluation/run_ablations.py --ablation dataset_size
python scripts/analysis/fidelity_correlation.py
python scripts/analysis/create_plots.py
python scripts/analysis/generate_tables.py
```

---

## Appendix B: Estimated Timeline

**Total Duration:** 4-6 weeks

| Phase | Duration | GPU-Hours | Notes |
|-------|----------|-----------|-------|
| Setup & Data Prep | 3-5 days | 20-40 | Downloading, processing |
| Teacher Response Gen | 1-2 days | 20-40 | Can run overnight |
| SFT Training | 1-2 days | 25-40 | Single method |
| Logit KD Training | 2-4 days | 35-55 | Parameter sweep |
| CoT Training | 2-3 days | 30-50 | Longer sequences |
| DPO Training | 2-3 days | 30-45 | Final method |
| Evaluation | 2-3 days | 10-20 | All benchmarks |
| Ablations | 3-5 days | 40-80 | Multiple experiments |
| Analysis & Report | 3-5 days | 0 | No GPU needed |
| **Total** | **4-6 weeks** | **210-370** | |

**Critical Path:** Teacher response generation and training phases can overlap if you have multiple GPUs.

---

## Appendix C: Data Format Examples

### Training Data Format (JSONL)
```json
{"question": "A 45-year-old man presents with chest pain...", "options": ["A) Myocardial infarction", "B) Pneumonia", "C) GERD", "D) Anxiety"], "answer": "A", "source": "medqa"}
```

### Teacher Response Format
```json
{"question_id": "001", "question": "...", "teacher_response": "The correct answer is A) Myocardial infarction...", "teacher_logits": [...], "cot_reasoning": "First, we consider the patient's age and symptoms..."}
```

### DPO Pairs Format
```json
{"prompt": "Question: ...\nAnswer:", "chosen": "The answer is A because...", "rejected": "The answer is C which is incorrect..."}
```

---

## Appendix D: Key Metrics Definitions

**Accuracy:** Percentage of questions answered correctly

**Macro-F1:** Average F1 score across all classes (for multi-class problems)

**Perplexity:** Measure of how "surprised" the model is by text; lower = better language understanding
- Formula: exp(average negative log-likelihood)

**KL Divergence:** Measures how different student's predictions are from teacher's
- Lower = more similar to teacher

**Top-k Overlap:** Percentage of times student's top-k predictions include teacher's top choice

**Citation Coverage:** Percentage of answers that cite provided evidence passages

**Hallucination Rate:** Percentage of statements not supported by evidence

---

## Appendix E: Checkpoints and Validation

### What to Check After Each Phase

**After Data Prep:**
- [ ] Training set has 108k examples
- [ ] No duplicates in dataset
- [ ] No overlap between train/validation/test
- [ ] Questions are well-formatted
- [ ] All required fields present

**After Teacher Generation:**
- [ ] All training examples have teacher responses
- [ ] Responses are medically sensible
- [ ] CoT responses contain reasoning steps
- [ ] DPO pairs have clear chosen/rejected differences

**After Each Training:**
- [ ] Training loss decreased significantly
- [ ] Validation loss decreased (not overfitting)
- [ ] Generated samples look reasonable
- [ ] Model checkpoint saved successfully
- [ ] LoRA adapters can be loaded

**After Evaluation:**
- [ ] All benchmarks completed without errors
- [ ] Scores are within reasonable range (40-80% accuracy)
- [ ] Results are reproducible (re-run gives similar scores)
- [ ] All metrics calculated correctly

---

## Notes for Success

1. **Document Everything:** Keep detailed logs of all experiments, including failed runs
2. **Version Control:** Commit code regularly to Git
3. **Save Checkpoints:** Training can fail; save frequently
4. **Monitor Resources:** Watch GPU memory and disk space
5. **Start Small:** Test on 1,000 examples before full 120k
6. **Validate Early:** Check data quality before starting expensive training
7. **Compare to Baselines:** Always evaluate teacher model first
8. **Reproducibility:** Set random seeds and document all hyperparameters

---

**End of Procedure Guide**

For questions or issues, refer to the Troubleshooting Guide or consult the project team.
