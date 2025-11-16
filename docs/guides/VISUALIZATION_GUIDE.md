# Visualization Guide for Med-Distillation

This guide covers all visualization capabilities in the Med-Distillation project, including automatic plot generation during training and evaluation.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Automatic Plot Generation](#automatic-plot-generation)
4. [Plot Types](#plot-types)
5. [Manual Plotting](#manual-plotting)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Med-Distillation project includes comprehensive visualization capabilities using **Seaborn** and **Matplotlib**. All plots are automatically generated during training and evaluation, saving high-resolution PNG files to the results directory.

### Features:
- **Training curves**: Loss and learning rate over epochs
- **Benchmark comparison**: Student vs teacher accuracy with gap analysis
- **Fidelity metrics**: KL divergence, BLEU, ROUGE, top-k overlap
- **FidelityBench radar chart**: Multi-metric evidence faithfulness comparison
- **Ablation study plots**: Hyperparameter sensitivity analysis

---

## Installation

Install visualization dependencies:

```bash
pip install matplotlib>=3.5.0 seaborn>=0.12.0
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

### Verify Installation:

```python
import matplotlib
import seaborn as sns
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")
```

---

## Automatic Plot Generation

All plots are **automatically generated** at the appropriate stages of training and evaluation. No manual intervention needed!

### During Training:

**When:** After `save_training_history()` is called at the end of training

**Generated Plot:**
- `outputs/{run_name}/results/training_curves.png`
  - Left panel: Training and validation loss curves
  - Right panel: Learning rate schedule

**Example:**
```bash
python src/Trainer.py --method sft --num_epochs 3
# → Automatically creates training_curves.png
```

---

### After Evaluation:

**When:** `run_comprehensive_evaluation()` completes (automatically after training)

**Generated Plots:**

1. **`benchmark_comparison.png`** (if benchmarks evaluated)
   - Student vs teacher accuracy bars
   - Accuracy gap visualization

2. **`fidelity_metrics.png`** (always generated)
   - KL/JS divergence
   - Top-k overlap
   - BLEU/ROUGE scores
   - Exact match rate

3. **`fidelitybench_radar.png`** (if FidelityBench-Med evaluated)
   - Radar chart with 5 metrics:
     - Factual correctness
     - Answer relevancy
     - Overall faithfulness
     - Citation coverage
     - No hallucination rate

**Location:** All saved to `outputs/{run_name}/results/`

---

### During Ablation Studies:

**When:** After ablation study completes

**Generated Plot:**
- `outputs/ablation_{type}/ablation_plot.png`
  - 6-panel plot showing all metrics vs hyperparameter value
  - Includes: validation loss, training time, benchmark accuracy, KL divergence, BLEU, faithfulness

**Example:**
```bash
python src/Trainer.py \
    --run_ablation \
    --ablation_type temperature \
    --ablation_values "2.0,3.0,4.0,5.0" \
    --method logit_kd
# → Creates ablation_temperature/ablation_plot.png
```

---

## Plot Types

### 1. Training Curves (`plot_training_history()`)

**Purpose:** Visualize training dynamics and learning rate schedule

**Features:**
- Training and validation loss over epochs
- Learning rate schedule (log scale)
- Markers for each epoch
- Grid for readability

**Code (Manual):**
```python
trainer.plot_training_history(save_path="custom_training_curves.png")
```

**Interpreting the Plot:**
- **Converging losses:** Good learning progress
- **Diverging losses:** Overfitting (student memorizes training data)
- **Flat validation loss:** Model has stopped learning, can stop early
- **Learning rate decay:** Should smoothly decrease over epochs

---

### 2. Benchmark Comparison (`plot_benchmark_comparison()`)

**Purpose:** Compare student and teacher performance across medical QA benchmarks

**Features:**
- Side-by-side accuracy bars (student vs teacher)
- Accuracy gap visualization (how much knowledge was lost)
- Percentage labels on bars
- Supports 4 benchmarks: MedQA, MedMCQA, PubMedQA, PubHealth

**Code (Manual):**
```python
benchmark_results = trainer.evaluate_medical_benchmarks(benchmark_paths)
trainer.plot_benchmark_comparison(benchmark_results, save_path="benchmarks.png")
```

**Interpreting the Plot:**
- **Small gap (<5%):** Excellent distillation, student retained most knowledge
- **Medium gap (5-10%):** Acceptable knowledge transfer
- **Large gap (>10%):** Poor distillation, consider adjusting hyperparameters
- **Student > Teacher:** Rare but possible, student may generalize better on test set

---

### 3. Fidelity Metrics (`plot_fidelity_metrics()`)

**Purpose:** Measure how closely student mimics teacher's output distribution

**Features:**
- 4-panel layout:
  1. **Divergence metrics:** KL and JS divergence (lower is better)
  2. **Overlap metrics:** Top-1 and Top-5 token agreement (higher is better)
  3. **Text similarity:** BLEU and ROUGE scores (higher is better)
  4. **Exact match rate:** Percentage of identical outputs (higher is better)

**Code (Manual):**
```python
fidelity_results = trainer.evaluate_fidelity(num_samples=100)
trainer.plot_fidelity_metrics(fidelity_results, save_path="fidelity.png")
```

**Interpreting the Plot:**
- **Low KL divergence (<1.0):** Student distribution closely matches teacher
- **High top-k overlap (>80%):** Student and teacher agree on most likely tokens
- **High BLEU (>70%):** Generated text is very similar
- **High exact match (>40%):** Strong fidelity for short, deterministic outputs

---

### 4. FidelityBench Radar Chart (`plot_fidelitybench_radar()`)

**Purpose:** Comprehensive evidence faithfulness comparison using radar chart

**Features:**
- Polar plot with 5 metrics:
  - **Factual Correctness:** NLI entailment with ground truth
  - **Answer Relevancy:** Semantic similarity to question
  - **Overall Faithfulness:** Combined metric
  - **Citation Coverage:** Use of provided evidence
  - **No Hallucination:** Inverse of contradiction rate
- Overlapping student and teacher polygons for comparison

**Code (Manual):**
```python
fidelitybench_results = trainer.evaluate_fidelitybench_med("data/fidelitybench_med.jsonl")
trainer.plot_fidelitybench_radar(fidelitybench_results, save_path="radar.png")
```

**Interpreting the Plot:**
- **Larger polygon area:** Better overall performance
- **Student polygon inside teacher:** Knowledge gap
- **Similar shapes:** Student learned teacher's reasoning pattern
- **All metrics >80%:** Excellent evidence-based reasoning
- **High hallucination (low "No Hallucination"):** Model invents facts, needs improvement

---

### 5. Ablation Study Plot (`plot_ablation_results()`)

**Purpose:** Visualize hyperparameter sensitivity across multiple metrics

**Features:**
- 6-panel layout:
  1. **Validation loss vs parameter** (with best value marked by red star)
  2. **Training time vs parameter** (computational cost analysis)
  3. **Benchmark accuracy vs parameter** (all benchmarks overlaid)
  4. **KL divergence vs parameter** (fidelity analysis)
  5. **BLEU score vs parameter** (generation quality)
  6. **Faithfulness vs parameter** (FidelityBench overall score)

**Code (Automatic during ablation):**
```bash
python src/Trainer.py \
    --run_ablation \
    --ablation_type alpha \
    --ablation_values "0.3,0.5,0.7,0.9"
```

**Interpreting the Plot:**
- **U-shaped validation loss:** Optimal value is at the bottom (minimum)
- **Monotonic training time:** Higher complexity = longer training
- **Plateau in accuracy:** Diminishing returns beyond certain threshold
- **Inversely correlated KL and accuracy:** Trade-off between fidelity and task performance
- **Red star on panel 1:** Best hyperparameter value for validation loss

---

## Manual Plotting

If you want to generate plots manually (e.g., for custom analysis):

### Example 1: Plot Training History from Saved JSON

```python
import json
from src.Trainer import Trainer, TrainingConfig
import argparse

# Load training history
with open("outputs/sft_run1/results/training_history.json", 'r') as f:
    history = json.load(f)

# Create minimal config
args = argparse.Namespace(
    output_dir="outputs/sft_run1",
    # ... other required args ...
)
config = TrainingConfig(args)

# Create trainer instance (models not needed for plotting)
class MockTrainer:
    def __init__(self, config, history):
        self.config = config
        self.training_history = history

mock_trainer = MockTrainer(config, history)
Trainer.plot_training_history(mock_trainer, save_path="custom_plot.png")
```

### Example 2: Generate Benchmark Plot from Evaluation Results

```python
import json
from src.Trainer import Trainer

# Load benchmark results
with open("outputs/sft_run1/results/benchmark_results.json", 'r') as f:
    benchmark_results = json.load(f)

# Assuming you have a trainer instance
trainer.plot_benchmark_comparison(
    benchmark_results, 
    save_path="custom_benchmark_comparison.png"
)
```

### Example 3: Custom Ablation Visualization

```python
# Load ablation summary
with open("outputs/ablation_temperature/ablation_summary.json", 'r') as f:
    ablation_data = json.load(f)

# Plot with custom path
trainer.plot_ablation_results(
    ablation_data,
    save_path="custom_ablation_analysis.png"
)
```

---

## Customization

### Change Plot Style

Edit `src/Trainer.py` after imports:

```python
# Current default style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Alternative styles:
# sns.set_style("darkgrid")  # Dark background
# sns.set_style("white")      # No grid
# sns.set_palette("Set2")     # Different color scheme
# sns.set_palette("colorblind")  # Colorblind-friendly
```

### Adjust Figure Size

```python
plt.rcParams['figure.figsize'] = (16, 8)  # Wider plots
plt.rcParams['font.size'] = 12            # Larger fonts
```

### Change DPI (Resolution)

In any `plt.savefig()` call:

```python
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # Publication quality
```

### Export to Vector Format

```python
# Save as PDF (scalable, great for papers)
plt.savefig("plot.pdf", format='pdf', bbox_inches='tight')

# Save as SVG (web-friendly vector)
plt.savefig("plot.svg", format='svg', bbox_inches='tight')
```

---

## Troubleshooting

### Issue 1: "No module named 'matplotlib'"

**Solution:**
```bash
pip install matplotlib seaborn
```

### Issue 2: Plots not appearing on server

**Symptom:** `UserWarning: Matplotlib is currently using agg, which is a non-GUI backend`

**Explanation:** This is expected! The `Agg` backend is designed for headless servers (no display). Plots are still saved to files correctly.

**Verify plots are saved:**
```bash
ls outputs/*/results/*.png
```

### Issue 3: "RuntimeError: main thread is not in main loop"

**Cause:** Trying to use interactive backend (`TkAgg`) on a server without display

**Solution:** Already handled! The code sets `matplotlib.use('Agg')` before importing pyplot.

### Issue 4: Font warnings

**Symptom:** `findfont: Font family ['foo'] not found`

**Solution:** Matplotlib falls back to default fonts. Ignore or install specific fonts:
```bash
# Ubuntu/Debian
sudo apt-get install fonts-dejavu-core

# Windows: Fonts install automatically
```

### Issue 5: Low-resolution plots

**Solution:** Increase DPI in `plt.savefig()`:
```python
plt.savefig(save_path, dpi=300)  # Default
plt.savefig(save_path, dpi=600)  # Publication quality
```

### Issue 6: Plot generation fails silently

**Symptom:** No error but no plot file created

**Debug:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Look for warnings in trainer output
```

**Common causes:**
- Empty data (e.g., no training history)
- Missing evaluation results
- Disk space full
- Permission errors

---

## Advanced Usage

### Combine Multiple Runs in One Plot

```python
import matplotlib.pyplot as plt
import json

# Load multiple training histories
runs = {
    'SFT': json.load(open('outputs/sft/results/training_history.json')),
    'Logit-KD': json.load(open('outputs/logit_kd/results/training_history.json')),
    'AdaKD': json.load(open('outputs/adakd/results/training_history.json'))
}

# Plot comparison
plt.figure(figsize=(12, 6))
for name, history in runs.items():
    epochs = range(1, len(history['val_loss']) + 1)
    plt.plot(epochs, history['val_loss'], marker='o', label=name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison Across Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Create Summary Report with All Plots

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Create multi-page PDF report
with PdfPages('training_report.pdf') as pdf:
    # Page 1: Training curves
    trainer.plot_training_history()
    pdf.savefig()
    plt.close()
    
    # Page 2: Benchmarks
    trainer.plot_benchmark_comparison(benchmark_results)
    pdf.savefig()
    plt.close()
    
    # Page 3: Fidelity
    trainer.plot_fidelity_metrics(fidelity_results)
    pdf.savefig()
    plt.close()
    
    # Page 4: FidelityBench
    trainer.plot_fidelitybench_radar(fidelitybench_results)
    pdf.savefig()
    plt.close()

print("Report saved to training_report.pdf")
```

---

## Summary

### Quick Reference:

| Plot Type | File Name | When Generated | Key Insights |
|-----------|-----------|----------------|--------------|
| Training Curves | `training_curves.png` | End of training | Loss convergence, LR schedule |
| Benchmark Comparison | `benchmark_comparison.png` | After evaluation | Student vs teacher accuracy |
| Fidelity Metrics | `fidelity_metrics.png` | After evaluation | Distribution matching quality |
| FidelityBench Radar | `fidelitybench_radar.png` | After evaluation | Evidence faithfulness |
| Ablation Study | `ablation_plot.png` | After ablation | Hyperparameter sensitivity |

### Best Practices:

1. **Always check plots after training** to verify learning dynamics
2. **Compare student and teacher** on radar charts to identify gaps
3. **Use ablation plots** to optimize hyperparameters systematically
4. **Export to PDF** for publication-quality figures
5. **Keep DPI at 300+** for presentations and papers

---

## Questions?

If you encounter issues not covered here:

1. Check `outputs/{run_name}/trainer.log` for error messages
2. Verify dependencies: `pip list | grep -E "matplotlib|seaborn"`
3. Test manual plotting with small dataset
4. Open an issue with:
   - Error message
   - Python version
   - OS and environment (local/server/colab)
   - Minimal reproducible example

---

**Happy Visualizing! 📊**
