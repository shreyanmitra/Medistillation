# Med-DistillMix Dataset Strategy

## Overview

This document describes the comprehensive dataset strategy for the medical knowledge distillation experiments. The strategy has been updated to **use ALL available data** from four high-quality medical sources, rather than sampling limited subsets.

---

## Dataset Composition

### Training Corpus Sources

| Dataset | HuggingFace Path | Approximate Size | Description |
|---------|-----------------|------------------|-------------|
| **MedMCQA** | `openlifescienceai/medmcqa` | ~182,000 | Indian medical entrance exam questions (multiple choice) |
| **MedQA** | `bigbio/med_qa` | ~10,000 | USMLE-style clinical questions (multiple choice) |
| **PubMedQA** | `qiaojin/PubMedQA` (artificial) | ~211,000 | Biomedical research questions from PubMed abstracts (yes/no/maybe) |
| **PubHealth** | `bigbio/pubhealth` | ~11,000 | Health claim verification (true/false/mixture/unproven) |
| **TOTAL** | - | **~414,000** | Combined training corpus |

### Data Splits

All datasets are combined, shuffled, and split:

- **Training**: 90% (~372,600 examples)
- **Validation**: 5% (~20,700 examples)
- **Holdout**: 5% (~20,700 examples)

**Files saved to**: `data/processed/`
- `train.jsonl`
- `validation.jsonl`
- `holdout.jsonl`

---

## Unified Data Format

All four datasets are automatically converted to a **unified long-form answer format**:

### Format Conversions

#### 1. MedMCQA (Multiple Choice)
**Original**: 4 options (opa, opb, opc, opd), correct option index (cop)

**Converted to**:
```
Question: [question text]

Options:
A) [option 1]
B) [option 2]
C) [option 3]
D) [option 4]

Answer: [Full text of correct option]
```

#### 2. MedQA (Multiple Choice)
**Original**: Options dict {'A': '...', 'B': '...', ...}, answer letter

**Converted to**:
```
Question: [question text]

Options:
A) [option 1]
B) [option 2]
C) [option 3]
D) [option 4]

Answer: [Full text of correct option]
```

#### 3. PubMedQA (Yes/No/Maybe)
**Original**: Question with PubMed context, yes/no/maybe answer

**Converted to**:
```
Question: [question text]

Answer: [yes/no/maybe with optional explanation]
```

#### 4. PubHealth (Claim Verification)
**Original**: Health claim, label (true/false/mixture/unproven), explanation

**Converted to**:
```
Question: Is the following health claim true or false?

Claim: [health claim]

Answer: [True/False/Mixture/Unproven. Explanation...]
```

### Why Long-Form Answers?

1. **Better for distillation**: Captures teacher's reasoning process, not just letter selection
2. **Natural language**: Models learn to explain, not just classify
3. **Fidelity metrics**: BLEU/ROUGE scores can measure generation similarity
4. **Flexible**: Works for both MCQ and open-ended questions

---

## Evaluation Benchmarks

Separate held-out test sets are downloaded for unbiased evaluation:

| Benchmark | HuggingFace Path | Split | Size | Purpose |
|-----------|-----------------|-------|------|---------|
| MedQA | `bigbio/med_qa` | test | ~1,200 | Clinical reasoning |
| MedMCQA | `openlifescienceai/medmcqa` | validation | ~4,100 | Medical knowledge |
| PubMedQA | `qiaojin/PubMedQA` (labeled) | test | ~500 | Biomedical QA |
| PubHealth | `bigbio/pubhealth` | test | ~2,300 | Claim verification |

**Files saved to**: `data/benchmarks/`
- `medqa_test.jsonl`
- `medmcqa_val.jsonl`
- `pubmedqa_test.jsonl`
- `pubhealth_test.jsonl`

---

## Perplexity Corpus

**MedPPL-10k**: 10,000 PubMed abstracts for language modeling evaluation

- **Source**: Sampled from PubMedQA context fields (both labeled and artificial subsets)
- **Purpose**: Measure domain-specific language modeling quality
- **Completely separate**: From both training data and QA benchmarks
- **File**: `data/medppl_10k.jsonl`

---

## Implementation Details

### UniversalMedicalDataset Class

The `UniversalMedicalDataset` class in `DataLoader.py` provides:

1. **Automatic format detection**: Identifies dataset type from field structure
2. **Unified conversion**: Standardizes all formats to long-form answers
3. **Metadata preservation**: Keeps original format info for analysis
4. **Source tracking**: Records which dataset each example came from

**Supported formats**:
- `medmcqa` - Detected by: `opa`, `opb` fields
- `medqa` - Detected by: `options` dict
- `pubmedqa` - Detected by: `long_answer` or `context.contexts`
- `pubhealth` - Detected by: `claim`, `explanation` fields

### Dataset Preparation Commands

```bash
# Prepare ALL datasets at once (recommended)
python src/DataLoader.py --prepare_all

# Or prepare individually:
python src/DataLoader.py --prepare_training --output_dir data/processed
python src/DataLoader.py --download_benchmarks --benchmark_dir data/benchmarks
python src/DataLoader.py --create_perplexity_corpus --perplexity_path data/medppl_10k.jsonl
```

### Custom Sample Sizes (Optional)

If you want to limit dataset sizes for faster experimentation:

```python
from src.DataLoader import prepare_med_distillmix_dataset

prepare_med_distillmix_dataset(
    output_dir='data/processed',
    num_medmcqa=50000,    # Limit to 50k
    num_medqa=10000,      # Limit to 10k
    num_pubmedqa=20000,   # Limit to 20k
    num_pubhealth=5000,   # Limit to 5k
    seed=42
)
```

Default (all None) = use ALL available data.

---

## Rationale for This Strategy

### ✅ Why Use ALL Data?

1. **Maximize knowledge**: More diverse training examples improve generalization
2. **Real-world scale**: ~400k examples is realistic for modern LLM training
3. **Avoid arbitrary limits**: No reason to artificially restrict high-quality data
4. **Better distillation**: Larger corpus helps student learn teacher's full capability

### ✅ Why Four Datasets?

1. **MedMCQA**: Indian medical exams, different perspective than US-centric datasets
2. **MedQA**: USMLE-style, gold standard for clinical reasoning
3. **PubMedQA**: Research-oriented, evidence-based reasoning from literature
4. **PubHealth**: Health misinformation detection, real-world claims

**Together**: Comprehensive coverage of clinical, research, and public health domains

### ✅ Why Long-Form Answers?

- **Distillation focus**: We care about teacher-student agreement, not just accuracy
- **Reasoning capture**: Full explanations reveal how teacher thinks
- **Metric compatibility**: BLEU/ROUGE measure generation similarity
- **Flexible evaluation**: Can measure both accuracy (ground truth) and fidelity (teacher agreement)

### ✅ Why Separate Benchmarks?

- **No data leakage**: Training on MedMCQA train ≠ evaluating on MedMCQA validation
- **Fair comparison**: Industry-standard test sets for reproducible results
- **Multiple domains**: Each benchmark tests different medical competencies

---

## Expected Outcomes

After running `python src/DataLoader.py --prepare_all`, you will have:

### Training Data
- `data/processed/train.jsonl` (~372k examples, 90%)
- `data/processed/validation.jsonl` (~21k examples, 5%)
- `data/processed/holdout.jsonl` (~21k examples, 5%)

### Evaluation Benchmarks
- `data/benchmarks/medqa_test.jsonl`
- `data/benchmarks/medmcqa_val.jsonl`
- `data/benchmarks/pubmedqa_test.jsonl`
- `data/benchmarks/pubhealth_test.jsonl`

### Perplexity Corpus
- `data/medppl_10k.jsonl` (10k PubMed abstracts)

### Total Disk Usage
- Approximately 2-3 GB for all JSONL files combined

---

## Next Steps

1. **Run dataset preparation**:
   ```bash
   python src/DataLoader.py --prepare_all
   ```

2. **Verify data quality**:
   ```python
   from src.DataLoader import UniversalMedicalDataset
   from transformers import AutoTokenizer
   
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
   dataset = UniversalMedicalDataset(
       data_path='data/processed/train.jsonl',
       tokenizer=tokenizer
   )
   
   print(f"Total examples: {len(dataset)}")
   print(f"Source distribution: {dataset._get_source_distribution()}")
   print(f"\nSample example:")
   print(dataset[0])
   ```

3. **Start training**:
   ```bash
   python src/Trainer.py \
       --teacher_model epfl-llm/meditron-70b \
       --student_model Qwen/Qwen2-1.5B \
       --distillation_method sft \
       --train_data data/processed/train.jsonl \
       --val_data data/processed/validation.jsonl
   ```

---

## Changelog

### v2.0 (Current)
- ✅ Use **ALL available data** from each source (~414k total)
- ✅ Added **PubHealth dataset** for health claim verification
- ✅ Default parameters: `num_*=None` (use all)
- ✅ Updated documentation to reflect full dataset strategy

### v1.0 (Previous)
- ❌ Sampled limited subsets (50k + 10k + 20k = 80k total)
- ❌ Missing PubHealth dataset
- ❌ Arbitrary sample size limits

---

**Last Updated**: November 15, 2025
