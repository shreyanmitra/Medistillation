# Data Directory

This directory contains all datasets used in the Med-Distillation project.

## Structure:
```
data/
├── raw/                      # Original downloaded datasets
├── processed/                # Prepared training/validation/test splits
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── benchmarks/               # Evaluation benchmarks
│   ├── medqa_test.jsonl
│   ├── medmcqa_val.jsonl
│   ├── pubmedqa_test.jsonl
│   └── pubhealth_test.jsonl
├── medppl_10k.jsonl         # Perplexity evaluation corpus
└── fidelitybench_med.jsonl  # Evidence faithfulness evaluation
```

## Data Preparation:

Run the data preparation script:
```bash
python src/DataLoader.py --prepare_all
```

This will download and prepare all required datasets.

**Note:** Large data files are gitignored. See `.gitignore` for details.
