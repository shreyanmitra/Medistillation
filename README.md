# Medistillation
Repository for the Medistillation Experiment. (See docs/project_details_latex.tex for more info). 



Examples for usage 

download this guy https://github.com/astral-sh/uv

uv sync 

# Standard SFT
uv run .src/Trainer.py \
    --method sft \
    --output_dir ./outputs/sft_run1 \
    --num_epochs 3 \
    --batch_size 8

# Logit-KD with custom hyperparameters
uv run .src/Trainer.py \
    --method logit_kd \
    --alpha 0.5 \
    --temperature 3.0 \
    --output_dir ./outputs/logit_kd_a05_t3

# Chain-of-Thought with multiple rationales
uv run .src/Trainer.py \
    --method cot \
    --num_rationales 3 \
    --sampling_temperature 0.7 \
    --output_dir ./outputs/cot_3rationales

# DPO
uv run .src/Trainer.py \
    --method dpo \
    --beta 0.1 \
    --output_dir ./outputs/dpo_b01