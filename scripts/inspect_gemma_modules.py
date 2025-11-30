#!/usr/bin/env python3
import sys
import traceback

MODEL_NAME = "google/gemma-3-4b-it"

try:
    print(f"Inspecting model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM
    import torch

    # Attempt to load model (low_cpu_mem_usage to reduce peak memory)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print("\nModel loaded. Scanning module and parameter names for LoRA target candidates...")

    targets = ['q_proj','k_proj','v_proj','o_proj','wq','wk','wv','wo','gate','down_proj','up_proj','proj','attn','dense','mha']

    module_names = set(n for n, _ in model.named_modules())
    param_names = [n for n, _ in model.named_parameters()]

    for t in targets:
        hits = sorted([n for n in module_names if t in n])
        if hits:
            print(f"\n=== matches for '{t}': (showing up to 200) ===")
            for h in hits[:200]:
                print(h)

    # Also print a short sample of parameter names to help choose target modules
    print("\n--- Sample parameter names (first 400) ---")
    for i, n in enumerate(param_names):
        print(n)
        if i >= 399:
            break

    print('\nInspection complete.')

except Exception as e:
    print("ERROR during inspection:")
    traceback.print_exc()
    print('\nIf downloading the model failed, ensure you have network access and an HF token if the model requires authentication.')
    sys.exit(1)


