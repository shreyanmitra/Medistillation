#!/usr/bin/env python3
# -*- coding: utf-8 -*-

TEACHER_MODEL = "epfl-llm/meditron-70b"
STUDENT_MODEL = "meta-llama/Llama-2-7b-hf"  
BASELINE_MODEL = "epfl-llm/meditron-7b"

NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-4 

ENABLE_CPU_OFFLOAD = True  # Disable CPU offload for better GPU utilization (RTX 5090 has 32GB VRAM)
ALIGN_VOCABULARIES = True

METHODS_TO_RUN = [
    "sft",        # Baseline: Supervised Fine-Tuning
    "logit_kd",   # Logit-based Knowledge Distillation
    "spin",       # SPIN
]

OUTPUT_BASE = "outputs"

print("="*80)
print("EXPERIMENT CONFIGURATION: DISTILLATION EFFICIENCY STUDY")
print("="*80)
print(f"🎓 Teacher Model:  {TEACHER_MODEL}")
print(f"👨‍🎓 Student Model:  {STUDENT_MODEL}")
print(f"📊 Baseline Model: {BASELINE_MODEL} (comparison target)")
print(f"\n📚 Training Settings:")
print(f"  • Epochs: {NUM_EPOCHS}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Gradient Accumulation: {GRADIENT_ACCUMULATION}")
print(f"  • Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  • Learning Rate: {LEARNING_RATE}")
print(f"  • GPU: RTX 5090 (32GB VRAM)")
print(f"\n🔬 Methods: {', '.join(METHODS_TO_RUN)}")
print(f"\n🎯 Research Goal:")
print(f"  Demonstrate distillation matches direct fine-tuning at fraction of cost")
print("="*80)


from huggingface_hub import login
import os

print("="*80)
print("🔐 HUGGINGFACE AUTHENTICATION")
print("="*80)

hf_token = ""
login(token=hf_token)

import time
from datetime import datetime
METHOD_CONFIGS = {
    'sft': {
        'description': 'Supervised Fine-Tuning (baseline)',
        'params': {}  # No extra parameters needed
    },
    'logit_kd': {
        'description': 'Knowledge Distillation with soft targets',
        'params': {
            '--alpha': 0.5,           # Distillation weight (balance hard/soft)
            '--temperature': 3.0       # Temperature for softening distributions
        }
    },
    'spin': {
        'description': 'Self-Play Fine-Tuning with contrastive learning',
        'params': {
            '--beta': 0.1,       # Contrastive loss weight
            '--temperature': 1.0  # Temperature for SPIN logits
        }
    },
    'adakd': {
        'description': 'Adaptive Knowledge Distillation',
        'params': {
            '--alpha': 0.5,           # Initial distillation weight
            '--base_temperature': 3.0, # Base temperature for soft targets
            '--min_temperature': 1.0,  # Minimum adaptive temperature
            '--max_temperature': 5.0   # Maximum adaptive temperature
        }
    },
    'cot': {
        'description': 'Chain-of-Thought distillation',
        'params': {
            '--cot_prompt': "Let's think step by step:",  # CoT trigger
            '--num_rationales': 3,     # Number of reasoning paths
            '--sampling_temperature': 0.7  # Temperature for diverse generation
        }
    },
    # Advanced RL methods (optional - longer training time)
    'on_policy': {
        'description': 'On-Policy RL Distillation (REINFORCE)',
        'params': {
            '--gamma': 0.99,          # Discount factor
            '--temperature': 1.0,      # Sampling temperature
            '--entropy_coef': 0.01     # Entropy regularization coefficient
        }
    },
    'ppo': {
        'description': 'PPO Distillation',
        'params': {
            '--epsilon': 0.2,         # PPO clipping parameter
            '--gamma': 0.99,          # Discount factor
            '--temperature': 1.0,      # Sampling temperature
            '--value_coef': 0.5,       # Value loss coefficient
            '--entropy_coef': 0.01     # Entropy coefficient
        }
    },
    'bond': {
        'description': 'Best-of-N Distillation',
        'params': {
            '--num_samples': 16,      # Number of candidate responses to sample
            '--temperature': 1.0       # Sampling temperature
        }
    }
}

# Training results tracker
training_results = {}

# Progress tracking function
def show_training_progress(method, epoch, total_epochs, loss, elapsed_time):
    """Display real-time training progress"""
    progress_pct = (epoch / total_epochs) * 100
    bar_length = 40
    filled = int(bar_length * epoch / total_epochs)
    bar = '█' * filled + '░' * (bar_length - filled)

    eta = (elapsed_time / epoch) * (total_epochs - epoch) if epoch > 0 else 0

    print(f"""🚀 Training: {method.upper()}
        Progress: [{bar}] {progress_pct:.1f}%
        Epoch: {epoch}/{total_epochs} | Loss: {loss:.4f}
        Elapsed: {elapsed_time/60:.1f}m | ETA: {eta/60:.1f}m
        """)


# Main training loop
print("="*80)
print("🚀 STARTING MEDICAL LLM DISTILLATION EXPERIMENTS")
print("="*80)
print(f"\n📋 Methods to run: {', '.join([m.upper() for m in METHODS_TO_RUN])}")
print(f"🎯 Teacher: {TEACHER_MODEL}")
print(f"🤖 Student: {STUDENT_MODEL}")
print(f"📊 Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print(f"💾 Data: data/processed/train.jsonl")
print("="*80)

for method_idx, method in enumerate(METHODS_TO_RUN, 1):
    method_start_time = time.time()

    # Validate method exists
    if method not in METHOD_CONFIGS:
        print(f"\n⚠️  WARNING: Unknown method '{method}' - skipping")
        print(f"   Available methods: {', '.join(METHOD_CONFIGS.keys())}")
        continue

    method_info = METHOD_CONFIGS[method]

    print("\n" + "="*80)
    print(f"🔬 EXPERIMENT {method_idx}/{len(METHODS_TO_RUN)}: {method.upper()}")
    print("="*80)
    print(f"📖 Description: {method_info['description']}")

    output_dir = f"{OUTPUT_BASE}/{method}"

    # Build base command
    # Note: With batch_size=512, ensure sufficient GPU memory (RTX 5090 32GB should handle this)
    # Consider reducing max_length if you encounter OOM errors
    cmd = f"""python src/Trainer.py \\
        --teacher_model {TEACHER_MODEL} \\
        --student_model {STUDENT_MODEL} \\
        --method {method} \\
        --train_data data/processed/train.jsonl \\
        --val_data data/processed/validation.jsonl \\
        --num_epochs {NUM_EPOCHS} \\
        --batch_size {BATCH_SIZE} \\
        --gradient_accumulation_steps {GRADIENT_ACCUMULATION} \\
        --learning_rate {LEARNING_RATE} \\
        --max_length 1024 \\
        --num_workers 16 \\
        --output_dir {output_dir}"""

    if ENABLE_CPU_OFFLOAD:
        cmd += " \\\n        --enable_cpu_offload"

    if ALIGN_VOCABULARIES:
        cmd += " \\\n        --align_vocabularies"

    # Add method-specific parameters
    method_params = method_info['params']
    if method_params:
        print(f"\n⚙️  Method-specific parameters:")
        for param_name, param_value in method_params.items():
            # Format based on value type
            if isinstance(param_value, bool):
                if param_value:
                    cmd += f" \\\n        {param_name}"
                    print(f"   • {param_name}: enabled")
            elif isinstance(param_value, str):
                if param_value:  # Only add non-empty strings
                    cmd += f" \\\n        {param_name} \"{param_value}\""
                    print(f"   • {param_name}: \"{param_value}\"")
            elif param_value is not None:
                cmd += f" \\\n        {param_name} {param_value}"
                print(f"   • {param_name}: {param_value}")
    else:
        print(f"\n⚙️  No additional parameters (baseline method)")

    # Special messages for specific methods
    if method == 'spin':
        print(f"\n🎯 SPIN Configuration:")
        print(f"   • Iteration: 0 (student vs teacher)")
        print(f"   • Opponent: Teacher model")
        print(f"   • Learning: Contrastive (prefer student over teacher)")
    elif method == 'cot':
        print(f"\n🧠 CoT Configuration:")
        print(f"   • Generating {method_params.get('--num_rationales', 1)} reasoning paths")
        print(f"   • Temperature: {method_params.get('--sampling_temperature', 0.7)}")
    elif method == 'adakd':
        print(f"\n🔄 AdaKD Configuration:")
        print(f"   • Adaptive temperature scheduling")
        print(f"   • Temperature range: {method_params.get('--min_temperature', 1.0)}-{method_params.get('--max_temperature', 5.0)}")
        print(f"   • Dynamic weight adjustment based on loss")
    elif method == 'on_policy':
        print(f"\n🎮 On-Policy RL Configuration:")
        print(f"   • REINFORCE algorithm")
        print(f"   • Discount factor: γ={method_params.get('--gamma', 0.99)}")
    elif method == 'ppo':
        print(f"\n🎭 PPO Configuration:")
        print(f"   • Proximal Policy Optimization")
        print(f"   • Clip range: ε={method_params.get('--epsilon', 0.2)}")
    elif method == 'bond':
        print(f"\n🏆 Best-of-N Configuration:")
        print(f"   • Sampling {method_params.get('--num_samples', 16)} candidate responses")
        print(f"   • Selecting best based on teacher scoring")

    print(f"\n⏱️  Starting training...")
    print(f"💾 Output directory: {output_dir}")
    print(f"📊 Progress will be displayed below:")
    print(f"💡 Tip: Check GPU utilization with: !nvidia-smi")
    print("-" * 80)

    import subprocess
    import sys
    
    # Run training with real-time output streaming to show progress bars
    print("\n📊 Training Progress (real-time):")
    print("   You will see batch-by-batch progress below (from Trainer.py)")
    print("-" * 80)
    
    # Use Popen to stream output in real-time so tqdm progress bars are visible
    # Note: tqdm works best when output is a TTY, so we'll stream directly
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered for real-time output
        universal_newlines=True
    )
    
    # Stream output in real-time to show progress bars
    output_lines = []
    try:
        # Read line by line and print immediately
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output, end='', flush=True)  # Print immediately
                output_lines.append(output)
    except KeyboardInterrupt:
        process.terminate()
        print("\n⚠️  Training interrupted by user")
        raise
    
    # Wait for process to complete and get return code
    return_code = process.poll()
    
    # Store full output for error reporting
    full_output = ''.join(output_lines)
    
    # Check if training failed
    if return_code != 0:
        print(f"\n❌ ERROR: Training failed with return code {return_code}")
        if full_output:
            # Show last 50 lines of output for debugging
            error_lines = full_output.split('\n')[-50:]
            print("\nLast 50 lines of output:")
            print('\n'.join(error_lines))
        print("Continuing with next method...\n")

    # Calculate elapsed time
    elapsed_time = time.time() - method_start_time
    elapsed_mins = elapsed_time / 60
    elapsed_hrs = elapsed_mins / 60

    training_results[method] = {
        'output_dir': output_dir,
        'training_time_hours': elapsed_hrs,
        'training_time_minutes': elapsed_mins,
        'training_time_seconds': elapsed_time,
        'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': method_info['description'],
        'parameters': method_params
    }

    print("-" * 80)
    print(f"\n✅ {method.upper()} TRAINING COMPLETE!")
    print(f"⏱️  Time: {elapsed_mins:.1f} minutes ({elapsed_hrs:.2f} hours)")
    print(f"📁 Results saved to: {output_dir}/results/")
    print("="*80)

    # Brief pause between methods to avoid resource conflicts
    if method_idx < len(METHODS_TO_RUN):
        print("\n⏸️  Pausing 10 seconds before next method...\n")
        time.sleep(10)

# Total experiment summary
print("\n" + "="*80)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("="*80)

print(f"\n📊 Experiment Summary:")
print(f"{'Method':<15} {'Time (min)':<12} {'Time (hr)':<10} {'Description':<50}")
print("-" * 90)
for method, metrics in training_results.items():
    print(f"{method.upper():<15} {metrics['training_time_minutes']:>10.1f}  {metrics['training_time_hours']:>8.2f}  {metrics['description']:<50}")

total_time_mins = sum(r['training_time_minutes'] for r in training_results.values())
total_time_hrs = total_time_mins / 60
print("-" * 90)
print(f"{'TOTAL':<15} {total_time_mins:>10.1f}  {total_time_hrs:>8.2f}")
print(f"\n⏱️  Total Experiment Time: {total_time_mins:.1f} minutes ({total_time_hrs:.2f} hours)")

print(f"\n💾 Output Structure:")
print("   outputs/")
for method in METHODS_TO_RUN:
    if method in METHOD_CONFIGS:  # Only show valid methods
        print(f"   ├── {method}_colab/")
        print(f"   │   ├── results/               # Metrics, plots, evaluations")
        print(f"   │   │   ├── training_curves.png")
        print(f"   │   │   ├── benchmark_comparison.png")
        print(f"   │   │   ├── fidelity_metrics.png")
        print(f"   │   │   └── comprehensive_evaluation.json")
        print(f"   │   └── checkpoints/           # Model checkpoints")

print(f"\n💡 Next Steps:")
print("   1. Review visualizations in outputs/*/results/")
print("   2. Compare methods using comprehensive_evaluation.json")
print("   3. Run interactive analysis in next cells")
print("   4. Save best model to Google Drive")

print("\n" + "="*80)


import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 80)
print("📊 VISUALIZATION AND COMPARISON")
print("=" * 80)

# Dictionary to store all evaluation results
all_results = {}
training_history = {}

for method in METHODS_TO_RUN:
    output_dir = training_results[method]['output_dir']
    results_dir = f"{output_dir}/results"

    # Load numerical metrics and training history
    if os.path.exists(f"{results_dir}/comprehensive_evaluation.json"):
        with open(f"{results_dir}/comprehensive_evaluation.json", 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            all_results[method] = metrics

        print("\n📋 Summary Metrics:")
        print(f"   Training Time: {training_results[method]['training_time_hours']:.2f} hours")

        if 'benchmarks' in metrics:
            print("\n   Benchmark Scores:")
            for bench, scores in metrics['benchmarks'].items():
                if isinstance(scores, dict) and 'student_accuracy' in scores:
                    print(f"      {bench}: {scores['student_accuracy']*100:.2f}%")

        if 'fidelity' in metrics:
            print("\n   Fidelity Metrics:")
            for metric, value in metrics['fidelity'].items():
                if isinstance(value, (int, float)):
                    print(f"      {metric}: {value:.4f}")
    else:
        print("   ⚠️ Comprehensive evaluation results not found")

    # Load training history for interactive plots
    if os.path.exists(f"{results_dir}/training_history.json"):
        with open(f"{results_dir}/training_history.json", 'r', encoding='utf-8') as f:
            training_history[method] = json.load(f)

# Create comparison table
if all_results:
    print("\n" + "=" * 80)
    print("📋 METHOD COMPARISON TABLE")
    print("=" * 80)

    comparison_data = []
    for method, results in all_results.items():
        row = {
            'Method': method.upper(),
            'Training Time (hrs)': f"{training_results[method]['training_time_hours']:.2f}"
        }

        # Add benchmark scores
        if 'benchmarks' in results:
            for bench, scores in results['benchmarks'].items():
                if isinstance(scores, dict) and 'student_accuracy' in scores:
                    row[f"{bench} Acc"] = f"{scores['student_accuracy']*100:.2f}%"

        # Add key fidelity metrics
        if 'fidelity' in results:
            if 'kl_divergence' in results['fidelity']:
                row['KL Divergence'] = f"{results['fidelity']['kl_divergence']:.4f}"
            if 'bleu' in results['fidelity']:
                row['BLEU'] = f"{results['fidelity']['bleu']:.4f}"

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv(f"{OUTPUT_BASE}/method_comparison.csv", index=False)
    print(f"\n✅ Comparison table saved to: {OUTPUT_BASE}/method_comparison.csv")
else:
    print("\n⚠️ No results to compare")

print("\n" + "=" * 80)
print("✅ STATIC VISUALIZATION COMPLETE")
print("=" * 80)


import json

print("=" * 80)
print("🏆 BASELINE COMPARISON: DISTILLATION EFFICIENCY")
print("=" * 80)

# Check if baseline comparison results exist
baseline_results_found = False

for method in METHODS_TO_RUN:
    output_dir = training_results[method]['output_dir']
    comparison_path = f"{output_dir}/results/baseline_comparison.json"

    if os.path.exists(comparison_path):
        baseline_results_found = True

        with open(comparison_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")

        if 'summary' in results:
            summary = results['summary']

            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"  Your Distilled Model:     {summary['avg_your_accuracy']*100:.2f}%")
            print(f"  Meditron-7B Baseline:     {summary['avg_baseline_accuracy']*100:.2f}%")
            print(f"  Accuracy Gap:             {summary['avg_accuracy_gap']*100:.2f}%")
            print(f"  Performance Retained:     {summary['avg_retention_pct']:.1f}%")

            print(f"\n💰 EFFICIENCY GAINS:")
            gains = summary['efficiency_gains']
            print(f"  Training Data Reduction:  {gains['training_data_reduction']}x")
            print(f"  Training Time Speedup:    {gains['training_time_speedup']}x")
            print(f"  Cost Reduction:           {gains['cost_reduction']}x")
            print(f"\n  Your data used:    {gains['data_used_tokens']}")
            print(f"  Baseline data:     {gains['baseline_data_tokens']}")

            # Interpretation
            retention = summary['avg_retention_pct']
            print(f"\n📝 INTERPRETATION:")
            if retention >= 95:
                print(f"  ✅ EXCELLENT: Achieved {retention:.1f}% of baseline!")
                print(f"     Distillation successfully matches direct fine-tuning")
                print(f"     with massive efficiency gains ({gains['training_data_reduction']}x less data, {gains['cost_reduction']}x cheaper)!")
            elif retention >= 90:
                print(f"  ✅ GOOD: Achieved {retention:.1f}% of baseline")
                print(f"     Strong cost-benefit tradeoff - small accuracy gap")
                print(f"     for {gains['training_data_reduction']}x data reduction")
            elif retention >= 85:
                print(f"  ⚠️  ACCEPTABLE: {retention:.1f}% of baseline")
                print(f"     Consider hyperparameter tuning to close the gap")
            else:
                print(f"  ❌ NEEDS WORK: Only {retention:.1f}% of baseline")
                print(f"     Investigate training dynamics, data quality, or method choice")

            # Per-benchmark breakdown
            if 'benchmarks' in results:
                print(f"\n📋 PER-BENCHMARK BREAKDOWN:")
                print(f"  {'Benchmark':<15} {'Your Acc':<12} {'Baseline':<12} {'Gap':<10} {'Retained':<10}")
                print(f"  {'-'*70}")
                for bench, metrics in results['benchmarks'].items():
                    your_acc = metrics['your_accuracy'] * 100
                    base_acc = metrics['baseline_accuracy'] * 100
                    gap = metrics['accuracy_gap'] * 100
                    retained = metrics['accuracy_retained_pct']

                    status = "✅" if retained >= 95 else "⚠️" if retained >= 90 else "❌"
                    print(f"  {status} {bench.upper():<13} {your_acc:>10.2f}%  {base_acc:>10.2f}%  {gap:>8.2f}%  {retained:>8.1f}%")


        print(f"\n{'='*80}")

if not baseline_results_found:
    print("\n⚠️  Baseline comparison results not found.")
    print("   This is automatically generated during comprehensive evaluation.")
    print("   Make sure the evaluation completed successfully!")

print("\n" + "=" * 80)
print("✅ BASELINE COMPARISON ANALYSIS COMPLETE")
print("=" * 80)

"""## 🎨 Interactive Visualizations (Plotly)

Create interactive charts for better exploration and comparison.
"""

print("=" * 80)
print("🎨 CREATING INTERACTIVE PLOTLY VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# 1. Interactive Training Loss Comparison
# ============================================================================
if training_history:
    print("\n📈 Creating interactive training loss comparison...")

    fig = go.Figure()

    for method, history in training_history.items():
        epochs = history.get('epochs', [])
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])

        # Add training loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_losses,
            mode='lines+markers',
            name=f'{method.upper()} - Train',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{fullData.name}</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))

        # Add validation loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode='lines+markers',
            name=f'{method.upper()} - Val',
            line=dict(width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>%{fullData.name}</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title='Interactive Training Loss Comparison Across Methods',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='closest',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    fig.show()
    print("✅ Training loss comparison complete")

# ============================================================================
# 2. Interactive Benchmark Comparison Bar Chart
# ============================================================================
if all_results:
    print("\n📊 Creating interactive benchmark comparison...")

    benchmark_data = []
    for method, results in all_results.items():
        if 'benchmarks' in results:
            for bench, scores in results['benchmarks'].items():
                if isinstance(scores, dict) and 'student_accuracy' in scores:
                    benchmark_data.append({
                        'Method': method.upper(),
                        'Benchmark': bench,
                        'Student': scores['student_accuracy'] * 100,
                        'Teacher': scores.get('teacher_accuracy', 0) * 100,
                        'Gap': (scores.get('teacher_accuracy', 0) - scores['student_accuracy']) * 100
                    })

    if benchmark_data:
        df = pd.DataFrame(benchmark_data)

        # Create grouped bar chart
        fig = go.Figure()

        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]

            fig.add_trace(go.Bar(
                name=f'{method} Student',
                x=method_data['Benchmark'],
                y=method_data['Student'],
                text=method_data['Student'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            title='Interactive Benchmark Performance Comparison',
            xaxis_title='Benchmark',
            yaxis_title='Accuracy (%)',
            barmode='group',
            template='plotly_white',
            height=500,
            hovermode='closest',
            yaxis=dict(range=[0, 100])
        )

        fig.show()
        print("✅ Benchmark comparison complete")

# ============================================================================
# 3. Interactive Radar Chart for FidelityBench Metrics
# ============================================================================
if all_results:
    print("\n🔍 Creating interactive FidelityBench radar chart...")

    fidelity_metrics = ['semantic_similarity', 'entailment_score', 'rouge_l', 'bleu', 'exact_match']
    radar_data = []

    for method, results in all_results.items():
        if 'fidelity' in results:
            values = []
            for metric in fidelity_metrics:
                # Normalize to 0-100 scale
                value = results['fidelity'].get(metric, 0)
                if metric in ['semantic_similarity', 'entailment_score']:
                    value = value * 100  # Already 0-1, scale to 0-100
                elif metric in ['rouge_l', 'bleu']:
                    value = value * 100  # Already 0-1, scale to 0-100
                elif metric == 'exact_match':
                    value = value * 100  # Already 0-1, scale to 0-100
                values.append(value)

            radar_data.append({
                'method': method.upper(),
                'values': values
            })

    if radar_data:
        fig = go.Figure()

        for data in radar_data:
            fig.add_trace(go.Scatterpolar(
                r=data['values'],
                theta=fidelity_metrics,
                fill='toself',
                name=data['method'],
                hovertemplate='<b>%{fullData.name}</b><br>%{theta}: %{r:.2f}<extra></extra>'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title='Interactive FidelityBench Metrics Radar Chart',
            template='plotly_white',
            height=600,
            showlegend=True
        )

        fig.show()
        print("✅ FidelityBench radar chart complete")

# ============================================================================
# 4. Interactive Heatmap - Method vs Metric
# ============================================================================
if all_results:
    print("\n🔥 Creating interactive performance heatmap...")

    # Prepare data for heatmap
    methods = []
    metrics_list = []
    values = []

    for method, results in all_results.items():
        methods.append(method.upper())
        method_metrics = []

        # Collect all metrics
        if 'benchmarks' in results:
            for bench, scores in results['benchmarks'].items():
                if isinstance(scores, dict) and 'student_accuracy' in scores:
                    metric_name = f"{bench}_acc"
                    if metric_name not in metrics_list:
                        metrics_list.append(metric_name)
                    method_metrics.append(scores['student_accuracy'] * 100)

        if 'fidelity' in results:
            for metric, value in results['fidelity'].items():
                if isinstance(value, (int, float)) and metric not in ['kl_divergence', 'js_divergence']:
                    if metric not in metrics_list:
                        metrics_list.append(metric)
                    method_metrics.append(value * 100 if value <= 1 else value)

        values.append(method_metrics)

    if values and metrics_list:
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=metrics_list,
            y=methods,
            colorscale='RdYlGn',
            text=[[f'{val:.1f}' for val in row] for row in values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>',
            colorbar=dict(title="Score")
        ))

        fig.update_layout(
            title='Interactive Performance Heatmap (Higher is Better)',
            xaxis_title='Metrics',
            yaxis_title='Methods',
            height=400,
            template='plotly_white'
        )

        fig.show()
        print("✅ Performance heatmap complete")

print("\n" + "=" * 80)
print("✅ INTERACTIVE VISUALIZATIONS COMPLETE")
print("=" * 80)

"""## 🏆 Final Summary Comparison

Side-by-side comparison of all methods across key metrics.
"""

print("=" * 80)
print("🏆 FINAL SUMMARY COMPARISON")
print("=" * 80)

if all_results and training_results:
    # ========================================================================
    # 1. Multi-Panel Summary Dashboard
    # ========================================================================
    print("\n📊 Creating comprehensive summary dashboard...")

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Time Comparison',
            'Average Benchmark Accuracy',
            'Fidelity Score (BLEU)',
            'Knowledge Retention (Semantic Similarity)'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    methods = list(training_results.keys())
    methods_upper = [m.upper() for m in methods]

    # Panel 1: Training Time
    times = [training_results[m]['training_time_hours'] for m in methods]
    fig.add_trace(
        go.Bar(
            x=methods_upper,
            y=times,
            name='Training Time',
            marker_color='lightblue',
            text=[f'{t:.2f}h' for t in times],
            textposition='outside'
        ),
        row=1, col=1
    )

    # Panel 2: Average Benchmark Accuracy
    avg_accuracies = []
    for method in methods:
        if method in all_results and 'benchmarks' in all_results[method]:
            accs = [scores['student_accuracy'] * 100
                    for scores in all_results[method]['benchmarks'].values()
                    if isinstance(scores, dict) and 'student_accuracy' in scores]
            avg_accuracies.append(sum(accs) / len(accs) if accs else 0)
        else:
            avg_accuracies.append(0)

    fig.add_trace(
        go.Bar(
            x=methods_upper,
            y=avg_accuracies,
            name='Avg Accuracy',
            marker_color='lightgreen',
            text=[f'{a:.1f}%' for a in avg_accuracies],
            textposition='outside'
        ),
        row=1, col=2
    )

    # Panel 3: BLEU Score
    bleu_scores = []
    for method in methods:
        if method in all_results and 'fidelity' in all_results[method]:
            bleu_scores.append(all_results[method]['fidelity'].get('bleu', 0) * 100)
        else:
            bleu_scores.append(0)

    fig.add_trace(
        go.Bar(
            x=methods_upper,
            y=bleu_scores,
            name='BLEU',
            marker_color='lightcoral',
            text=[f'{b:.1f}' for b in bleu_scores],
            textposition='outside'
        ),
        row=2, col=1
    )

    # Panel 4: Semantic Similarity
    sem_scores = []
    for method in methods:
        if method in all_results and 'fidelity' in all_results[method]:
            sem_scores.append(all_results[method]['fidelity'].get('semantic_similarity', 0) * 100)
        else:
            sem_scores.append(0)

    fig.add_trace(
        go.Bar(
            x=methods_upper,
            y=sem_scores,
            name='Semantic Sim',
            marker_color='lightyellow',
            text=[f'{s:.1f}' for s in sem_scores],
            textposition='outside'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)

    fig.update_yaxes(title_text="Hours", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="BLEU Score", row=2, col=1)
    fig.update_yaxes(title_text="Similarity (%)", row=2, col=2)

    fig.update_layout(
        title_text="Comprehensive Method Comparison Dashboard",
        showlegend=False,
        height=800,
        template='plotly_white'
    )

    fig.show()
    print("✅ Summary dashboard complete")

    # ========================================================================
    # 2. Efficiency vs Performance Scatter Plot
    # ========================================================================
    print("\n📈 Creating efficiency vs performance analysis...")

    fig = go.Figure()

    for i, method in enumerate(methods):
        avg_acc = avg_accuracies[i]
        time = times[i]
        bleu = bleu_scores[i]

        fig.add_trace(go.Scatter(
            x=[time],
            y=[avg_acc],
            mode='markers+text',
            marker=dict(
                size=bleu * 2,  # Size proportional to BLEU score
                color=bleu,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="BLEU Score"),
                line=dict(width=2, color='white')
            ),
            text=[methods_upper[i]],
            textposition='top center',
            textfont=dict(size=12, color='black'),
            name=methods_upper[i],
            hovertemplate=f'<b>{methods_upper[i]}</b><br>' +
                         'Time: %{x:.2f}h<br>' +
                         'Accuracy: %{y:.2f}%<br>' +
                         f'BLEU: {bleu:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title='Efficiency vs Performance Analysis<br><sub>Bubble size = BLEU score (fidelity)</sub>',
        xaxis_title='Training Time (hours)',
        yaxis_title='Average Benchmark Accuracy (%)',
        template='plotly_white',
        height=600,
        showlegend=False,
        hovermode='closest'
    )

    # Add quadrant lines
    if avg_accuracies:
        median_acc = sorted(avg_accuracies)[len(avg_accuracies)//2]
        median_time = sorted(times)[len(times)//2]

        fig.add_hline(y=median_acc, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_time, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig.add_annotation(x=max(times)*0.9, y=max(avg_accuracies)*0.95,
                          text="🏆 Best", showarrow=False, font=dict(size=14, color="green"))
        fig.add_annotation(x=min(times)*1.1, y=max(avg_accuracies)*0.95,
                          text="⚡ Fast & Accurate", showarrow=False, font=dict(size=14, color="darkgreen"))

    fig.show()
    print("✅ Efficiency analysis complete")

    # ========================================================================
    # 3. Winner Summary Table
    # ========================================================================
    print("\n🏅 Determining category winners...")

    winners = {
        '⚡ Fastest Training': methods_upper[times.index(min(times))],
        '🎯 Highest Accuracy': methods_upper[avg_accuracies.index(max(avg_accuracies))],
        '🔍 Best Fidelity (BLEU)': methods_upper[bleu_scores.index(max(bleu_scores))],
        '💎 Best Semantic Similarity': methods_upper[sem_scores.index(max(sem_scores))],
    }

    # Calculate efficiency score (accuracy / time)
    efficiency_scores = [acc / time if time > 0 else 0
                         for acc, time in zip(avg_accuracies, times)]
    if efficiency_scores:
        winners['⚖️ Best Efficiency (Acc/Time)'] = methods_upper[efficiency_scores.index(max(efficiency_scores))]

    print("\n" + "="*60)
    print("🏆 CATEGORY WINNERS")
    print("="*60)
    for category, winner in winners.items():
        print(f"{category}: {winner}")

    # Overall recommendation
    print("\n" + "="*60)
    print("💡 RECOMMENDATION")
    print("="*60)

    # Find most frequent winner
    from collections import Counter
    winner_counts = Counter(winners.values())
    most_wins = winner_counts.most_common(1)[0]

    print(f"\n🌟 Overall Best Method: {most_wins[0]}")
    print(f"   (Winner in {most_wins[1]}/{len(winners)} categories)")

    if most_wins[1] < len(winners):
        print(f"\n📊 Trade-offs:")
        for category, winner in winners.items():
            if winner != most_wins[0]:
                print(f"   • For {category.lower()}: use {winner}")

    print("="*60)

else:
    print("\n⚠️ No results available for comparison")

print("\n" + "=" * 80)
print("✅ FINAL SUMMARY COMPLETE")
print("=" * 80)


# Create a quick reference markdown file
import torch
readme_content = f"""# MedDistillation Experiment Results

**Experiment Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

## Configuration
- **Teacher Model:** {TEACHER_MODEL}
- **Student Model:** {STUDENT_MODEL}
- **Epochs:** {NUM_EPOCHS}
- **Batch Size:** {BATCH_SIZE}
- **Learning Rate:** {LEARNING_RATE}

## Methods Trained
"""

for method in METHODS_TO_RUN:
    readme_content += f"\n### {method.upper()}\n"
    readme_content += f"- Training Time: {training_results[method]['training_time_hours']:.2f} hours\n"
    readme_content += f"- Output Directory: `{training_results[method]['output_dir']}`\n"
    readme_content += f"- Completed: {training_results[method]['completed_at']}\n"

readme_content += f"""
## Files Included
- `training_curves.png` - Training and validation loss curves
- `benchmark_comparison.png` - Performance on medical benchmarks
- `fidelity_metrics.png` - Teacher-student fidelity analysis
- `fidelitybench_radar.png` - Detailed FidelityBench metrics
- `comprehensive_evaluation.json` - All numerical results
- `training_history.json` - Epoch-by-epoch training logs
- `method_comparison.csv` - Cross-method comparison table

## Next Steps
1. Review visualizations in each method's `results/` directory
2. Compare methods using `method_comparison.csv`
3. Fine-tune hyperparameters based on results
4. Run ablation studies to optimize performance
"""

# Use OUTPUT_BASE instead of drive_output_dir (which was notebook-specific)
readme_path = f"{OUTPUT_BASE}/README.md"
try:
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ README created: {readme_path}")
except Exception as e:
    print(f"❌ Error creating README: {e}")

# Display summary
print("\n" + "=" * 80)
print("📊 EXPERIMENT SUMMARY")
print("=" * 80)
print(f"\n🕒 Total Experiment Duration: {sum(r['training_time_hours'] for r in training_results.values()):.2f} hours")
print(f"📁 Results Location: {OUTPUT_BASE}")
print(f"\n✅ All outputs saved successfully!")
print("=" * 80)

# List saved files
print("\n📋 Saved Files:")
if os.path.exists(OUTPUT_BASE):
    for root, dirs, files in os.walk(OUTPUT_BASE):
        level = root.replace(OUTPUT_BASE, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:20]:  # Limit to first 20 files per directory
            print(f"{subindent}{file}")
        if len(files) > 20:
            print(f"{subindent}... and {len(files) - 20} more files")

print("\n" + "=" * 80)
print("🎉 EXPERIMENT COMPLETE!")
print("=" * 80)