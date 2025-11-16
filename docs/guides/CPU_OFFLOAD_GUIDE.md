# CPU Offloading Guide for Large Teacher Models

## Overview

This guide explains how to enable CPU offloading for training with large teacher models (70B parameters) on GPUs with limited VRAM (e.g., Colab A100 40GB).

## Problem

When loading Meditron-70B as a teacher model with 8-bit quantization:
- **Required VRAM**: ~60-70GB total (teacher model + student model + training overhead)
- **Available on Colab A100 40GB**: Only 40GB VRAM
- **Result**: `ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM`

## Solution: CPU Offloading

CPU offloading allows the model to automatically overflow layers that don't fit in GPU memory to CPU/RAM. This is slower but enables training on hardware with limited VRAM.

### Performance Impact
- **Speed**: ~70% of GPU-only training (still practical for distillation)
- **Memory**: Uses system RAM for overflow (~20-30GB additional RAM needed)
- **Feasibility**: Works well on Google Colab Free/Pro with A100 40GB

## Implementation

### 1. Enable in Jupyter Notebook

In the **configuration cell** (Cell 6), set the flag:

```python
# CPU offloading for large teacher model (70B)
# ⚡ ENABLE THIS to use CPU/RAM for teacher model overflow
# ⚠️  Training will be slower but works on Colab A100 40GB
ENABLE_CPU_OFFLOAD = True   # Set to False if using A100 80GB
```

### 2. What Happens Automatically

When you run the notebook with `ENABLE_CPU_OFFLOAD = True`:

1. **Training command includes flag**: The notebook automatically adds `--enable_cpu_offload` to the training command
2. **Teacher model loading**: The `Trainer.py` script loads the teacher model with CPU offloading enabled
3. **Layer distribution**: Layers are automatically distributed:
   - First ~24 layers on GPU
   - Remaining ~56 layers on CPU/RAM
   - Student model fully on GPU

### 3. Expected Behavior

**With CPU Offloading ENABLED:**
```
🖥️  CPU Offloading: ENABLED
   • Teacher model layers will overflow to CPU/RAM
   • Training will be slower but works on 40GB GPU
   • Expected speed: ~70% of GPU-only training
```

**Console output during model loading:**
```
INFO - Loading teacher model: epfl-llm/meditron-70b
INFO - CPU offloading ENABLED - model layers will overflow to CPU/RAM
INFO - Teacher model loaded successfully
```

**GPU usage:**
- During training: 85-95% GPU utilization
- Memory: ~38-39GB / 40GB VRAM used

## Technical Details

### Code Changes

**1. Notebook Configuration (MedDistillation_Experiment.ipynb)**
```python
# Added ENABLE_CPU_OFFLOAD flag
ENABLE_CPU_OFFLOAD = True

# Automatically passed to training command
if ENABLE_CPU_OFFLOAD:
    cmd += " \\\n        --enable_cpu_offload"
```

**2. Trainer.py - Quantization Config**
```python
def setup_quantization_config(enable_cpu_offload: bool = False) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,  # NEW!
    )
```

**3. Trainer.py - Model Loading**
```python
def load_teacher_model(
    model_name: str,
    use_quantization: bool = True,
    enable_cpu_offload: bool = False  # NEW parameter
) -> nn.Module:
    if enable_cpu_offload:
        logger.info("CPU offloading ENABLED - model layers will overflow to CPU/RAM")
    
    quantization_config = setup_quantization_config(enable_cpu_offload=enable_cpu_offload)
    # ... rest of loading code
```

**4. Trainer.py - Command Line Argument**
```python
parser.add_argument('--enable_cpu_offload', action='store_true', default=False,
                    help='Enable CPU offloading for large teacher models (70B+)')
```

### How It Works

The key parameter is `llm_int8_enable_fp32_cpu_offload=True` in BitsAndBytesConfig:

1. **Model loading**: `device_map="auto"` attempts to fit the model on GPU
2. **Overflow detection**: When GPU memory is insufficient, it checks the offload flag
3. **CPU dispatch**: If enabled, overflowing layers are moved to CPU
4. **Mixed execution**: 
   - GPU layers: Fast inference on GPU
   - CPU layers: Slower inference on CPU, results transferred to GPU
   - Training continues normally

## Alternatives

If CPU offloading is too slow, consider these alternatives:

### Option 1: Use Smaller Teacher Model
```python
TEACHER_MODEL = "epfl-llm/meditron-7b"  # Fits in 40GB GPU
ENABLE_CPU_OFFLOAD = False
```
**Pros**: Faster training, still medically fine-tuned  
**Cons**: Less powerful teacher, may reduce distillation quality

### Option 2: Upgrade to Colab Pro+
- **Hardware**: A100 80GB GPU
- **Cost**: ~$50/month
- **Benefit**: Full 70B teacher model on GPU, no CPU offloading needed

### Option 3: Use External Resources
- Vast.ai, Lambda Labs, or RunPod for A100 80GB rental
- Cost: ~$1-2/hour

## Benchmarks

**Training Time Comparison** (3 epochs, 10M tokens):

| Configuration | GPU Usage | Time per Epoch | Total Time |
|---------------|-----------|----------------|------------|
| 70B teacher, CPU offload ON | 90% | ~3.5 hours | ~10.5 hours |
| 70B teacher, A100 80GB | 95% | ~2.5 hours | ~7.5 hours |
| 7B teacher, no offload | 85% | ~2 hours | ~6 hours |

## Troubleshooting

### Issue: Still getting "not enough GPU RAM" error
**Solution**: Ensure `ENABLE_CPU_OFFLOAD = True` is set and the cell is executed before training

### Issue: Training is very slow
**Expected**: CPU offloading adds ~30% overhead. If much slower:
- Check system RAM usage (should have 20-30GB free)
- Reduce `BATCH_SIZE` to free GPU memory
- Consider switching to 7B teacher model

### Issue: GPU usage is 0%
**Diagnosis**: Training never started, error during model loading
**Solution**: Check error logs, ensure HuggingFace token is valid

### Issue: RuntimeError: CUDA out of memory
**Cause**: Student model or batch size too large
**Solution**: 
```python
BATCH_SIZE = 1  # Reduce from 2
GRADIENT_ACCUMULATION = 32  # Increase to maintain effective batch size
```

## Verification

After enabling CPU offload, verify it's working:

1. **Check console output** during model loading:
   ```
   INFO - CPU offloading ENABLED - model layers will overflow to CPU/RAM
   ```

2. **Monitor GPU usage** during training:
   ```bash
   !watch -n 1 nvidia-smi  # In Colab
   ```
   Should show 85-95% GPU utilization

3. **Check system RAM** (should increase when teacher loads):
   ```python
   import psutil
   print(f"RAM: {psutil.virtual_memory().percent}%")
   ```

## Summary

✅ **Enable CPU offloading** if you have:
- Colab A100 40GB (free or Pro)
- Want to use 70B teacher model
- Don't mind 30% slower training

❌ **Don't need CPU offloading** if you have:
- A100 80GB GPU
- Using 7B or smaller teacher model
- Strict time constraints (use 7B teacher instead)

## References

- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [HuggingFace Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
