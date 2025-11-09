# GPU Training and Batch Experiments Guide

This guide explains how to use GPU acceleration and run multiple experiments in batch mode.

## GPU Training

### Automatic GPU Detection

The training script automatically detects and uses available GPUs. No manual configuration needed for basic usage.

### GPU Configuration Options

Add these settings to your `config.yaml` under the `runtime` section:

```yaml
runtime:
  mixed_precision: mixed_float16  # Use float16 for faster GPU training
  gpu_id: 0                       # Use specific GPU (0, 1, 2...) or null for all GPUs
```

### GPU Settings Explained

**`mixed_precision`:**
- `float32` (default): Standard precision, works on CPU and GPU
- `mixed_float16`: Faster training on modern GPUs (requires GPU with compute capability >= 7.0)
  - Can provide 2-3x speedup on compatible GPUs
  - Uses less GPU memory

**`gpu_id`:**
- `null` (default): Use all available GPUs
- `0`, `1`, `2`, etc.: Use specific GPU by ID
- Useful when you have multiple GPUs and want to use a specific one

### Example GPU Configurations

**Single GPU with mixed precision:**
```yaml
runtime:
  mixed_precision: mixed_float16
  gpu_id: 0
```

**Use second GPU:**
```yaml
runtime:
  mixed_precision: float32
  gpu_id: 1
```

**Use all GPUs:**
```yaml
runtime:
  mixed_precision: mixed_float16
  gpu_id: null
```

### GPU Memory Management

The script automatically enables memory growth, which means:
- GPU memory is allocated as needed
- Prevents TensorFlow from allocating all GPU memory at startup
- Allows multiple processes to share the same GPU

### Checking GPU Availability

When you run training, you'll see GPU information:

```
GPU(s) available: 1
  GPU 0: /physical_device:GPU:0
Using GPU 0
Mixed precision (float16) enabled
```

If no GPU is available:
```
No GPU available, using CPU
```

---

## Batch Experiments

### Overview

Run multiple experiments with different configurations automatically using the `run_experiments.py` script.

### Quick Start

1. **Create an experiments configuration file** (e.g., `experiments.yaml`)
2. **Run the experiments:**
   ```bash
   python run_experiments.py --experiments experiments.yaml
   ```

### Experiments Configuration File

The experiments file has two main sections:

```yaml
# Base configuration to start from
base_config: config.yaml

# List of experiments
experiments:
  - name: experiment_1
    config:
      # Override any settings here
      
  - name: experiment_2
    config:
      # Different overrides
```

### Example Experiments File

```yaml
base_config: config.yaml

experiments:
  # Test different dropout rates
  - name: dropout_0.2
    config:
      experiment:
        seed: 42
      model:
        dropout_rate: 0.2
      
  - name: dropout_0.5
    config:
      experiment:
        seed: 42
      model:
        dropout_rate: 0.5

  # Test different learning rates
  - name: lr_0.001
    config:
      experiment:
        seed: 42
      training:
        optimizer:
          learning_rate: 0.001
          
  - name: lr_0.01
    config:
      experiment:
        seed: 42
      training:
        optimizer:
          learning_rate: 0.01
```

### How It Works

1. **Base configuration** is loaded from the specified file
2. **Each experiment** merges its config with the base config
3. **Experiments run sequentially** (one after another)
4. **Results** are saved in separate directories under `runs/`
5. **Summary** is printed at the end showing success/failure

### Experiment Configuration

You can override any setting from the base config:

**Model architecture:**
```yaml
- name: deeper_model
  config:
    model:
      architecture:
        conv_blocks:
          - {filters: 64, kernel_size: 3, pool: 2}
          - {filters: 128, kernel_size: 3, pool: 2}
          - {filters: 256, kernel_size: 3, pool: 2}
        dense_units: [256, 128]
```

**Training parameters:**
```yaml
- name: long_training
  config:
    training:
      epochs: 100
      batch_size: 32
      optimizer:
        learning_rate: 0.0005
```

**Data augmentation:**
```yaml
- name: heavy_augmentation
  config:
    data:
      augment:
        random_flip: true
        random_brightness: 0.15
        random_contrast: 0.15
```

**GPU settings:**
```yaml
- name: mixed_precision_exp
  config:
    runtime:
      mixed_precision: mixed_float16
      gpu_id: 0
```

### Running Experiments

**Basic usage:**
```bash
python run_experiments.py --experiments experiments.yaml
```

**With custom base config:**
```yaml
# In experiments.yaml
base_config: my_custom_config.yaml

experiments:
  - name: exp1
    config:
      # ...
```

### Output

Each experiment creates its own directory:
```
runs/
├── 20241109_121500__dropout_0.2/
│   ├── config.yaml
│   ├── metrics.json
│   ├── final_model.h5
│   └── ...
├── 20241109_122000__dropout_0.5/
│   ├── config.yaml
│   ├── metrics.json
│   ├── final_model.h5
│   └── ...
└── ...
```

### Experiment Summary

At the end, you'll see a summary:

```
================================================================================
EXPERIMENTS SUMMARY
================================================================================

Total experiments: 5
Successful: 4
Failed: 1

✓ Successful experiments:
  - dropout_0.2: 245.3s → runs/20241109_121500__dropout_0.2
  - dropout_0.5: 238.7s → runs/20241109_122000__dropout_0.5
  - lr_0.001: 251.2s → runs/20241109_122500__lr_0.001
  - lr_0.01: 189.4s → runs/20241109_123000__lr_0.01

✗ Failed experiments:
  - bad_config: ValueError: Invalid learning rate
```

---

## Best Practices

### GPU Training

1. **Start with float32** to ensure everything works
2. **Try mixed_float16** if you have a modern GPU for 2-3x speedup
3. **Monitor GPU memory** usage with `nvidia-smi` (on Linux/Windows)
4. **Use smaller batch sizes** if you run out of GPU memory

### Batch Experiments

1. **Start small**: Test with 2-3 experiments first
2. **Use meaningful names**: Makes it easier to track results
3. **Keep seeds consistent**: For fair comparison across experiments
4. **Monitor progress**: Check the output to catch errors early
5. **Save experiments.yaml**: Version control your experiment configurations

### Hyperparameter Search

**Grid search example:**
```yaml
experiments:
  # Test all combinations of dropout and learning rate
  - name: dropout_0.2_lr_0.001
    config:
      model:
        dropout_rate: 0.2
      training:
        optimizer:
          learning_rate: 0.001
          
  - name: dropout_0.2_lr_0.01
    config:
      model:
        dropout_rate: 0.2
      training:
        optimizer:
          learning_rate: 0.01
          
  - name: dropout_0.5_lr_0.001
    config:
      model:
        dropout_rate: 0.5
      training:
        optimizer:
          learning_rate: 0.001
          
  - name: dropout_0.5_lr_0.01
    config:
      model:
        dropout_rate: 0.5
      training:
        optimizer:
          learning_rate: 0.01
```

---

## Troubleshooting

### GPU Not Detected

**Problem:** "No GPU available, using CPU"

**Solutions:**
1. Check if TensorFlow GPU is installed: `pip install tensorflow[and-cuda]`
2. Verify GPU drivers are installed
3. Check CUDA compatibility with your TensorFlow version

### Out of Memory

**Problem:** "ResourceExhaustedError: OOM when allocating tensor"

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 32  # or 16
   ```
2. Reduce model size:
   ```yaml
   model:
     architecture:
       conv_blocks:
         - {filters: 16, kernel_size: 3, pool: 2}
         - {filters: 32, kernel_size: 3, pool: 2}
   ```
3. Disable caching:
   ```yaml
   data:
     cache: false
   ```

### Mixed Precision Errors

**Problem:** Errors when using `mixed_float16`

**Solutions:**
1. Check GPU compute capability (needs >= 7.0)
2. Fall back to `float32`:
   ```yaml
   runtime:
     mixed_precision: float32
   ```

### Experiment Failures

**Problem:** Some experiments fail in batch mode

**Solutions:**
1. Check the error message in the summary
2. Run the failed experiment individually:
   ```bash
   python main.py --config config.yaml model.dropout_rate=0.5
   ```
3. Fix the configuration and re-run

---

## Example Workflow

### 1. Single GPU Training

```bash
# Edit config.yaml
# Set: runtime.mixed_precision: mixed_float16

python main.py --config config.yaml
```

### 2. Hyperparameter Search

```bash
# Create experiments.yaml with different configurations
python run_experiments.py --experiments experiments.yaml
```

### 3. Analyze Results

```bash
# Compare metrics.json from different runs
# Check figs/ for training curves
# Look at qualitative/ for prediction examples
```

---

## Performance Tips

1. **Use mixed precision** on modern GPUs (2-3x faster)
2. **Increase batch size** on GPU (64, 128, or 256)
3. **Enable data caching** for faster epoch iterations
4. **Use data augmentation** only on training set
5. **Monitor GPU utilization** with `nvidia-smi`

## Summary

- ✅ **GPU training** is automatic with memory growth
- ✅ **Mixed precision** for 2-3x speedup on compatible GPUs
- ✅ **Batch experiments** for systematic hyperparameter search
- ✅ **Flexible configuration** override system
- ✅ **Automatic result tracking** in separate directories
