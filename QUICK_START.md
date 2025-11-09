# Quick Start Guide

## Single Experiment

### CPU Training
```bash
python main.py --config config.yaml
```

### GPU Training (Automatic)
```bash
# GPU is automatically detected and used
python main.py --config config.yaml
```

### GPU Training (Mixed Precision - Faster)
```bash
# Edit config.yaml first:
# runtime:
#   mixed_precision: mixed_float16

python main.py --config config.yaml
```

### Override Settings
```bash
python main.py --config config.yaml \
    data.root_dir=./data \
    training.epochs=100 \
    model.dropout_rate=0.5
```

---

## Multiple Experiments

### Create experiments.yaml
```yaml
base_config: config.yaml

experiments:
  - name: baseline
    config:
      experiment:
        seed: 42
      
  - name: high_dropout
    config:
      experiment:
        seed: 42
      model:
        dropout_rate: 0.5
```

### Run All Experiments
```bash
python run_experiments.py --experiments experiments.yaml
```

---

## Common Configurations

### Faster Training (GPU)
```yaml
runtime:
  mixed_precision: mixed_float16
training:
  batch_size: 128
```

### Better Accuracy
```yaml
training:
  epochs: 100
  early_stopping:
    patience: 15
model:
  dropout_rate: 0.3
  l2_weight: 0.0001
```

### Deeper Model
```yaml
model:
  architecture:
    conv_blocks:
      - {filters: 32, kernel_size: 3, pool: 2}
      - {filters: 64, kernel_size: 3, pool: 2}
      - {filters: 128, kernel_size: 3, pool: 2}
      - {filters: 256, kernel_size: 3, pool: 2}
    dense_units: [256, 128]
```

### Different Loss Function
```yaml
training:
  loss: mae  # or huber
```

---

## File Structure

```
cs559hw_simple/
├── main.py                    # Single experiment
├── run_experiments.py         # Multiple experiments
├── train.py                   # Training logic
├── model.py                   # Model architecture
├── config.yaml               # Base configuration
├── experiments.yaml          # Experiments definition
└── data/                     # Your data
    ├── training/
    ├── validation/
    └── test/
```

---

## Results Location

```
runs/
└── 20241109_121500__experiment_name/
    ├── config.yaml           # Configuration used
    ├── metrics.json          # Final metrics
    ├── final_model.h5        # Trained model
    ├── checkpoints/
    │   └── best_model.h5     # Best checkpoint
    ├── figs/
    │   ├── loss.png
    │   ├── mae.png
    │   └── mae_rounded.png
    └── qualitative/
        ├── success_examples.png
        └── failure_examples.png
```

---

## Tips

✅ **Use GPU**: Automatic, just make sure TensorFlow GPU is installed  
✅ **Mixed Precision**: 2-3x faster on modern GPUs  
✅ **Batch Experiments**: Test multiple configurations automatically  
✅ **Monitor Training**: Check `figs/` for training curves  
✅ **Compare Results**: Look at `metrics.json` across runs  

---

## Troubleshooting

**No GPU detected?**
```bash
pip install tensorflow[and-cuda]
```

**Out of memory?**
```yaml
training:
  batch_size: 32  # Reduce this
data:
  cache: false    # Disable caching
```

**Need help?**
- See [README.md](README.md) for detailed documentation
- See [GPU_AND_EXPERIMENTS.md](GPU_AND_EXPERIMENTS.md) for GPU and batch experiments
