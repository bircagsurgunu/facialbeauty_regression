# Migration Guide: From cs559hw_5pro to cs559hw_simple

This document explains how the original repository structure maps to the simplified version.

## File Mapping

### Original → Simplified

| Original File | New Location | Notes |
|--------------|--------------|-------|
| `src/train.py` | `main.py` + `train.py` | Split into entry point and training logic |
| `src/models/custom_cnn.py` | `model.py` | Model architecture |
| `src/data/dataset.py` | `train.py` | Dataset utilities integrated |
| `src/utils/metrics.py` | `model.py` | Custom metrics in model file |
| `src/utils/seed.py` | `train.py` | Seed setting integrated into train function |
| `src/utils/plotting.py` | `train.py` | Plotting functions integrated |
| `src/utils/qualitative.py` | `train.py` | Qualitative analysis integrated |
| `src/utils/config.py` | `main.py` | Config utilities in entry point |
| `src/configs/base.yaml` | `config.yaml` | Simplified config file |
| `requirements.txt` | `requirements.txt` | Unchanged |

## Code Organization

### `model.py` contains:
- `RoundingMAE` class (from `src/utils/metrics.py`)
- `build_custom_cnn()` (from `src/models/custom_cnn.py`)
- `get_optimizer()` (from `src/train.py`)
- `get_loss()` (from `src/train.py`)
- `_get_initializer()` (from `src/models/custom_cnn.py`)

### `train.py` contains:
- Dataset utilities (from `src/data/dataset.py`):
  - `_parse_label_from_filename()`
  - `_list_images()`
  - `create_splits()`
  - `build_datasets()`
  - All dataset building functions
- Visualization (from `src/utils/plotting.py` and `src/utils/qualitative.py`):
  - `plot_history()`
  - `save_success_failure_examples()`
- Training function (from `src/train.py`):
  - `train()` - main training loop
  - Seed setting (from `src/utils/seed.py`)

### `main.py` contains:
- Config utilities (from `src/utils/config.py`):
  - `load_config()`
  - `apply_overrides()`
  - `deep_update()`
- Entry point (from `src/train.py`):
  - Argument parsing
  - Main execution flow

## Usage Comparison

### Original Repository

```bash
# Original way
python -m src.train --config src/configs/base.yaml data.root_dir=/path/to/data
```

### Simplified Repository

```bash
# New way
python main.py --config config.yaml data.root_dir=/path/to/data
```

## Data Loading - Identical Behavior

Both repositories load data **exactly the same way**:

1. **File discovery**: Recursively finds all `.jpg`, `.jpeg`, `.png` files
2. **Label parsing**: Extracts integer label from filename prefix (e.g., `3_image.jpg` → label 3)
3. **Splitting**: Creates stratified train/val/test splits
4. **Caching**: Caches datasets in memory for faster training
5. **Augmentation**: Applies random flip, brightness, and contrast to training set
6. **Batching**: Creates batched datasets with prefetching

### Example: Loading Data

Both versions use the same config structure:

```yaml
data:
  root_dir: /path/to/dataset
  image_size: [80, 80]
  batch_size: 64
  split:
    method: auto
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
    seed: 42
  augment:
    random_flip: true
    random_brightness: 0.08
    random_contrast: 0.08
```

The function `build_datasets(cfg)` works identically in both versions.

## Features Removed

The simplified version does **not** include:

- ❌ `src/experiments/run_sweep.py` - Hyperparameter sweeps
- ❌ `src/reporting/aggregate_results.py` - Multi-run aggregation
- ❌ `src/reporting/generate_report.py` - LaTeX report generation
- ❌ Multiple config files for different experiments
- ❌ Makefile and Docker support
- ❌ Separate CLI for dataset splitting

These features can be added back if needed, but the core training functionality is fully preserved.

## What's Preserved

✅ **Exact same data loading logic**  
✅ **Exact same model architecture**  
✅ **Exact same training loop**  
✅ **Exact same metrics and evaluation**  
✅ **Exact same visualization outputs**  
✅ **Exact same configuration system**  
✅ **Exact same reproducibility (seeds)**  

## Quick Start with Simplified Version

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Update config.yaml** with your data path:
   ```yaml
   data:
     root_dir: /absolute/path/to/your/dataset
   ```

3. **Run training**:
   ```bash
   python main.py --config config.yaml
   ```

4. **Results** will be in `runs/<timestamp>__<experiment_name>/`

## Converting Existing Configs

If you have configs from the original repository, they work directly with the simplified version:

```bash
# Use original config file
python main.py --config ../cs559hw_5pro/src/configs/base.yaml
```

No changes needed! The config structure is identical.

## Benefits of Simplified Structure

1. **Easier to understand**: 3 files vs. 17 files
2. **Easier to modify**: All related code is together
3. **Easier to debug**: Less jumping between files
4. **Easier to share**: Single directory, no complex imports
5. **Same functionality**: All core features preserved

## When to Use Original vs. Simplified

### Use Original (`cs559hw_5pro`) when:
- Running hyperparameter sweeps
- Generating LaTeX reports
- Need Docker deployment
- Managing multiple experiments
- Need advanced reporting tools

### Use Simplified (`cs559hw_simple`) when:
- Learning the codebase
- Running single experiments
- Prototyping new ideas
- Teaching/sharing code
- Want minimal dependencies

## Example: Running the Same Experiment

### Original:
```bash
cd cs559hw_5pro
python -m src.train --config src/configs/base.yaml \
    data.root_dir=/data/SCUT_FBP5500_downsampled \
    training.epochs=50 \
    model.dropout_rate=0.3
```

### Simplified:
```bash
cd cs559hw_simple
python main.py --config config.yaml \
    data.root_dir=/data/SCUT_FBP5500_downsampled \
    training.epochs=50 \
    model.dropout_rate=0.3
```

Both will produce identical results (given the same random seed).
