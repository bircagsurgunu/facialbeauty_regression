# Attractiveness Regression - Simplified Repository

## Repository Structure

```
cs559hw_simple/
├── main.py           # Entry point with config handling
├── model.py          # CNN architecture and custom metrics
├── train.py          # Dataset loading and training logic
├── config.yaml       # Configuration file
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Files Overview

### `main.py`
- Entry point for the application
- Handles command-line arguments
- Loads and parses YAML configuration
- Supports configuration overrides via command line
- Calls the training function

### `model.py`
- Custom CNN architecture builder (`build_custom_cnn`)
- Custom metric: `RoundingMAE` (rounds predictions before computing MAE)
- Optimizer factory (`get_optimizer`)
- Loss function factory (`get_loss`)
- Weight initializer utilities

### `train.py`
- Dataset utilities:
  - Image file discovery
  - Label parsing from filenames (format: `<label>_<filename>.jpg`)
  - Train/val/test split creation and loading
  - TensorFlow dataset building with augmentation
- Training function with:
  - Reproducible random seed setting
  - Model compilation and training
  - Callbacks (early stopping, checkpointing)
  - Evaluation on validation and test sets
  - History plotting
  - Qualitative example visualization

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --config config.yaml
```

### GPU Training

GPU is automatically detected and used if available. For advanced GPU settings:

```yaml
# In config.yaml
runtime:
  mixed_precision: mixed_float16  # 2-3x faster on modern GPUs
  gpu_id: 0                       # Use specific GPU
```

See [GPU_AND_EXPERIMENTS.md](GPU_AND_EXPERIMENTS.md) for detailed GPU configuration.

### Batch Experiments

Run multiple experiments with different configurations:

```bash
python run_experiments.py --experiments experiments.yaml
```

Example `experiments.yaml`:
```yaml
base_config: config.yaml

experiments:
  - name: high_dropout
    config:
      model:
        dropout_rate: 0.5
        
  - name: low_dropout
    config:
      model:
        dropout_rate: 0.2
```

See [GPU_AND_EXPERIMENTS.md](GPU_AND_EXPERIMENTS.md) for detailed experiment configuration.

### With Configuration Overrides

Override the data directory:
```bash
python main.py --config config.yaml data.root_dir=/path/to/your/dataset
```

Override multiple parameters:
```bash
python main.py --config config.yaml \
    data.root_dir=/path/to/data \
    training.epochs=100 \
    training.batch_size=32 \
    model.dropout_rate=0.5
```

## Data Format

The dataset should be organized in pre-split folders:

```
data/
├── training/
│   ├── 1_img001.jpg
│   ├── 2_img002.jpg
│   ├── 3_img003.jpg
│   └── ...
├── validation/
│   ├── 1_img101.jpg
│   ├── 2_img102.jpg
│   └── ...
└── test/
    ├── 1_img201.jpg
    ├── 2_img202.jpg
    └── ...
```

**Requirements:**
- Images should be named with the format: `<label>_<identifier>.<ext>`
  - Example: `3_image001.jpg` (label = 3)
  - Supported extensions: `.jpg`, `.jpeg`, `.png`
- Three folders required: `training/`, `validation/`, `test/`
- All `.jpg` files in each folder will be loaded automatically

## Configuration

The `config.yaml` file contains all hyperparameters and settings:

- **experiment**: Name and random seed
- **output**: Where to save results
- **data**: Dataset path, image size, augmentation settings, split ratios
- **model**: Architecture configuration (conv blocks, dense layers, regularization)
- **training**: Optimizer, loss, epochs, batch size, callbacks

## Data Loading

The data loading is simplified to work with pre-organized folders:

1. **Folder-based loading**: Simply place your images in `training/`, `validation/`, and `test/` folders
2. **No splitting needed**: Data should already be split before training
3. **Automatic discovery**: All `.jpg`, `.jpeg`, `.png` files in each folder are loaded automatically
4. **Caching**: Dataset caching is enabled by default for faster training
5. **Augmentation**: Random flip, brightness, and contrast augmentation for training set only

## Output

After training, results are saved to `runs/<timestamp>__<experiment_name>/`:

```
runs/20241109_013000__baseline/
├── config.yaml                    # Resolved configuration
├── metrics.json                   # Final evaluation metrics
├── final_model.h5                 # Trained model
├── checkpoints/
│   └── best_model.h5             # Best model checkpoint
├── figs/
│   ├── loss.png                  # Loss curves
│   ├── mae.png                   # MAE curves
│   └── mae_rounded.png           # Rounded MAE curves
└── qualitative/
    ├── success_examples.png      # Best predictions
    └── failure_examples.png      # Worst predictions
```


## Example Workflow

1. **Organize your dataset** into the required folder structure:
   ```
   data/
   ├── training/
   ├── validation/
   └── test/
   ```
2. **Edit config.yaml** to set your data path (or use default `./data`):
   ```yaml
   data:
     root_dir: ./data
   ```
3. **Run training**:
   ```bash
   python main.py --config config.yaml
   ```
4. **Check results** in the `runs/` directory

## Tips

- Start with the default configuration and adjust as needed
- Use configuration overrides for quick experiments
- Monitor the training curves in the `figs/` directory
- Check qualitative examples to understand model behavior
- The `mae_rounded` metric is most relevant for integer label prediction

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- See `requirements.txt` for complete list

