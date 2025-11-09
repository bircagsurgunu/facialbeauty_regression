# Data Loading Changes

## Summary

The data loading logic has been simplified to work with pre-organized folders instead of creating splits automatically.

## What Changed

### Before (Complex)
- Recursively searched for all images in a root directory
- Automatically created train/val/test splits using sklearn
- Saved splits to text files
- Supported "auto" and "provided" split methods
- Required stratified splitting logic

### After (Simple)
- Directly loads images from pre-organized folders
- Expects data to already be split into `training/`, `validation/`, and `test/` folders
- No split creation or file writing
- Simple folder scanning

## New Folder Structure Required

```
data/
├── training/
│   ├── 1_img001.jpg
│   ├── 2_img002.jpg
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

## Configuration Changes

### Old config.yaml
```yaml
data:
  root_dir: /path/to/dataset
  split:
    method: auto
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
    seed: 42
```

### New config.yaml
```yaml
data:
  root_dir: ./data  # Should contain training/, validation/, test/ subfolders
```

## Code Changes

### Removed Functions
- `create_splits()` - No longer needed
- `_load_or_create_splits()` - No longer needed
- `_save_split_txt()` - No longer needed
- `_load_split_txt()` - No longer needed
- `_list_images()` - Replaced with simpler version

### New Functions
- `_list_images_in_folder()` - Lists images in a single folder (non-recursive)
- `_load_from_folders()` - Loads from pre-organized training/validation/test folders

### Modified Functions
- `build_datasets()` - Now uses `_load_from_folders()` instead of split creation logic

### Removed Dependencies
- `sklearn.model_selection.train_test_split` - No longer needed
- `pathlib.Path` - No longer needed

## Usage

### Before
```bash
# Data could be in any structure, splits created automatically
python main.py --config config.yaml data.root_dir=/path/to/images
```

### After
```bash
# Data must be pre-organized into training/validation/test folders
python main.py --config config.yaml data.root_dir=./data
```

## Benefits

1. **Simpler code**: Removed ~80 lines of splitting logic
2. **Faster startup**: No need to scan and split data on every run
3. **More control**: You decide how to split your data
4. **Clearer structure**: Explicit folder organization
5. **Fewer dependencies**: No sklearn needed for splitting

## Migration Guide

If you have data in the old format, you need to organize it into folders:

```python
# Example script to organize existing data
import os
import shutil

# Assuming you have split files from the old system
def organize_data(old_root, new_root):
    os.makedirs(f"{new_root}/training", exist_ok=True)
    os.makedirs(f"{new_root}/validation", exist_ok=True)
    os.makedirs(f"{new_root}/test", exist_ok=True)
    
    # Read old split files
    with open(f"{old_root}/splits/train.txt") as f:
        train_files = [line.strip() for line in f]
    with open(f"{old_root}/splits/val.txt") as f:
        val_files = [line.strip() for line in f]
    with open(f"{old_root}/splits/test.txt") as f:
        test_files = [line.strip() for line in f]
    
    # Copy files to new structure
    for f in train_files:
        shutil.copy(f, f"{new_root}/training/")
    for f in val_files:
        shutil.copy(f, f"{new_root}/validation/")
    for f in test_files:
        shutil.copy(f, f"{new_root}/test/")
```

## Notes

- The label parsing from filenames (`<label>_<name>.jpg`) remains unchanged
- Data augmentation still works the same way
- Caching and batching logic is unchanged
- All other functionality is preserved
