import os
import re
import json
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import build_custom_cnn, get_optimizer, get_loss, RoundingMAE

_LABEL_REGEX = re.compile(r"^(\d+)_")

def _parse_label_from_filename(filename: str):
    """Extract integer label from filename (e.g., '3_image.jpg' -> 3)."""
    m = _LABEL_REGEX.match(os.path.basename(filename))
    if not m:
        raise ValueError(f"Filename does not start with integer label_: {filename}")
    return int(m.group(1))


def _list_images_in_folder(folder_path: str):
    """List all image files in a specific folder (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png"}
    files: List[str] = []
    
    if not os.path.exists(folder_path):
        return []
    
    for fn in os.listdir(folder_path):
        file_path = os.path.join(folder_path, fn)
        if os.path.isfile(file_path) and os.path.splitext(fn.lower())[1] in exts:
            files.append(file_path)
    
    return sorted(files)


def _load_from_folders(data_dir: str):
    """
    Load images from pre-organized folders.
    
    Expected structure:
        data_dir/
            training/
                *.jpg
            validation/
                *.jpg
            test/
                *.jpg
    
    Args:
        data_dir: Root data directory (e.g., './data')
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    
    train_files = _list_images_in_folder(train_dir)
    val_files = _list_images_in_folder(val_dir)
    test_files = _list_images_in_folder(test_dir)
    
    if not train_files:
        raise FileNotFoundError(f"No training images found in: {train_dir}")
    if not val_files:
        raise FileNotFoundError(f"No validation images found in: {val_dir}")
    

    if not test_files:
        print(f"Warning: No test images found in: {test_dir}")
    
    return train_files, val_files, test_files


def _decode_image(path: tf.Tensor, image_size: Tuple[int, int]):
    """Load and preprocess an image."""
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _build_dataset(
    file_paths: List[str],
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    shuffle_buffer: int,
    cache: bool,
    augment_cfg: Optional[Dict],
    num_parallel_calls: int,
):
    """Build a TensorFlow dataset from file paths."""
    paths = tf.constant(file_paths)
    labels = tf.constant([_parse_label_from_filename(p) for p in file_paths], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = _decode_image(path, image_size)
        return img, label

    ds = ds.map(_load, num_parallel_calls=num_parallel_calls)

    # Data augmentation
    if augment_cfg:
        if augment_cfg.get("random_flip", False):
            ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=num_parallel_calls)
        if (b := augment_cfg.get("random_brightness", 0)) and b > 0:
            ds = ds.map(lambda x, y: (tf.image.random_brightness(x, max_delta=float(b)), y), num_parallel_calls=num_parallel_calls)
        if (c := augment_cfg.get("random_contrast", 0)) and c > 0:
            ds = ds.map(lambda x, y: (tf.image.random_contrast(x, 1.0 - float(c), 1.0 + float(c)), y), num_parallel_calls=num_parallel_calls)

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(cfg: Dict):
    """
    Build train, validation, and test datasets from pre-organized folders.
    
    Expected folder structure:
        root_dir/
            training/
                *.jpg
            validation/
                *.jpg
            test/
                *.jpg
    
    Args:
        cfg: Data configuration dictionary
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds, sizes_dict)
    """
    data_dir = cfg["root_dir"]
    image_size = tuple(cfg.get("image_size", [80, 80]))
    batch_size = int(cfg.get("batch_size", 64))
    shuffle_buffer = int(cfg.get("shuffle_buffer", 2048))
    num_parallel_calls = int(cfg.get("num_parallel_calls", 4))

    # Load images from pre-organized folders
    train_files, val_files, test_files = _load_from_folders(data_dir)

    sizes = {"train": len(train_files), "val": len(val_files), "test": len(test_files)}
    print(f"Loaded {sizes['train']} training, {sizes['val']} validation, {sizes['test']} test images")

    augment_cfg = cfg.get("augment", {})
    cache = bool(cfg.get("cache", True))

    train_ds = _build_dataset(
        train_files, image_size, batch_size, shuffle=True, shuffle_buffer=shuffle_buffer, cache=cache, augment_cfg=augment_cfg, num_parallel_calls=num_parallel_calls,
    )
    val_ds = _build_dataset(
        val_files, image_size, batch_size, shuffle=False, shuffle_buffer=shuffle_buffer, cache=cache, augment_cfg=None, num_parallel_calls=num_parallel_calls,
    )
    test_ds = _build_dataset(
        test_files, image_size, batch_size, shuffle=False, shuffle_buffer=shuffle_buffer, cache=cache, augment_cfg=None, num_parallel_calls=num_parallel_calls,
    ) if sizes["test"] > 0 else None

    return train_ds, val_ds, test_ds, sizes




def plot_history(history: Dict[str, List[float]], out_dir: str):
    """Plot training history curves."""
    os.makedirs(out_dir, exist_ok=True)

    # Loss curves
    plt.figure(figsize=(6, 4))
    if "loss" in history:
        plt.plot(history["loss"], label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # MAE curves
    if "mae" in history or "val_mae" in history:
        plt.figure(figsize=(6, 4))
        if "mae" in history:
            plt.plot(history["mae"], label="train")
        if "val_mae" in history:
            plt.plot(history["val_mae"], label="val")
        plt.title("MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mae.png"))
        plt.close()

    # Rounded MAE curves
    if "mae_rounded" in history or "val_mae_rounded" in history:
        plt.figure(figsize=(6, 4))
        if "mae_rounded" in history:
            plt.plot(history["mae_rounded"], label="train")
        if "val_mae_rounded" in history:
            plt.plot(history["val_mae_rounded"], label="val")
        plt.title("Rounded MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE (rounded)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mae_rounded.png"))
        plt.close()


def save_success_failure_examples(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    out_dir: str,
    num_examples: int = 4,
    limit_batches: int = 50,
):
    """Save visualization of best and worst predictions."""
    os.makedirs(out_dir, exist_ok=True)

    imgs = []
    gts = []
    preds = []

    for i, (xb, yb) in enumerate(dataset):
        yhat = model.predict(xb, verbose=0)
        imgs.append(xb.numpy())
        gts.append(yb.numpy())
        preds.append(yhat)
        if i + 1 >= limit_batches:
            break

    if not imgs:
        return

    X = np.concatenate(imgs, axis=0)
    Y = np.concatenate(gts, axis=0).reshape(-1)
    P = np.concatenate(preds, axis=0).reshape(-1)

    rounded = np.rint(P)
    errors = np.abs(Y - rounded)

    # Get best and worst indices
    order = np.argsort(errors)
    best_idx = order[:num_examples]
    worst_idx = order[::-1][:num_examples]

    def _save_grid(idxs, fname):
        cols = min(4, len(idxs))
        rows = int(np.ceil(len(idxs) / cols))
        plt.figure(figsize=(cols * 2.5, rows * 2.5))
        for j, idx in enumerate(idxs):
            plt.subplot(rows, cols, j + 1)
            plt.imshow(np.clip(X[idx], 0, 1))
            plt.axis("off")
            plt.title(f"GT:{int(Y[idx])} Pred:{P[idx]:.2f} (|e|={errors[idx]:.0f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    _save_grid(best_idx, "success_examples.png")
    _save_grid(worst_idx, "failure_examples.png")



# ============================================================================
# GPU Configuration
# ============================================================================

def configure_gpu(cfg: Dict) -> None:
    """
    Configure GPU settings for training.
    
    Args:
        cfg: Configuration dictionary with runtime settings
    """
    runtime_cfg = cfg.get("runtime", {})
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"GPU(s) available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Set visible devices if specified
            gpu_id = runtime_cfg.get("gpu_id", None)
            if gpu_id is not None:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                print(f"Using GPU {gpu_id}")
            else:
                print("Using all available GPUs")
                
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")
    
    # Mixed precision training
    mixed_precision = runtime_cfg.get("mixed_precision", "float32")
    if mixed_precision == "mixed_float16" and gpus:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy('mixed_float16')
            print("Mixed precision (float16) enabled")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")


# ============================================================================
# Training function
# ============================================================================

def train(cfg: Dict):
    """
    Main training function.
    
    Args:
        cfg: Complete configuration dictionary
        
    Returns:
        Path to the run directory
    """
    exp_name = cfg.get("experiment", {}).get("name", "exp")
    seed = int(cfg.get("experiment", {}).get("seed", 2400))

    # Configure GPU
    configure_gpu(cfg)

    # Set seeds for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prepare run directory
    runs_dir = cfg.get("output", {}).get("runs_dir", "runs")
    run_dir = os.path.join(runs_dir, f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}__{exp_name}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    figs_dir = os.path.join(run_dir, "figs")
    qual_dir = os.path.join(run_dir, "qualitative")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(qual_dir, exist_ok=True)

    # Save resolved config
    import yaml
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Build datasets
    data_cfg = cfg.get("data", {})
    data_cfg = {**data_cfg, "batch_size": int(cfg.get("training", {}).get("batch_size", 64))}
    train_ds, val_ds, test_ds, sizes = build_datasets(data_cfg)
    
    print(f"Dataset sizes: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}")

    # Build model
    image_size = tuple(data_cfg.get("image_size", [80, 80]))
    input_shape = (image_size[0], image_size[1], 3)
    model = build_custom_cnn(input_shape, cfg.get("model", {}))

    # Compile model
    opt = get_optimizer(cfg.get("training", {}).get("optimizer", {}))
    loss = get_loss(cfg.get("training", {}).get("loss", "mse"))
    model.compile(
        optimizer=opt, 
        loss=loss, 
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), RoundingMAE(name="mae_rounded")]
    )

    # Setup callbacks
    callbacks = []
    
    # Early stopping
    es_cfg = cfg.get("training", {}).get("early_stopping", {"enabled": True, "monitor": "val_mae_rounded", "patience": 8, "mode": "min"})
    if es_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.get("monitor", "val_mae_rounded"),
                patience=int(es_cfg.get("patience", 8)),
                restore_best_weights=True,
                mode=es_cfg.get("mode", "min"),
            )
        )

    # Model checkpoint
    ck_cfg = cfg.get("training", {}).get("checkpoint", {"save_best_only": True, "monitor": "val_mae_rounded"})
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best_model.h5"),
            save_best_only=bool(ck_cfg.get("save_best_only", True)),
            monitor=ck_cfg.get("monitor", "val_mae_rounded"),
            mode="min",
            save_format="h5",
        )
    )

    # Train the model
    print("\nStarting training...")
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.get("training", {}).get("epochs", 50)),
        verbose=1,
        callbacks=callbacks,
    )

    # Save history plots
    plot_history(hist.history, figs_dir)

    # Evaluate on validation and test sets
    results = {}
    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    results["val"] = val_eval
    print(f"\nValidation metrics: {val_eval}")

    if test_ds is not None:
        test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
        results["test"] = test_eval
        print(f"Test metrics: {test_eval}")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save qualitative examples
    save_success_failure_examples(model, val_ds, qual_dir, num_examples=4)

    # Save final model
    model.save(os.path.join(run_dir, "final_model.h5"), save_format="h5")

    print(f"\nRun directory: {run_dir}")
    return run_dir
