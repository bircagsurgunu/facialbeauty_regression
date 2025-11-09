"""
Main entry point for training our attractiveness regression models.

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml data.root_dir=/path/to/data
"""
import argparse
import json
from copy import deepcopy
from typing import Any, Dict, Iterable

import yaml

from train import train


def deep_update(config: Dict[str, Any], key_path: str, value: Any):
    """
    Update a nested dictionary using dot-separated key path.
    
    Example:
        deep_update(cfg, "data.root_dir", "/new/path")
    """
    keys = key_path.split(".")
    cur = config
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_config(cfg_path: str):
    """Load YAML configuration file."""
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _parse_value(value: str):
    """Parse string value to appropriate type (JSON, bool, int, float, or string)."""
    # Try JSON first
    try:
        return json.loads(value)
    except Exception:
        pass
    # Try bool
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    # Try int
    try:
        return int(value)
    except Exception:
        pass
    # Try float
    try:
        return float(value)
    except Exception:
        pass
    # Return as string
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]):
    """
    Apply command-line overrides to configuration.
    
    Args:
        cfg: Base configuration dictionary
        overrides: List of "key=value" strings
        
    Returns:
        Updated configuration dictionary
    """
    if not overrides:
        return cfg
    cfg = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        deep_update(cfg, key, _parse_value(val))
    return cfg


def build_argparser():
    """Build command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Train attractiveness regression models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python main.py --config config.yaml
    python main.py --config config.yaml data.root_dir=/absolute/path/to/data
    python main.py --config config.yaml training.epochs=100 training.batch_size=32
        """
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("overrides", nargs=argparse.REMAINDER, 
                   help="Optional KEY=VALUE overrides (e.g., data.root_dir=/abs/path)")
    return p


def main():
    """Main entry point."""
    parser = build_argparser()
    args = parser.parse_args()

    # Load and apply overrides
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(yaml.dump(cfg, sort_keys=False))
    print("=" * 80)

    # Run training
    run_dir = train(cfg)
    
    print("\n" + "=" * 80)
    print(f"Training complete! Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
