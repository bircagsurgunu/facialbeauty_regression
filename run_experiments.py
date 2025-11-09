"""
Run multiple experiments from a configuration file.

Usage:
    python run_experiments.py --experiments experiments.yaml
"""
import argparse
import os
import sys
import time
from typing import Dict, List, Any
from copy import deepcopy

import yaml

from train import train


def load_experiments_config(config_path: str) -> Dict:
    """Load experiments configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base_cfg: Dict, override_cfg: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base_cfg: Base configuration
        override_cfg: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    result = deepcopy(base_cfg)
    
    for key, value in override_cfg.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


def run_experiments(experiments_config: Dict) -> None:
    """
    Run a series of experiments.
    
    Args:
        experiments_config: Configuration containing base config and experiment variations
    """
    # Load base configuration
    base_config_path = experiments_config.get("base_config", "config.yaml")
    print(f"Loading base configuration from: {base_config_path}")
    
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    
    # Get list of experiments
    experiments = experiments_config.get("experiments", [])
    
    if not experiments:
        print("No experiments defined!")
        return
    
    print(f"\n{'='*80}")
    print(f"Running {len(experiments)} experiments")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        exp_name = exp.get("name", f"experiment_{i}")
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(experiments)}: {exp_name}")
        print(f"{'='*80}")
        
        # Merge base config with experiment-specific config
        exp_cfg = merge_configs(base_cfg, exp.get("config", {}))
        
        # Set experiment name
        if "experiment" not in exp_cfg:
            exp_cfg["experiment"] = {}
        exp_cfg["experiment"]["name"] = exp_name
        
        # Print experiment configuration
        print("\nExperiment configuration overrides:")
        print(yaml.dump(exp.get("config", {}), sort_keys=False))
        
        try:
            start_time = time.time()
            
            # Run training
            run_dir = train(exp_cfg)
            
            elapsed_time = time.time() - start_time
            
            results.append({
                "name": exp_name,
                "status": "success",
                "run_dir": run_dir,
                "time": elapsed_time
            })
            
            print(f"\n✓ Experiment '{exp_name}' completed in {elapsed_time:.1f}s")
            print(f"  Results saved to: {run_dir}")
            
        except Exception as e:
            print(f"\n✗ Experiment '{exp_name}' failed with error:")
            print(f"  {type(e).__name__}: {e}")
            
            results.append({
                "name": exp_name,
                "status": "failed",
                "error": str(e)
            })
            
            # Continue with next experiment
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENTS SUMMARY")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successful experiments:")
        for r in successful:
            print(f"  - {r['name']}: {r['time']:.1f}s → {r['run_dir']}")
    
    if failed:
        print("\n✗ Failed experiments:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")
    
    print(f"\n{'='*80}\n")


def build_argparser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Run multiple experiments from a configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example experiments.yaml:

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
      
  - name: more_epochs
    config:
      experiment:
        seed: 42
      training:
        epochs: 100
        """
    )
    p.add_argument("--experiments", required=True, help="Path to experiments YAML config file")
    return p


def main():
    """Main entry point."""
    parser = build_argparser()
    args = parser.parse_args()
    
    # Load experiments configuration
    experiments_config = load_experiments_config(args.experiments)
    
    # Run experiments
    run_experiments(experiments_config)


if __name__ == "__main__":
    main()
