#!/usr/bin/env python3
"""
Hyperparameter sweep runner for imitation learning.

Reads a YAML config, generates parameter combinations (grid or random sample),
runs train_imitation.py for each, and produces a summary table.

Results are logged to W&B (if available) and saved to a local CSV fallback.

Usage:
    # Random sample of 20 configs (recommended)
    uv run python3 scripts/run_sweep.py --config configs/sweep_imitation.yaml --max-configs 20

    # Full grid search (972 configs — expensive!)
    uv run python3 scripts/run_sweep.py --config configs/sweep_imitation.yaml --strategy grid

    # Dry run — print configs without training
    uv run python3 scripts/run_sweep.py --config configs/sweep_imitation.yaml --dry-run
"""

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config. Falls back to a simple parser if PyYAML is missing."""
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    # Minimal fallback: only works for flat/simple YAML used in this project
    raise ImportError(
        "PyYAML is required for sweep configs. Install with: uv pip install pyyaml"
    )


# ---------------------------------------------------------------------------
# Sweep generation
# ---------------------------------------------------------------------------

def generate_grid(sweep_params: Dict[str, list]) -> List[Dict[str, Any]]:
    """Generate all combinations (Cartesian product) of sweep parameters."""
    keys = sorted(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def sample_configs(
    sweep_params: Dict[str, list],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Randomly sample N unique configs from the grid without generating all combos."""
    rng = random.Random(seed)
    keys = sorted(sweep_params.keys())
    values = [sweep_params[k] for k in keys]

    total = 1
    for v in values:
        total *= len(v)

    if n >= total:
        return generate_grid(sweep_params)

    seen = set()
    configs = []
    while len(configs) < n:
        combo = tuple(rng.choice(v) for v in values)
        if combo not in seen:
            seen.add(combo)
            configs.append(dict(zip(keys, combo)))
    return configs


# ---------------------------------------------------------------------------
# Training invocation
# ---------------------------------------------------------------------------

def build_train_command(
    base: Dict[str, Any],
    hparams: Dict[str, Any],
    run_idx: int,
    output_dir: str,
) -> List[str]:
    """Build the CLI command to invoke train_imitation.py for one config."""
    cmd = [
        sys.executable, "scripts/train_imitation.py",
        "--data-dir", str(base.get("data_dir", "training_data/imitation_aws")),
        "--encoder-version", str(base.get("encoder_version", "v2")),
        "--max-actions", str(base.get("max_actions", 203)),
        "--mechanics-h5", str(base.get("mechanics_h5", "data/card_mechanics_commander.h5")),
        "--epochs", str(hparams.get("epochs", 20)),
        "--batch-size", str(hparams.get("batch_size", 128)),
        "--lr", str(hparams.get("learning_rate", 3e-4)),
        "--output", str(Path(output_dir) / f"sweep_{run_idx:03d}.pt"),
        "--wandb-project", str(base.get("wandb_project", "forgerl")),
        "--wandb-entity", str(base.get("wandb_entity", "sgoncia-self")),
        "--log-dir", str(Path("logs/sweep") / f"run_{run_idx:03d}"),
    ]
    return cmd


def set_sweep_env(hparams: Dict[str, Any]) -> Dict[str, str]:
    """Set environment variables for hyperparams not exposed as CLI flags.

    The train_imitation.py script currently uses hardcoded weight_decay=0.01,
    CosineAnnealingLR, and dropout=0.1. We pass overrides via environment
    variables so the training script can optionally pick them up. If the
    training script doesn't read these yet, the sweep runner logs them for
    post-hoc analysis regardless.
    """
    env = os.environ.copy()
    env["SWEEP_WEIGHT_DECAY"] = str(hparams.get("weight_decay", 0))
    env["SWEEP_WARMUP_STEPS"] = str(hparams.get("warmup_steps", 0))
    env["SWEEP_SCHEDULER"] = str(hparams.get("scheduler", "cosine"))
    env["SWEEP_DROPOUT"] = str(hparams.get("dropout", 0.1))
    return env


def run_training(
    base: Dict[str, Any],
    hparams: Dict[str, Any],
    run_idx: int,
    output_dir: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single training configuration. Returns result dict."""
    cmd = build_train_command(base, hparams, run_idx, output_dir)
    env = set_sweep_env(hparams)

    result = {
        "run_idx": run_idx,
        **hparams,
        "status": "pending",
        "val_accuracy": None,
        "val_top3": None,
        "best_epoch": None,
        "duration_s": None,
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*70}")
    print(f"SWEEP RUN {run_idx + 1}")
    print(f"{'='*70}")
    for k, v in sorted(hparams.items()):
        print(f"  {k}: {v}")
    print(f"  cmd: {' '.join(cmd)}")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2-hour safety timeout per run
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        elapsed = time.time() - start
        result["duration_s"] = round(elapsed, 1)

        if proc.returncode != 0:
            result["status"] = "error"
            # Print last 20 lines of stderr for diagnosis
            stderr_lines = proc.stderr.strip().split("\n")
            print(f"  ERROR (exit {proc.returncode}):")
            for line in stderr_lines[-20:]:
                print(f"    {line}")
            return result

        result["status"] = "ok"

        # Parse the summary JSON if it exists
        summary_path = Path(output_dir) / f"sweep_{run_idx:03d}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            result["val_accuracy"] = summary.get("best_val_acc")
            result["val_top3"] = summary.get("final_top3")
            result["best_epoch"] = summary.get("best_epoch")
        else:
            # Try parsing from stdout as fallback
            for line in proc.stdout.split("\n"):
                if "Best validation accuracy" in line:
                    # "Best validation accuracy: 45.2% (epoch 12)"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        acc_str = parts[-1].strip().split("%")[0]
                        try:
                            result["val_accuracy"] = float(acc_str) / 100.0
                        except ValueError:
                            pass
                if "Top-3 Accuracy" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        top3_str = parts[-1].strip().split("%")[0]
                        try:
                            result["val_top3"] = float(top3_str) / 100.0
                        except ValueError:
                            pass

        print(f"  Completed in {elapsed:.0f}s — "
              f"val_acc={result['val_accuracy']}, val_top3={result['val_top3']}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration_s"] = 7200
        print("  TIMEOUT (>2h)")
    except Exception as e:
        result["status"] = "exception"
        print(f"  EXCEPTION: {e}")

    return result


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def save_csv(results: List[Dict[str, Any]], path: str):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {path}")


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a ranked summary table to stdout."""
    # Sort by val_accuracy descending (None values last)
    ranked = sorted(
        results,
        key=lambda r: r.get("val_accuracy") or -1,
        reverse=True,
    )

    print(f"\n{'='*90}")
    print("SWEEP RESULTS (ranked by val_accuracy)")
    print(f"{'='*90}")

    header = (
        f"{'Rank':>4} {'Run':>4} {'LR':>8} {'BS':>4} {'WD':>8} "
        f"{'Warm':>5} {'Ep':>3} {'Sched':>7} {'Drop':>5} "
        f"{'ValAcc':>7} {'Top3':>7} {'Time':>6} {'Status':>8}"
    )
    print(header)
    print("-" * 90)

    for rank, r in enumerate(ranked, 1):
        val_acc = f"{r['val_accuracy']*100:.1f}%" if r.get("val_accuracy") else "N/A"
        val_top3 = f"{r['val_top3']*100:.1f}%" if r.get("val_top3") else "N/A"
        dur = f"{r['duration_s']:.0f}s" if r.get("duration_s") else "N/A"
        print(
            f"{rank:>4} {r['run_idx']:>4} "
            f"{r.get('learning_rate', 'N/A'):>8} {r.get('batch_size', 'N/A'):>4} "
            f"{r.get('weight_decay', 'N/A'):>8} {r.get('warmup_steps', 'N/A'):>5} "
            f"{r.get('epochs', 'N/A'):>3} {r.get('scheduler', 'N/A'):>7} "
            f"{r.get('dropout', 'N/A'):>5} "
            f"{val_acc:>7} {val_top3:>7} {dur:>6} {r['status']:>8}"
        )

    # Top-3 highlight
    successful = [r for r in ranked if r.get("val_accuracy") is not None]
    if successful:
        print(f"\nBest config (run {successful[0]['run_idx']}):")
        for k in ["learning_rate", "batch_size", "weight_decay", "warmup_steps",
                   "epochs", "scheduler", "dropout"]:
            print(f"  {k}: {successful[0].get(k)}")
        print(f"  val_accuracy: {successful[0]['val_accuracy']*100:.2f}%")
        if successful[0].get("val_top3"):
            print(f"  val_top3:     {successful[0]['val_top3']*100:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep runner for imitation learning"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to sweep YAML config (e.g. configs/sweep_imitation.yaml)",
    )
    parser.add_argument(
        "--strategy", choices=["random", "grid"], default="random",
        help="Sweep strategy: random sampling (default) or full grid",
    )
    parser.add_argument(
        "--max-configs", type=int, default=30,
        help="Max configs to evaluate (random strategy only, default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without running training",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Path for results CSV (default: checkpoints/sweep/sweep_results_TIMESTAMP.csv)",
    )
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)
    base = config.get("base", {})
    sweep_params = config.get("sweep", {})
    eval_cfg = config.get("evaluation", {})

    # Normalize sweep values (YAML may parse '1e-4' as string)
    for key, values in sweep_params.items():
        sweep_params[key] = [
            float(v) if isinstance(v, (int, float)) else
            (float(v) if _is_numeric(v) else v)
            for v in values
        ]

    # Generate configs
    if args.strategy == "grid":
        configs = generate_grid(sweep_params)
        print(f"Grid search: {len(configs)} total configs")
    else:
        configs = sample_configs(sweep_params, args.max_configs, seed=args.seed)
        total_grid = 1
        for v in sweep_params.values():
            total_grid *= len(v)
        print(f"Random sample: {len(configs)} of {total_grid} possible configs")

    # Setup output directory
    output_dir = base.get("output_dir", "checkpoints/sweep")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"\nDRY RUN — {len(configs)} configs would be evaluated:\n")
        for i, cfg in enumerate(configs):
            print(f"  [{i}] {cfg}")
        return

    # Run sweep
    print(f"\nStarting sweep: {len(configs)} configs")
    print(f"Output: {output_dir}")
    print(f"Metric: {eval_cfg.get('metric', 'val_accuracy')}")

    results = []
    for idx, hparams in enumerate(configs):
        result = run_training(base, hparams, idx, output_dir)
        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_csv or str(Path(output_dir) / f"sweep_results_{timestamp}.csv")
    save_csv(results, csv_path)

    # Also save full results as JSON for programmatic access
    json_path = str(Path(csv_path).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump({
            "config_path": args.config,
            "strategy": args.strategy,
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)
    print(f"Full results JSON: {json_path}")

    # Print summary
    print_summary_table(results)


def _is_numeric(s) -> bool:
    """Check if a string-like value can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    main()
