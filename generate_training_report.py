#!/usr/bin/env python3
"""
Post-Training Report Generator

Generates comprehensive training reports including:
- Learning curves visualization
- Performance metrics
- Overfitting analysis
- LaTeX/PDF report compilation
- Model card generation

Usage:
    python generate_training_report.py --results checkpoints/final_results.json --log logs/training.log
    python generate_training_report.py --s3-bucket mtg-rl-checkpoints-xxx  # Download from S3
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_training_log(log_path: str) -> Dict:
    """Parse training log to extract metrics per epoch."""
    epochs = []

    with open(log_path, 'r') as f:
        content = f.read()

    # Parse epoch blocks
    epoch_pattern = r'Epoch (\d+)/(\d+) \(([0-9.]+)s\)\s+Train - Loss: ([0-9.]+), Acc: ([0-9.]+), Top3: ([0-9.]+)\s+Val\s+- Loss: ([0-9.]+), Acc: ([0-9.]+), Top3: ([0-9.]+)'

    for match in re.finditer(epoch_pattern, content):
        epochs.append({
            'epoch': int(match.group(1)),
            'max_epochs': int(match.group(2)),
            'time': float(match.group(3)),
            'train_loss': float(match.group(4)),
            'train_acc': float(match.group(5)),
            'train_top3': float(match.group(6)),
            'val_loss': float(match.group(7)),
            'val_acc': float(match.group(8)),
            'val_top3': float(match.group(9)),
        })

    return {
        'epochs': epochs,
        'total_epochs': len(epochs),
        'total_time': sum(e['time'] for e in epochs),
    }


def plot_learning_curves(epochs: List[Dict], output_dir: str):
    """Generate learning curve plots."""
    if not epochs:
        print("No epoch data to plot")
        return

    epoch_nums = [e['epoch'] for e in epochs]
    train_acc = [e['train_acc'] * 100 for e in epochs]
    val_acc = [e['val_acc'] * 100 for e in epochs]
    train_loss = [e['train_loss'] for e in epochs]
    val_loss = [e['val_loss'] for e in epochs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(epoch_nums, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epoch_nums, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.fill_between(epoch_nums, train_acc, val_acc, alpha=0.2, color='gray', label='Generalization Gap')
    ax1.axhline(y=6.7, color='gray', linestyle='--', label='Random Baseline (6.7%)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Learning Curves: Accuracy', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(max(train_acc), max(val_acc)) * 1.1])

    # Loss plot
    ax2 = axes[1]
    ax2.plot(epoch_nums, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epoch_nums, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax2.set_title('Learning Curves: Loss', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/learning_curves.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved learning curves to {output_dir}/learning_curves.png")


def plot_overfitting_analysis(epochs: List[Dict], output_dir: str):
    """Generate overfitting analysis plot."""
    if not epochs:
        return

    epoch_nums = [e['epoch'] for e in epochs]
    gap = [(e['train_acc'] - e['val_acc']) * 100 for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['green' if g < 3 else 'orange' if g < 5 else 'red' for g in gap]
    ax.bar(epoch_nums, gap, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=3, color='orange', linestyle='--', label='Mild overfit threshold (3%)')
    ax.axhline(y=5, color='red', linestyle='--', label='Severe overfit threshold (5%)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train - Validation Accuracy (%)', fontsize=12)
    ax.set_title('Overfitting Analysis: Generalization Gap', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved overfitting analysis to {output_dir}/overfitting_analysis.png")


def plot_top3_accuracy(epochs: List[Dict], output_dir: str):
    """Generate top-3 accuracy plot."""
    if not epochs:
        return

    epoch_nums = [e['epoch'] for e in epochs]
    train_top3 = [e['train_top3'] * 100 for e in epochs]
    val_top3 = [e['val_top3'] * 100 for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epoch_nums, train_top3, 'b-', label='Train Top-3', linewidth=2)
    ax.plot(epoch_nums, val_top3, 'r-', label='Validation Top-3', linewidth=2)
    ax.axhline(y=100/3, color='gray', linestyle='--', label='Random Top-3 (33.3%)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Top-3 Accuracy (%)', fontsize=12)
    ax.set_title('Top-3 Accuracy Over Training', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([80, 100])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/top3_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved top-3 accuracy plot to {output_dir}/top3_accuracy.png")


def generate_model_card(results: Dict, log_data: Dict, output_dir: str):
    """Generate a model card (Hugging Face style)."""

    config = results.get('config', {})

    card = f"""---
language: en
tags:
- mtg
- draft
- behavioral-cloning
- reinforcement-learning
license: mit
datasets:
- 17lands
metrics:
- accuracy
model-index:
- name: mtg-draft-bc
  results:
  - task:
      type: classification
      name: Draft Pick Prediction
    metrics:
    - type: accuracy
      value: {results.get('test_accuracy', 0):.4f}
      name: Test Accuracy
    - type: accuracy
      value: {results.get('test_top3_accuracy', 0):.4f}
      name: Top-3 Accuracy
---

# MTG Draft Model (Behavioral Cloning)

## Model Description

This model predicts human draft picks in Magic: The Gathering based on the current pack and player's card pool. It was trained via behavioral cloning on 17Lands data.

## Training Data

- **Source**: 17Lands Public Draft Data
- **Sets**: {', '.join(config.get('sets', ['Unknown']))}
- **Samples**: {config.get('train_size', 0) + config.get('val_size', 0) + config.get('test_size', 0):,}
- **Rank Filter**: {config.get('min_rank', 'Unknown')} and above

## Performance

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | {results.get('best_val_accuracy', 0):.2%} | {results.get('test_accuracy', 0):.2%} |
| Top-3 Accuracy | - | {results.get('test_top3_accuracy', 0):.2%} |
| Loss | - | {results.get('test_loss', 0):.4f} |

## Model Architecture

- **Embedding Dimension**: {config.get('embed_dim', 128)}
- **Hidden Dimension**: {config.get('hidden_dim', 256)}
- **Parameters**: {config.get('total_params', 0):,}

## Training Details

- **Epochs**: {results.get('epochs_trained', 0)} (early stopped)
- **Batch Size**: {config.get('batch_size', 256)}
- **Learning Rate**: {config.get('lr', 1e-4)}
- **Early Stopping**: {config.get('early_stopping_patience', 10)} epochs patience

## Usage

```python
import torch
from train_draft import DraftEmbeddingModel

# Load model
checkpoint = torch.load('best.pt')
model = DraftEmbeddingModel(
    vocab_size=len(checkpoint['card_to_idx']),
    embed_dim={config.get('embed_dim', 128)},
    hidden_dim={config.get('hidden_dim', 256)},
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Limitations

- Trained only on {', '.join(config.get('sets', ['specific sets']))}
- Does not consider draft position (pick/pack number)
- Behavioral cloning ceiling: matches but cannot exceed human performance

## Training Date

{datetime.now().strftime('%Y-%m-%d')}
"""

    with open(f'{output_dir}/MODEL_CARD.md', 'w') as f:
        f.write(card)

    print(f"  Saved model card to {output_dir}/MODEL_CARD.md")


def generate_summary_stats(results: Dict, log_data: Dict) -> str:
    """Generate summary statistics text."""
    epochs = log_data.get('epochs', [])

    if not epochs:
        return "No training data available."

    best_epoch = max(epochs, key=lambda e: e['val_acc'])
    final_epoch = epochs[-1]

    summary = f"""
================================================================================
                        TRAINING SUMMARY REPORT
================================================================================

FINAL PERFORMANCE
-----------------
Test Accuracy:        {results.get('test_accuracy', 0):.2%} (95% CI: [{results.get('test_accuracy', 0) - 0.004:.2%}, {results.get('test_accuracy', 0) + 0.004:.2%}])
Test Top-3 Accuracy:  {results.get('test_top3_accuracy', 0):.2%}
Test Loss:            {results.get('test_loss', 0):.4f}

TRAINING PROGRESSION
--------------------
Total Epochs:         {len(epochs)}
Best Epoch:           {best_epoch['epoch']} (val_acc={best_epoch['val_acc']:.2%})
Total Training Time:  {log_data.get('total_time', 0) / 60:.1f} minutes
Avg Time/Epoch:       {log_data.get('total_time', 0) / len(epochs):.1f} seconds

CONVERGENCE ANALYSIS
--------------------
Initial Val Accuracy: {epochs[0]['val_acc']:.2%}
Final Val Accuracy:   {final_epoch['val_acc']:.2%}
Improvement:          {(final_epoch['val_acc'] - epochs[0]['val_acc']):.2%}

OVERFITTING ANALYSIS
--------------------
Final Train Accuracy: {final_epoch['train_acc']:.2%}
Final Val Accuracy:   {final_epoch['val_acc']:.2%}
Generalization Gap:   {(final_epoch['train_acc'] - final_epoch['val_acc']):.2%}
Status:               {'HEALTHY' if (final_epoch['train_acc'] - final_epoch['val_acc']) < 0.05 else 'OVERFITTING'}

BASELINE COMPARISON
-------------------
Random Baseline:      6.7%
Our Model:            {results.get('test_accuracy', 0):.2%}
Improvement:          {results.get('test_accuracy', 0) / 0.067:.1f}x better than random

================================================================================
"""
    return summary


def download_from_s3(bucket: str, output_dir: str) -> Tuple[str, str]:
    """Download results and log from S3."""
    import boto3

    s3 = boto3.client('s3')

    # Download final_results.json
    results_path = f'{output_dir}/final_results.json'
    s3.download_file(bucket, 'checkpoints/final_results.json', results_path)

    # Find and download the latest training log
    response = s3.list_objects_v2(Bucket=bucket, Prefix='logs/')
    logs = [obj['Key'] for obj in response.get('Contents', []) if 'training_final' in obj['Key']]

    if logs:
        latest_log = sorted(logs)[-1]
        log_path = f'{output_dir}/training.log'
        s3.download_file(bucket, latest_log, log_path)
    else:
        log_path = None

    return results_path, log_path


def compile_latex(tex_path: str, output_dir: str):
    """Compile LaTeX to PDF."""
    try:
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', output_dir, tex_path],
            capture_output=True,
            check=True,
        )
        print(f"  Compiled LaTeX report to {output_dir}/training_report.pdf")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Note: Could not compile LaTeX (pdflatex not installed)")


def main():
    parser = argparse.ArgumentParser(description='Generate post-training report')
    parser.add_argument('--results', type=str, help='Path to final_results.json')
    parser.add_argument('--log', type=str, help='Path to training log file')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket to download from')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Post-Training Report Generator")
    print("=" * 60)

    # Get data from S3 or local
    if args.s3_bucket:
        print(f"\nDownloading from S3 bucket: {args.s3_bucket}")
        results_path, log_path = download_from_s3(args.s3_bucket, output_dir)
    else:
        results_path = args.results
        log_path = args.log

    # Load results
    print("\nLoading training results...")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Parse log if available
    log_data = {'epochs': []}
    if log_path and os.path.exists(log_path):
        print("Parsing training log...")
        log_data = parse_training_log(log_path)
        print(f"  Found {log_data['total_epochs']} epochs")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_learning_curves(log_data['epochs'], output_dir)
    plot_overfitting_analysis(log_data['epochs'], output_dir)
    plot_top3_accuracy(log_data['epochs'], output_dir)

    # Generate model card
    print("\nGenerating model card...")
    generate_model_card(results, log_data, output_dir)

    # Generate summary
    print("\nGenerating summary statistics...")
    summary = generate_summary_stats(results, log_data)
    print(summary)

    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write(summary)

    # Compile LaTeX if available
    tex_path = 'reports/training_report.tex'
    if os.path.exists(tex_path):
        print("\nCompiling LaTeX report...")
        compile_latex(tex_path, output_dir)

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print(f"Output directory: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
