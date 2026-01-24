#!/usr/bin/env python3
"""
Cloud-Ready Draft Model Training

Enhanced training script with:
- Proper train/val/test splits (80/10/10)
- S3 checkpoint uploading
- Graceful shutdown handling (Ctrl+C, spot termination)
- Early stopping with patience
- Remote monitoring support

Usage:
    # Local test
    python train_draft_cloud.py --sets FDN DSK BLB TLA --epochs 20 --quick

    # Full AWS training
    python train_draft_cloud.py --sets FDN DSK BLB TLA --epochs 50 \
        --s3-bucket mtg-rl-checkpoints --early-stopping-patience 5
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from data_loader_17lands import SeventeenLandsDataset, collate_picks
from train_draft import DraftEmbeddingModel, DraftTrainer


class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM."""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n[SIGNAL] Graceful shutdown requested...")
        self.kill_now = True


def check_spot_termination() -> bool:
    """Check if AWS spot instance is being terminated (2-min warning)."""
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            timeout=1
        )
        return response.status_code == 200
    except (requests.RequestException, Exception):
        return False


def upload_to_s3(local_path: str, bucket: str, s3_key: str) -> bool:
    """Upload file to S3 using boto3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, s3_key)
        print(f"  [S3] Uploaded {local_path} -> s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"  [S3] Upload failed: {e}")
        return False


def create_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create proper train/val/test splits.

    Args:
        dataset: Full dataset
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    import numpy as np

    n = len(dataset)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def evaluate_test_set(trainer: DraftTrainer, test_loader: DataLoader) -> Dict:
    """
    Final evaluation on held-out test set.
    Only run after training is complete.
    """
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 60)

    metrics = trainer.validate(test_loader)

    print(f"Test Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Test Top-3 Acc:    {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)")
    print(f"Test Loss:         {metrics['loss']:.4f}")
    print("=" * 60)

    return metrics


def save_training_state(
    trainer: DraftTrainer,
    epoch: int,
    metrics: Dict,
    card_to_idx: Dict,
    path: str,
    s3_bucket: Optional[str] = None,
    is_emergency: bool = False,
):
    """Save checkpoint locally and optionally to S3."""
    prefix = "EMERGENCY_" if is_emergency else ""
    local_path = f"{path}/{prefix}checkpoint_epoch{epoch}.pt"

    trainer.save_checkpoint(local_path, epoch, metrics, card_to_idx)

    if s3_bucket:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"checkpoints/{prefix}checkpoint_{timestamp}_epoch{epoch}.pt"
        upload_to_s3(local_path, s3_bucket, s3_key)

        # Also save as "latest"
        upload_to_s3(local_path, s3_bucket, f"checkpoints/{prefix}latest.pt")


def main():
    parser = argparse.ArgumentParser(description="Cloud-ready draft model training")

    # Data args
    parser.add_argument("--sets", nargs="+", default=["FDN", "DSK", "BLB", "TLA"],
                       help="Set codes (default: current Standard sets)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples per set (None = all)")
    parser.add_argument("--min-rank", type=str, default="gold",
                       help="Minimum player rank filter")

    # Model args
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)

    # Training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Stop if no improvement for N epochs (0 = disabled)")

    # Cloud args
    parser.add_argument("--s3-bucket", type=str, default=None,
                       help="S3 bucket for checkpoints (enables cloud mode)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--spot-check-interval", type=int, default=100,
                       help="Check spot termination every N batches")

    # Convenience
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (10K samples, 5 epochs)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.max_samples = 10000
        args.epochs = 5
        args.early_stopping_patience = 0

    # Setup graceful shutdown
    killer = GracefulKiller()

    # Print config
    print("=" * 60)
    print("Draft Model Training (Cloud-Ready)")
    print("=" * 60)
    print(f"Sets: {args.sets}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples per set: {args.max_samples or 'all'}")
    print(f"Early stopping patience: {args.early_stopping_patience or 'disabled'}")
    print(f"S3 bucket: {args.s3_bucket or 'disabled (local only)'}")
    print("=" * 60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    load_start = time.time()

    dataset = SeventeenLandsDataset(
        data_dir="data/17lands",
        sets=args.sets,
        max_samples=args.max_samples,
        min_rank=args.min_rank,
    )

    print(f"Data loaded in {time.time() - load_start:.1f}s")
    print(f"Total samples: {len(dataset):,}")
    print(f"Unique cards: {len(dataset.card_to_idx):,}")

    # Create train/val/test splits
    print("\nCreating train/val/test splits (80/10/10)...")
    train_dataset, val_dataset, test_dataset = create_splits(dataset)
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val:   {len(val_dataset):,}")
    print(f"  Test:  {len(test_dataset):,} (held out until end)")

    # Data loaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_picks,
        "num_workers": 4 if device == "cuda" else 0,
        "pin_memory": device == "cuda",
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Model
    print("\nInitializing model...")
    model = DraftEmbeddingModel(
        vocab_size=len(dataset.card_to_idx),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Trainer
    trainer = DraftTrainer(model, device=device, lr=args.lr)

    # Resume from checkpoint
    start_epoch = 0
    best_val_accuracy = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint.get("metrics", {}).get("accuracy", 0)
        print(f"  Resuming from epoch {start_epoch}, best acc: {best_val_accuracy:.4f}")

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    run_name = f"draft_{'-'.join(args.sets)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"logs/{run_name}")

    # Save config
    config = vars(args)
    config["run_name"] = run_name
    config["device"] = device
    config["total_params"] = total_params
    config["train_size"] = len(train_dataset)
    config["val_size"] = len(val_dataset)
    config["test_size"] = len(test_dataset)

    with open(f"logs/{run_name}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    epochs_without_improvement = 0
    training_interrupted = False

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Check for shutdown signal
        if killer.kill_now:
            print("\n[SHUTDOWN] Saving emergency checkpoint...")
            save_training_state(
                trainer, epoch, {"accuracy": best_val_accuracy},
                dataset.card_to_idx, "checkpoints",
                args.s3_bucket, is_emergency=True
            )
            training_interrupted = True
            break

        # Check spot termination (on AWS)
        if args.s3_bucket and check_spot_termination():
            print("\n[SPOT] Instance termination detected! Saving emergency checkpoint...")
            save_training_state(
                trainer, epoch, {"accuracy": best_val_accuracy},
                dataset.card_to_idx, "checkpoints",
                args.s3_bucket, is_emergency=True
            )
            training_interrupted = True
            break

        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch, writer)

        # Validate
        val_metrics = trainer.validate(val_loader)

        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/top3_accuracy", val_metrics["top3_accuracy"], epoch)
        writer.add_scalar("train/epoch_time", epoch_time, epoch)

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Top3: {train_metrics['top3_accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Top3: {val_metrics['top3_accuracy']:.4f}")

        # Check for improvement
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0

            # Save best model
            save_training_state(
                trainer, epoch, val_metrics, dataset.card_to_idx,
                "checkpoints", args.s3_bucket
            )

            # Also save as "best"
            trainer.save_checkpoint(
                "checkpoints/draft_best.pt", epoch, val_metrics, dataset.card_to_idx
            )
            if args.s3_bucket:
                upload_to_s3("checkpoints/draft_best.pt", args.s3_bucket, "checkpoints/best.pt")

            print(f"  [NEW BEST] accuracy={best_val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save_training_state(
                trainer, epoch, val_metrics, dataset.card_to_idx,
                "checkpoints", args.s3_bucket
            )

        # Early stopping
        if args.early_stopping_patience > 0:
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n[EARLY STOPPING] No improvement for {args.early_stopping_patience} epochs")
                break

    writer.close()

    # Final evaluation on held-out test set
    if not training_interrupted:
        print("\nLoading best model for test evaluation...")
        trainer.load_checkpoint("checkpoints/draft_best.pt")
        test_metrics = evaluate_test_set(trainer, test_loader)

        # Save final results
        results = {
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "test_top3_accuracy": test_metrics["top3_accuracy"],
            "test_loss": test_metrics["loss"],
            "epochs_trained": epoch + 1,
            "config": config,
        }

        results_path = f"checkpoints/final_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if args.s3_bucket:
            upload_to_s3(results_path, args.s3_bucket, "checkpoints/final_results.json")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        print(f"Held-out Test Accuracy:   {test_metrics['accuracy']:.4f}")
        print(f"Model saved to: checkpoints/draft_best.pt")
        if args.s3_bucket:
            print(f"S3 location: s3://{args.s3_bucket}/checkpoints/")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED")
        print("=" * 60)
        print("Emergency checkpoint saved. Resume with --resume flag.")
        print("=" * 60)


if __name__ == "__main__":
    main()
