#!/usr/bin/env python3
"""
Train pass/play policy from collected Forge AI data.

Since the collected data only distinguishes pass (-1) from play (0+),
this trains a binary classifier to predict when to pass priority.
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
import json


class PassPlayDataset(Dataset):
    """Dataset for pass/play binary classification."""

    def __init__(self, data_dir: str, decision_type: int = 0):
        """
        Args:
            data_dir: Directory with HDF5 files
            decision_type: 0=choose_action, 1=declare_attackers, 2=declare_blockers
        """
        data_path = Path(data_dir)
        h5_files = sorted(data_path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No HDF5 files found in {data_dir}")

        # Use largest file
        largest = max(h5_files, key=lambda f: f.stat().st_size)
        print(f"Loading data from: {largest}")

        with h5py.File(largest, 'r') as f:
            states = f['states'][:]
            choices = f['choices'][:]
            decision_types = f['decision_types'][:]
            num_actions = f['num_actions'][:]

        # Filter to specified decision type
        dtype_mask = decision_types == decision_type
        self.states = states[dtype_mask]
        self.choices = choices[dtype_mask]
        self.num_actions = num_actions[dtype_mask]

        # Binary labels: 0 = pass, 1 = play
        # choice=-1 means pass, choice>=0 means play
        self.labels = (self.choices >= 0).astype(np.int64)

        print(f"Loaded {len(self.states):,} decisions (type={decision_type})")
        print(f"Pass: {(self.labels == 0).sum():,} ({100*(self.labels == 0).sum()/len(self.labels):.1f}%)")
        print(f"Play: {(self.labels == 1).sum():,} ({100*(self.labels == 1).sum()/len(self.labels):.1f}%)")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': torch.FloatTensor(self.states[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'num_actions': self.num_actions[idx]
        }

    def get_class_weights(self):
        """Return class weights for balanced sampling."""
        n_pass = (self.labels == 0).sum()
        n_play = (self.labels == 1).sum()
        total = len(self.labels)
        return torch.tensor([total / (2 * n_pass), total / (2 * n_play)], dtype=torch.float32)


class PassPlayPolicy(nn.Module):
    """Binary classifier for pass/play decisions."""

    def __init__(self, state_dim: int = 17, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # Binary: pass or play
        )

    def forward(self, state):
        return self.net(state)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        state = batch['state'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(state)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(state)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += len(state)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-class accuracy
    pass_correct = 0
    pass_total = 0
    play_correct = 0
    play_total = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            label = batch['label'].to(device)

            logits = model(state)
            loss = criterion(logits, label)

            total_loss += loss.item() * len(state)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += len(state)

            # Per-class
            pass_mask = label == 0
            play_mask = label == 1
            pass_correct += ((preds == label) & pass_mask).sum().item()
            pass_total += pass_mask.sum().item()
            play_correct += ((preds == label) & play_mask).sum().item()
            play_total += play_mask.sum().item()

    return {
        'loss': total_loss / total,
        'acc': correct / total,
        'pass_acc': pass_correct / max(pass_total, 1),
        'play_acc': play_correct / max(play_total, 1)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train pass/play policy")
    parser.add_argument("--data-dir", default="training_data/imitation_aws", help="Data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--output", default="checkpoints/pass_policy.pt", help="Output model path")
    parser.add_argument("--balanced", action="store_true", help="Use balanced sampling")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    dataset = PassPlayDataset(args.data_dir)

    # Train/val split
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Optionally use balanced sampling
    if args.balanced:
        # Create weights for each sample
        weights = dataset.get_class_weights()
        sample_weights = weights[dataset.labels]
        train_indices = train_dataset.indices
        train_weights = sample_weights[train_indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = PassPlayPolicy(state_dim=17, hidden_dim=args.hidden_dim).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Use class weights in loss
    class_weights = dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING PASS/PLAY POLICY")
    print(f"{'='*60}")

    best_val_acc = 0
    best_epoch = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']*100:.1f}% "
              f"(Pass: {val_metrics['pass_acc']*100:.1f}%, Play: {val_metrics['play_acc']*100:.1f}%)")

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, output_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}% (epoch {best_epoch})")
    print(f"Model saved to: {output_path}")

    # Final evaluation
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = evaluate(model, val_loader, criterion, device)

    print(f"\nFinal model performance:")
    print(f"  Overall Accuracy: {final_metrics['acc']*100:.1f}%")
    print(f"  Pass Accuracy: {final_metrics['pass_acc']*100:.1f}%")
    print(f"  Play Accuracy: {final_metrics['play_acc']*100:.1f}%")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'model_params': sum(p.numel() for p in model.parameters())
    }

    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
