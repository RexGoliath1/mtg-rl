#!/usr/bin/env python3
"""
Train imitation learning model on collected Forge AI data.

Uses a simple MLP policy network to learn from expert decisions.
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import json


class ImitationDataset(Dataset):
    """Dataset for imitation learning from HDF5 files."""

    def __init__(self, data_dir: str, max_actions: int = 64):
        self.max_actions = max_actions

        # Load all HDF5 files and use the largest one
        data_path = Path(data_dir)
        h5_files = sorted(data_path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No HDF5 files found in {data_dir}")

        # Use largest file
        largest = max(h5_files, key=lambda f: f.stat().st_size)
        print(f"Loading data from: {largest}")

        with h5py.File(largest, 'r') as f:
            self.states = f['states'][:]
            self.choices = f['choices'][:]
            self.num_actions = f['num_actions'][:]
            self.decision_types = f['decision_types'][:]
            self.turns = f['turns'][:]

        # Filter out invalid/trivial decisions:
        # 1. choice must be >= 0 (not -1 = auto-pass)
        # 2. choice must be < num_actions (valid expert choice)
        # 3. choice must be < max_actions (fits in our action space)
        # 4. num_actions must be > 1 (meaningful choice, not forced)
        valid_mask = (
            (self.choices >= 0) &
            (self.choices < self.num_actions) &
            (self.choices < self.max_actions) &
            (self.num_actions > 1)  # Only decisions with actual choice
        )
        self.states = self.states[valid_mask]
        self.choices = self.choices[valid_mask]
        self.num_actions = self.num_actions[valid_mask]
        self.decision_types = self.decision_types[valid_mask]
        self.turns = self.turns[valid_mask]

        # Clip num_actions to max_actions for masking
        self.num_actions = np.clip(self.num_actions, 0, self.max_actions)

        print(f"Loaded {len(self.states):,} valid decisions")
        print(f"State shape: {self.states.shape}")
        print(f"Max actions in data: {self.num_actions.max()}")
        print(f"Max choice in data: {self.choices.max()}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.FloatTensor(self.states[idx])
        choice = int(self.choices[idx])
        num_actions = int(self.num_actions[idx])

        # Ensure choice is within bounds (should already be filtered, but double-check)
        assert 0 <= choice < self.max_actions, f"Invalid choice {choice} >= {self.max_actions}"

        # Create action mask (1 for valid actions, 0 for invalid)
        action_mask = torch.zeros(self.max_actions)
        action_mask[:num_actions] = 1.0

        return {
            'state': state,
            'choice': torch.tensor(choice, dtype=torch.long),
            'action_mask': action_mask,
            'num_actions': num_actions
        }


class ImitationPolicy(nn.Module):
    """MLP policy network for imitation learning."""

    def __init__(self, state_dim: int = 17, hidden_dim: int = 256, max_actions: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.action_head = nn.Linear(hidden_dim, max_actions)

    def forward(self, state, action_mask=None):
        """Forward pass with optional action masking."""
        features = self.encoder(state)
        logits = self.action_head(features)

        if action_mask is not None:
            # Mask invalid actions with large negative value
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        state = batch['state'].to(device)
        choice = batch['choice'].to(device)
        action_mask = batch['action_mask'].to(device)

        optimizer.zero_grad()
        logits = model(state, action_mask)
        loss = criterion(logits, choice)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(state)
        preds = logits.argmax(dim=1)
        correct += (preds == choice).sum().item()
        total += len(state)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            choice = batch['choice'].to(device)
            action_mask = batch['action_mask'].to(device)

            logits = model(state, action_mask)
            loss = criterion(logits, choice)

            total_loss += loss.item() * len(state)
            preds = logits.argmax(dim=1)
            correct += (preds == choice).sum().item()

            # Top-3 accuracy
            top3_preds = logits.topk(3, dim=1).indices
            top3_correct += (top3_preds == choice.unsqueeze(1)).any(dim=1).sum().item()

            total += len(state)

    return total_loss / total, correct / total, top3_correct / total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train imitation learning model")
    parser.add_argument("--data-dir", default="training_data/imitation_aws", help="Data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--max-actions", type=int, default=64, help="Max action space size")
    parser.add_argument("--output", default="checkpoints/imitation_policy.pt", help="Output model path")
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
    dataset = ImitationDataset(args.data_dir, max_actions=args.max_actions)

    # Train/val split
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = ImitationPolicy(
        state_dim=17,
        hidden_dim=args.hidden_dim,
        max_actions=args.max_actions
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING IMITATION POLICY")
    print(f"{'='*60}")

    best_val_acc = 0
    best_epoch = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_top3 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Top3: {val_top3*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top3': val_top3,
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
    _, final_acc, final_top3 = evaluate(model, val_loader, criterion, device)

    print("\nFinal model performance:")
    print(f"  Top-1 Accuracy: {final_acc*100:.1f}%")
    print(f"  Top-3 Accuracy: {final_top3*100:.1f}%")

    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'final_top1': float(final_acc),
        'final_top3': float(final_top3),
        'model_params': sum(p.numel() for p in model.parameters())
    }

    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
