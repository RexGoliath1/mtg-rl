#!/usr/bin/env python3
"""
Train imitation learning model on collected Forge AI data.

Supports two encoder versions:
- v1: 17-dim state vectors (legacy MLP policy, fast)
- v2: Full ForgeGameStateEncoder (768-dim, 32M params, uses game_state_json from HDF5)

Auto-detects from HDF5 encoding_version attribute, or use --encoder-version flag.
"""

import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import json

from src.utils.wandb_integration import WandbTracker


# =============================================================================
# V1: Legacy 17-dim MLP
# =============================================================================

class ImitationDataset(Dataset):
    """Dataset for imitation learning from HDF5 files (v1: 17-dim states)."""

    def __init__(self, data_dir: str, max_actions: int = 64):
        self.max_actions = max_actions

        # Load all HDF5 files and use the largest one
        data_path = Path(data_dir)
        h5_files = sorted(data_path.rglob("*.h5"))
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

        # Filter: choice must be >= 0 (not -1 = auto-pass) and fit in action space
        valid_mask = (
            (self.choices >= 0) &
            (self.choices < self.max_actions)
        )
        self.states = self.states[valid_mask]
        self.choices = self.choices[valid_mask]
        self.num_actions = self.num_actions[valid_mask]
        self.decision_types = self.decision_types[valid_mask]
        self.turns = self.turns[valid_mask]

        # Fix num_actions: if 0 or less than choice, infer from choice index
        # (collection v1 didn't always populate num_actions for valid decisions)
        needs_fix = self.num_actions <= self.choices
        self.num_actions[needs_fix] = self.choices[needs_fix] + 1

        # Clip num_actions to max_actions for masking
        self.num_actions = np.clip(self.num_actions, 1, self.max_actions)

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


# =============================================================================
# V2: Full ForgeGameStateEncoder
# =============================================================================

class ImitationDatasetV2(Dataset):
    """Dataset that re-encodes game_state_json with ForgeGameStateEncoder at training time."""

    def __init__(self, data_dir: str, max_actions: int = 64, mechanics_h5_path: str = "data/card_mechanics_commander.h5"):
        from src.forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig, parse_forge_json

        self.max_actions = max_actions
        self._parse_forge_json = parse_forge_json

        # Create encoder (CPU, for tensor preparation only — not the trainable copy)
        config = GameStateConfig(mechanics_h5_path=mechanics_h5_path)
        self._encoder = ForgeGameStateEncoder(config)
        self._encoder.eval()

        # Load all HDF5 files and use the largest one
        data_path = Path(data_dir)
        h5_files = sorted(data_path.rglob("*.h5"))
        if not h5_files:
            raise ValueError(f"No HDF5 files found in {data_dir}")

        largest = max(h5_files, key=lambda f: f.stat().st_size)
        print(f"Loading v2 data from: {largest}")

        with h5py.File(largest, 'r') as f:
            encoding_version = f.attrs.get('encoding_version', 1)
            if encoding_version < 2:
                raise ValueError(
                    f"HDF5 file has encoding_version={encoding_version}, "
                    "but v2 encoder requires encoding_version=2 with game_state_json"
                )
            if 'game_state_json' not in f:
                raise ValueError("HDF5 file missing 'game_state_json' dataset")

            self.game_state_jsons = f['game_state_json'][:]
            self.choices = f['choices'][:]
            self.num_actions = f['num_actions'][:]

        # Filter: choice must be >= 0 and fit in action space
        valid_mask = (
            (self.choices >= 0) &
            (self.choices < self.max_actions)
        )
        self.game_state_jsons = self.game_state_jsons[valid_mask]
        self.choices = self.choices[valid_mask]
        self.num_actions = self.num_actions[valid_mask]

        # Fix num_actions
        needs_fix = self.num_actions <= self.choices
        self.num_actions[needs_fix] = self.choices[needs_fix] + 1
        self.num_actions = np.clip(self.num_actions, 1, self.max_actions)

        print(f"Loaded {len(self.choices):,} valid v2 decisions")
        print(f"Max actions in data: {self.num_actions.max()}")
        print(f"Max choice in data: {self.choices.max()}")

    def __len__(self):
        return len(self.choices)

    def __getitem__(self, idx):
        # Decode JSON and prepare tensors
        json_bytes = self.game_state_jsons[idx]
        if isinstance(json_bytes, bytes):
            json_str = json_bytes.decode('utf-8')
        else:
            json_str = str(json_bytes)

        game_state = json.loads(json_str)
        parsed = self._parse_forge_json(game_state)

        # Get tensor dict from encoder's _prepare_tensors (no grad needed)
        with torch.no_grad():
            tensor_dict = self._encoder._prepare_tensors(parsed)

        choice = int(self.choices[idx])
        num_actions = int(self.num_actions[idx])

        assert 0 <= choice < self.max_actions, f"Invalid choice {choice} >= {self.max_actions}"

        action_mask = torch.zeros(self.max_actions)
        action_mask[:num_actions] = 1.0

        return {
            'tensor_dict': tensor_dict,
            'choice': torch.tensor(choice, dtype=torch.long),
            'action_mask': action_mask,
            'num_actions': num_actions,
        }


def collate_v2(batch):
    """Custom collate for v2 dataset — stacks tensor dicts across batch."""
    choices = torch.stack([b['choice'] for b in batch])
    action_masks = torch.stack([b['action_mask'] for b in batch])

    # Stack zone_cards and zone_masks (each is dict of [1, max_cards, feat_dim])
    zone_cards = {}
    zone_masks = {}
    for zone_name in ["hand", "battlefield", "graveyard", "exile"]:
        zone_cards[zone_name] = torch.cat([b['tensor_dict']['zone_cards'][zone_name] for b in batch], dim=0)
        zone_masks[zone_name] = torch.cat([b['tensor_dict']['zone_masks'][zone_name] for b in batch], dim=0)

    stack_features = torch.cat([b['tensor_dict']['stack_features'] for b in batch], dim=0)
    stack_mask = torch.cat([b['tensor_dict']['stack_mask'] for b in batch], dim=0)

    life_totals = torch.cat([b['tensor_dict']['life_totals'] for b in batch], dim=0)
    mana_pools = torch.cat([b['tensor_dict']['mana_pools'] for b in batch], dim=0)
    turn_number = torch.cat([b['tensor_dict']['turn_number'] for b in batch], dim=0)
    phase = torch.cat([b['tensor_dict']['phase'] for b in batch], dim=0)
    active_player = torch.cat([b['tensor_dict']['active_player'] for b in batch], dim=0)
    priority_player = torch.cat([b['tensor_dict']['priority_player'] for b in batch], dim=0)
    extra_features = torch.cat([b['tensor_dict']['extra_features'] for b in batch], dim=0)

    return {
        'zone_cards': zone_cards,
        'zone_masks': zone_masks,
        'stack_features': stack_features,
        'stack_mask': stack_mask,
        'life_totals': life_totals,
        'mana_pools': mana_pools,
        'turn_number': turn_number,
        'phase': phase,
        'active_player': active_player,
        'priority_player': priority_player,
        'extra_features': extra_features,
        'choice': choices,
        'action_mask': action_masks,
    }


class ImitationPolicyV2(nn.Module):
    """Full ForgeGameStateEncoder + action head for imitation learning."""

    def __init__(self, max_actions: int = 64, mechanics_h5_path: str = "data/card_mechanics_commander.h5"):
        super().__init__()
        from src.forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig

        config = GameStateConfig(mechanics_h5_path=mechanics_h5_path)
        self.encoder = ForgeGameStateEncoder(config)
        self.action_head = nn.Sequential(
            nn.Linear(config.output_dim, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, max_actions),
        )
        self.max_actions = max_actions

    def forward(
        self,
        zone_cards, zone_masks, stack_features, stack_mask,
        life_totals, mana_pools, turn_number, phase,
        active_player, priority_player, extra_features=None,
        action_mask=None,
    ):
        """Forward pass: encoder → action logits."""
        state_emb = self.encoder(
            zone_cards, zone_masks, stack_features, stack_mask,
            life_totals, mana_pools, turn_number, phase,
            active_player, priority_player, extra_features,
        )
        logits = self.action_head(state_emb)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        return logits


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch_v1(model, dataloader, optimizer, criterion, device):
    """Train for one epoch (v1 path)."""
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


def train_epoch_v2(model, dataloader, optimizer, criterion, device):
    """Train for one epoch (v2 path — full encoder)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        choice = batch['choice'].to(device)
        action_mask = batch['action_mask'].to(device)

        # Move all tensor inputs to device
        zone_cards = {k: v.to(device) for k, v in batch['zone_cards'].items()}
        zone_masks = {k: v.to(device) for k, v in batch['zone_masks'].items()}
        stack_features = batch['stack_features'].to(device)
        stack_mask = batch['stack_mask'].to(device)
        life_totals = batch['life_totals'].to(device)
        mana_pools = batch['mana_pools'].to(device)
        turn_number = batch['turn_number'].to(device)
        phase = batch['phase'].to(device)
        active_player = batch['active_player'].to(device)
        priority_player = batch['priority_player'].to(device)
        extra_features = batch['extra_features'].to(device)

        optimizer.zero_grad()
        logits = model(
            zone_cards, zone_masks, stack_features, stack_mask,
            life_totals, mana_pools, turn_number, phase,
            active_player, priority_player, extra_features,
            action_mask,
        )
        loss = criterion(logits, choice)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(choice)
        preds = logits.argmax(dim=1)
        correct += (preds == choice).sum().item()
        total += len(choice)

    return total_loss / total, correct / total


def evaluate_v1(model, dataloader, criterion, device):
    """Evaluate model (v1 path)."""
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


def evaluate_v2(model, dataloader, criterion, device):
    """Evaluate model (v2 path)."""
    model.eval()
    total_loss = 0
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            choice = batch['choice'].to(device)
            action_mask = batch['action_mask'].to(device)

            zone_cards = {k: v.to(device) for k, v in batch['zone_cards'].items()}
            zone_masks = {k: v.to(device) for k, v in batch['zone_masks'].items()}
            stack_features = batch['stack_features'].to(device)
            stack_mask = batch['stack_mask'].to(device)
            life_totals = batch['life_totals'].to(device)
            mana_pools = batch['mana_pools'].to(device)
            turn_number = batch['turn_number'].to(device)
            phase = batch['phase'].to(device)
            active_player = batch['active_player'].to(device)
            priority_player = batch['priority_player'].to(device)
            extra_features = batch['extra_features'].to(device)

            logits = model(
                zone_cards, zone_masks, stack_features, stack_mask,
                life_totals, mana_pools, turn_number, phase,
                active_player, priority_player, extra_features,
                action_mask,
            )
            loss = criterion(logits, choice)

            total_loss += loss.item() * len(choice)
            preds = logits.argmax(dim=1)
            correct += (preds == choice).sum().item()

            top3_preds = logits.topk(3, dim=1).indices
            top3_correct += (top3_preds == choice.unsqueeze(1)).any(dim=1).sum().item()

            total += len(choice)

    return total_loss / total, correct / total, top3_correct / total


# =============================================================================
# AUTO-DETECTION
# =============================================================================

def detect_encoder_version(data_dir: str) -> str:
    """Auto-detect encoder version from HDF5 file."""
    data_path = Path(data_dir)
    h5_files = sorted(data_path.rglob("*.h5"))
    if not h5_files:
        return "v1"

    largest = max(h5_files, key=lambda f: f.stat().st_size)
    with h5py.File(largest, 'r') as f:
        version = f.attrs.get('encoding_version', 1)
        has_json = 'game_state_json' in f
    if version >= 2 and has_json:
        return "v2"
    return "v1"


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train imitation learning model")
    parser.add_argument("--data-dir", default="training_data/imitation_aws", help="Data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension (v1 only)")
    parser.add_argument("--max-actions", type=int, default=64, help="Max action space size")
    parser.add_argument("--output", default="checkpoints/imitation_policy.pt", help="Output model path")
    parser.add_argument("--wandb-project", default="forgerl", help="W&B project name")
    parser.add_argument("--wandb-entity", default="sgoncia-self", help="W&B entity")
    parser.add_argument("--log-dir", default="logs/imitation", help="TensorBoard log dir")
    parser.add_argument("--encoder-version", choices=["v1", "v2", "auto"], default="auto",
                        help="Encoder version: v1 (17-dim MLP), v2 (full encoder), auto (detect from HDF5)")
    parser.add_argument("--mechanics-h5", default="data/card_mechanics_commander.h5",
                        help="Path to mechanics HDF5 (v2 only)")
    args = parser.parse_args()

    # Auto-detect encoder version
    encoder_version = args.encoder_version
    if encoder_version == "auto":
        encoder_version = detect_encoder_version(args.data_dir)
        print(f"Auto-detected encoder version: {encoder_version}")
    else:
        print(f"Using encoder version: {encoder_version}")

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
    if encoder_version == "v2":
        dataset = ImitationDatasetV2(args.data_dir, max_actions=args.max_actions, mechanics_h5_path=args.mechanics_h5)
        collate_fn = collate_v2
    else:
        dataset = ImitationDataset(args.data_dir, max_actions=args.max_actions)
        collate_fn = None

    # Train/val split
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # V2 uses smaller batch size by default (large model, more memory)
    batch_size = args.batch_size
    if encoder_version == "v2" and args.batch_size == 256:
        batch_size = 32
        print(f"V2 encoder: reducing batch size to {batch_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    if encoder_version == "v2":
        model = ImitationPolicyV2(
            max_actions=args.max_actions,
            mechanics_h5_path=args.mechanics_h5,
        ).to(device)
        train_fn = train_epoch_v2
        eval_fn = evaluate_v2
    else:
        model = ImitationPolicy(
            state_dim=17,
            hidden_dim=args.hidden_dim,
            max_actions=args.max_actions
        ).to(device)
        train_fn = train_epoch_v1
        eval_fn = evaluate_v1

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # LR default: lower for v2 (larger model)
    lr = args.lr
    if encoder_version == "v2" and args.lr == 1e-3:
        lr = 3e-4
        print(f"V2 encoder: using lr={lr}")

    # W&B tracking (uses WANDB_API_KEY env var if set, else disabled)
    tracker = WandbTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"imitation_{encoder_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=vars(args) | {"model_params": n_params, "device": str(device),
                              "train_samples": len(train_dataset), "val_samples": len(val_dataset),
                              "encoder_version": encoder_version, "effective_lr": lr,
                              "effective_batch_size": batch_size},
        tags=["imitation-learning", f"encoder-{encoder_version}"],
        enabled=os.environ.get("WANDB_MODE") != "disabled",
    )

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\n{'='*60}")
    print(f"TRAINING IMITATION POLICY ({encoder_version.upper()})")
    print(f"{'='*60}")

    best_val_acc = 0
    best_epoch = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_top3 = eval_fn(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Top3: {val_top3*100:.1f}%")

        # Log to W&B + TensorBoard
        tracker.log_epoch(epoch + 1,
            train_metrics={"loss": train_loss, "accuracy": train_acc},
            val_metrics={"loss": val_loss, "accuracy": val_acc, "top3_accuracy": val_top3},
            extra={"lr": scheduler.get_last_lr()[0]})
        tb_writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        tb_writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc, "val_top3": val_top3}, epoch + 1)
        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch + 1)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top3': val_top3,
                'args': vars(args),
                'encoder_version': encoder_version,
            }, output_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}% (epoch {best_epoch})")
    print(f"Model saved to: {output_path}")

    # Final evaluation
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, final_acc, final_top3 = eval_fn(model, val_loader, criterion, device)

    print("\nFinal model performance:")
    print(f"  Top-1 Accuracy: {final_acc*100:.1f}%")
    print(f"  Top-3 Accuracy: {final_top3*100:.1f}%")

    # Log final metrics + save model artifact to W&B
    tracker.save_model(str(output_path), name="imitation-policy",
        metadata={"val_acc": float(final_acc), "val_top3": float(final_top3),
                  "encoder_version": encoder_version},
        aliases=["latest"])
    tracker.alert("Training Complete",
        f"Imitation policy ({encoder_version}): {final_acc*100:.1f}% top-1, {final_top3*100:.1f}% top-3")
    tracker.finish()
    tb_writer.close()

    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'encoder_version': encoder_version,
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
