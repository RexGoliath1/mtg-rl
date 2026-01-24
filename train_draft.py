#!/usr/bin/env python3
"""
Draft Model Training

Trains a neural network to predict draft picks using 17lands data.
Uses an embedding-based approach where cards are represented by learned vectors.

Usage:
    python train_draft.py --sets FDN DSK --epochs 10
    python train_draft.py --sets FDN --epochs 5 --quick  # Quick test
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data_loader_17lands import (
    SeventeenLandsDataset,
    collate_picks,
    build_card_vocabulary,
)


class DraftEmbeddingModel(nn.Module):
    """
    Draft model using learned card embeddings.

    Architecture:
    1. Card embeddings: Learn 128-dim vector for each card
    2. Pool encoder: Summarize drafted cards via attention
    3. Pack-pool cross-attention: Relate pack to pool
    4. Pick scorer: MLP to score each card in pack
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Card embeddings
        self.card_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Pool encoder (attention over drafted cards)
        self.pool_proj = nn.Linear(vocab_size, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # Pack-pool cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pick scorer
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Value head for RL (optional, trained with BC for now)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        pack_indices: torch.Tensor,    # [batch, pack_size]
        pack_mask: torch.Tensor,       # [batch, pack_size]
        pool_tensor: torch.Tensor,     # [batch, vocab_size]
    ) -> tuple:
        """
        Forward pass.

        Returns:
            pick_logits: [batch, pack_size] - logits for each card
            value: [batch, 1] - estimated draft value
        """
        batch_size, pack_size = pack_indices.shape

        # Embed pack cards
        pack_emb = self.card_embedding(pack_indices)  # [batch, pack_size, embed_dim]

        # Encode pool (using pool tensor directly)
        pool_context = self.pool_proj(pool_tensor)  # [batch, hidden_dim]
        pool_context = self.pool_norm(pool_context)
        pool_context = pool_context.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross attention: pack attends to pool
        # First, project pool context to embed_dim for attention
        pool_emb = pool_context.expand(-1, 1, -1)  # [batch, 1, hidden_dim]

        # Simple approach: concatenate pool context to each pack card
        pool_expanded = pool_context.expand(-1, pack_size, -1)[:, :, :self.embed_dim]

        # Combine pack embeddings with pool context
        combined = torch.cat([pack_emb, pool_expanded], dim=-1)  # [batch, pack_size, embed_dim*2]

        # Score each card
        scores = self.scorer(combined).squeeze(-1)  # [batch, pack_size]

        # Apply mask
        scores = scores.masked_fill(pack_mask == 0, float('-inf'))

        # Value estimate
        value = self.value_head(pool_context.squeeze(1))

        return scores, value


class DraftTrainer:
    """Trainer for draft model."""

    def __init__(
        self,
        model: DraftEmbeddingModel,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader, epoch: int, writer: Optional[SummaryWriter] = None) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        top3_correct = 0

        for batch_idx, batch in enumerate(dataloader):
            pack_indices = batch['pack_indices'].to(self.device)
            pack_mask = batch['pack_mask'].to(self.device)
            pool_tensor = batch['pool_tensor'].to(self.device)
            pick_idx = batch['pick_idx'].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits, _ = self.model(pack_indices, pack_mask, pool_tensor)

            # Loss
            loss = self.criterion(logits, pick_idx)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == pick_idx).sum().item()
            total += pick_idx.shape[0]

            # Top-3 accuracy
            _, top3 = logits.topk(3, dim=-1)
            for i in range(len(pick_idx)):
                if pick_idx[i] in top3[i]:
                    top3_correct += 1

            # Log
            if batch_idx % 100 == 0 and writer:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/accuracy', correct / total, step)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'top3_accuracy': top3_correct / total,
        }

    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate on held-out data."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        top3_correct = 0

        with torch.no_grad():
            for batch in dataloader:
                pack_indices = batch['pack_indices'].to(self.device)
                pack_mask = batch['pack_mask'].to(self.device)
                pool_tensor = batch['pool_tensor'].to(self.device)
                pick_idx = batch['pick_idx'].to(self.device)

                logits, _ = self.model(pack_indices, pack_mask, pool_tensor)
                loss = self.criterion(logits, pick_idx)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == pick_idx).sum().item()
                total += pick_idx.shape[0]

                _, top3 = logits.topk(3, dim=-1)
                for i in range(len(pick_idx)):
                    if pick_idx[i] in top3[i]:
                        top3_correct += 1

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'top3_accuracy': top3_correct / total,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict, card_to_idx: Dict):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'card_to_idx': card_to_idx,
            'vocab_size': self.model.vocab_size,
            'embed_dim': self.model.embed_dim,
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train draft model on 17lands data")
    parser.add_argument("--sets", nargs="+", default=["FDN"], help="Set codes")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per set")
    parser.add_argument("--min-rank", type=str, default="gold", help="Minimum player rank")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.max_samples = 10000
        args.epochs = 3

    print("=" * 60)
    print("Draft Model Training")
    print("=" * 60)
    print(f"Sets: {args.sets}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min rank: {args.min_rank}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading data...")
    start = time.time()

    dataset = SeventeenLandsDataset(
        data_dir="data/17lands",
        sets=args.sets,
        max_samples=args.max_samples,
        min_rank=args.min_rank,
    )

    print(f"Data loaded in {time.time() - start:.1f}s")

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_picks,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_picks,
        num_workers=0,
    )

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
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = trainer.load_checkpoint(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # TensorBoard
    Path("logs").mkdir(exist_ok=True)
    writer = SummaryWriter(f"logs/draft_{'-'.join(args.sets)}")

    # Checkpoints dir
    Path("checkpoints").mkdir(exist_ok=True)

    # Training loop
    best_accuracy = 0
    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, writer)

        # Validate
        val_metrics = trainer.validate(val_loader)

        epoch_time = time.time() - epoch_start

        # Log
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('val/top3_accuracy', val_metrics['top3_accuracy'], epoch)

        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Top3: {train_metrics['top3_accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Top3: {val_metrics['top3_accuracy']:.4f}")

        # Save best
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            trainer.save_checkpoint(
                "checkpoints/draft_best.pt",
                epoch,
                val_metrics,
                dataset.card_to_idx,
            )
            print(f"  New best model saved! (acc={best_accuracy:.4f})")

        # Save periodic
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(
                f"checkpoints/draft_epoch{epoch+1}.pt",
                epoch,
                val_metrics,
                dataset.card_to_idx,
            )

    writer.close()

    print("\n" + "=" * 60)
    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: checkpoints/draft_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
