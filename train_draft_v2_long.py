#!/usr/bin/env python3
"""
Draft Model Training v2 - Extended Training Run

Changes from base v2:
- 100 epochs instead of 50
- Early stopping patience: 20 epochs (up from 10)
- Learning rate scheduler: Cosine annealing with warmup
- Monitor both loss and accuracy
- More aggressive dropout (0.2 instead of 0.1)

Usage:
    python train_draft_v2_long.py --sets FDN DSK BLB TLA --epochs 100
"""

import argparse
import json
import gzip
import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from hybrid_card_encoder import HybridCardEncoder, HybridEncoderConfig, StructuralFeatureExtractor


def upload_to_s3(local_path: str, bucket: str, s3_key: str) -> bool:
    """Upload file to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, s3_key)
        print(f"  [S3] Uploaded {local_path} -> s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"  [S3] Upload failed: {e}")
        return False


class V2DraftDataset(Dataset):
    """Draft dataset with pre-computed embeddings."""

    def __init__(
        self,
        data_dir: str,
        sets: List[str],
        metadata_path: str = "data/card_metadata.json",
        embeddings_path: str = "data/card_embeddings.pt",
        max_samples: Optional[int] = None,
        min_rank: str = "gold",
    ):
        self.data_dir = Path(data_dir)
        self.sets = sets

        print(f"Loading card metadata from {metadata_path}...")
        with open(metadata_path) as f:
            self.card_metadata = json.load(f)
        print(f"  Loaded metadata for {len(self.card_metadata)} cards")

        print(f"Loading text embeddings from {embeddings_path}...")
        embedding_data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
        self.text_embeddings = embedding_data['embeddings']
        self.embedding_dim = embedding_data['embedding_dim']
        print(f"  Loaded embeddings: {len(self.text_embeddings)} cards, {self.embedding_dim}d")

        self.card_to_idx: Dict[str, int] = {}
        self.idx_to_card: Dict[int, str] = {}
        for i, name in enumerate(self.card_metadata.keys()):
            self.card_to_idx[name] = i
            self.idx_to_card[i] = name

        config = HybridEncoderConfig()
        self.feature_extractor = StructuralFeatureExtractor(config)

        print("Pre-computing structural features...")
        self.structural_features = {}
        for name, metadata in self.card_metadata.items():
            self.structural_features[name] = self.feature_extractor.extract(metadata)
        self.structural_dim = config.structural_dim

        print(f"Loading draft data for sets: {sets}...")
        self.samples: List[Dict] = []
        self._load_draft_data(max_samples, min_rank)
        print(f"  Total samples: {len(self.samples)}")

    def _load_draft_data(self, max_samples: Optional[int], min_rank: str):
        rank_order = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'mythic']
        min_rank_idx = rank_order.index(min_rank.lower())
        samples_per_set = max_samples // len(self.sets) if max_samples else None

        for set_code in self.sets:
            csv_path = self.data_dir / f"draft_data_public.{set_code}.PremierDraft.csv.gz"
            if not csv_path.exists():
                print(f"  Skipping {set_code} (not found)")
                continue

            print(f"  Loading {set_code}...")
            count = 0

            with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                pack_cols = [c for c in reader.fieldnames if c.startswith('pack_card_')]
                pool_cols = [c for c in reader.fieldnames if c.startswith('pool_')]

                for row in reader:
                    rank = row.get('rank', 'bronze').lower()
                    if rank in rank_order and rank_order.index(rank) < min_rank_idx:
                        continue

                    pack_cards = []
                    for col in pack_cols:
                        if row.get(col) == '1':
                            card_name = col.replace('pack_card_', '')
                            if card_name in self.card_metadata:
                                pack_cards.append(card_name)

                    picked_card = row.get('pick')
                    if not picked_card or picked_card not in self.card_metadata:
                        continue
                    if picked_card not in pack_cards:
                        continue

                    pool_cards = []
                    for col in pool_cols:
                        card_name = col.replace('pool_', '')
                        try:
                            count_val = int(row.get(col, 0))
                            for _ in range(count_val):
                                if card_name in self.card_metadata:
                                    pool_cards.append(card_name)
                        except ValueError:
                            pass

                    self.samples.append({
                        'pack_cards': pack_cards,
                        'picked_card': picked_card,
                        'picked_idx': pack_cards.index(picked_card),
                        'pool_cards': pool_cards,
                    })

                    count += 1
                    if samples_per_set and count >= samples_per_set:
                        break

            print(f"    Loaded {count} samples from {set_code}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        pack_text = torch.stack([
            self.text_embeddings[name] for name in sample['pack_cards']
        ])
        pack_struct = torch.from_numpy(np.array([
            self.structural_features[name] for name in sample['pack_cards']
        ], dtype=np.float32))

        if sample['pool_cards']:
            pool_text = torch.stack([
                self.text_embeddings[name] for name in sample['pool_cards']
            ])
            pool_struct = torch.from_numpy(np.array([
                self.structural_features[name] for name in sample['pool_cards']
            ], dtype=np.float32))
        else:
            pool_text = torch.zeros(1, self.embedding_dim)
            pool_struct = torch.zeros(1, self.structural_dim)

        return (pack_text, pack_struct, sample['picked_idx'], pool_text, pool_struct)


def collate_v2_picks(batch):
    pack_texts, pack_structs, picked_idxs, pool_texts, pool_structs = zip(*batch)

    max_pack = max(pt.size(0) for pt in pack_texts)
    max_pool = max(pt.size(0) for pt in pool_texts)

    batch_size = len(batch)
    text_dim = pack_texts[0].size(1)
    struct_dim = pack_structs[0].size(1)

    pack_text_padded = torch.zeros(batch_size, max_pack, text_dim)
    pack_struct_padded = torch.zeros(batch_size, max_pack, struct_dim)
    pack_mask = torch.zeros(batch_size, max_pack)

    for i, (pt, ps) in enumerate(zip(pack_texts, pack_structs)):
        n = pt.size(0)
        pack_text_padded[i, :n] = pt
        pack_struct_padded[i, :n] = ps
        pack_mask[i, :n] = 1

    pool_text_padded = torch.zeros(batch_size, max_pool, text_dim)
    pool_struct_padded = torch.zeros(batch_size, max_pool, struct_dim)
    pool_mask = torch.zeros(batch_size, max_pool)

    for i, (pt, ps) in enumerate(zip(pool_texts, pool_structs)):
        n = pt.size(0)
        pool_text_padded[i, :n] = pt
        pool_struct_padded[i, :n] = ps
        pool_mask[i, :n] = 1

    picked_tensor = torch.tensor(picked_idxs, dtype=torch.long)

    return {
        'pack_text': pack_text_padded,
        'pack_struct': pack_struct_padded,
        'pack_mask': pack_mask,
        'pool_text': pool_text_padded,
        'pool_struct': pool_struct_padded,
        'pool_mask': pool_mask,
        'picked_idx': picked_tensor,
    }


class V2DraftModelWithDropout(nn.Module):
    """Draft model with increased dropout for regularization."""

    def __init__(self, encoder_config: Optional[HybridEncoderConfig] = None, dropout: float = 0.2):
        super().__init__()

        self.encoder_config = encoder_config or HybridEncoderConfig(dropout=dropout)
        self.encoder = HybridCardEncoder(self.encoder_config)
        self.dropout = nn.Dropout(dropout)

        self.pool_aggregator = nn.Sequential(
            nn.Linear(self.encoder_config.output_dim, self.encoder_config.output_dim),
            nn.LayerNorm(self.encoder_config.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.scorer = nn.Sequential(
            nn.Linear(self.encoder_config.output_dim * 2, self.encoder_config.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_config.output_dim, 1),
        )

    def forward(self, pack_text, pack_struct, pack_mask, pool_text, pool_struct, pool_mask):
        pack_emb = self.encoder(pack_text, pack_struct, pack_mask, use_interactions=True)
        pack_emb = self.dropout(pack_emb)

        pool_emb = self.encoder(pool_text, pool_struct, pool_mask, use_interactions=True)

        pool_mask_exp = pool_mask.unsqueeze(-1)
        pool_sum = (pool_emb * pool_mask_exp).sum(dim=1)
        pool_count = pool_mask_exp.sum(dim=1).clamp(min=1)
        pool_context = pool_sum / pool_count
        pool_context = self.pool_aggregator(pool_context)

        batch_size, max_pack, emb_dim = pack_emb.shape
        pool_context_exp = pool_context.unsqueeze(1).expand(-1, max_pack, -1)

        combined = torch.cat([pack_emb, pool_context_exp], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        scores = scores.masked_fill(pack_mask == 0, float('-inf'))

        return scores


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def create_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    n = len(dataset)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    return (
        Subset(dataset, indices[:train_end]),
        Subset(dataset, indices[train_end:val_end]),
        Subset(dataset, indices[val_end:]),
    )


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        pack_text = batch['pack_text'].to(device)
        pack_struct = batch['pack_struct'].to(device)
        pack_mask = batch['pack_mask'].to(device)
        pool_text = batch['pool_text'].to(device)
        pool_struct = batch['pool_struct'].to(device)
        pool_mask = batch['pool_mask'].to(device)
        targets = batch['picked_idx'].to(device)

        optimizer.zero_grad()
        scores = model(pack_text, pack_struct, pack_mask, pool_text, pool_struct, pool_mask)

        loss = F.cross_entropy(scores, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(targets)
        preds = scores.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += len(targets)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'lr': scheduler.get_last_lr()[0],
    }


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_top3 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            pack_text = batch['pack_text'].to(device)
            pack_struct = batch['pack_struct'].to(device)
            pack_mask = batch['pack_mask'].to(device)
            pool_text = batch['pool_text'].to(device)
            pool_struct = batch['pool_struct'].to(device)
            pool_mask = batch['pool_mask'].to(device)
            targets = batch['picked_idx'].to(device)

            scores = model(pack_text, pack_struct, pack_mask, pool_text, pool_struct, pool_mask)
            loss = F.cross_entropy(scores, targets)

            total_loss += loss.item() * len(targets)
            preds = scores.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()

            top3 = scores.topk(min(3, scores.size(1)), dim=-1).indices
            total_top3 += (top3 == targets.unsqueeze(1)).any(dim=1).sum().item()

            total_samples += len(targets)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'top3_accuracy': total_top3 / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Extended v2 training")
    parser.add_argument("--sets", nargs="+", default=["FDN", "DSK", "BLB", "TLA"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)  # Higher initial LR
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--s3-bucket", type=str, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("Draft Model Training v2 - EXTENDED RUN")
    print("=" * 60)
    print(f"Sets: {args.sets}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr} (with warmup + cosine decay)")
    print(f"Dropout: {args.dropout}")
    print(f"Early stopping patience: {args.early_stopping_patience}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    dataset = V2DraftDataset(
        data_dir="data/17lands",
        sets=args.sets,
        max_samples=args.max_samples,
    )

    print("\nCreating train/val/test splits...")
    train_ds, val_ds, test_ds = create_splits(dataset)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_v2_picks, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v2_picks, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v2_picks, num_workers=4, pin_memory=True
    )

    print("\nInitializing model...")
    model = V2DraftModelWithDropout(dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate schedule: warmup + cosine decay
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    run_name = f"draft_v2_long_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"logs/{run_name}")

    print("\n" + "=" * 60)
    print("Starting extended training...")
    print("=" * 60)

    best_val_acc = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        # Log metrics
        writer.add_scalar("train/loss", train_metrics['loss'], epoch)
        writer.add_scalar("train/accuracy", train_metrics['accuracy'], epoch)
        writer.add_scalar("train/lr", train_metrics['lr'], epoch)
        writer.add_scalar("val/loss", val_metrics['loss'], epoch)
        writer.add_scalar("val/accuracy", val_metrics['accuracy'], epoch)
        writer.add_scalar("val/top3_accuracy", val_metrics['top3_accuracy'], epoch)

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) lr={train_metrics['lr']:.2e}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Top3: {val_metrics['top3_accuracy']:.4f}")

        # Check for improvement (use both accuracy AND loss)
        improved = False
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            improved = True
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            improved = True

        if improved:
            epochs_without_improvement = 0
            checkpoint_path = "checkpoints/draft_v2_long_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'encoder_version': 'v2_hybrid_extended',
            }, checkpoint_path)
            print(f"  [NEW BEST] acc={best_val_acc:.4f}, loss={best_val_loss:.4f}")

            if args.s3_bucket:
                upload_to_s3(checkpoint_path, args.s3_bucket, "checkpoints/best.pt")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0 and args.s3_bucket:
            periodic_path = f"checkpoints/v2_long_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
            }, periodic_path)
            upload_to_s3(periodic_path, args.s3_bucket, f"checkpoints/v2_long_epoch{epoch}.pt")

        # Early stopping
        if args.early_stopping_patience > 0:
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n[EARLY STOPPING] No improvement for {args.early_stopping_patience} epochs")
                break

    writer.close()

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    checkpoint = torch.load("checkpoints/draft_v2_long_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Test Top-3:    {test_metrics['top3_accuracy']:.4f} ({test_metrics['top3_accuracy']*100:.2f}%)")
    print(f"Test Loss:     {test_metrics['loss']:.4f}")

    results = {
        'encoder_version': 'v2_hybrid_extended',
        'test_accuracy': test_metrics['accuracy'],
        'test_top3_accuracy': test_metrics['top3_accuracy'],
        'test_loss': test_metrics['loss'],
        'best_val_accuracy': best_val_acc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'dropout': args.dropout,
            'warmup_epochs': args.warmup_epochs,
            'early_stopping_patience': args.early_stopping_patience,
        },
        'sets': args.sets,
        'max_samples': args.max_samples,
    }

    with open("checkpoints/v2_long_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    if args.s3_bucket:
        upload_to_s3("checkpoints/v2_long_results.json", args.s3_bucket, "checkpoints/final_results.json")
        upload_to_s3("checkpoints/draft_v2_long_best.pt", args.s3_bucket, "checkpoints/best.pt")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f}")
    print(f"Epochs trained: {epoch + 1}")
    if args.s3_bucket:
        print(f"S3: s3://{args.s3_bucket}/checkpoints/")


if __name__ == "__main__":
    main()
