#!/usr/bin/env python3
"""
Draft Model Training v2 - With Hybrid Card Encoder

Uses pre-computed text embeddings + structural features for better
generalization to new mechanics.

Usage:
    # Quick local test (1K samples, 3 epochs)
    python train_draft_v2.py --quick

    # Full local training
    python train_draft_v2.py --sets FDN DSK BLB TLA --epochs 20

    # Cloud training
    python train_draft_v2.py --sets FDN DSK BLB TLA --epochs 50 \
        --s3-bucket mtg-rl-checkpoints
"""

import argparse
import json
import gzip
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.models.hybrid_card_encoder import HybridCardEncoder, HybridEncoderConfig, StructuralFeatureExtractor


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
    """
    Draft dataset with pre-computed embeddings and structural features.

    Loads:
    - 17lands pick data (who picked what)
    - Pre-computed text embeddings (from Scryfall oracle text)
    - Structural features (mana, types, stats)
    """

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

        # Load card metadata
        print(f"Loading card metadata from {metadata_path}...")
        with open(metadata_path) as f:
            self.card_metadata = json.load(f)
        print(f"  Loaded metadata for {len(self.card_metadata)} cards")

        # Load pre-computed text embeddings
        print(f"Loading text embeddings from {embeddings_path}...")
        embedding_data = torch.load(embeddings_path, weights_only=False)
        self.text_embeddings = embedding_data['embeddings']
        self.embedding_dim = embedding_data['embedding_dim']
        print(f"  Loaded embeddings: {len(self.text_embeddings)} cards, {self.embedding_dim}d")

        # Build card vocabulary from metadata
        self.card_to_idx: Dict[str, int] = {}
        self.idx_to_card: Dict[int, str] = {}
        for i, name in enumerate(self.card_metadata.keys()):
            self.card_to_idx[name] = i
            self.idx_to_card[i] = name

        # Initialize structural feature extractor
        config = HybridEncoderConfig()
        self.feature_extractor = StructuralFeatureExtractor(config)

        # Pre-compute structural features for all cards
        print("Pre-computing structural features...")
        self.structural_features = {}
        for name, metadata in self.card_metadata.items():
            self.structural_features[name] = self.feature_extractor.extract(metadata)
        self.structural_dim = config.structural_dim

        # Load draft data
        print(f"Loading draft data for sets: {sets}...")
        self.samples: List[Dict] = []
        self._load_draft_data(max_samples, min_rank)
        print(f"  Total samples: {len(self.samples)}")

    def _load_draft_data(self, max_samples: Optional[int], min_rank: str):
        """Load draft picks from 17lands CSV files."""
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

                # Get card columns
                pack_cols = [c for c in reader.fieldnames if c.startswith('pack_card_')]
                pool_cols = [c for c in reader.fieldnames if c.startswith('pool_')]

                for row in reader:
                    # Filter by rank
                    rank = row.get('rank', 'bronze').lower()
                    if rank in rank_order and rank_order.index(rank) < min_rank_idx:
                        continue

                    # Extract pack cards
                    pack_cards = []
                    for col in pack_cols:
                        if row.get(col) == '1':
                            card_name = col.replace('pack_card_', '')
                            if card_name in self.card_metadata:
                                pack_cards.append(card_name)

                    # Extract picked card
                    picked_card = row.get('pick')
                    if not picked_card or picked_card not in self.card_metadata:
                        continue
                    if picked_card not in pack_cards:
                        continue

                    # Extract pool cards
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pack_text_emb: [num_pack_cards, embedding_dim]
            pack_structural: [num_pack_cards, structural_dim]
            picked_idx: int (index of picked card in pack)
            pool_text_emb: [num_pool_cards, embedding_dim]
            pool_structural: [num_pool_cards, structural_dim]
        """
        sample = self.samples[idx]

        # Pack embeddings
        pack_text = torch.stack([
            self.text_embeddings[name] for name in sample['pack_cards']
        ])
        pack_struct = torch.from_numpy(np.array([
            self.structural_features[name] for name in sample['pack_cards']
        ], dtype=np.float32))

        # Pool embeddings
        if sample['pool_cards']:
            pool_text = torch.stack([
                self.text_embeddings[name] for name in sample['pool_cards']
            ])
            pool_struct = torch.from_numpy(np.array([
                self.structural_features[name] for name in sample['pool_cards']
            ], dtype=np.float32))
        else:
            # Empty pool at start of draft
            pool_text = torch.zeros(1, self.embedding_dim)
            pool_struct = torch.zeros(1, self.structural_dim)

        return (
            pack_text,
            pack_struct,
            sample['picked_idx'],
            pool_text,
            pool_struct,
        )


def collate_v2_picks(batch):
    """
    Collate function that handles variable pack/pool sizes.
    Pads to max size in batch.
    """
    pack_texts, pack_structs, picked_idxs, pool_texts, pool_structs = zip(*batch)

    # Find max sizes
    max_pack = max(pt.size(0) for pt in pack_texts)
    max_pool = max(pt.size(0) for pt in pool_texts)

    batch_size = len(batch)
    text_dim = pack_texts[0].size(1)
    struct_dim = pack_structs[0].size(1)

    # Pad pack tensors
    pack_text_padded = torch.zeros(batch_size, max_pack, text_dim)
    pack_struct_padded = torch.zeros(batch_size, max_pack, struct_dim)
    pack_mask = torch.zeros(batch_size, max_pack)

    for i, (pt, ps) in enumerate(zip(pack_texts, pack_structs)):
        n = pt.size(0)
        pack_text_padded[i, :n] = pt
        pack_struct_padded[i, :n] = ps
        pack_mask[i, :n] = 1

    # Pad pool tensors
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


class V2DraftModel(nn.Module):
    """
    Draft model using v2 hybrid card encoder.

    Architecture:
    1. Encode pack cards with hybrid encoder
    2. Encode pool cards with hybrid encoder
    3. Pool pool cards to single representation
    4. Score each pack card against pool context
    """

    def __init__(self, encoder_config: Optional[HybridEncoderConfig] = None):
        super().__init__()

        self.encoder_config = encoder_config or HybridEncoderConfig()
        self.encoder = HybridCardEncoder(self.encoder_config)

        # Pool context aggregation
        self.pool_aggregator = nn.Sequential(
            nn.Linear(self.encoder_config.output_dim, self.encoder_config.output_dim),
            nn.LayerNorm(self.encoder_config.output_dim),
            nn.GELU(),
        )

        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(self.encoder_config.output_dim * 2, self.encoder_config.output_dim),
            nn.GELU(),
            nn.Linear(self.encoder_config.output_dim, 1),
        )

    def forward(
        self,
        pack_text: torch.Tensor,
        pack_struct: torch.Tensor,
        pack_mask: torch.Tensor,
        pool_text: torch.Tensor,
        pool_struct: torch.Tensor,
        pool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pick scores for pack cards.

        Returns:
            [batch, max_pack] scores for each card
        """
        # Encode pack cards
        pack_emb = self.encoder(
            pack_text, pack_struct, pack_mask, use_interactions=True
        )

        # Encode pool cards
        pool_emb = self.encoder(
            pool_text, pool_struct, pool_mask, use_interactions=True
        )

        # Aggregate pool to single vector (masked mean)
        pool_mask_exp = pool_mask.unsqueeze(-1)
        pool_sum = (pool_emb * pool_mask_exp).sum(dim=1)
        pool_count = pool_mask_exp.sum(dim=1).clamp(min=1)
        pool_context = pool_sum / pool_count
        pool_context = self.pool_aggregator(pool_context)

        # Score each pack card against pool context
        batch_size, max_pack, emb_dim = pack_emb.shape
        pool_context_exp = pool_context.unsqueeze(1).expand(-1, max_pack, -1)

        combined = torch.cat([pack_emb, pool_context_exp], dim=-1)
        scores = self.scorer(combined).squeeze(-1)

        # Mask invalid positions
        scores = scores.masked_fill(pack_mask == 0, float('-inf'))

        return scores


def create_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Create train/val/test splits."""
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


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        # Move to device
        pack_text = batch['pack_text'].to(device)
        pack_struct = batch['pack_struct'].to(device)
        pack_mask = batch['pack_mask'].to(device)
        pool_text = batch['pool_text'].to(device)
        pool_struct = batch['pool_struct'].to(device)
        pool_mask = batch['pool_mask'].to(device)
        targets = batch['picked_idx'].to(device)

        # Forward pass
        optimizer.zero_grad()
        scores = model(
            pack_text, pack_struct, pack_mask,
            pool_text, pool_struct, pool_mask
        )

        # Cross-entropy loss
        loss = F.cross_entropy(scores, targets)
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * len(targets)
        preds = scores.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += len(targets)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


def validate(model, loader, device):
    """Validate on a data loader."""
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

            scores = model(
                pack_text, pack_struct, pack_mask,
                pool_text, pool_struct, pool_mask
            )

            loss = F.cross_entropy(scores, targets)

            total_loss += loss.item() * len(targets)
            preds = scores.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()

            # Top-3 accuracy
            top3 = scores.topk(min(3, scores.size(1)), dim=-1).indices
            total_top3 += (top3 == targets.unsqueeze(1)).any(dim=1).sum().item()

            total_samples += len(targets)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'top3_accuracy': total_top3 / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Draft model training v2")
    parser.add_argument("--sets", nargs="+", default=["FDN", "DSK", "BLB", "TLA"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true", help="Quick test (1K samples, 3 epochs)")
    parser.add_argument("--s3-bucket", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Stop if no improvement for N epochs (0=disabled)")

    args = parser.parse_args()

    if args.quick:
        args.max_samples = 1000
        args.epochs = 3

    print("=" * 60)
    print("Draft Model Training v2 (Hybrid Encoder)")
    print("=" * 60)
    print(f"Sets: {args.sets}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading data...")
    dataset = V2DraftDataset(
        data_dir="data/17lands",
        sets=args.sets,
        max_samples=args.max_samples,
    )

    # Create splits
    print("\nCreating train/val/test splits...")
    train_ds, val_ds, test_ds = create_splits(dataset)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Data loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_v2_picks, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v2_picks, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v2_picks, num_workers=0
    )

    # Model
    print("\nInitializing model...")
    model = V2DraftModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    run_name = f"draft_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"logs/{run_name}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_acc = 0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        # Log metrics
        writer.add_scalar("train/loss", train_metrics['loss'], epoch)
        writer.add_scalar("train/accuracy", train_metrics['accuracy'], epoch)
        writer.add_scalar("val/loss", val_metrics['loss'], epoch)
        writer.add_scalar("val/accuracy", val_metrics['accuracy'], epoch)
        writer.add_scalar("val/top3_accuracy", val_metrics['top3_accuracy'], epoch)

        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Top3: {val_metrics['top3_accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            epochs_without_improvement = 0
            checkpoint_path = "checkpoints/draft_v2_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'encoder_version': 'v2_hybrid',
            }, checkpoint_path)
            print(f"  [NEW BEST] accuracy={best_val_acc:.4f}")

            if args.s3_bucket:
                upload_to_s3(checkpoint_path, args.s3_bucket, "checkpoints/best.pt")
                upload_to_s3(checkpoint_path, args.s3_bucket, f"checkpoints/v2_epoch{epoch}.pt")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 and args.s3_bucket:
            periodic_path = f"checkpoints/v2_checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, periodic_path)
            upload_to_s3(periodic_path, args.s3_bucket, f"checkpoints/v2_epoch{epoch}.pt")

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

    checkpoint = torch.load("checkpoints/draft_v2_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Test Top-3:    {test_metrics['top3_accuracy']:.4f} ({test_metrics['top3_accuracy']*100:.2f}%)")
    print(f"Test Loss:     {test_metrics['loss']:.4f}")

    # Save results
    results = {
        'encoder_version': 'v2_hybrid',
        'test_accuracy': test_metrics['accuracy'],
        'test_top3_accuracy': test_metrics['top3_accuracy'],
        'test_loss': test_metrics['loss'],
        'best_val_accuracy': best_val_acc,
        'epochs': args.epochs,
        'sets': args.sets,
        'max_samples': args.max_samples,
    }

    with open("checkpoints/v2_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    if args.s3_bucket:
        upload_to_s3("checkpoints/v2_results.json", args.s3_bucket, "checkpoints/final_results.json")
        upload_to_s3("checkpoints/draft_v2_best.pt", args.s3_bucket, "checkpoints/best.pt")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val: {best_val_acc:.4f}")
    print(f"Test: {test_metrics['accuracy']:.4f}")
    print("Model: checkpoints/draft_v2_best.pt")
    if args.s3_bucket:
        print(f"S3: s3://{args.s3_bucket}/checkpoints/")


if __name__ == "__main__":
    main()
