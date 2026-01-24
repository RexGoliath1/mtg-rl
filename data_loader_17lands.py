#!/usr/bin/env python3
"""
17lands Data Loader

Loads draft data from 17lands CSV files in their native wide format.
Each row is a pick with pack_card_* and pool_* columns indicating
which cards are available and already drafted.
"""

import gzip
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import json
import re


@dataclass
class DraftPick:
    """Represents a single draft pick."""
    draft_id: str
    pack_number: int
    pick_number: int
    picked_card: str
    pack_cards: List[str]      # Cards available in pack
    pool_cards: Dict[str, int]  # Cards already drafted (name -> count)
    rank: str
    event_wins: int
    event_losses: int


class SeventeenLandsDataset(Dataset):
    """
    Dataset for 17lands draft data.

    Loads data into memory for fast access during training.
    Use StreamingSeventeenLandsDataset for larger datasets.
    """

    def __init__(
        self,
        data_dir: str,
        sets: List[str],
        max_samples: Optional[int] = None,
        min_rank: Optional[str] = None,  # Filter by rank (e.g., "platinum")
    ):
        self.data_dir = Path(data_dir)
        self.sets = sets
        self.samples: List[DraftPick] = []
        self.card_to_idx: Dict[str, int] = {}
        self.idx_to_card: Dict[int, str] = {}

        self._load_data(max_samples, min_rank)

    def _load_data(self, max_samples: Optional[int], min_rank: Optional[str]):
        """Load data from CSV files."""
        rank_order = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'mythic']
        min_rank_idx = rank_order.index(min_rank.lower()) if min_rank else 0

        for set_code in self.sets:
            csv_path = self.data_dir / f"draft_data_public.{set_code}.PremierDraft.csv.gz"

            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue

            print(f"Loading {set_code}...")
            count = 0

            with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Get card columns from header
                pack_cols = [c for c in reader.fieldnames if c.startswith('pack_card_')]
                pool_cols = [c for c in reader.fieldnames if c.startswith('pool_')]

                # Build card vocabulary
                for col in pack_cols:
                    card_name = col.replace('pack_card_', '')
                    if card_name not in self.card_to_idx:
                        idx = len(self.card_to_idx)
                        self.card_to_idx[card_name] = idx
                        self.idx_to_card[idx] = card_name

                for row in reader:
                    # Filter by rank if specified
                    rank = row.get('rank', 'bronze').lower()
                    if rank in rank_order:
                        rank_idx = rank_order.index(rank)
                        if rank_idx < min_rank_idx:
                            continue

                    # Extract pack cards (where value == '1')
                    pack_cards = []
                    for col in pack_cols:
                        if row.get(col) == '1':
                            card_name = col.replace('pack_card_', '')
                            pack_cards.append(card_name)

                    # Skip if pack is empty (shouldn't happen)
                    if not pack_cards:
                        continue

                    # Extract pool cards (count > 0)
                    pool_cards = {}
                    for col in pool_cols:
                        count_str = row.get(col, '0')
                        try:
                            count_val = int(count_str)
                            if count_val > 0:
                                card_name = col.replace('pool_', '')
                                pool_cards[card_name] = count_val
                        except ValueError:
                            pass

                    # Get picked card
                    picked = row.get('pick', '')
                    if not picked or picked not in pack_cards:
                        continue

                    # Create sample
                    sample = DraftPick(
                        draft_id=row.get('draft_id', ''),
                        pack_number=int(row.get('pack_number', 0)),
                        pick_number=int(row.get('pick_number', 0)),
                        picked_card=picked,
                        pack_cards=pack_cards,
                        pool_cards=pool_cards,
                        rank=rank,
                        event_wins=int(row.get('event_match_wins', 0)),
                        event_losses=int(row.get('event_match_losses', 0)),
                    )

                    self.samples.append(sample)
                    count += 1

                    if max_samples and len(self.samples) >= max_samples:
                        break

            print(f"  Loaded {count} picks from {set_code}")

            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"Total: {len(self.samples)} picks, {len(self.card_to_idx)} unique cards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Create pack tensor (one-hot for each card in vocabulary)
        pack_size = len(sample.pack_cards)
        pack_indices = [self.card_to_idx[c] for c in sample.pack_cards]

        # Create pool tensor (counts for each card)
        pool_tensor = torch.zeros(len(self.card_to_idx))
        for card, count in sample.pool_cards.items():
            if card in self.card_to_idx:
                pool_tensor[self.card_to_idx[card]] = count

        # Get pick index within pack
        pick_idx = sample.pack_cards.index(sample.picked_card)

        return {
            'pack_indices': torch.tensor(pack_indices, dtype=torch.long),
            'pack_size': pack_size,
            'pool_tensor': pool_tensor,
            'pick_idx': pick_idx,
            'pack_number': sample.pack_number,
            'pick_number': sample.pick_number,
        }


class StreamingSeventeenLandsDataset(IterableDataset):
    """
    Streaming dataset for large 17lands files.

    Reads directly from gzipped CSV without loading all into memory.
    """

    def __init__(
        self,
        csv_path: str,
        card_to_idx: Dict[str, int],
        min_rank: Optional[str] = None,
        shuffle_buffer: int = 10000,
    ):
        self.csv_path = Path(csv_path)
        self.card_to_idx = card_to_idx
        self.min_rank = min_rank
        self.shuffle_buffer = shuffle_buffer

        rank_order = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'mythic']
        self.min_rank_idx = rank_order.index(min_rank.lower()) if min_rank else 0
        self.rank_order = rank_order

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []

        with gzip.open(self.csv_path, 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            pack_cols = [c for c in reader.fieldnames if c.startswith('pack_card_')]
            pool_cols = [c for c in reader.fieldnames if c.startswith('pool_')]

            for row in reader:
                # Filter by rank
                rank = row.get('rank', 'bronze').lower()
                if rank in self.rank_order:
                    rank_idx = self.rank_order.index(rank)
                    if rank_idx < self.min_rank_idx:
                        continue

                # Extract pack cards
                pack_cards = []
                for col in pack_cols:
                    if row.get(col) == '1':
                        card_name = col.replace('pack_card_', '')
                        if card_name in self.card_to_idx:
                            pack_cards.append(card_name)

                if not pack_cards:
                    continue

                # Get picked card
                picked = row.get('pick', '')
                if not picked or picked not in pack_cards:
                    continue

                # Extract pool
                pool_tensor = torch.zeros(len(self.card_to_idx))
                for col in pool_cols:
                    try:
                        count = int(row.get(col, '0'))
                        if count > 0:
                            card_name = col.replace('pool_', '')
                            if card_name in self.card_to_idx:
                                pool_tensor[self.card_to_idx[card_name]] = count
                    except ValueError:
                        pass

                # Create sample
                pack_indices = [self.card_to_idx[c] for c in pack_cards]
                pick_idx = pack_cards.index(picked)

                sample = {
                    'pack_indices': torch.tensor(pack_indices, dtype=torch.long),
                    'pack_size': len(pack_cards),
                    'pool_tensor': pool_tensor,
                    'pick_idx': pick_idx,
                    'pack_number': int(row.get('pack_number', 0)),
                    'pick_number': int(row.get('pick_number', 0)),
                }

                # Shuffle buffer
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer:
                    np.random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        # Yield remaining
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item


def collate_picks(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads pack indices to same length and creates attention masks.
    """
    max_pack_size = max(b['pack_size'] for b in batch)
    batch_size = len(batch)

    # Pad pack indices
    pack_indices = torch.zeros(batch_size, max_pack_size, dtype=torch.long)
    pack_mask = torch.zeros(batch_size, max_pack_size)

    for i, b in enumerate(batch):
        size = b['pack_size']
        pack_indices[i, :size] = b['pack_indices']
        pack_mask[i, :size] = 1

    # Stack pool tensors
    pool_tensors = torch.stack([b['pool_tensor'] for b in batch])

    # Pick indices
    pick_indices = torch.tensor([b['pick_idx'] for b in batch], dtype=torch.long)

    return {
        'pack_indices': pack_indices,
        'pack_mask': pack_mask,
        'pool_tensor': pool_tensors,
        'pick_idx': pick_indices,
    }


def build_card_vocabulary(data_dir: str, sets: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build card vocabulary from CSV headers.

    Returns:
        card_to_idx: Dict mapping card name to index
        idx_to_card: Dict mapping index to card name
    """
    card_to_idx = {}
    idx_to_card = {}

    for set_code in sets:
        csv_path = Path(data_dir) / f"draft_data_public.{set_code}.PremierDraft.csv.gz"

        if not csv_path.exists():
            continue

        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

            for col in header:
                if col.startswith('pack_card_'):
                    card_name = col.replace('pack_card_', '')
                    if card_name not in card_to_idx:
                        idx = len(card_to_idx)
                        card_to_idx[card_name] = idx
                        idx_to_card[idx] = card_name

    return card_to_idx, idx_to_card


def test_data_loader():
    """Test the data loader."""
    print("Testing 17lands Data Loader")
    print("=" * 60)

    data_dir = "data/17lands"
    sets = ["FDN"]

    # Test dataset
    print("\nLoading dataset...")
    dataset = SeventeenLandsDataset(
        data_dir=data_dir,
        sets=sets,
        max_samples=1000,
        min_rank="gold",  # Only gold+ players
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.card_to_idx)}")

    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Pack size: {sample['pack_size']}")
    print(f"Pick index: {sample['pick_idx']}")
    print(f"Pack indices shape: {sample['pack_indices'].shape}")
    print(f"Pool tensor shape: {sample['pool_tensor'].shape}")

    # Test DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_picks,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Pack indices shape: {batch['pack_indices'].shape}")
    print(f"Pack mask shape: {batch['pack_mask'].shape}")
    print(f"Pool tensor shape: {batch['pool_tensor'].shape}")
    print(f"Pick indices shape: {batch['pick_idx'].shape}")

    print("\n" + "=" * 60)
    print("Data loader test complete!")


if __name__ == "__main__":
    test_data_loader()
