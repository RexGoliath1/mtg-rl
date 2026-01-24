#!/usr/bin/env python3
"""
DEPRECATED: Use data_loader_17lands.py instead.

This module assumes a different CSV format than the actual 17lands public data.
The data_loader_17lands.py module handles the native 17lands wide-format CSVs
with pack_card_* and pool_* columns.

Migration:
    # Old
    from data_17lands import SeventeenLandsConfig, DraftDataset

    # New
    from data_loader_17lands import SeventeenLandsDataset, collate_picks, create_data_splits

---

17lands Data Integration for MTG Draft Training (LEGACY)

17lands.com provides free public datasets of draft picks from MTGA.
This module handles:
1. Downloading and caching 17lands data
2. Parsing draft pick CSV files
3. Creating training datasets for behavioral cloning
4. Data augmentation and preprocessing

Data format from 17lands:
- Draft pick data: CSV with columns for draft ID, pack cards, pick, pool, etc.
- Game data: CSV with match outcomes

Key insight: We can pre-train the draft policy using behavioral cloning
on millions of human draft picks before fine-tuning with RL in Forge.

Download data from: https://www.17lands.com/public_datasets
"""

import warnings
warnings.warn(
    "data_17lands.py is deprecated. Use data_loader_17lands.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import csv
import gzip
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SeventeenLandsConfig:
    """Configuration for 17lands data loading."""
    # Data paths
    data_dir: str = "data/17lands"
    cache_dir: str = "data/17lands/cache"

    # Set code for filtering (e.g., "NEO", "MKM", "BRO")
    set_code: Optional[str] = None

    # Filtering
    min_user_game_count: int = 10  # Filter out inexperienced drafters
    min_win_rate: float = 0.0      # Filter by drafter win rate (0.0 = all)
    max_win_rate: float = 1.0

    # Data splits
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    # Processing
    max_picks_per_file: int = 100000  # Limit for memory
    use_multiprocessing: bool = True
    num_workers: int = 4


@dataclass
class DraftPick:
    """Represents a single draft pick decision."""
    draft_id: str
    pack_number: int  # 1-3
    pick_number: int  # 1-15
    pack_cards: List[str]  # Cards in the pack
    pick: str              # Card that was picked
    pool: List[str]        # Cards drafted so far
    user_win_rate: float   # Drafter's overall win rate
    draft_time: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'draft_id': self.draft_id,
            'pack_number': self.pack_number,
            'pick_number': self.pick_number,
            'pack_cards': self.pack_cards,
            'pick': self.pick,
            'pool': self.pool,
            'user_win_rate': self.user_win_rate,
        }


class CardDatabase:
    """
    Database of card information for feature extraction.

    Loads card data from Scryfall or cached JSON.
    """

    def __init__(self, cache_path: str = "data/cards.json"):
        self.cache_path = cache_path
        self.cards: Dict[str, Dict] = {}
        self._load_or_fetch()

    def _load_or_fetch(self):
        """Load card database from cache or fetch from Scryfall."""
        if os.path.exists(self.cache_path):
            logger.info(f"Loading card database from {self.cache_path}")
            with open(self.cache_path, 'r') as f:
                self.cards = json.load(f)
            logger.info(f"Loaded {len(self.cards)} cards")
        else:
            logger.info("Card database not found, creating empty database")
            self.cards = {}

    def get_card(self, name: str) -> Optional[Dict]:
        """Get card data by name."""
        # Normalize name
        name_lower = name.lower().strip()

        if name_lower in self.cards:
            return self.cards[name_lower]

        # Try without set suffix (e.g., "Lightning Bolt (M21)")
        if '(' in name:
            base_name = name.split('(')[0].strip().lower()
            if base_name in self.cards:
                return self.cards[base_name]

        return None

    def add_card(self, name: str, data: Dict):
        """Add or update card data."""
        self.cards[name.lower().strip()] = data

    def save(self):
        """Save card database to cache."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cards, f, indent=2)


class SeventeenLandsParser:
    """
    Parses 17lands draft pick CSV files.

    Expected CSV format (from 17lands public datasets):
    - draft_id: Unique identifier for the draft
    - pack_number: 1, 2, or 3
    - pick_number: 1-15
    - pack_card_1 through pack_card_15: Cards in the pack
    - pick: The card that was picked
    - pool_1 through pool_N: Cards already in pool
    """

    def __init__(self, config: SeventeenLandsConfig = None):
        self.config = config or SeventeenLandsConfig()
        self.card_db = CardDatabase()

    def parse_csv(
        self,
        filepath: str,
        max_picks: Optional[int] = None
    ) -> Iterator[DraftPick]:
        """
        Parse a 17lands draft picks CSV file.

        Args:
            filepath: Path to CSV file (can be .csv or .csv.gz)
            max_picks: Maximum number of picks to parse

        Yields:
            DraftPick objects
        """
        open_fn = gzip.open if filepath.endswith('.gz') else open
        mode = 'rt' if filepath.endswith('.gz') else 'r'

        count = 0
        with open_fn(filepath, mode, encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                if max_picks and count >= max_picks:
                    break

                try:
                    pick = self._parse_row(row)
                    if pick is not None:
                        yield pick
                        count += 1
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

        logger.info(f"Parsed {count} picks from {filepath}")

    def _parse_row(self, row: Dict[str, str]) -> Optional[DraftPick]:
        """Parse a single CSV row into a DraftPick."""
        # Extract pack cards
        pack_cards = []
        for i in range(1, 16):
            col = f'pack_card_{i}'
            if col in row and row[col]:
                pack_cards.append(row[col])

        if not pack_cards:
            return None

        # Get the pick
        pick = row.get('pick', row.get('pick_1', ''))
        if not pick:
            return None

        # Extract pool
        pool = []
        for i in range(1, 46):  # Max 45 cards in pool
            col = f'pool_{i}'
            if col in row and row[col]:
                pool.append(row[col])

        # Get pack/pick numbers
        try:
            pack_number = int(row.get('pack_number', 1))
            pick_number = int(row.get('pick_number', 1))
        except ValueError:
            pack_number = 1
            pick_number = 1

        # Get user win rate if available
        try:
            user_win_rate = float(row.get('user_game_win_rate_bucket', 0.5))
        except ValueError:
            user_win_rate = 0.5

        return DraftPick(
            draft_id=row.get('draft_id', ''),
            pack_number=pack_number,
            pick_number=pick_number,
            pack_cards=pack_cards,
            pick=pick,
            pool=pool,
            user_win_rate=user_win_rate,
            draft_time=row.get('draft_time'),
        )

    def filter_pick(self, pick: DraftPick) -> bool:
        """Check if a pick passes the filter criteria."""
        # Check win rate filter
        if pick.user_win_rate < self.config.min_win_rate:
            return False
        if pick.user_win_rate > self.config.max_win_rate:
            return False

        return True


class DraftDataset(Dataset):
    """
    PyTorch Dataset for draft pick training.

    Loads and preprocesses 17lands data for behavioral cloning.
    """

    def __init__(
        self,
        picks: List[DraftPick],
        feature_extractor,
        card_db: CardDatabase,
        max_pack_size: int = 15,
        max_pool_size: int = 45,
    ):
        self.picks = picks
        self.feature_extractor = feature_extractor
        self.card_db = card_db
        self.max_pack_size = max_pack_size
        self.max_pool_size = max_pool_size
        self.input_dim = feature_extractor.config.input_dim

    def __len__(self) -> int:
        return len(self.picks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pick = self.picks[idx]

        # Extract pack features
        pack_features = np.zeros((self.max_pack_size, self.input_dim), dtype=np.float32)
        pack_mask = np.zeros(self.max_pack_size, dtype=np.float32)

        for i, card_name in enumerate(pick.pack_cards[:self.max_pack_size]):
            card_data = self._get_card_data(card_name)
            pack_features[i] = self.feature_extractor.extract(card_data)
            pack_mask[i] = 1.0

        # Find the index of the picked card
        pick_index = -1
        for i, card_name in enumerate(pick.pack_cards):
            if card_name == pick.pick:
                pick_index = i
                break

        if pick_index == -1:
            # Pick not found in pack (data error) - use first card
            pick_index = 0

        # Extract pool features
        pool_features = np.zeros((self.max_pool_size, self.input_dim), dtype=np.float32)
        pool_mask = np.zeros(self.max_pool_size, dtype=np.float32)

        for i, card_name in enumerate(pick.pool[:self.max_pool_size]):
            card_data = self._get_card_data(card_name)
            pool_features[i] = self.feature_extractor.extract(card_data)
            pool_mask[i] = 1.0

        # If pool is empty, add a dummy entry
        if not pick.pool:
            pool_mask[0] = 0.0  # Still masked, but avoids empty tensor issues

        return {
            'pack_features': torch.tensor(pack_features, dtype=torch.float32),
            'pool_features': torch.tensor(pool_features, dtype=torch.float32),
            'pack_mask': torch.tensor(pack_mask, dtype=torch.float32),
            'pool_mask': torch.tensor(pool_mask, dtype=torch.float32),
            'pick_index': torch.tensor(pick_index, dtype=torch.long),
            'pack_number': torch.tensor(pick.pack_number, dtype=torch.long),
            'pick_number': torch.tensor(pick.pick_number, dtype=torch.long),
        }

    def _get_card_data(self, card_name: str) -> Dict:
        """Get card data for feature extraction."""
        card_data = self.card_db.get_card(card_name)
        if card_data:
            return card_data

        # Create minimal card data from name
        return {
            'name': card_name,
            'mana_cost': '',
            'type': '',
            'keywords': [],
            'rarity': 'common',
        }


class StreamingDraftDataset(IterableDataset):
    """
    Streaming dataset for large 17lands files.

    Useful when the data doesn't fit in memory.
    """

    def __init__(
        self,
        filepaths: List[str],
        feature_extractor,
        card_db: CardDatabase,
        parser: SeventeenLandsParser,
        max_pack_size: int = 15,
        max_pool_size: int = 45,
        shuffle_buffer_size: int = 10000,
    ):
        self.filepaths = filepaths
        self.feature_extractor = feature_extractor
        self.card_db = card_db
        self.parser = parser
        self.max_pack_size = max_pack_size
        self.max_pool_size = max_pool_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.input_dim = feature_extractor.config.input_dim

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through all files and yield processed samples."""
        buffer = []

        for filepath in self.filepaths:
            for pick in self.parser.parse_csv(filepath):
                if not self.parser.filter_pick(pick):
                    continue

                sample = self._process_pick(pick)
                buffer.append(sample)

                # Shuffle buffer when full
                if len(buffer) >= self.shuffle_buffer_size:
                    np.random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        # Yield remaining items
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item

    def _process_pick(self, pick: DraftPick) -> Dict[str, torch.Tensor]:
        """Process a single pick into tensors."""
        pack_features = np.zeros((self.max_pack_size, self.input_dim), dtype=np.float32)
        pack_mask = np.zeros(self.max_pack_size, dtype=np.float32)

        for i, card_name in enumerate(pick.pack_cards[:self.max_pack_size]):
            card_data = self._get_card_data(card_name)
            pack_features[i] = self.feature_extractor.extract(card_data)
            pack_mask[i] = 1.0

        pick_index = 0
        for i, card_name in enumerate(pick.pack_cards):
            if card_name == pick.pick:
                pick_index = i
                break

        pool_features = np.zeros((self.max_pool_size, self.input_dim), dtype=np.float32)
        pool_mask = np.zeros(self.max_pool_size, dtype=np.float32)

        for i, card_name in enumerate(pick.pool[:self.max_pool_size]):
            card_data = self._get_card_data(card_name)
            pool_features[i] = self.feature_extractor.extract(card_data)
            pool_mask[i] = 1.0

        return {
            'pack_features': torch.tensor(pack_features, dtype=torch.float32),
            'pool_features': torch.tensor(pool_features, dtype=torch.float32),
            'pack_mask': torch.tensor(pack_mask, dtype=torch.float32),
            'pool_mask': torch.tensor(pool_mask, dtype=torch.float32),
            'pick_index': torch.tensor(pick_index, dtype=torch.long),
            'pack_number': torch.tensor(pick.pack_number, dtype=torch.long),
            'pick_number': torch.tensor(pick.pick_number, dtype=torch.long),
        }

    def _get_card_data(self, card_name: str) -> Dict:
        card_data = self.card_db.get_card(card_name)
        if card_data:
            return card_data
        return {
            'name': card_name,
            'mana_cost': '',
            'type': '',
            'keywords': [],
            'rarity': 'common',
        }


def download_17lands_data(
    set_code: str,
    output_dir: str = "data/17lands",
    data_type: str = "draft"
) -> Optional[str]:
    """
    Download 17lands data for a specific set.

    Note: 17lands data must be manually downloaded from their website.
    This function checks for existing data and provides instructions.

    Args:
        set_code: Set code (e.g., "NEO", "MKM")
        output_dir: Directory to save data
        data_type: "draft" or "game"

    Returns:
        Path to the data file, or None if not found
    """
    os.makedirs(output_dir, exist_ok=True)

    # Expected filename patterns
    patterns = [
        f"{data_type}_data_public.{set_code}.csv.gz",
        f"{data_type}_data_public.{set_code}.csv",
        f"{set_code}_{data_type}.csv.gz",
        f"{set_code}_{data_type}.csv",
    ]

    # Check for existing files
    for pattern in patterns:
        filepath = os.path.join(output_dir, pattern)
        if os.path.exists(filepath):
            logger.info(f"Found existing data file: {filepath}")
            return filepath

    # Data not found - provide instructions
    logger.warning(f"17lands data for {set_code} not found in {output_dir}")
    logger.info("Please download data from https://www.17lands.com/public_datasets")
    logger.info(f"Save the {data_type} data file to: {output_dir}")

    return None


def create_data_splits(
    picks: List[DraftPick],
    config: SeventeenLandsConfig,
) -> Tuple[List[DraftPick], List[DraftPick], List[DraftPick]]:
    """
    Split picks into train/val/test sets by draft ID.

    Splits by draft_id to avoid data leakage (all picks from a draft
    go into the same split).
    """
    # Group by draft ID
    drafts = defaultdict(list)
    for pick in picks:
        drafts[pick.draft_id].append(pick)

    draft_ids = list(drafts.keys())
    np.random.shuffle(draft_ids)

    # Calculate split indices
    n_drafts = len(draft_ids)
    train_end = int(n_drafts * config.train_ratio)
    val_end = int(n_drafts * (config.train_ratio + config.val_ratio))

    train_ids = set(draft_ids[:train_end])
    val_ids = set(draft_ids[train_end:val_end])
    test_ids = set(draft_ids[val_end:])

    # Collect picks for each split
    train_picks = [p for p in picks if p.draft_id in train_ids]
    val_picks = [p for p in picks if p.draft_id in val_ids]
    test_picks = [p for p in picks if p.draft_id in test_ids]

    logger.info(f"Data splits: train={len(train_picks)}, val={len(val_picks)}, test={len(test_picks)}")

    return train_picks, val_picks, test_picks


def create_synthetic_picks(n_picks: int = 1000) -> List[DraftPick]:
    """
    Create synthetic draft picks for testing.

    Useful when 17lands data is not available.
    """
    picks = []

    # Fake card names
    card_pool = [
        "Lightning Bolt", "Counterspell", "Giant Growth", "Dark Ritual", "Swords to Plowshares",
        "Llanowar Elves", "Birds of Paradise", "Sol Ring", "Path to Exile", "Brainstorm",
        "Force of Will", "Mana Drain", "Ancestral Recall", "Black Lotus", "Time Walk",
        "Goblin Guide", "Monastery Swiftspear", "Delver of Secrets", "Tarmogoyf", "Thoughtseize",
        "Fatal Push", "Aether Vial", "Stoneforge Mystic", "Snapcaster Mage", "Dark Confidant",
        "Mountain", "Island", "Forest", "Swamp", "Plains",
    ]

    for i in range(n_picks):
        pack_num = (i % 45) // 15 + 1
        pick_num = (i % 15) + 1
        pool_size = (i % 45)

        # Generate pack
        pack_size = 16 - pick_num
        pack_cards = np.random.choice(card_pool, min(pack_size, len(card_pool)), replace=False).tolist()

        # Pick a card
        pick = np.random.choice(pack_cards)

        # Generate pool
        pool = np.random.choice(card_pool, min(pool_size, len(card_pool)), replace=False).tolist() if pool_size > 0 else []

        picks.append(DraftPick(
            draft_id=f"synthetic_{i // 45}",
            pack_number=pack_num,
            pick_number=pick_num,
            pack_cards=pack_cards,
            pick=pick,
            pool=pool,
            user_win_rate=0.5 + np.random.randn() * 0.1,
        ))

    return picks


def test_17lands_integration():
    """Test the 17lands data loading pipeline."""
    print("Testing 17lands Data Integration")
    print("=" * 60)

    # Create synthetic data for testing
    print("\nCreating synthetic picks...")
    picks = create_synthetic_picks(1000)
    print(f"Created {len(picks)} synthetic picks")

    # Test parser
    print("\nTesting data splits...")
    config = SeventeenLandsConfig()
    train, val, test = create_data_splits(picks, config)
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Test dataset
    print("\nTesting DraftDataset...")
    from shared_card_encoder import CardFeatureExtractor, CardEncoderConfig
    extractor = CardFeatureExtractor(CardEncoderConfig())
    card_db = CardDatabase("data/cards.json")

    dataset = DraftDataset(train[:100], extractor, card_db)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Pack features shape: {sample['pack_features'].shape}")
    print(f"Pool features shape: {sample['pool_features'].shape}")
    print(f"Pick index: {sample['pick_index']}")

    # Test DataLoader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch pack_features shape: {batch['pack_features'].shape}")
    print(f"Batch pick_index shape: {batch['pick_index'].shape}")

    print("\n" + "=" * 60)
    print("17lands integration test completed!")


if __name__ == "__main__":
    test_17lands_integration()
