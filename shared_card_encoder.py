#!/usr/bin/env python3
"""
Shared Card Encoder for MTG RL

This module implements a shared card encoder that can be used across:
- Draft training (learning card quality and synergy)
- Gameplay training (understanding board state)
- Deckbuilding (evaluating card choices)

The encoder is designed to be pre-trained on 17lands draft data and then
transferred to gameplay training, similar to how language models are
pre-trained on text and fine-tuned for specific tasks.

Architecture inspired by:
- BERT's pre-training approach
- Set Transformers for permutation-invariant representations
- AlphaStar's entity encoder
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class CardEncoderConfig:
    """Configuration for the shared card encoder."""
    # Input feature dimensions
    keyword_dim: int = 40      # One-hot keywords
    mana_dim: int = 11         # Mana cost features
    type_dim: int = 30         # Card type features
    stats_dim: int = 8         # P/T, loyalty, etc.
    rarity_dim: int = 5        # Common, Uncommon, Rare, Mythic, Special

    # Encoder architecture
    d_model: int = 256         # Internal representation dimension
    n_heads: int = 4           # Attention heads for card interactions
    n_layers: int = 2          # Number of transformer layers
    d_ff: int = 512            # Feed-forward hidden dimension
    dropout: float = 0.1

    # Card name embedding (optional)
    use_card_names: bool = False
    vocab_size: int = 50000    # For card name tokenization
    max_name_length: int = 10  # Max tokens in card name

    # Output
    output_dim: int = 256      # Final card embedding dimension

    @property
    def input_dim(self) -> int:
        """Total input feature dimension."""
        return self.keyword_dim + self.mana_dim + self.type_dim + self.stats_dim + self.rarity_dim


class CardFeatureProjection(nn.Module):
    """Projects raw card features to embedding space."""

    def __init__(self, config: CardEncoderConfig):
        super().__init__()
        self.config = config

        # Separate projections for different feature types
        # This allows the model to learn appropriate representations for each
        self.keyword_proj = nn.Linear(config.keyword_dim, config.d_model // 4)
        self.mana_proj = nn.Linear(config.mana_dim, config.d_model // 4)
        self.type_proj = nn.Linear(config.type_dim, config.d_model // 4)
        self.stats_proj = nn.Linear(config.stats_dim + config.rarity_dim, config.d_model // 4)

        # Combine projections
        self.combine = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, card_features: torch.Tensor) -> torch.Tensor:
        """
        Project card features to embedding space.

        Args:
            card_features: [batch, num_cards, input_dim] raw features

        Returns:
            [batch, num_cards, d_model] projected embeddings
        """
        # Split features by type
        idx = 0
        keywords = card_features[..., idx:idx + self.config.keyword_dim]
        idx += self.config.keyword_dim

        mana = card_features[..., idx:idx + self.config.mana_dim]
        idx += self.config.mana_dim

        types = card_features[..., idx:idx + self.config.type_dim]
        idx += self.config.type_dim

        stats_rarity = card_features[..., idx:]

        # Project each feature type
        kw_emb = self.keyword_proj(keywords)
        mana_emb = self.mana_proj(mana)
        type_emb = self.type_proj(types)
        stats_emb = self.stats_proj(stats_rarity)

        # Concatenate and combine
        combined = torch.cat([kw_emb, mana_emb, type_emb, stats_emb], dim=-1)
        return self.combine(combined)


class CardInteractionLayer(nn.Module):
    """
    Models interactions between cards using self-attention.

    This allows the encoder to understand synergies:
    - "Goblin Lord" with other Goblins
    - Equipment with creatures
    - Removal with threats
    """

    def __init__(self, config: CardEncoderConfig):
        super().__init__()
        self.config = config

        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply self-attention and feed-forward.

        Args:
            x: [batch, num_cards, d_model]
            mask: [batch, num_cards] - 1 for valid, 0 for padding

        Returns:
            [batch, num_cards, d_model]
        """
        # Convert mask to attention mask format
        attn_mask = None
        if mask is not None:
            # PyTorch expects True for positions to mask OUT
            attn_mask = (mask == 0)

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class SharedCardEncoder(nn.Module):
    """
    Shared card encoder for MTG.

    This encoder produces fixed-size embeddings for cards that capture:
    1. Individual card properties (stats, types, keywords)
    2. Contextual information (synergies with other cards)

    The encoder can operate in two modes:
    - Independent: Each card encoded separately (faster)
    - Contextual: Cards attend to each other (captures synergies)
    """

    def __init__(self, config: Optional[CardEncoderConfig] = None):
        super().__init__()
        self.config = config or CardEncoderConfig()

        # Feature projection
        self.feature_proj = CardFeatureProjection(self.config)

        # Card interaction layers
        self.interaction_layers = nn.ModuleList([
            CardInteractionLayer(self.config)
            for _ in range(self.config.n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        card_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_interactions: bool = True
    ) -> torch.Tensor:
        """
        Encode cards to embeddings.

        Args:
            card_features: [batch, num_cards, input_dim] raw card features
            mask: [batch, num_cards] - 1 for valid cards, 0 for padding
            use_interactions: Whether to use card interaction layers

        Returns:
            [batch, num_cards, output_dim] card embeddings
        """
        # Project features
        x = self.feature_proj(card_features)

        # Apply interaction layers if requested
        if use_interactions:
            for layer in self.interaction_layers:
                x = layer(x, mask)

        # Output projection
        return self.output_proj(x)

    def encode_single(self, card_features: torch.Tensor) -> torch.Tensor:
        """
        Encode a single card without interactions.

        Args:
            card_features: [input_dim] or [batch, input_dim]

        Returns:
            [output_dim] or [batch, output_dim]
        """
        if card_features.dim() == 1:
            card_features = card_features.unsqueeze(0).unsqueeze(0)
            result = self.forward(card_features, use_interactions=False)
            return result.squeeze(0).squeeze(0)
        else:
            card_features = card_features.unsqueeze(1)
            result = self.forward(card_features, use_interactions=False)
            return result.squeeze(1)

    def get_pooled_representation(
        self,
        card_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get a single representation for a set of cards.

        Args:
            card_features: [batch, num_cards, input_dim]
            mask: [batch, num_cards]
            pooling: "mean", "max", or "attention"

        Returns:
            [batch, output_dim]
        """
        embeddings = self.forward(card_features, mask)

        if pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                return (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            return embeddings.mean(dim=1)

        elif pooling == "max":
            if mask is not None:
                embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            return embeddings.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def save(self, path: str):
        """Save encoder to file."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'SharedCardEncoder':
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(checkpoint['config'])
        encoder.load_state_dict(checkpoint['state_dict'])
        return encoder


class CardFeatureExtractor:
    """
    Extracts features from card data for the encoder.

    This bridges between card representations (JSON, dict) and
    the tensor format expected by the encoder.
    """

    # Keyword list for one-hot encoding
    KEYWORDS = [
        "Flying", "First strike", "Double strike", "Trample", "Haste",
        "Vigilance", "Lifelink", "Deathtouch", "Reach", "Menace",
        "Defender", "Indestructible", "Hexproof", "Shroud", "Flash",
        "Fear", "Intimidate", "Infect", "Wither", "Prowess",
        "Exalted", "Undying", "Persist", "Flanking", "Shadow",
        "Changeling", "Devoid", "Decayed", "Cascade", "Storm",
        "Convoke", "Delve", "Affinity", "Flashback", "Kicker",
        "Cycling", "Morph", "Madness", "Suspend", "Evoke",
    ]

    # Card types for one-hot encoding
    CARD_TYPES = [
        "Creature", "Instant", "Sorcery", "Enchantment", "Artifact",
        "Land", "Planeswalker", "Battle", "Legendary", "Basic",
    ]

    CREATURE_TYPES = [
        "Human", "Soldier", "Wizard", "Warrior", "Knight",
        "Zombie", "Vampire", "Elf", "Goblin", "Merfolk",
        "Spirit", "Dragon", "Angel", "Demon", "Beast",
        "Elemental", "Giant", "Bird", "Cat", "Rogue",
    ]

    RARITIES = ["Common", "Uncommon", "Rare", "Mythic", "Special"]

    def __init__(self, config: CardEncoderConfig):
        self.config = config
        self.keyword_to_idx = {kw.lower(): i for i, kw in enumerate(self.KEYWORDS)}
        self.type_to_idx = {t.lower(): i for i, t in enumerate(self.CARD_TYPES)}
        self.creature_type_to_idx = {t.lower(): i for i, t in enumerate(self.CREATURE_TYPES)}
        self.rarity_to_idx = {r.lower(): i for i, r in enumerate(self.RARITIES)}

    def extract(self, card_data: Dict) -> np.ndarray:
        """
        Extract features from a single card.

        Args:
            card_data: Dict with card properties

        Returns:
            numpy array of shape [input_dim]
        """
        features = []

        # Keywords (40 dims)
        keywords = np.zeros(self.config.keyword_dim, dtype=np.float32)
        for kw in card_data.get('keywords', []):
            kw_lower = kw.lower().split()[0]  # Handle "Flying" and "Flying 2"
            if kw_lower in self.keyword_to_idx:
                keywords[self.keyword_to_idx[kw_lower]] = 1.0
        features.append(keywords)

        # Mana cost (11 dims)
        mana = self._parse_mana_cost(card_data.get('mana_cost', ''))
        features.append(mana)

        # Card types (30 dims = 10 types + 20 creature types)
        types = np.zeros(self.config.type_dim, dtype=np.float32)
        type_line = card_data.get('type', card_data.get('types', '')).lower()

        for i, t in enumerate(self.CARD_TYPES):
            if t.lower() in type_line:
                types[i] = 1.0

        for i, ct in enumerate(self.CREATURE_TYPES):
            if ct.lower() in type_line:
                types[10 + i] = 1.0
        features.append(types)

        # Stats (8 dims)
        stats = np.zeros(self.config.stats_dim, dtype=np.float32)

        power = card_data.get('power')
        if power is not None:
            try:
                stats[0] = float(power) / 15.0
            except (ValueError, TypeError):
                stats[0] = 0.0

        toughness = card_data.get('toughness')
        if toughness is not None:
            try:
                stats[1] = float(toughness) / 15.0
            except (ValueError, TypeError):
                stats[1] = 0.0

        loyalty = card_data.get('loyalty')
        if loyalty is not None:
            try:
                stats[2] = float(loyalty) / 10.0
            except (ValueError, TypeError):
                stats[2] = 0.0

        cmc = card_data.get('cmc', 0)
        stats[3] = cmc / 15.0

        # Efficiency metrics
        if cmc > 0:
            if power is not None:
                try:
                    stats[4] = float(power) / cmc
                except (ValueError, TypeError):
                    pass
            if power is not None and toughness is not None:
                try:
                    stats[5] = (float(power) + float(toughness)) / cmc
                except (ValueError, TypeError):
                    pass

        features.append(stats)

        # Rarity (5 dims)
        rarity = np.zeros(self.config.rarity_dim, dtype=np.float32)
        rarity_str = card_data.get('rarity', 'common').lower()
        if rarity_str in self.rarity_to_idx:
            rarity[self.rarity_to_idx[rarity_str]] = 1.0
        features.append(rarity)

        return np.concatenate(features)

    def _parse_mana_cost(self, cost_str: str) -> np.ndarray:
        """Parse mana cost string to feature vector."""
        result = np.zeros(self.config.mana_dim, dtype=np.float32)

        if not cost_str:
            return result

        import re
        symbols = re.findall(r'\{([^}]+)\}', cost_str)

        total_cmc = 0
        for symbol in symbols:
            symbol = symbol.upper()

            if symbol == 'W':
                result[1] += 1
                total_cmc += 1
            elif symbol == 'U':
                result[2] += 1
                total_cmc += 1
            elif symbol == 'B':
                result[3] += 1
                total_cmc += 1
            elif symbol == 'R':
                result[4] += 1
                total_cmc += 1
            elif symbol == 'G':
                result[5] += 1
                total_cmc += 1
            elif symbol == 'C':
                result[6] += 1
                total_cmc += 1
            elif symbol.isdigit():
                result[7] += int(symbol)
                total_cmc += int(symbol)
            elif '/' in symbol:
                if 'P' in symbol:
                    result[9] += 1
                else:
                    result[8] += 1
                total_cmc += 1
            elif symbol == 'X':
                result[10] += 1

        result[0] = total_cmc / 15.0

        # Normalize color counts
        for i in range(1, 8):
            result[i] = result[i] / 5.0

        return result

    def extract_batch(self, cards: List[Dict]) -> np.ndarray:
        """Extract features for a batch of cards."""
        return np.array([self.extract(card) for card in cards])


def test_shared_encoder():
    """Test the shared card encoder."""
    print("Testing Shared Card Encoder")
    print("=" * 60)

    config = CardEncoderConfig()
    encoder = SharedCardEncoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder parameters: {total_params:,}")
    print(f"Input dimension: {config.input_dim}")
    print(f"Output dimension: {config.output_dim}")

    # Test with random input
    batch_size = 4
    num_cards = 15

    card_features = torch.randn(batch_size, num_cards, config.input_dim)
    mask = torch.ones(batch_size, num_cards)
    mask[:, 10:] = 0  # Last 5 cards are padding

    print(f"\nInput shape: {card_features.shape}")
    print(f"Mask shape: {mask.shape}")

    # Forward pass
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(card_features, mask)
        pooled = encoder.get_pooled_representation(card_features, mask, "mean")

    print(f"\nOutput shape: {embeddings.shape}")
    print(f"Pooled shape: {pooled.shape}")

    # Test feature extractor
    print("\nTesting feature extractor...")
    extractor = CardFeatureExtractor(config)

    test_card = {
        'name': 'Lightning Bolt',
        'mana_cost': '{R}',
        'type': 'Instant',
        'keywords': [],
        'rarity': 'Common',
    }

    features = extractor.extract(test_card)
    print(f"Lightning Bolt features: shape={features.shape}, sum={features.sum():.2f}")

    test_creature = {
        'name': 'Tarmogoyf',
        'mana_cost': '{1}{G}',
        'type': 'Creature - Lhurgoyf',
        'power': 0,
        'toughness': 1,
        'keywords': [],
        'rarity': 'Mythic',
    }

    features = extractor.extract(test_creature)
    print(f"Tarmogoyf features: shape={features.shape}, sum={features.sum():.2f}")

    # Test save/load
    print("\nTesting save/load...")
    encoder.save("/tmp/test_encoder.pt")
    loaded = SharedCardEncoder.load("/tmp/test_encoder.pt")
    print("Save/load successful!")

    print("\n" + "=" * 60)
    print("Shared Card Encoder test completed!")


if __name__ == "__main__":
    test_shared_encoder()
