#!/usr/bin/env python3
"""
Hybrid Card Encoder v2 for MTG RL

This module implements a hybrid card encoder that combines:
1. Text embeddings (MiniLM) - semantic understanding of oracle text
2. Structural features - precise numerical data (mana, P/T, types)
3. Self-attention - card interaction modeling (synergies)

Key advantages over v1 (keyword-based):
- Handles NEW mechanics automatically (text embedding captures semantics)
- Parameterized abilities work ("Mill 3" vs "Mill 5" get different embeddings)
- Same words with different meanings distinguished by context
- Transfer learning to unseen cards: ~42% accuracy (vs 0% for keyword-based)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                      Card Input                          │
    │  oracle_text, mana_cost, types, power, toughness, etc.  │
    └─────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │    Text     │   │    Mana     │   │   Types/    │
    │  Embedding  │   │  Features   │   │   Stats     │
    │ (MiniLM)    │   │  (11 dim)   │   │  (43 dim)   │
    │  384 dim    │   │             │   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   Linear    │   │   Linear    │   │   Linear    │
    │  384 → 128  │   │  11 → 64    │   │  43 → 64    │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │ Concat
                             ▼
                   ┌─────────────────┐
                   │  Fusion Layer   │
                   │  256 → 256      │
                   │  + LayerNorm    │
                   │  + GELU         │
                   └────────┬────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Card Interaction Layers    │
              │  (Self-Attention × 2)       │
              │  - Models pack synergies    │
              │  - Pool context             │
              └─────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Output (256d)  │
                   │  Card Embedding │
                   └─────────────────┘

References:
- arXiv:2407.05879 (July 2024): Text embeddings for MTG
- minimaxir/mtg-embeddings: ModernBERT approach
- See CARD_ENCODING.md for detailed rationale
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import text embedding components
from text_embeddings import (
    PretrainedTextEmbedder,
    TextEmbeddingConfig,
    MTGTextPreprocessor,
)


@dataclass
class HybridEncoderConfig:
    """Configuration for the hybrid card encoder v2."""

    # Text embedding
    text_embedding_dim: int = 384      # MiniLM output dimension
    text_model_name: str = "all-MiniLM-L6-v2"
    use_text_cache: bool = True
    text_cache_dir: str = "data/embeddings_cache"

    # Structural feature dimensions
    mana_dim: int = 11                 # CMC, WUBRG counts, colorless, generic, hybrid, phyrexian, X
    type_dim: int = 30                 # Card types (10) + creature types (20)
    stats_dim: int = 8                 # P/T, loyalty, CMC ratio, efficiency metrics
    rarity_dim: int = 5                # Common, Uncommon, Rare, Mythic, Special

    # Projection dimensions
    text_proj_dim: int = 128           # Text embedding projection
    mana_proj_dim: int = 64            # Mana features projection
    type_stats_proj_dim: int = 64      # Types + stats projection

    # Encoder architecture
    d_model: int = 256                 # Internal representation dimension
    n_heads: int = 4                   # Attention heads for card interactions
    n_layers: int = 2                  # Number of transformer layers
    d_ff: int = 512                    # Feed-forward hidden dimension
    dropout: float = 0.1

    # Output
    output_dim: int = 256              # Final card embedding dimension

    # Training options
    freeze_text_embeddings: bool = True  # Don't backprop through text model
    use_text_refinement: bool = False    # Add learnable refinement layer

    @property
    def structural_dim(self) -> int:
        """Total structural feature dimension."""
        return self.mana_dim + self.type_dim + self.stats_dim + self.rarity_dim

    @property
    def fusion_input_dim(self) -> int:
        """Dimension after projections, before fusion."""
        return self.text_proj_dim + self.mana_proj_dim + self.type_stats_proj_dim


class TextEmbeddingModule(nn.Module):
    """
    Handles text embedding with optional refinement.

    Can operate in two modes:
    1. Frozen: Use pre-computed embeddings directly (faster, no gradients)
    2. Refined: Add learnable refinement layer on top (can fine-tune)
    """

    def __init__(self, config: HybridEncoderConfig):
        super().__init__()
        self.config = config

        # Text embedding config
        text_config = TextEmbeddingConfig(
            model_name=config.text_model_name,
            embedding_dim=config.text_embedding_dim,
            cache_dir=config.text_cache_dir,
            use_cache=config.use_text_cache,
        )

        # Initialize embedder (loads sentence-transformers model)
        self.embedder = PretrainedTextEmbedder(text_config)

        # Projection layer
        self.proj = nn.Linear(config.text_embedding_dim, config.text_proj_dim)

        # Optional refinement layer
        if config.use_text_refinement:
            self.refinement = nn.Sequential(
                nn.Linear(config.text_proj_dim, config.text_proj_dim),
                nn.LayerNorm(config.text_proj_dim),
                nn.GELU(),
                nn.Linear(config.text_proj_dim, config.text_proj_dim),
            )
            self.refinement_weight = nn.Parameter(torch.zeros(1))
        else:
            self.refinement = None

    def get_text_embeddings(
        self,
        oracle_texts: List[str],
        card_names: List[str],
    ) -> torch.Tensor:
        """
        Get text embeddings for a batch of cards.

        Args:
            oracle_texts: List of oracle text strings
            card_names: List of card names (for self-reference replacement)

        Returns:
            torch.Tensor of shape [batch, text_embedding_dim]
        """
        # Pair up texts with names
        texts = list(zip(oracle_texts, card_names))

        # Get embeddings from pretrained model
        embeddings = self.embedder.embed_batch(texts)

        # Convert to tensor
        return torch.from_numpy(embeddings).float()

    def forward(
        self,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project and optionally refine text embeddings.

        Args:
            text_embeddings: [batch, num_cards, text_embedding_dim] or
                           [batch, text_embedding_dim]

        Returns:
            [batch, num_cards, text_proj_dim] or [batch, text_proj_dim]
        """
        # Project
        x = self.proj(text_embeddings)

        # Optional refinement
        if self.refinement is not None:
            refined = self.refinement(x)
            weight = torch.sigmoid(self.refinement_weight)
            x = x + weight * refined

        return x


class StructuralFeatureExtractor:
    """
    Extracts structural features from card data.

    These features are precise numerical/categorical data that
    complement the semantic text embeddings:
    - Mana cost (exact colors and amounts)
    - Card types (creature, instant, etc.)
    - Stats (P/T, loyalty)
    - Rarity
    """

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

    def __init__(self, config: HybridEncoderConfig):
        self.config = config
        self.type_to_idx = {t.lower(): i for i, t in enumerate(self.CARD_TYPES)}
        self.creature_type_to_idx = {t.lower(): i for i, t in enumerate(self.CREATURE_TYPES)}
        self.rarity_to_idx = {r.lower(): i for i, r in enumerate(self.RARITIES)}

    def extract(self, card_data: Dict) -> np.ndarray:
        """
        Extract structural features from a single card.

        Args:
            card_data: Dict with card properties (from Scryfall/MTGJSON format)

        Returns:
            numpy array of shape [mana_dim + type_dim + stats_dim + rarity_dim]
        """
        features = []

        # Mana cost features (11 dims)
        mana = self._parse_mana_cost(card_data.get('mana_cost', ''))
        features.append(mana)

        # Card types (30 dims = 10 types + 20 creature types)
        types = np.zeros(self.config.type_dim, dtype=np.float32)
        type_line = card_data.get('type_line', card_data.get('type', '')).lower()

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
                stats[0] = float(power) / 15.0  # Normalize
            except (ValueError, TypeError):
                stats[0] = 0.0  # Handle '*' power

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
        stats[3] = cmc / 15.0  # Normalized CMC

        # Efficiency metrics
        if cmc > 0:
            if power is not None:
                try:
                    stats[4] = float(power) / cmc  # Power per mana
                except (ValueError, TypeError):
                    pass
            if power is not None and toughness is not None:
                try:
                    stats[5] = (float(power) + float(toughness)) / cmc  # Total stats per mana
                except (ValueError, TypeError):
                    pass

        # Color count (multicolor indicator)
        colors = card_data.get('colors', [])
        stats[6] = len(colors) / 5.0  # Normalized color count

        # Is it a creature? (for efficiency metrics)
        stats[7] = 1.0 if 'creature' in type_line else 0.0

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
                result[6] += 1  # Colorless
                total_cmc += 1
            elif symbol.isdigit():
                result[7] += int(symbol)  # Generic
                total_cmc += int(symbol)
            elif '/' in symbol:
                if 'P' in symbol:
                    result[9] += 1  # Phyrexian
                else:
                    result[8] += 1  # Hybrid
                total_cmc += 1
            elif symbol == 'X':
                result[10] += 1

        result[0] = total_cmc / 15.0  # Normalized total CMC

        # Normalize color counts
        for i in range(1, 8):
            result[i] = result[i] / 5.0

        return result

    def extract_batch(self, cards: List[Dict]) -> np.ndarray:
        """Extract structural features for a batch of cards."""
        return np.array([self.extract(card) for card in cards])


class StructuralProjection(nn.Module):
    """Projects structural features to embedding space."""

    def __init__(self, config: HybridEncoderConfig):
        super().__init__()
        self.config = config

        # Separate projections for different feature types
        self.mana_proj = nn.Linear(config.mana_dim, config.mana_proj_dim)
        self.type_stats_proj = nn.Linear(
            config.type_dim + config.stats_dim + config.rarity_dim,
            config.type_stats_proj_dim
        )

    def forward(self, structural_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project structural features.

        Args:
            structural_features: [batch, num_cards, structural_dim] or [batch, structural_dim]

        Returns:
            Tuple of (mana_embedding, type_stats_embedding)
        """
        # Split features
        mana = structural_features[..., :self.config.mana_dim]
        type_stats = structural_features[..., self.config.mana_dim:]

        # Project
        mana_emb = self.mana_proj(mana)
        type_stats_emb = self.type_stats_proj(type_stats)

        return mana_emb, type_stats_emb


class CardInteractionLayer(nn.Module):
    """
    Models interactions between cards using self-attention.

    This allows the encoder to understand synergies:
    - "Goblin Lord" with other Goblins
    - Equipment with creatures
    - Removal with threats in pool
    """

    def __init__(self, config: HybridEncoderConfig):
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
            # PyTorch expects True for positions to MASK OUT
            attn_mask = (mask == 0)

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class HybridCardEncoder(nn.Module):
    """
    Hybrid Card Encoder v2 - combines text embeddings with structural features.

    This encoder produces fixed-size embeddings for cards that capture:
    1. Semantic meaning of abilities (from text embeddings)
    2. Precise numerical data (from structural features)
    3. Card interactions (from self-attention)

    Usage:
        config = HybridEncoderConfig()
        encoder = HybridCardEncoder(config)

        # Option 1: Pass pre-computed embeddings
        embeddings = encoder(text_emb, structural_features)

        # Option 2: Pass card data directly
        embeddings = encoder.encode_cards(card_data_list)
    """

    def __init__(self, config: Optional[HybridEncoderConfig] = None):
        super().__init__()
        self.config = config or HybridEncoderConfig()

        # Text embedding module
        self.text_module = TextEmbeddingModule(self.config)

        # Structural feature projection
        self.structural_proj = StructuralProjection(self.config)

        # Feature extractor (non-nn, for convenience)
        self.feature_extractor = StructuralFeatureExtractor(self.config)

        # Fusion layer - combines text and structural embeddings
        self.fusion = nn.Sequential(
            nn.Linear(self.config.fusion_input_dim, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Card interaction layers (self-attention)
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
        text_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_interactions: bool = True,
    ) -> torch.Tensor:
        """
        Encode cards to embeddings.

        Args:
            text_embeddings: [batch, num_cards, text_embedding_dim] pre-computed text embeddings
            structural_features: [batch, num_cards, structural_dim] structural features
            mask: [batch, num_cards] - 1 for valid cards, 0 for padding
            use_interactions: Whether to use card interaction layers

        Returns:
            [batch, num_cards, output_dim] card embeddings
        """
        # Project text embeddings
        text_proj = self.text_module(text_embeddings)

        # Project structural features
        mana_emb, type_stats_emb = self.structural_proj(structural_features)

        # Concatenate all projections
        combined = torch.cat([text_proj, mana_emb, type_stats_emb], dim=-1)

        # Fuse features
        x = self.fusion(combined)

        # Apply interaction layers if requested
        if use_interactions:
            for layer in self.interaction_layers:
                x = layer(x, mask)

        # Output projection
        return self.output_proj(x)

    def encode_cards(
        self,
        cards: List[Dict],
        use_interactions: bool = True,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode cards directly from card data dicts.

        Args:
            cards: List of card data dicts (Scryfall/MTGJSON format)
            use_interactions: Whether to use card interaction layers
            device: Device to put tensors on

        Returns:
            [num_cards, output_dim] card embeddings
        """
        if device is None:
            device = next(self.parameters()).device

        # Extract oracle texts and names
        oracle_texts = [c.get('oracle_text', c.get('text', '')) for c in cards]
        card_names = [c.get('name', '') for c in cards]

        # Get text embeddings
        text_emb = self.text_module.get_text_embeddings(oracle_texts, card_names)
        text_emb = text_emb.to(device).unsqueeze(0)  # [1, num_cards, text_dim]

        # Get structural features
        structural = self.feature_extractor.extract_batch(cards)
        structural = torch.from_numpy(structural).float().to(device).unsqueeze(0)

        # Encode
        with torch.no_grad() if self.config.freeze_text_embeddings else torch.enable_grad():
            embeddings = self.forward(text_emb, structural, use_interactions=use_interactions)

        return embeddings.squeeze(0)

    def encode_single(
        self,
        card: Dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode a single card.

        Args:
            card: Card data dict

        Returns:
            [output_dim] card embedding
        """
        return self.encode_cards([card], use_interactions=False, device=device).squeeze(0)

    def get_pooled_representation(
        self,
        text_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get a single representation for a set of cards (e.g., a pool).

        Args:
            text_embeddings: [batch, num_cards, text_embedding_dim]
            structural_features: [batch, num_cards, structural_dim]
            mask: [batch, num_cards]
            pooling: "mean" or "max"

        Returns:
            [batch, output_dim]
        """
        embeddings = self.forward(text_embeddings, structural_features, mask)

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
    def load(cls, path: str, device: torch.device = None) -> 'HybridCardEncoder':
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(checkpoint['config'])
        encoder.load_state_dict(checkpoint['state_dict'])
        return encoder


# =============================================================================
# CARD DATABASE WITH HYBRID EMBEDDINGS
# =============================================================================

class HybridCardDatabase:
    """
    Maintains a database of cards with pre-computed embeddings.

    Stores both text embeddings and structural features for fast lookup
    during training/inference.
    """

    def __init__(self, config: HybridEncoderConfig):
        self.config = config
        self.feature_extractor = StructuralFeatureExtractor(config)

        # Text embedding config
        text_config = TextEmbeddingConfig(
            model_name=config.text_model_name,
            embedding_dim=config.text_embedding_dim,
            cache_dir=config.text_cache_dir,
            use_cache=config.use_text_cache,
        )
        self.text_embedder = PretrainedTextEmbedder(text_config)

        # Storage
        self.card_data: Dict[str, Dict] = {}
        self.text_embeddings: Dict[str, np.ndarray] = {}
        self.structural_features: Dict[str, np.ndarray] = {}

    def add_card(self, card: Dict):
        """Add a single card to the database."""
        name = card.get('name', '')
        if not name:
            return

        self.card_data[name] = card

        # Compute text embedding
        oracle = card.get('oracle_text', card.get('text', ''))
        self.text_embeddings[name] = self.text_embedder.embed(oracle, name)

        # Compute structural features
        self.structural_features[name] = self.feature_extractor.extract(card)

    def add_cards(self, cards: List[Dict]):
        """Add multiple cards to the database."""
        for card in cards:
            self.add_card(card)

    def load_from_scryfall(self, set_code: str):
        """Load cards from Scryfall API."""
        try:
            import requests

            url = f"https://api.scryfall.com/cards/search?q=set:{set_code}"
            cards = []

            while url:
                response = requests.get(url)
                data = response.json()
                cards.extend(data.get('data', []))
                url = data.get('next_page')

            self.add_cards(cards)
            print(f"Loaded {len(cards)} cards from set {set_code}")

        except Exception as e:
            print(f"Error loading from Scryfall: {e}")

    def get_embeddings(self, card_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get text and structural embeddings for a card."""
        text_emb = self.text_embeddings.get(card_name)
        struct_feat = self.structural_features.get(card_name)

        if text_emb is None or struct_feat is None:
            raise KeyError(f"Card not found: {card_name}")

        return text_emb, struct_feat

    def get_batch_embeddings(
        self,
        card_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get embeddings for multiple cards."""
        text_embs = []
        struct_feats = []

        for name in card_names:
            text_emb, struct_feat = self.get_embeddings(name)
            text_embs.append(text_emb)
            struct_feats.append(struct_feat)

        return np.stack(text_embs), np.stack(struct_feats)

    def save(self, path: str):
        """Save database to disk."""
        data = {
            'card_data': self.card_data,
            'text_embeddings': {k: v.tolist() for k, v in self.text_embeddings.items()},
            'structural_features': {k: v.tolist() for k, v in self.structural_features.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(self.card_data)} cards to {path}")

    def load(self, path: str):
        """Load database from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.card_data = data.get('card_data', {})
        self.text_embeddings = {k: np.array(v) for k, v in data.get('text_embeddings', {}).items()}
        self.structural_features = {k: np.array(v) for k, v in data.get('structural_features', {}).items()}
        print(f"Loaded {len(self.card_data)} cards from {path}")

    def save_cache(self):
        """Save text embedding cache."""
        self.text_embedder.save_cache()


# =============================================================================
# TESTING
# =============================================================================

def test_hybrid_encoder():
    """Test the hybrid card encoder."""
    print("=" * 70)
    print("Testing Hybrid Card Encoder v2")
    print("=" * 70)

    config = HybridEncoderConfig()
    print(f"\nConfig:")
    print(f"  Text embedding dim: {config.text_embedding_dim}")
    print(f"  Structural dim: {config.structural_dim}")
    print(f"  Fusion input dim: {config.fusion_input_dim}")
    print(f"  Output dim: {config.output_dim}")

    # Create encoder
    encoder = HybridCardEncoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test with synthetic data
    print("\n" + "-" * 70)
    print("Test 1: Forward pass with synthetic data")
    print("-" * 70)

    batch_size = 4
    num_cards = 15

    # Random text embeddings (simulating MiniLM output)
    text_emb = torch.randn(batch_size, num_cards, config.text_embedding_dim)

    # Random structural features
    struct_feat = torch.randn(batch_size, num_cards, config.structural_dim)

    # Mask (last 5 cards are padding)
    mask = torch.ones(batch_size, num_cards)
    mask[:, 10:] = 0

    print(f"  Input shapes:")
    print(f"    Text embeddings: {text_emb.shape}")
    print(f"    Structural features: {struct_feat.shape}")
    print(f"    Mask: {mask.shape}")

    # Forward pass
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(text_emb, struct_feat, mask)
        pooled = encoder.get_pooled_representation(text_emb, struct_feat, mask, "mean")

    print(f"  Output shapes:")
    print(f"    Card embeddings: {embeddings.shape}")
    print(f"    Pooled representation: {pooled.shape}")

    # Test with real card data
    print("\n" + "-" * 70)
    print("Test 2: Encode real card data")
    print("-" * 70)

    test_cards = [
        {
            'name': 'Lightning Bolt',
            'mana_cost': '{R}',
            'type_line': 'Instant',
            'oracle_text': 'Lightning Bolt deals 3 damage to any target.',
            'rarity': 'common',
        },
        {
            'name': 'Tarmogoyf',
            'mana_cost': '{1}{G}',
            'type_line': 'Creature — Lhurgoyf',
            'oracle_text': "Tarmogoyf's power is equal to the number of card types among cards in all graveyards and its toughness is equal to that number plus 1.",
            'power': '*',
            'toughness': '1+*',
            'rarity': 'mythic',
        },
        {
            'name': 'Psychic Drain',
            'mana_cost': '{X}{U}{B}',
            'type_line': 'Sorcery',
            'oracle_text': 'Target player mills X cards. You gain life equal to the number of cards milled this way.',
            'rarity': 'uncommon',
        },
    ]

    print(f"  Encoding {len(test_cards)} cards:")
    for card in test_cards:
        print(f"    - {card['name']}")

    embeddings = encoder.encode_cards(test_cards)
    print(f"  Output shape: {embeddings.shape}")

    # Show similarity between cards
    print("\n  Pairwise cosine similarities:")
    for i in range(len(test_cards)):
        for j in range(i + 1, len(test_cards)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()
            print(f"    {test_cards[i]['name']} vs {test_cards[j]['name']}: {sim:.4f}")

    # Test save/load
    print("\n" + "-" * 70)
    print("Test 3: Save/Load")
    print("-" * 70)

    encoder.save("/tmp/hybrid_encoder_test.pt")
    loaded = HybridCardEncoder.load("/tmp/hybrid_encoder_test.pt")
    print("  Save/load successful!")

    # Verify loaded encoder produces same output
    loaded.eval()
    with torch.no_grad():
        loaded_emb = loaded(text_emb, struct_feat, mask)

    diff = (embeddings - loaded_emb[:, :len(test_cards)]).abs().max().item()
    print(f"  Max difference after reload: {diff:.6f}")

    # Test card database
    print("\n" + "-" * 70)
    print("Test 4: Card Database")
    print("-" * 70)

    db = HybridCardDatabase(config)
    db.add_cards(test_cards)
    print(f"  Added {len(test_cards)} cards to database")

    text_emb, struct_feat = db.get_batch_embeddings(['Lightning Bolt', 'Tarmogoyf'])
    print(f"  Retrieved embeddings shape: text={text_emb.shape}, struct={struct_feat.shape}")

    print("\n" + "=" * 70)
    print("Hybrid Card Encoder v2 tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_hybrid_encoder()
