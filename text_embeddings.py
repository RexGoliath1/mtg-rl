#!/usr/bin/env python3
"""
Text Embeddings for MTG Card Abilities

This module provides text embeddings for Magic card oracle text, enabling the
neural network to understand ability semantics beyond keyword matching.

Approaches:
1. PRETRAINED: Use sentence-transformers (MiniLM, MPNet) - best quality
2. LEARNED: Train embeddings from scratch during RL - more flexible
3. HYBRID: Pretrained + fine-tuning during training

For production, we recommend pretrained embeddings because:
- Card text is natural language, benefits from language model pretraining
- Reduces training compute (embeddings are cached)
- Captures semantic similarity ("destroy target creature" ≈ "exile target creature")
"""

import os
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class TextEmbeddingConfig:
    """Configuration for text embeddings."""
    # Model selection
    model_name: str = "all-MiniLM-L6-v2"  # Good balance of speed/quality
    embedding_dim: int = 384  # MiniLM output dim

    # Caching
    cache_dir: str = "data/embeddings_cache"
    use_cache: bool = True

    # Fallback for no sentence-transformers
    fallback_dim: int = 256
    use_learned_fallback: bool = True

    # Preprocessing
    max_text_length: int = 512
    lowercase: bool = True
    remove_reminder_text: bool = True  # Remove (reminder text in parentheses)


class MTGTextPreprocessor:
    """
    Preprocesses MTG card text for embedding.

    Handles:
    - Mana symbol normalization ({W} -> white mana)
    - Reminder text removal
    - Card name replacement (for self-references)
    - Keyword extraction
    """

    # Mana symbol mappings
    MANA_SYMBOLS = {
        '{W}': 'white mana',
        '{U}': 'blue mana',
        '{B}': 'black mana',
        '{R}': 'red mana',
        '{G}': 'green mana',
        '{C}': 'colorless mana',
        '{X}': 'X mana',
        '{T}': 'tap',
        '{Q}': 'untap',
    }

    def __init__(self, config: TextEmbeddingConfig):
        self.config = config

    def preprocess(self, oracle_text: str, card_name: str = "this creature") -> str:
        """
        Preprocess oracle text for embedding.

        Args:
            oracle_text: Raw oracle text from card
            card_name: Card's name (for self-reference replacement)

        Returns:
            Preprocessed text suitable for embedding
        """
        if not oracle_text:
            return ""

        text = oracle_text

        # Remove reminder text if configured
        if self.config.remove_reminder_text:
            text = re.sub(r'\([^)]*\)', '', text)

        # Replace card name with generic reference
        if card_name:
            text = text.replace(card_name, "this card")

        # Normalize mana symbols
        for symbol, replacement in self.MANA_SYMBOLS.items():
            text = text.replace(symbol, replacement)

        # Handle generic mana costs {1}, {2}, etc.
        text = re.sub(r'\{(\d+)\}', r'\1 generic mana', text)

        # Handle hybrid mana {W/U}, {2/W}, etc.
        text = re.sub(r'\{([WUBRGC])/([WUBRGC])\}', r'\1 or \2 mana', text)
        text = re.sub(r'\{(\d)/([WUBRGC])\}', r'\1 or \2 mana', text)

        # Handle Phyrexian mana {W/P}, etc.
        text = re.sub(r'\{([WUBRGC])/P\}', r'\1 mana or 2 life', text)

        # Lowercase if configured
        if self.config.lowercase:
            text = text.lower()

        # Clean up whitespace
        text = ' '.join(text.split())

        # Truncate if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]

        return text


class PretrainedTextEmbedder:
    """
    Uses pretrained sentence-transformers for text embedding.

    Recommended models:
    - all-MiniLM-L6-v2: Fast, 384-dim, good for most cases
    - all-mpnet-base-v2: Best quality, 768-dim, slower
    - paraphrase-MiniLM-L3-v2: Fastest, 384-dim, slightly lower quality
    """

    def __init__(self, config: TextEmbeddingConfig):
        self.config = config
        self.preprocessor = MTGTextPreprocessor(config)
        self.model = None
        self._load_model()

        # Setup cache
        self.cache_path = Path(config.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()

    def _load_model(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name)
            print(f"Loaded text embedding model: {self.config.model_name}")
        except ImportError:
            print("Warning: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            print("Using fallback random embeddings.")
            self.model = None

    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_path / "text_embeddings.json"
        if cache_file.exists() and self.config.use_cache:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = {k: np.array(v) for k, v in data.items()}
                print(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.config.use_cache:
            return
        cache_file = self.cache_path / "text_embeddings.json"
        try:
            data = {k: v.tolist() for k, v in self.cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, oracle_text: str, card_name: str = "") -> np.ndarray:
        """
        Embed a single card's oracle text.

        Args:
            oracle_text: Card's oracle text
            card_name: Card's name for self-reference replacement

        Returns:
            numpy array of shape [embedding_dim]
        """
        # Preprocess
        text = self.preprocessor.preprocess(oracle_text, card_name)

        if not text:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate embedding
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: random but deterministic embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.config.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

        # Cache
        self.cache[cache_key] = embedding

        return embedding

    def embed_batch(self, texts: List[Tuple[str, str]]) -> np.ndarray:
        """
        Embed multiple cards' oracle texts.

        Args:
            texts: List of (oracle_text, card_name) tuples

        Returns:
            numpy array of shape [num_cards, embedding_dim]
        """
        # Preprocess all
        processed = [self.preprocessor.preprocess(t, n) for t, n in texts]

        # Check cache for all
        results = []
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(processed):
            if not text:
                results.append((i, np.zeros(self.config.embedding_dim, dtype=np.float32)))
            else:
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    results.append((i, self.cache[cache_key]))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        # Embed uncached
        if uncached_texts and self.model is not None:
            embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embeddings[i]
                results.append((idx, embeddings[i]))
        elif uncached_texts:
            # Fallback
            for idx, text in zip(uncached_indices, uncached_texts):
                np.random.seed(hash(text) % (2**32))
                embedding = np.random.randn(self.config.embedding_dim).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                results.append((idx, embedding))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.stack([r[1] for r in results])

    def save_cache(self):
        """Explicitly save cache to disk."""
        self._save_cache()


class LearnedTextEmbedder(nn.Module):
    """
    Learns text embeddings from scratch using a simple transformer.

    This is a fallback when pretrained models aren't available or
    when you want end-to-end training of text representations.

    Architecture:
    - Character-level tokenization (no vocabulary needed)
    - Small transformer encoder
    - Mean pooling for fixed-size output
    """

    def __init__(self, config: TextEmbeddingConfig):
        super().__init__()
        self.config = config

        # Character embedding (ASCII printable + special tokens)
        self.vocab_size = 128  # ASCII
        self.char_embedding = nn.Embedding(self.vocab_size, 64)

        # Position embedding
        self.max_len = config.max_text_length
        self.pos_embedding = nn.Embedding(self.max_len, 64)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(64, config.fallback_dim),
            nn.LayerNorm(config.fallback_dim),
        )

        self.preprocessor = MTGTextPreprocessor(config)

    def tokenize(self, text: str) -> torch.Tensor:
        """Convert text to character indices."""
        text = self.preprocessor.preprocess(text, "")
        indices = [min(ord(c), self.vocab_size - 1) for c in text[:self.max_len]]

        # Pad to max length
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Embed a batch of texts.

        Args:
            texts: List of oracle texts

        Returns:
            torch.Tensor of shape [batch, fallback_dim]
        """
        # Tokenize
        tokens = torch.stack([self.tokenize(t) for t in texts])
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        # Create mask (0 = padding)
        mask = (tokens == 0)

        # Embed characters
        char_emb = self.char_embedding(tokens)

        # Add position embeddings
        positions = torch.arange(self.max_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = char_emb + pos_emb

        # Transform
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling (ignoring padding)
        mask_expanded = (~mask).unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Output projection
        return self.output_proj(x)


class HybridTextEmbedder(nn.Module):
    """
    Combines pretrained embeddings with learnable refinement.

    Uses pretrained embeddings as initialization, then allows
    fine-tuning through a small MLP.
    """

    def __init__(self, config: TextEmbeddingConfig, pretrained: PretrainedTextEmbedder):
        super().__init__()
        self.config = config
        self.pretrained = pretrained

        # Refinement MLP
        self.refine = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

        # Residual weight (starts at 0, learns to add refinement)
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, texts: List[Tuple[str, str]]) -> torch.Tensor:
        """
        Embed texts with refinement.

        Args:
            texts: List of (oracle_text, card_name) tuples

        Returns:
            torch.Tensor of shape [batch, embedding_dim]
        """
        # Get pretrained embeddings
        pretrained_emb = self.pretrained.embed_batch(texts)
        pretrained_emb = torch.from_numpy(pretrained_emb).to(next(self.parameters()).device)

        # Refine
        refined = self.refine(pretrained_emb)

        # Residual connection with learnable weight
        weight = torch.sigmoid(self.residual_weight)
        output = pretrained_emb + weight * refined

        return output


# =============================================================================
# CARD DATABASE WITH EMBEDDINGS
# =============================================================================

class CardEmbeddingDatabase:
    """
    Maintains a database of card embeddings for fast lookup.

    Pre-computes embeddings for all cards in a set/format and
    stores them for efficient retrieval during training/inference.
    """

    def __init__(self, config: TextEmbeddingConfig):
        self.config = config
        self.embedder = PretrainedTextEmbedder(config)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.card_data: Dict[str, Dict] = {}

    def load_cards_from_json(self, json_path: str):
        """
        Load cards from Scryfall-format JSON.

        Args:
            json_path: Path to JSON file with card data
        """
        with open(json_path, 'r') as f:
            cards = json.load(f)

        if isinstance(cards, dict):
            cards = list(cards.values())

        for card in cards:
            name = card.get('name', '')
            oracle = card.get('oracle_text', card.get('text', ''))

            if name:
                self.card_data[name] = card
                self.embeddings[name] = self.embedder.embed(oracle, name)

        print(f"Loaded {len(self.embeddings)} card embeddings")
        self.embedder.save_cache()

    def load_cards_from_scryfall(self, set_code: str):
        """
        Load cards from Scryfall API for a specific set.

        Args:
            set_code: Set code (e.g., "FDN", "DSK")
        """
        try:
            import requests

            url = f"https://api.scryfall.com/cards/search?q=set:{set_code}"
            cards = []

            while url:
                response = requests.get(url)
                data = response.json()
                cards.extend(data.get('data', []))
                url = data.get('next_page')

            for card in cards:
                name = card.get('name', '')
                oracle = card.get('oracle_text', '')

                if name:
                    self.card_data[name] = card
                    self.embeddings[name] = self.embedder.embed(oracle, name)

            print(f"Loaded {len(self.embeddings)} cards from set {set_code}")
            self.embedder.save_cache()

        except Exception as e:
            print(f"Error loading from Scryfall: {e}")

    def get_embedding(self, card_name: str) -> Optional[np.ndarray]:
        """Get embedding for a card by name."""
        return self.embeddings.get(card_name)

    def get_embedding_or_compute(self, card_name: str, oracle_text: str = "") -> np.ndarray:
        """Get embedding, computing if not cached."""
        if card_name in self.embeddings:
            return self.embeddings[card_name]

        embedding = self.embedder.embed(oracle_text, card_name)
        self.embeddings[card_name] = embedding
        return embedding

    def save(self, path: str):
        """Save database to disk."""
        data = {
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'card_data': self.card_data,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(self.embeddings)} embeddings to {path}")

    def load(self, path: str):
        """Load database from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
        self.card_data = data.get('card_data', {})
        print(f"Loaded {len(self.embeddings)} embeddings from {path}")


# =============================================================================
# TESTING
# =============================================================================

def test_text_embeddings():
    """Test text embedding components."""
    print("Testing Text Embeddings")
    print("=" * 70)

    config = TextEmbeddingConfig()

    # Test preprocessor
    print("\n1. Testing preprocessor...")
    preprocessor = MTGTextPreprocessor(config)

    test_texts = [
        ("Lightning Bolt deals 3 damage to any target.", "Lightning Bolt"),
        ("When Siege Rhino enters the battlefield, each opponent loses 3 life and you gain 3 life.", "Siege Rhino"),
        ("{T}: Add {W}.", "Plains"),
        ("Flying, vigilance (This creature can attack and block at the same time.)", "Serra Angel"),
        ("Counter target spell. Its controller may pay {3}. If they do, return this card to your hand.", "Remand"),
    ]

    for oracle, name in test_texts:
        processed = preprocessor.preprocess(oracle, name)
        print(f"  {name}: {processed[:60]}...")

    # Test pretrained embedder
    print("\n2. Testing pretrained embedder...")
    embedder = PretrainedTextEmbedder(config)

    # Embed test cards
    for oracle, name in test_texts[:3]:
        embedding = embedder.embed(oracle, name)
        print(f"  {name}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")

    # Test semantic similarity
    print("\n3. Testing semantic similarity...")

    similar_pairs = [
        ("Destroy target creature.", "Exile target creature."),
        ("Draw a card.", "Look at the top card of your library."),
        ("Counter target spell.", "Target spell's controller sacrifices it."),
    ]

    for text1, text2 in similar_pairs:
        emb1 = embedder.embed(text1, "")
        emb2 = embedder.embed(text2, "")
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"  '{text1[:30]}...' vs '{text2[:30]}...': {similarity:.3f}")

    # Test batch embedding
    print("\n4. Testing batch embedding...")
    batch = [(oracle, name) for oracle, name in test_texts]
    embeddings = embedder.embed_batch(batch)
    print(f"  Batch shape: {embeddings.shape}")

    # Test learned embedder
    print("\n5. Testing learned embedder...")
    learned = LearnedTextEmbedder(config)

    with torch.no_grad():
        texts = [oracle for oracle, _ in test_texts]
        learned_emb = learned(texts)
        print(f"  Learned embedding shape: {learned_emb.shape}")

    # Test card database
    print("\n6. Testing card database...")
    db = CardEmbeddingDatabase(config)

    # Add some test cards
    for oracle, name in test_texts:
        db.embeddings[name] = embedder.embed(oracle, name)
        db.card_data[name] = {"name": name, "oracle_text": oracle}

    # Test retrieval
    emb = db.get_embedding("Lightning Bolt")
    print(f"  Retrieved Lightning Bolt: shape={emb.shape if emb is not None else None}")

    # Save cache
    embedder.save_cache()

    print("\n" + "=" * 70)
    print("Text Embeddings test completed!")


if __name__ == "__main__":
    test_text_embeddings()
