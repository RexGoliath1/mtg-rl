#!/usr/bin/env python3
"""
MTG Transformer Policy Network

Implements a Transformer-based architecture for MTG decision making.
Designed to handle the unique challenges of MTG:
- Variable number of cards in each zone
- Complex card interactions
- Large action space with masking
- Long-term credit assignment

Architecture Overview:
┌─────────────────────────────────────────────────────────────┐
│                      Input Processing                        │
│  [Card Embeddings] + [Zone Positions] + [Global Features]   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   Transformer Encoder                        │
│  - Self-attention over all cards (handles interactions)     │
│  - Cross-attention between zones                            │
│  - 4-6 layers, 4-8 heads                                   │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
┌─────────────▼─────────────┐  ┌───────────▼───────────┐
│      Policy Head          │  │      Value Head       │
│  (Action probabilities)   │  │   (State value V(s))  │
│  + Action Masking         │  │                       │
└───────────────────────────┘  └───────────────────────┘
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TransformerConfig:
    """Configuration for the Transformer policy network."""
    # Card embedding dimensions
    card_embedding_dim: int = 92  # From card_embeddings.py (89 + 3 context)

    # Transformer dimensions
    d_model: int = 256  # Internal representation dimension
    n_heads: int = 8    # Number of attention heads
    n_layers: int = 4   # Number of transformer layers
    d_ff: int = 512     # Feed-forward hidden dimension
    dropout: float = 0.1

    # Zone configuration
    max_hand_size: int = 10
    max_battlefield_size: int = 20
    max_graveyard_size: int = 20
    max_exile_size: int = 10
    max_stack_size: int = 10

    # Global features (from GameState.to_observation())
    global_feature_dim: int = 38

    # Action space
    max_actions: int = 50

    # Zone type embeddings
    n_zones: int = 6  # hand, battlefield, graveyard, exile, stack, opponent_battlefield

    @property
    def max_sequence_length(self) -> int:
        """Maximum total cards we can process."""
        return (self.max_hand_size +
                self.max_battlefield_size * 2 +  # Our + opponent
                self.max_graveyard_size +
                self.max_exile_size +
                self.max_stack_size)


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ZoneEncoding(nn.Module):
    """Learnable zone type embeddings."""

    def __init__(self, n_zones: int, d_model: int):
        super().__init__()
        self.zone_embeddings = nn.Embedding(n_zones, d_model)

    def forward(self, zone_ids: torch.Tensor) -> torch.Tensor:
        """Get zone embeddings.

        Args:
            zone_ids: Tensor of shape (batch, seq_len) with zone type indices
        """
        return self.zone_embeddings(zone_ids)


# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for attention: (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        """
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


# =============================================================================
# STATE ENCODER
# =============================================================================

class GameStateEncoder(nn.Module):
    """Encodes the full game state into a fixed-size representation.

    Takes cards from multiple zones and produces a unified representation
    using attention to capture card interactions.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Project card embeddings to model dimension
        self.card_projection = nn.Linear(config.card_embedding_dim, config.d_model)

        # Zone embeddings
        self.zone_encoding = ZoneEncoding(config.n_zones, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            config.max_sequence_length,
            config.dropout
        )

        # Global features projection
        self.global_projection = nn.Sequential(
            nn.Linear(config.global_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # CLS token for aggregation (like BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        card_embeddings: torch.Tensor,
        zone_ids: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode game state.

        Args:
            card_embeddings: (batch, max_cards, card_embedding_dim) - embedded cards
            zone_ids: (batch, max_cards) - zone type for each card
            card_mask: (batch, max_cards) - 1 for real cards, 0 for padding
            global_features: (batch, global_feature_dim) - game state features

        Returns:
            cls_output: (batch, d_model) - aggregated state representation
            sequence_output: (batch, max_cards+1, d_model) - per-card representations
        """
        batch_size = card_embeddings.size(0)

        # Project cards to model dimension
        x = self.card_projection(card_embeddings)  # (batch, max_cards, d_model)

        # Add zone embeddings
        x = x + self.zone_encoding(zone_ids)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+max_cards, d_model)

        # Extend mask for CLS token (always valid)
        cls_mask = torch.ones(batch_size, 1, device=card_mask.device)
        extended_mask = torch.cat([cls_mask, card_mask], dim=1)

        # Add projected global features to CLS token
        global_proj = self.global_projection(global_features)  # (batch, d_model)
        x[:, 0, :] = x[:, 0, :] + global_proj

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, extended_mask)

        # Final normalization
        x = self.final_norm(x)

        # Split CLS and sequence outputs
        cls_output = x[:, 0, :]  # (batch, d_model)
        sequence_output = x[:, 1:, :]  # (batch, max_cards, d_model)

        return cls_output, sequence_output


# =============================================================================
# POLICY AND VALUE HEADS
# =============================================================================

class PolicyHead(nn.Module):
    """Policy head for action selection with masking."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Action scoring network
        self.action_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.max_actions)
        )

    def forward(
        self,
        state_encoding: torch.Tensor,
        action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action logits with masking.

        Args:
            state_encoding: (batch, d_model) - encoded state
            action_mask: (batch, max_actions) - 1 for valid actions, 0 for invalid

        Returns:
            action_logits: (batch, max_actions) - masked logits
            action_probs: (batch, max_actions) - action probabilities
        """
        # Compute raw logits
        logits = self.action_net(state_encoding)  # (batch, max_actions)

        # Apply action mask (set invalid actions to -inf)
        masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # Compute probabilities
        action_probs = F.softmax(masked_logits, dim=-1)

        return masked_logits, action_probs


class ValueHead(nn.Module):
    """Value head for state value estimation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_ff // 2),
            nn.GELU(),
            nn.Linear(config.d_ff // 2, 1)
        )

    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.

        Args:
            state_encoding: (batch, d_model) - encoded state

        Returns:
            value: (batch, 1) - estimated state value
        """
        return self.value_net(state_encoding)


# =============================================================================
# FULL POLICY NETWORK
# =============================================================================

class MTGPolicyNetwork(nn.Module):
    """
    Complete Transformer-based policy network for MTG.

    Combines:
    - Game state encoding via Transformer
    - Policy head with action masking
    - Value head for PPO training
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()

        # State encoder
        self.state_encoder = GameStateEncoder(self.config)

        # Policy and value heads
        self.policy_head = PolicyHead(self.config)
        self.value_head = ValueHead(self.config)

    def forward(
        self,
        card_embeddings: torch.Tensor,
        zone_ids: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            card_embeddings: (batch, max_cards, card_embedding_dim)
            zone_ids: (batch, max_cards)
            card_mask: (batch, max_cards)
            global_features: (batch, global_feature_dim)
            action_mask: (batch, max_actions)

        Returns:
            action_logits: (batch, max_actions)
            action_probs: (batch, max_actions)
            state_value: (batch, 1)
        """
        # Encode state
        cls_output, _ = self.state_encoder(
            card_embeddings, zone_ids, card_mask, global_features
        )

        # Get policy and value
        action_logits, action_probs = self.policy_head(cls_output, action_mask)
        state_value = self.value_head(cls_output)

        return action_logits, action_probs, state_value

    def get_action(
        self,
        card_embeddings: torch.Tensor,
        zone_ids: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action for environment interaction.

        Args:
            deterministic: If True, select argmax action; else sample

        Returns:
            action: (batch,) - selected action indices
            action_log_prob: (batch,) - log probability of selected actions
            action_probs: (batch, max_actions) - full action distribution
            state_value: (batch, 1) - estimated state value
        """
        action_logits, action_probs, state_value = self.forward(
            card_embeddings, zone_ids, card_mask, global_features, action_mask
        )

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            # Sample from distribution
            dist = Categorical(action_probs)
            action = dist.sample()

        # Compute log probability
        action_log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze(1)

        return action, action_log_prob, action_probs, state_value

    def evaluate_actions(
        self,
        card_embeddings: torch.Tensor,
        zone_ids: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        Used during PPO training.

        Args:
            actions: (batch,) - actions to evaluate

        Returns:
            action_log_probs: (batch,) - log probs of actions
            state_values: (batch, 1) - state values
            entropy: (batch,) - policy entropy
        """
        action_logits, action_probs, state_value = self.forward(
            card_embeddings, zone_ids, card_mask, global_features, action_mask
        )

        # Log probability of taken actions
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8).squeeze(1)

        # Entropy for exploration bonus
        dist = Categorical(action_probs)
        entropy = dist.entropy()

        return action_log_probs, state_value, entropy


# =============================================================================
# STATE PREPARATION UTILITIES
# =============================================================================

class StatePreprocessor:
    """Prepares game state for the policy network."""

    def __init__(self, config: TransformerConfig):
        self.config = config

        # Zone type mapping
        self.zone_to_id = {
            'hand': 0,
            'battlefield': 1,
            'graveyard': 2,
            'exile': 3,
            'stack': 4,
            'opponent_battlefield': 5,
        }

    def prepare_state(
        self,
        game_state,  # Dict or GameState object
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert game state to tensors for the network.

        Args:
            game_state: Dict or GameState object with player data
            device: PyTorch device

        Returns:
            card_embeddings, zone_ids, card_mask, global_features, action_mask
        """
        from card_embeddings import CardEmbedding
        embedder = CardEmbedding(use_text_embeddings=False)

        all_cards = []
        all_zones = []

        def pad_embedding(emb: np.ndarray) -> np.ndarray:
            """Pad embedding to match config dimension."""
            if len(emb) < self.config.card_embedding_dim:
                return np.pad(emb, (0, self.config.card_embedding_dim - len(emb)))
            return emb[:self.config.card_embedding_dim]

        # Handle both dict and GameState objects
        if hasattr(game_state, 'our_player'):
            # GameState object
            our_player = game_state.our_player
            opponent = game_state.opponent
            stack = game_state.stack
            turn = game_state.turn
            our_hand = our_player.hand if hasattr(our_player, 'hand') else []
            our_battlefield = our_player.battlefield if hasattr(our_player, 'battlefield') else []
            our_graveyard = our_player.graveyard if hasattr(our_player, 'graveyard') else []
            opp_battlefield = opponent.battlefield if hasattr(opponent, 'battlefield') else []
            our_life = our_player.life if hasattr(our_player, 'life') else 20
            opp_life = opponent.life if hasattr(opponent, 'life') else 20
        else:
            # Dict format
            our_player = game_state.get('our_player', {})
            opponent = game_state.get('opponent', {})
            stack = game_state.get('stack', [])
            turn = game_state.get('turn', 1)
            our_hand = our_player.get('hand', [])
            our_battlefield = our_player.get('battlefield', [])
            our_graveyard = our_player.get('graveyard', [])
            opp_battlefield = opponent.get('battlefield', [])
            our_life = our_player.get('life', 20)
            opp_life = opponent.get('life', 20)

        # Helper to convert CardState to dict for embedding
        def card_to_dict(card):
            if hasattr(card, 'name'):
                return {
                    'name': card.name,
                    'mana_cost': card.mana_cost,
                    'types': card.card_type,
                    'oracle_text': getattr(card, 'oracle_text', ''),
                    'power': card.power,
                    'toughness': card.toughness,
                    'keywords': getattr(card, 'keywords', []),
                }
            return card

        # Our hand
        for card in our_hand[:self.config.max_hand_size]:
            card_dict = card_to_dict(card)
            emb = embedder.embed_from_game_state(card_dict)
            all_cards.append(pad_embedding(emb))
            all_zones.append(self.zone_to_id['hand'])

        # Our battlefield
        for card in our_battlefield[:self.config.max_battlefield_size]:
            card_dict = card_to_dict(card)
            emb = embedder.embed_from_game_state(card_dict)
            all_cards.append(pad_embedding(emb))
            all_zones.append(self.zone_to_id['battlefield'])

        # Our graveyard
        for card in our_graveyard[:self.config.max_graveyard_size]:
            card_dict = card_to_dict(card)
            emb = embedder.embed_from_game_state(card_dict)
            all_cards.append(pad_embedding(emb))
            all_zones.append(self.zone_to_id['graveyard'])

        # Opponent battlefield
        for card in opp_battlefield[:self.config.max_battlefield_size]:
            card_dict = card_to_dict(card)
            emb = embedder.embed_from_game_state(card_dict)
            all_cards.append(pad_embedding(emb))
            all_zones.append(self.zone_to_id['opponent_battlefield'])

        # Stack
        for entry in stack[:self.config.max_stack_size]:
            # Stack entries are different - create simple embedding
            stack_emb = np.zeros(self.config.card_embedding_dim, dtype=np.float32)
            all_cards.append(stack_emb)
            all_zones.append(self.zone_to_id['stack'])

        # Pad to max sequence length
        max_len = self.config.max_sequence_length
        n_cards = len(all_cards)

        if n_cards == 0:
            # Empty state - add dummy card
            all_cards.append(np.zeros(self.config.card_embedding_dim, dtype=np.float32))
            all_zones.append(0)
            n_cards = 1

        # Create tensors
        card_embeddings = np.zeros((max_len, self.config.card_embedding_dim), dtype=np.float32)
        zone_ids = np.zeros(max_len, dtype=np.int64)
        card_mask = np.zeros(max_len, dtype=np.float32)

        for i, (emb, zone) in enumerate(zip(all_cards, all_zones)):
            if i >= max_len:
                break
            card_embeddings[i] = emb
            zone_ids[i] = zone
            card_mask[i] = 1.0

        # Global features
        global_features = np.zeros(self.config.global_feature_dim, dtype=np.float32)
        global_features[0] = turn / 100.0
        global_features[5] = our_life / 40.0
        global_features[20] = opp_life / 40.0

        # Action mask (all valid for now - will be set by environment)
        action_mask = np.ones(self.config.max_actions, dtype=np.float32)

        # Convert to tensors and add batch dimension
        return (
            torch.tensor(card_embeddings, device=device).unsqueeze(0),
            torch.tensor(zone_ids, device=device).unsqueeze(0),
            torch.tensor(card_mask, device=device).unsqueeze(0),
            torch.tensor(global_features, device=device).unsqueeze(0),
            torch.tensor(action_mask, device=device).unsqueeze(0),
        )

    def create_empty_state(
        self,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create an empty state (for initialization or when no game state is available).

        Returns:
            Tuple of tensors: (card_embeddings, zone_ids, card_mask, global_features, action_mask)
        """
        max_len = self.config.max_sequence_length

        # Empty card embeddings with single dummy entry
        card_embeddings = np.zeros((max_len, self.config.card_embedding_dim), dtype=np.float32)
        zone_ids = np.zeros(max_len, dtype=np.int64)
        card_mask = np.zeros(max_len, dtype=np.float32)
        card_mask[0] = 1.0  # At least one "card" to avoid empty attention

        # Neutral global features
        global_features = np.zeros(self.config.global_feature_dim, dtype=np.float32)
        global_features[5] = 0.5  # Neutral life
        global_features[20] = 0.5

        # All actions invalid initially
        action_mask = np.zeros(self.config.max_actions, dtype=np.float32)
        action_mask[0] = 1.0  # At least one valid action (pass)

        return (
            torch.tensor(card_embeddings, device=device).unsqueeze(0),
            torch.tensor(zone_ids, device=device).unsqueeze(0),
            torch.tensor(card_mask, device=device).unsqueeze(0),
            torch.tensor(global_features, device=device).unsqueeze(0),
            torch.tensor(action_mask, device=device).unsqueeze(0),
        )


# =============================================================================
# TESTING
# =============================================================================

def test_policy_network():
    """Test the policy network with random inputs."""
    print("Testing MTG Policy Network")
    print("=" * 60)

    config = TransformerConfig()
    print("\nConfiguration:")
    print(f"  Card embedding dim: {config.card_embedding_dim}")
    print(f"  Model dim (d_model): {config.d_model}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  Transformer layers: {config.n_layers}")
    print(f"  Max sequence length: {config.max_sequence_length}")
    print(f"  Max actions: {config.max_actions}")

    # Create network
    network = MTGPolicyNetwork(config)

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("\nNetwork parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create random batch
    batch_size = 4
    n_cards = 30  # Simulated number of cards

    card_embeddings = torch.randn(batch_size, config.max_sequence_length, config.card_embedding_dim)
    zone_ids = torch.randint(0, config.n_zones, (batch_size, config.max_sequence_length))
    card_mask = torch.zeros(batch_size, config.max_sequence_length)
    card_mask[:, :n_cards] = 1.0  # First n_cards are real
    global_features = torch.randn(batch_size, config.global_feature_dim)
    action_mask = torch.ones(batch_size, config.max_actions)
    action_mask[:, 40:] = 0  # Only first 40 actions valid

    print("\nInput shapes:")
    print(f"  card_embeddings: {card_embeddings.shape}")
    print(f"  zone_ids: {zone_ids.shape}")
    print(f"  card_mask: {card_mask.shape}")
    print(f"  global_features: {global_features.shape}")
    print(f"  action_mask: {action_mask.shape}")

    # Forward pass
    network.eval()
    with torch.no_grad():
        action_logits, action_probs, state_value = network(
            card_embeddings, zone_ids, card_mask, global_features, action_mask
        )

    print("\nOutput shapes:")
    print(f"  action_logits: {action_logits.shape}")
    print(f"  action_probs: {action_probs.shape}")
    print(f"  state_value: {state_value.shape}")

    # Verify masking
    print("\nAction masking verification:")
    print(f"  Prob sum for valid actions: {action_probs[:, :40].sum(dim=1)}")
    print(f"  Prob sum for invalid actions: {action_probs[:, 40:].sum(dim=1)}")

    # Test action selection
    action, log_prob, probs, value = network.get_action(
        card_embeddings, zone_ids, card_mask, global_features, action_mask,
        deterministic=False
    )

    print("\nAction selection:")
    print(f"  Selected actions: {action}")
    print(f"  Log probabilities: {log_prob}")
    print(f"  State values: {value.squeeze()}")

    # Test action evaluation
    actions = torch.randint(0, 40, (batch_size,))  # Random valid actions
    log_probs, values, entropy = network.evaluate_actions(
        card_embeddings, zone_ids, card_mask, global_features, action_mask, actions
    )

    print("\nAction evaluation:")
    print(f"  Actions evaluated: {actions}")
    print(f"  Log probs: {log_probs}")
    print(f"  Entropy: {entropy}")

    print("\n" + "=" * 60)
    print("Policy network test completed successfully!")


if __name__ == "__main__":
    test_policy_network()
