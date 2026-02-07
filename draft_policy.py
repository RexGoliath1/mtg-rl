#!/usr/bin/env python3
"""
Draft Policy Network for MTG

Implements a neural network specifically designed for draft decisions.
Uses the shared card encoder and adds draft-specific components:
- Pool context encoding (what cards have been drafted)
- Pack-pool cross-attention (how pack cards relate to pool)
- Pick scoring with synergy awareness

Key insight: Draft is a SET-TO-SET problem, not sequence-to-action:
- Input: Set of cards in pack + Set of cards in pool
- Output: Distribution over which card to pick

This is fundamentally different from gameplay, which requires:
- Temporal reasoning (turn order, phases)
- Hidden information (opponent's hand)
- Complex action sequences
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from shared_card_encoder import SharedCardEncoder, CardEncoderConfig, CardFeatureExtractor


@dataclass
class DraftPolicyConfig:
    """Configuration for the draft policy network."""
    # Card encoder config (shared with gameplay)
    card_encoder_config: CardEncoderConfig = None

    # Pool encoder
    pool_encoder_layers: int = 2
    pool_encoder_heads: int = 4

    # Cross-attention
    cross_attention_heads: int = 4

    # Pick scorer
    scorer_hidden_dim: int = 256

    # Context features
    use_pack_position: bool = True  # Pack number, pick number
    use_color_counts: bool = True   # Color distribution in pool

    # Training
    dropout: float = 0.1

    def __post_init__(self):
        if self.card_encoder_config is None:
            self.card_encoder_config = CardEncoderConfig()


class PoolEncoder(nn.Module):
    """
    Encodes the current draft pool to capture:
    - Overall pool strength
    - Color distribution
    - Curve considerations
    - Synergy clusters
    """

    def __init__(self, config: DraftPolicyConfig):
        super().__init__()
        self.config = config
        d_model = config.card_encoder_config.output_dim

        # Transformer encoder for pool
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.pool_encoder_heads,
            dim_feedforward=d_model * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.pool_encoder_layers,
        )

        # Pool summary token (like CLS)
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        pool_embeddings: torch.Tensor,
        pool_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the draft pool.

        Args:
            pool_embeddings: [batch, pool_size, d_model]
            pool_mask: [batch, pool_size] - 1 for valid, 0 for padding

        Returns:
            pool_context: [batch, pool_size+1, d_model] - all pool representations
            pool_summary: [batch, d_model] - single summary vector
        """
        batch_size = pool_embeddings.size(0)

        # Prepend pool summary token
        pool_tokens = self.pool_token.expand(batch_size, -1, -1)
        x = torch.cat([pool_tokens, pool_embeddings], dim=1)

        # Extend mask for pool token
        if pool_mask is not None:
            token_mask = torch.ones(batch_size, 1, device=pool_mask.device)
            extended_mask = torch.cat([token_mask, pool_mask], dim=1)
            # Convert to attention mask (True = mask out)
            attn_mask = (extended_mask == 0)
        else:
            attn_mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)

        # Split summary and context
        pool_summary = x[:, 0, :]
        pool_context = x[:, 1:, :]

        return pool_context, pool_summary


class PackPoolCrossAttention(nn.Module):
    """
    Cross-attention between pack cards and pool cards.

    For each card in the pack, this computes how well it synergizes
    with the current pool by attending to pool cards.
    """

    def __init__(self, config: DraftPolicyConfig):
        super().__init__()
        d_model = config.card_encoder_config.output_dim

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            config.cross_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        pack_embeddings: torch.Tensor,
        pool_context: torch.Tensor,
        pack_mask: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention from pack to pool.

        Args:
            pack_embeddings: [batch, pack_size, d_model]
            pool_context: [batch, pool_size, d_model]
            pack_mask: [batch, pack_size]
            pool_mask: [batch, pool_size]

        Returns:
            [batch, pack_size, d_model] - pack embeddings with pool context
        """
        # Key padding mask for pool
        key_padding_mask = None
        if pool_mask is not None:
            key_padding_mask = (pool_mask == 0)

        # Cross attention
        attn_out, _ = self.cross_attn(
            pack_embeddings,
            pool_context,
            pool_context,
            key_padding_mask=key_padding_mask,
        )

        # Residual connection
        return self.norm(pack_embeddings + self.dropout(attn_out))


class PickScorer(nn.Module):
    """
    Scores each card in the pack for picking.

    Combines:
    - Card's intrinsic quality (from embedding)
    - Synergy with pool (from cross-attention)
    - Draft context (pack/pick number, colors)
    """

    def __init__(self, config: DraftPolicyConfig):
        super().__init__()
        self.config = config
        d_model = config.card_encoder_config.output_dim

        # Context features dimension
        context_dim = 0
        if config.use_pack_position:
            context_dim += 4  # pack_num (one-hot 3), pick_num (normalized)
        if config.use_color_counts:
            context_dim += 5  # WUBRG counts

        # Input: pack embedding + pool-attended embedding + pool summary + context
        input_dim = d_model * 3 + context_dim

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, config.scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.scorer_hidden_dim, config.scorer_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.scorer_hidden_dim // 2, 1),
        )

    def forward(
        self,
        pack_embeddings: torch.Tensor,
        pack_with_context: torch.Tensor,
        pool_summary: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
        pack_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score each card in the pack.

        Args:
            pack_embeddings: [batch, pack_size, d_model] - original pack embeddings
            pack_with_context: [batch, pack_size, d_model] - after cross-attention
            pool_summary: [batch, d_model] - pool summary
            context_features: [batch, context_dim] - pack position, colors, etc.
            pack_mask: [batch, pack_size] - 1 for valid, 0 for padding

        Returns:
            [batch, pack_size] - score for each card (higher = better pick)
        """
        batch_size, pack_size, d_model = pack_embeddings.shape

        # Expand pool summary to match pack size
        pool_summary_expanded = pool_summary.unsqueeze(1).expand(-1, pack_size, -1)

        # Combine all features
        combined = [pack_embeddings, pack_with_context, pool_summary_expanded]

        if context_features is not None:
            # Expand context to match pack size
            context_expanded = context_features.unsqueeze(1).expand(-1, pack_size, -1)
            combined.append(context_expanded)

        x = torch.cat(combined, dim=-1)

        # Score each card
        scores = self.scorer(x).squeeze(-1)  # [batch, pack_size]

        # Mask invalid cards
        if pack_mask is not None:
            scores = scores.masked_fill(pack_mask == 0, float('-inf'))

        return scores


class DraftPolicyNetwork(nn.Module):
    """
    Complete draft policy network.

    Architecture:
    1. Shared card encoder encodes all cards
    2. Pool encoder summarizes drafted cards
    3. Cross-attention relates pack cards to pool
    4. Pick scorer outputs pick distribution
    """

    def __init__(
        self,
        config: Optional[DraftPolicyConfig] = None,
        card_encoder: Optional[SharedCardEncoder] = None,
    ):
        super().__init__()
        self.config = config or DraftPolicyConfig()

        # Shared card encoder (can be pre-trained)
        if card_encoder is not None:
            self.card_encoder = card_encoder
        else:
            self.card_encoder = SharedCardEncoder(self.config.card_encoder_config)

        # Draft-specific components
        self.pool_encoder = PoolEncoder(self.config)
        self.cross_attention = PackPoolCrossAttention(self.config)
        self.pick_scorer = PickScorer(self.config)

        # Value head for RL training
        d_model = self.config.card_encoder_config.output_dim
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        pack_features: torch.Tensor,
        pool_features: torch.Tensor,
        pack_mask: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pack_num: Optional[int] = None,
        pick_num: Optional[int] = None,
        color_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for draft decision.

        Args:
            pack_features: [batch, pack_size, input_dim] - raw features for pack cards
            pool_features: [batch, pool_size, input_dim] - raw features for pool cards
            pack_mask: [batch, pack_size] - 1 for valid, 0 for padding
            pool_mask: [batch, pool_size] - 1 for valid, 0 for padding
            pack_num: Current pack number (1-3)
            pick_num: Current pick number (1-15)
            color_counts: [batch, 5] - WUBRG counts in pool

        Returns:
            pick_logits: [batch, pack_size] - logits for each card
            value: [batch, 1] - estimated value of current draft state
        """
        # Encode cards using shared encoder
        pack_embeddings = self.card_encoder(pack_features, pack_mask, use_interactions=True)
        pool_embeddings = self.card_encoder(pool_features, pool_mask, use_interactions=True)

        # Encode pool
        pool_context, pool_summary = self.pool_encoder(pool_embeddings, pool_mask)

        # Cross-attention from pack to pool
        pack_with_context = self.cross_attention(
            pack_embeddings, pool_context, pack_mask, pool_mask
        )

        # Build context features
        context_features = self._build_context_features(
            pack_features.size(0), pack_features.device,
            pack_num, pick_num, color_counts
        )

        # Score picks
        pick_logits = self.pick_scorer(
            pack_embeddings, pack_with_context, pool_summary,
            context_features, pack_mask
        )

        # Compute value
        value = self.value_head(pool_summary)

        return pick_logits, value

    def _build_context_features(
        self,
        batch_size: int,
        device: torch.device,
        pack_num: Optional[int],
        pick_num: Optional[int],
        color_counts: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Build context feature vector."""
        features = []

        if self.config.use_pack_position:
            pos_features = torch.zeros(batch_size, 4, device=device)
            if pack_num is not None:
                # One-hot for pack (1, 2, or 3)
                pos_features[:, min(pack_num - 1, 2)] = 1.0
            if pick_num is not None:
                # Normalized pick number
                pos_features[:, 3] = pick_num / 15.0
            features.append(pos_features)

        if self.config.use_color_counts:
            if color_counts is not None:
                # Normalize color counts
                features.append(color_counts / 10.0)
            else:
                features.append(torch.zeros(batch_size, 5, device=device))

        if features:
            return torch.cat(features, dim=-1)
        return None

    def get_action(
        self,
        pack_features: torch.Tensor,
        pool_features: torch.Tensor,
        pack_mask: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pack_num: Optional[int] = None,
        pick_num: Optional[int] = None,
        color_counts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select a pick action.

        Args:
            deterministic: If True, pick highest-scored card; else sample

        Returns:
            action: [batch] - selected card indices
            log_prob: [batch] - log probability of selection
            probs: [batch, pack_size] - full distribution
            value: [batch, 1] - state value estimate
        """
        pick_logits, value = self.forward(
            pack_features, pool_features, pack_mask, pool_mask,
            pack_num, pick_num, color_counts
        )

        # Softmax to get probabilities
        probs = F.softmax(pick_logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        # Log probability of selected action
        log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze(1)

        return action, log_prob, probs, value

    def evaluate_actions(
        self,
        pack_features: torch.Tensor,
        pool_features: torch.Tensor,
        actions: torch.Tensor,
        pack_mask: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pack_num: Optional[int] = None,
        pick_num: Optional[int] = None,
        color_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities for given actions (for PPO).

        Returns:
            log_probs: [batch] - log probability of actions
            values: [batch, 1] - state values
            entropy: [batch] - policy entropy
        """
        pick_logits, values = self.forward(
            pack_features, pool_features, pack_mask, pool_mask,
            pack_num, pick_num, color_counts
        )

        probs = F.softmax(pick_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy

    def freeze_encoder(self):
        """Freeze the shared card encoder (for transfer learning)."""
        for param in self.card_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the shared card encoder."""
        for param in self.card_encoder.parameters():
            param.requires_grad = True

    def save(self, path: str, save_encoder_separately: bool = False):
        """Save the draft policy."""
        state = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }

        if save_encoder_separately:
            # Save encoder state separately for sharing
            state['encoder_state_dict'] = self.card_encoder.state_dict()

        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        card_encoder: Optional[SharedCardEncoder] = None,
        device: torch.device = None
    ) -> 'DraftPolicyNetwork':
        """Load the draft policy."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        policy = cls(checkpoint['config'], card_encoder)

        # Load state dict
        policy.load_state_dict(checkpoint['state_dict'])

        return policy


class DraftStatePreprocessor:
    """
    Converts draft environment state to tensors for the policy network.
    """

    def __init__(self, config: DraftPolicyConfig = None):
        self.config = config or DraftPolicyConfig()
        self.extractor = CardFeatureExtractor(self.config.card_encoder_config)

    def prepare_state(
        self,
        pack: List[Dict],
        pool: List[str],
        pack_num: int = 1,
        pick_num: int = 1,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare draft state for the policy network.

        Args:
            pack: List of card dicts in current pack
            pool: List of card names in pool (simplified)
            pack_num: Current pack number (1-3)
            pick_num: Current pick number (1-15)
            device: PyTorch device

        Returns:
            Dict with all tensors needed for forward pass
        """
        device = device or torch.device('cpu')

        # Extract pack features
        pack_features = self.extractor.extract_batch(pack)
        pack_tensor = torch.tensor(pack_features, dtype=torch.float32, device=device)
        pack_tensor = pack_tensor.unsqueeze(0)  # Add batch dimension

        pack_mask = torch.ones(1, len(pack), device=device)

        # For pool, we need to convert card names to features
        # This is simplified - in practice you'd look up full card data
        pool_size = len(pool) if pool else 1
        pool_features = np.zeros((pool_size, self.config.card_encoder_config.input_dim), dtype=np.float32)
        pool_tensor = torch.tensor(pool_features, dtype=torch.float32, device=device)
        pool_tensor = pool_tensor.unsqueeze(0)

        pool_mask = torch.ones(1, pool_size, device=device)
        if not pool:
            pool_mask[0, 0] = 0  # Mask dummy entry

        return {
            'pack_features': pack_tensor,
            'pool_features': pool_tensor,
            'pack_mask': pack_mask,
            'pool_mask': pool_mask,
            'pack_num': pack_num,
            'pick_num': pick_num,
        }


def test_draft_policy():
    """Test the draft policy network."""
    print("Testing Draft Policy Network")
    print("=" * 60)

    config = DraftPolicyConfig()
    policy = DraftPolicyNetwork(config)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    encoder_params = sum(p.numel() for p in policy.card_encoder.parameters())
    draft_params = total_params - encoder_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"  Shared encoder: {encoder_params:,}")
    print(f"  Draft-specific: {draft_params:,}")

    # Test forward pass
    batch_size = 2
    pack_size = 15
    pool_size = 20
    input_dim = config.card_encoder_config.input_dim

    pack_features = torch.randn(batch_size, pack_size, input_dim)
    pool_features = torch.randn(batch_size, pool_size, input_dim)
    pack_mask = torch.ones(batch_size, pack_size)
    pool_mask = torch.ones(batch_size, pool_size)
    pool_mask[:, 15:] = 0  # Mask last 5 pool cards

    print("\nInput shapes:")
    print(f"  pack_features: {pack_features.shape}")
    print(f"  pool_features: {pool_features.shape}")

    policy.eval()
    with torch.no_grad():
        pick_logits, value = policy(
            pack_features, pool_features, pack_mask, pool_mask,
            pack_num=1, pick_num=5
        )

    print("\nOutput shapes:")
    print(f"  pick_logits: {pick_logits.shape}")
    print(f"  value: {value.shape}")

    # Test action selection
    action, log_prob, probs, value = policy.get_action(
        pack_features, pool_features, pack_mask, pool_mask,
        pack_num=1, pick_num=5
    )

    print("\nAction selection:")
    print(f"  Selected actions: {action}")
    print(f"  Log probs: {log_prob}")
    print(f"  Value: {value.squeeze()}")

    # Test probability distribution
    print(f"  Prob distribution (first batch): {probs[0, :5].tolist()}")

    # Test freezing encoder
    print("\nTesting encoder freezing...")
    policy.freeze_encoder()
    frozen = sum(1 for p in policy.card_encoder.parameters() if not p.requires_grad)
    total = sum(1 for p in policy.card_encoder.parameters())
    print(f"  Frozen encoder params: {frozen}/{total}")

    policy.unfreeze_encoder()
    unfrozen = sum(1 for p in policy.card_encoder.parameters() if p.requires_grad)
    print(f"  Unfrozen encoder params: {unfrozen}/{total}")

    # Test save/load
    print("\nTesting save/load...")
    policy.save("/tmp/test_draft_policy.pt")
    _ = DraftPolicyNetwork.load("/tmp/test_draft_policy.pt")
    print("  Save/load successful!")

    print("\n" + "=" * 60)
    print("Draft Policy Network test completed!")


if __name__ == "__main__":
    test_draft_policy()
