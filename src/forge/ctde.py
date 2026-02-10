"""
Centralized Training, Decentralized Execution (CTDE) - Dual Value Heads.

Implements the PerfectDou/Suphx-style information asymmetry handling:

- **Oracle Value Head (training only)**: Sees full game state including
  opponent's hand and hidden information. Used as the critic during training
  for more accurate value estimation.

- **Observable Value Head (inference)**: Sees only legally observable
  information (own hand, all battlefields, graveyards, etc). This is
  the standard ValueHead, used during actual play.

- **Oracle Dropout (Suphx-style)**: During training, oracle features are
  randomly dropped with increasing probability over time. This gradually
  weans the critic off perfect information, preventing the network from
  becoming dependent on information it won't have at inference time.

Usage:
    ctde = CTDEValueHeads(state_dim=768, oracle_extra_dim=128)

    # Training:
    obs_value, oracle_value = ctde(observable_state, oracle_features, epoch=50)
    loss = obs_value_loss + 0.5 * oracle_value_loss

    # Inference (oracle features zeroed or not provided):
    obs_value = ctde.observable_value(observable_state)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CTDEConfig:
    """Configuration for CTDE dual value heads."""
    # Observable state dimension (from encoder)
    state_dim: int = 768

    # Oracle-exclusive features dimension
    # These are features only available during training:
    # opponent hand contents, hidden zone cards, etc.
    oracle_extra_dim: int = 128

    # Value head architecture
    hidden_dim: int = 384
    n_layers: int = 2
    dropout: float = 0.1

    # Oracle dropout schedule
    oracle_dropout_start: float = 0.0    # dropout prob at epoch 0
    oracle_dropout_end: float = 0.8      # dropout prob at final epoch
    oracle_dropout_warmup: int = 200     # epochs to reach end dropout


def oracle_dropout_schedule(epoch: int, config: CTDEConfig) -> float:
    """
    Compute oracle dropout probability for a given epoch.

    Linearly increases from oracle_dropout_start to oracle_dropout_end
    over oracle_dropout_warmup epochs.
    """
    if epoch >= config.oracle_dropout_warmup:
        return config.oracle_dropout_end

    t = epoch / max(config.oracle_dropout_warmup, 1)
    return config.oracle_dropout_start + t * (config.oracle_dropout_end - config.oracle_dropout_start)


class OracleValueHead(nn.Module):
    """
    Value head with access to oracle (hidden) information.

    Takes the observable state PLUS oracle-exclusive features as input.
    Only used during training as a critic with better signal.
    """

    def __init__(
        self,
        config: CTDEConfig,
        num_players: int = 2,
    ):
        super().__init__()
        self.config = config
        self.num_players = num_players

        # Oracle features projection
        self.oracle_proj = nn.Sequential(
            nn.Linear(config.oracle_extra_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Combined state + oracle -> value
        combined_dim = config.state_dim + config.hidden_dim

        layers = []
        in_dim = combined_dim
        for _ in range(config.n_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.hidden_dim

        self.hidden = nn.Sequential(*layers)

        if num_players <= 2:
            self.output = nn.Linear(config.hidden_dim, 1)
        else:
            self.output = nn.Linear(config.hidden_dim, num_players)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        oracle_features: torch.Tensor,
        oracle_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute value with oracle information.

        Args:
            state: [batch, state_dim] - observable game state
            oracle_features: [batch, oracle_extra_dim] - hidden info features
            oracle_dropout_prob: probability of dropping oracle features to zero

        Returns:
            value: [batch, 1] for 1v1 or [batch, num_players] for multiplayer
        """
        # Apply oracle dropout (Suphx-style: applies during both train and eval
        # since it's a deliberate information ablation, not regularization)
        if oracle_dropout_prob > 0:
            mask = torch.bernoulli(
                torch.full((state.shape[0], 1), 1.0 - oracle_dropout_prob,
                           device=state.device)
            )
            oracle_features = oracle_features * mask

        # Project oracle features
        oracle_proj = self.oracle_proj(oracle_features)

        # Combine
        combined = torch.cat([state, oracle_proj], dim=-1)
        hidden = self.hidden(combined)
        output = self.output(hidden)

        if self.num_players <= 2:
            return torch.tanh(output)
        else:
            return F.softmax(output, dim=-1)


class CTDEValueHeads(nn.Module):
    """
    Dual value heads for Centralized Training, Decentralized Execution.

    Wraps both the observable value head (standard) and oracle value head
    (training-only critic with hidden information access).
    """

    def __init__(
        self,
        config: Optional[CTDEConfig] = None,
        num_players: int = 2,
    ):
        super().__init__()
        self.config = config or CTDEConfig()
        self.num_players = num_players

        # Observable value head (used at inference)
        from src.forge.policy_value_heads import PolicyValueConfig, ValueHead
        pv_config = PolicyValueConfig(
            state_dim=self.config.state_dim,
            value_hidden_dim=self.config.hidden_dim,
            value_n_layers=self.config.n_layers,
            value_dropout=self.config.dropout,
        )
        self.observable_head = ValueHead(pv_config, num_players)

        # Oracle value head (used during training only)
        self.oracle_head = OracleValueHead(self.config, num_players)

    def forward(
        self,
        state: torch.Tensor,
        oracle_features: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute both observable and oracle values.

        Args:
            state: [batch, state_dim] - observable game state
            oracle_features: [batch, oracle_extra_dim] - hidden info (training only)
            epoch: current training epoch (for oracle dropout schedule)

        Returns:
            observable_value: [batch, 1] or [batch, num_players]
            oracle_value: same shape, or None if oracle_features not provided
        """
        # Observable value (always computed)
        obs_value = self.observable_head(state)

        # Oracle value (only during training with oracle features)
        oracle_value = None
        if oracle_features is not None:
            dropout_prob = oracle_dropout_schedule(epoch, self.config)
            oracle_value = self.oracle_head(state, oracle_features, dropout_prob)

        return obs_value, oracle_value

    def observable_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute only observable value (for inference)."""
        return self.observable_head(state)


def compute_ctde_loss(
    obs_value: torch.Tensor,
    oracle_value: Optional[torch.Tensor],
    target: torch.Tensor,
    oracle_weight: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined CTDE value loss.

    Args:
        obs_value: [batch, 1] observable value prediction
        oracle_value: [batch, 1] oracle value prediction (or None)
        target: [batch, 1] actual game outcome
        oracle_weight: weight for oracle loss (0-1)

    Returns:
        total_loss: combined scalar loss
        metrics: dict with individual loss components
    """
    obs_loss = F.mse_loss(obs_value, target)

    metrics = {'obs_value_loss': obs_loss.item()}

    if oracle_value is not None:
        oracle_loss = F.mse_loss(oracle_value, target)
        total_loss = obs_loss + oracle_weight * oracle_loss
        metrics['oracle_value_loss'] = oracle_loss.item()
        metrics['total_value_loss'] = total_loss.item()
    else:
        total_loss = obs_loss
        metrics['total_value_loss'] = obs_loss.item()

    return total_loss, metrics


def extract_oracle_features(
    game_state: dict,
    player_idx: int,
    feature_dim: int = 128,
) -> torch.Tensor:
    """
    Extract oracle-exclusive features from a game state.

    These are features the player cannot legally see:
    - Opponent's hand contents
    - Top cards of opponent's library
    - Face-down cards' identities

    Args:
        game_state: parsed Forge game state dict
        player_idx: which player's perspective (0-3)
        feature_dim: output dimension

    Returns:
        oracle_features: [feature_dim] tensor
    """
    features = torch.zeros(feature_dim)

    players = game_state.get('players', [])
    feat_idx = 0

    for i, player in enumerate(players):
        if i == player_idx:
            continue  # Skip self (already in observable state)

        # Opponent hand info (hidden from the player)
        hand = player.get('cards', {}).get(0, [])  # Zone.HAND = 0

        # Hand size (already observable, but exact contents aren't)
        if feat_idx < feature_dim:
            features[feat_idx] = min(len(hand), 15) / 15.0
            feat_idx += 1

        # Aggregate hand stats (oracle info)
        total_cmc = 0
        creature_count = 0
        spell_count = 0
        land_count = 0

        for card in hand[:15]:  # Cap at 15
            cmc = getattr(card, 'cmc', 0) if hasattr(card, 'cmc') else card.get('cmc', 0)
            total_cmc += cmc

            is_creature = getattr(card, 'is_creature', False) if hasattr(card, 'is_creature') else card.get('is_creature', False)
            is_land = getattr(card, 'is_land', False) if hasattr(card, 'is_land') else card.get('is_land', False)
            is_instant = getattr(card, 'is_instant', False) if hasattr(card, 'is_instant') else card.get('is_instant', False)
            is_sorcery = getattr(card, 'is_sorcery', False) if hasattr(card, 'is_sorcery') else card.get('is_sorcery', False)

            if is_creature:
                creature_count += 1
            if is_instant or is_sorcery:
                spell_count += 1
            if is_land:
                land_count += 1

        if feat_idx + 4 <= feature_dim:
            features[feat_idx] = min(total_cmc, 50) / 50.0
            features[feat_idx + 1] = min(creature_count, 7) / 7.0
            features[feat_idx + 2] = min(spell_count, 7) / 7.0
            features[feat_idx + 3] = min(land_count, 7) / 7.0
            feat_idx += 4

    return features
