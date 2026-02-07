"""
Policy and Value Heads for AlphaZero-style MTG Agent

The policy head outputs action probabilities over legal moves.
The value head outputs expected win probability for the current state.

These are used together with MCTS:
- Policy guides the tree search (which moves to explore)
- Value provides state evaluation (without full rollout)

Action Space:
MTG has a variable action space - different legal actions each state.
We use action masking: network outputs over all possible action types,
then mask to legal actions before softmax.

Action Types:
1. Pass (always legal when you have priority)
2. Cast spell from hand (one per card in hand)
3. Activate ability (one per activatable permanent)
4. Attack with creature (during declare attackers)
5. Block with creature (during declare blockers)
6. Pay for triggered ability
7. Choose target
8. Choose mode/order

Usage:
    encoder = ForgeGameStateEncoder()
    policy_head = PolicyHead(config)
    value_head = ValueHead(config)

    state = encoder.encode_json(forge_state)
    action_probs = policy_head(state, action_mask)
    win_prob = value_head(state)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ACTION SPACE
# =============================================================================

class ActionType(IntEnum):
    """
    High-level action types in MTG.

    Each type maps to multiple specific actions (e.g., CAST_SPELL
    has one action per castable card in hand).
    """
    PASS = 0
    CAST_SPELL = 1
    ACTIVATE_ABILITY = 2
    ATTACK = 3
    BLOCK = 4
    PAY_COST = 5
    CHOOSE_TARGET = 6
    CHOOSE_MODE = 7
    MULLIGAN = 8
    CONCEDE = 9


@dataclass
class ActionConfig:
    """Configuration for the action space."""

    # Maximum actions per type
    max_hand_size: int = 15          # Max spells to cast
    max_battlefield: int = 50         # Max abilities/attackers/blockers
    max_targets: int = 20             # Max targets to choose
    max_modes: int = 5                # Max modes to choose (e.g., Commands)
    max_costs: int = 10               # Max costs to pay (e.g., triggers)

    # Special actions (always available when relevant)
    num_special: int = 3              # pass, mulligan, concede

    @property
    def total_actions(self) -> int:
        """Total size of action space."""
        return (
            self.num_special +        # pass, mulligan, concede
            self.max_hand_size +      # cast from hand
            self.max_battlefield * 2 + # activate or attack/block
            self.max_targets +         # choose target
            self.max_modes +           # choose mode
            self.max_costs             # pay cost
        )

    def get_action_indices(self, action_type: ActionType) -> Tuple[int, int]:
        """Get start and end indices for an action type."""
        offsets = {
            ActionType.PASS: (0, 1),
            ActionType.MULLIGAN: (1, 2),
            ActionType.CONCEDE: (2, 3),
            ActionType.CAST_SPELL: (3, 3 + self.max_hand_size),
            ActionType.ACTIVATE_ABILITY: (
                3 + self.max_hand_size,
                3 + self.max_hand_size + self.max_battlefield
            ),
            ActionType.ATTACK: (
                3 + self.max_hand_size + self.max_battlefield,
                3 + self.max_hand_size + self.max_battlefield * 2
            ),
            ActionType.BLOCK: (
                3 + self.max_hand_size + self.max_battlefield * 2,
                3 + self.max_hand_size + self.max_battlefield * 3
            ),
            ActionType.CHOOSE_TARGET: (
                3 + self.max_hand_size + self.max_battlefield * 3,
                3 + self.max_hand_size + self.max_battlefield * 3 + self.max_targets
            ),
            ActionType.CHOOSE_MODE: (
                3 + self.max_hand_size + self.max_battlefield * 3 + self.max_targets,
                3 + self.max_hand_size + self.max_battlefield * 3 + self.max_targets + self.max_modes
            ),
            ActionType.PAY_COST: (
                3 + self.max_hand_size + self.max_battlefield * 3 + self.max_targets + self.max_modes,
                3 + self.max_hand_size + self.max_battlefield * 3 + self.max_targets + self.max_modes + self.max_costs
            ),
        }
        return offsets.get(action_type, (0, 0))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PolicyValueConfig:
    """Configuration for policy and value heads."""

    # Input dimension (from game state encoder)
    state_dim: int = 512

    # Policy head
    policy_hidden_dim: int = 256
    policy_n_layers: int = 2
    policy_dropout: float = 0.1

    # Value head
    value_hidden_dim: int = 256
    value_n_layers: int = 2
    value_dropout: float = 0.1

    # Action space
    action_config: ActionConfig = None

    def __post_init__(self):
        if self.action_config is None:
            self.action_config = ActionConfig()

    @property
    def action_dim(self) -> int:
        return self.action_config.total_actions


# =============================================================================
# POLICY HEAD
# =============================================================================

class PolicyHead(nn.Module):
    """
    Policy network head for action selection.

    Takes state embedding, outputs action probabilities.
    Uses action masking to handle variable legal actions.
    """

    def __init__(self, config: Optional[PolicyValueConfig] = None):
        super().__init__()
        self.config = config or PolicyValueConfig()

        layers = []
        in_dim = self.config.state_dim

        for i in range(self.config.policy_n_layers):
            out_dim = self.config.policy_hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.config.policy_dropout),
            ])
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer (logits for each action)
        self.output = nn.Linear(self.config.policy_hidden_dim, self.config.action_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute action probabilities.

        Args:
            state: [batch, state_dim] - encoded game state
            action_mask: [batch, action_dim] - 1 for legal, 0 for illegal
            return_logits: If True, return raw logits instead of probabilities

        Returns:
            If return_logits: [batch, action_dim] logits
            Else: [batch, action_dim] probabilities (softmax over legal actions)
        """
        # Hidden layers
        hidden = self.hidden(state)

        # Output logits
        logits = self.output(hidden)

        if return_logits:
            return logits

        # Apply action mask (set illegal actions to -inf before softmax)
        if action_mask is not None:
            # Ensure at least one action is legal
            # Set illegal to -inf (becomes 0 after softmax)
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # Softmax over actions
        probs = F.softmax(logits, dim=-1)

        # Handle all-masked case (return uniform over actions)
        if action_mask is not None:
            # Where all actions were masked, use uniform
            all_masked = (action_mask.sum(dim=-1) == 0).unsqueeze(-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = torch.where(all_masked, uniform, probs)

        return probs

    def sample_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: [batch, state_dim]
            action_mask: [batch, action_dim]
            temperature: Sampling temperature (higher = more random)

        Returns:
            action: [batch] - selected action indices
            log_prob: [batch] - log probability of selected actions
        """
        # Get logits
        logits = self.forward(state, action_mask, return_logits=True)

        # Apply temperature
        logits = logits / temperature

        # Mask illegal actions
        logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def get_action_prob(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get probability of a specific action.

        Args:
            state: [batch, state_dim]
            action_mask: [batch, action_dim]
            action: [batch] - action indices

        Returns:
            prob: [batch] - probability of each action
        """
        probs = self.forward(state, action_mask)
        return probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)


# =============================================================================
# VALUE HEAD
# =============================================================================

class ValueHead(nn.Module):
    """
    Value network head for state evaluation.

    Takes state embedding, outputs expected win probability.
    For multiplayer (Commander), outputs win probabilities for each player.
    """

    def __init__(
        self,
        config: Optional[PolicyValueConfig] = None,
        num_players: int = 1,
    ):
        super().__init__()
        self.config = config or PolicyValueConfig()
        self.num_players = num_players

        layers = []
        in_dim = self.config.state_dim

        for i in range(self.config.value_n_layers):
            out_dim = self.config.value_hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.config.value_dropout),
            ])
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer
        # For 1v1: single value in [-1, 1] (tanh activation)
        # For multiplayer: win probability per player (softmax)
        if num_players <= 2:
            self.output = nn.Linear(self.config.value_hidden_dim, 1)
        else:
            self.output = nn.Linear(self.config.value_hidden_dim, num_players)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            state: [batch, state_dim]

        Returns:
            For 1v1: [batch, 1] values in [-1, 1]
            For multiplayer: [batch, num_players] win probabilities
        """
        hidden = self.hidden(state)
        output = self.output(hidden)

        if self.num_players <= 2:
            # Tanh for 1v1 (like AlphaZero)
            return torch.tanh(output)
        else:
            # Softmax for multiplayer win probabilities
            return F.softmax(output, dim=-1)


# =============================================================================
# COMBINED NETWORK
# =============================================================================

class PolicyValueNetwork(nn.Module):
    """
    Combined policy-value network (AlphaZero style).

    Shares the state encoder between policy and value heads
    for efficient computation and shared representations.
    """

    def __init__(
        self,
        state_encoder: nn.Module,
        config: Optional[PolicyValueConfig] = None,
        num_players: int = 2,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.config = config or PolicyValueConfig()

        self.policy_head = PolicyHead(self.config)
        self.value_head = ValueHead(self.config, num_players)

    def forward(
        self,
        **encoder_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and both heads.

        Args:
            **encoder_inputs: Inputs for the state encoder

        Returns:
            policy_logits: [batch, action_dim]
            value: [batch, 1] or [batch, num_players]
        """
        # Encode state
        state = self.state_encoder(**encoder_inputs)

        # Get policy logits and value
        policy_logits = self.policy_head(state, return_logits=True)
        value = self.value_head(state)

        return policy_logits, value

    def get_action_and_value(
        self,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
        **encoder_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value in one forward pass.

        Args:
            action_mask: [batch, action_dim]
            temperature: Sampling temperature
            **encoder_inputs: Inputs for state encoder

        Returns:
            action: [batch] sampled actions
            log_prob: [batch] log probabilities
            entropy: [batch] policy entropy
            value: [batch, 1] or [batch, num_players]
        """
        # Encode state
        state = self.state_encoder(**encoder_inputs)

        # Get logits and value
        logits = self.policy_head(state, return_logits=True)
        value = self.value_head(state)

        # Apply temperature and mask
        logits = logits / temperature
        logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # Sample action
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_action(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        **encoder_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given action (for training).

        Args:
            action: [batch] action indices
            action_mask: [batch, action_dim]
            **encoder_inputs: Inputs for state encoder

        Returns:
            log_prob: [batch] log probability of action
            entropy: [batch] policy entropy
            value: [batch, 1] or [batch, num_players]
        """
        # Encode state
        state = self.state_encoder(**encoder_inputs)

        # Get logits and value
        logits = self.policy_head(state, return_logits=True)
        value = self.value_head(state)

        # Mask and compute log prob
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, value

    def save(self, path: str):
        """Save network to file."""
        torch.save({
            'config': self.config,
            'num_players': self.value_head.num_players,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        state_encoder: nn.Module,
        device: Optional[torch.device] = None
    ) -> 'PolicyValueNetwork':
        """Load network from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        network = cls(
            state_encoder,
            checkpoint['config'],
            checkpoint['num_players']
        )
        network.load_state_dict(checkpoint['state_dict'])
        return network


# =============================================================================
# ACTION MASK UTILITIES
# =============================================================================

def create_action_mask(
    legal_actions: List[Dict],
    config: Optional[ActionConfig] = None
) -> np.ndarray:
    """
    Create action mask from Forge's legal actions list.

    Args:
        legal_actions: List of legal action dicts from Forge
            Format: [{"type": "cast", "card_idx": 0}, {"type": "pass"}, ...]
        config: Action configuration

    Returns:
        mask: [action_dim] binary array, 1 for legal actions
    """
    if config is None:
        config = ActionConfig()

    mask = np.zeros(config.total_actions, dtype=np.float32)

    for action in legal_actions:
        action_type = action.get("type", "").lower()
        idx = action.get("index", 0)

        if action_type == "pass":
            start, _ = config.get_action_indices(ActionType.PASS)
            mask[start] = 1.0

        elif action_type == "mulligan":
            start, _ = config.get_action_indices(ActionType.MULLIGAN)
            mask[start] = 1.0

        elif action_type == "concede":
            start, _ = config.get_action_indices(ActionType.CONCEDE)
            mask[start] = 1.0

        elif action_type in ["cast", "play"]:
            start, end = config.get_action_indices(ActionType.CAST_SPELL)
            if start + idx < end:
                mask[start + idx] = 1.0

        elif action_type in ["activate", "ability"]:
            start, end = config.get_action_indices(ActionType.ACTIVATE_ABILITY)
            if start + idx < end:
                mask[start + idx] = 1.0

        elif action_type == "attack":
            start, end = config.get_action_indices(ActionType.ATTACK)
            if start + idx < end:
                mask[start + idx] = 1.0

        elif action_type == "block":
            start, end = config.get_action_indices(ActionType.BLOCK)
            if start + idx < end:
                mask[start + idx] = 1.0

        elif action_type == "target":
            start, end = config.get_action_indices(ActionType.CHOOSE_TARGET)
            if start + idx < end:
                mask[start + idx] = 1.0

        elif action_type == "mode":
            start, end = config.get_action_indices(ActionType.CHOOSE_MODE)
            if start + idx < end:
                mask[start + idx] = 1.0

        elif action_type in ["pay", "cost"]:
            start, end = config.get_action_indices(ActionType.PAY_COST)
            if start + idx < end:
                mask[start + idx] = 1.0

    return mask


def decode_action(
    action_idx: int,
    config: Optional[ActionConfig] = None
) -> Dict:
    """
    Decode action index to action type and parameters.

    Args:
        action_idx: Index in the action space
        config: Action configuration

    Returns:
        Action dict with type and index
    """
    if config is None:
        config = ActionConfig()

    # Check each action type range
    for action_type in ActionType:
        start, end = config.get_action_indices(action_type)
        if start <= action_idx < end:
            return {
                "type": action_type.name.lower(),
                "index": action_idx - start,
            }

    return {"type": "unknown", "index": action_idx}


# =============================================================================
# TESTING
# =============================================================================

def test_policy_value_heads():
    """Test policy and value heads."""
    print("=" * 70)
    print("Testing Policy and Value Heads")
    print("=" * 70)

    config = PolicyValueConfig()

    print(f"\nAction space size: {config.action_dim}")
    print("Action breakdown:")
    for action_type in ActionType:
        start, end = config.action_config.get_action_indices(action_type)
        print(f"  {action_type.name}: indices {start}-{end} ({end - start} actions)")

    # Test policy head
    print("\n--- Policy Head ---")
    policy = PolicyHead(config)
    policy_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {policy_params:,}")

    batch_size = 4
    state = torch.randn(batch_size, config.state_dim)
    action_mask = torch.zeros(batch_size, config.action_dim)
    action_mask[:, 0] = 1  # Pass always legal
    action_mask[:, 3:6] = 1  # 3 spells castable

    with torch.no_grad():
        probs = policy(state, action_mask)
        print(f"Output shape: {probs.shape}")
        print(f"Probs sum: {probs.sum(dim=-1)}")
        print(f"Non-zero probs per sample: {(probs > 0).sum(dim=-1)}")

        # Sample action
        action, log_prob = policy.sample_action(state, action_mask)
        print(f"Sampled actions: {action}")
        print(f"Log probs: {log_prob}")

    # Test value head (1v1)
    print("\n--- Value Head (1v1) ---")
    value_1v1 = ValueHead(config, num_players=2)
    value_params = sum(p.numel() for p in value_1v1.parameters())
    print(f"Parameters: {value_params:,}")

    with torch.no_grad():
        value = value_1v1(state)
        print(f"Output shape: {value.shape}")
        print(f"Values (should be in [-1, 1]): {value.squeeze()}")

    # Test value head (Commander 4-player)
    print("\n--- Value Head (4-player Commander) ---")
    value_4p = ValueHead(config, num_players=4)

    with torch.no_grad():
        win_probs = value_4p(state)
        print(f"Output shape: {win_probs.shape}")
        print(f"Win probs sum (should be 1): {win_probs.sum(dim=-1)}")
        print(f"Sample probs: {win_probs[0]}")

    # Test action mask utilities
    print("\n--- Action Mask Utilities ---")
    legal_actions = [
        {"type": "pass"},
        {"type": "cast", "index": 0},
        {"type": "cast", "index": 2},
        {"type": "activate", "index": 5},
    ]
    mask = create_action_mask(legal_actions)
    print(f"Mask shape: {mask.shape}")
    print(f"Legal actions: {np.nonzero(mask)[0]}")

    # Decode actions
    for idx in np.nonzero(mask)[0]:
        decoded = decode_action(idx)
        print(f"  Index {idx}: {decoded}")

    print("\n" + "=" * 70)
    print("Policy and Value Heads test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_policy_value_heads()
