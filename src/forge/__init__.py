"""
Forge Integration Module

Components for integrating with the Forge MTG engine:
- GameStateEncoder: Converts Forge JSON state to neural network input
- PolicyHead: Outputs action probabilities over legal moves
- ValueHead: Outputs expected win probability
- PolicyValueNetwork: Combined AlphaZero-style network
- MCTS: Monte Carlo Tree Search for gameplay
"""

from src.forge.game_state_encoder import (
    ForgeGameStateEncoder,
    GameStateConfig,
    MechanicsCache,
    Zone,
    Phase,
)

from src.forge.policy_value_heads import (
    PolicyHead,
    ValueHead,
    PolicyValueNetwork,
    PolicyValueConfig,
    ActionConfig,
    ActionType,
    create_action_mask,
    decode_action,
)

from src.forge.mcts import (
    MCTS,
    MCTSConfig,
    MCTSNode,
    BatchedMCTS,
    ForgeClientInterface,
    SimulatedForgeClient,
)

__all__ = [
    # Game state encoder
    "ForgeGameStateEncoder",
    "GameStateConfig",
    "MechanicsCache",
    "Zone",
    "Phase",
    # Policy/Value heads
    "PolicyHead",
    "ValueHead",
    "PolicyValueNetwork",
    "PolicyValueConfig",
    "ActionConfig",
    "ActionType",
    "create_action_mask",
    "decode_action",
    # MCTS
    "MCTS",
    "MCTSConfig",
    "MCTSNode",
    "BatchedMCTS",
    "ForgeClientInterface",
    "SimulatedForgeClient",
]
