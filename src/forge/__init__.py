"""
Forge Integration Module

Components for integrating with the Forge MTG engine:
- GameStateEncoder: Converts Forge JSON state to neural network input
- (Future) PolicyHead: Outputs action probabilities
- (Future) ValueHead: Outputs expected win probability
- (Future) MCTS: Tree search using Forge as simulator
"""

from src.forge.game_state_encoder import (
    ForgeGameStateEncoder,
    GameStateConfig,
    MechanicsCache,
    Zone,
    Phase,
)

__all__ = [
    "ForgeGameStateEncoder",
    "GameStateConfig",
    "MechanicsCache",
    "Zone",
    "Phase",
]
