"""
Training Module for MTG RL Agent

Components:
- SelfPlayTrainer: Full AlphaZero-style training loop
- SelfPlayActor: Plays games using MCTS
- Learner: Trains network on self-play data
- ReplayBuffer: Stores game trajectories
"""

from src.training.self_play import (
    SelfPlayTrainer,
    SelfPlayConfig,
    SelfPlayActor,
    Learner,
    ReplayBuffer,
    TrainingSample,
    AlphaZeroNetwork,
)

__all__ = [
    "SelfPlayTrainer",
    "SelfPlayConfig",
    "SelfPlayActor",
    "Learner",
    "ReplayBuffer",
    "TrainingSample",
    "AlphaZeroNetwork",
]
