"""
Training Module for MTG RL Agent

Components:
- SelfPlayTrainer: Full AlphaZero-style training loop
- SelfPlayActor: Plays games using MCTS
- Learner: Trains network on self-play data
- ReplayBuffer: Stores game trajectories
- ParallelSelfPlayTrainer: Multi-actor parallel training
- TrainingProfiler: Performance profiling and benchmarking
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

from src.training.profiler import (
    TrainingProfiler,
    TimingStats,
    MemoryStats,
    get_gpu_info,
    estimate_batch_memory,
    estimate_training_time,
    compare_configurations,
)

from src.training.parallel_selfplay import (
    ParallelSelfPlayTrainer,
    ParallelConfig,
    BatchedInferenceServer,
    ParallelActor,
)

__all__ = [
    # Core self-play
    "SelfPlayTrainer",
    "SelfPlayConfig",
    "SelfPlayActor",
    "Learner",
    "ReplayBuffer",
    "TrainingSample",
    "AlphaZeroNetwork",
    # Parallel training
    "ParallelSelfPlayTrainer",
    "ParallelConfig",
    "BatchedInferenceServer",
    "ParallelActor",
    # Profiling
    "TrainingProfiler",
    "TimingStats",
    "MemoryStats",
    "get_gpu_info",
    "estimate_batch_memory",
    "estimate_training_time",
    "compare_configurations",
]
