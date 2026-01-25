#!/usr/bin/env python3
"""
Local Self-Play Test

Runs a quick self-play training test to verify everything works
before deploying to cloud.

Usage:
    python scripts/test_selfplay_local.py
    python scripts/test_selfplay_local.py --iterations 5 --games 5
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.self_play import SelfPlayTrainer, SelfPlayConfig


def main():
    parser = argparse.ArgumentParser(description="Test self-play training locally")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=3,
                        help="Games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=20,
                        help="MCTS simulations per move (more = better but slower)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/selfplay_test",
                        help="Directory for checkpoints")
    args = parser.parse_args()

    print("=" * 70)
    print("LOCAL SELF-PLAY TEST")
    print("=" * 70)
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print()

    # Small config for testing
    config = SelfPlayConfig(
        games_per_iteration=args.games,
        mcts_simulations=args.mcts_sims,
        epochs_per_iteration=2,
        batch_size=32,
        min_buffer_size=10,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=1,
    )

    # Create trainer
    trainer = SelfPlayTrainer(config)

    print(f"Network parameters: {sum(p.numel() for p in trainer.network.parameters()):,}")
    print(f"Device: {config.device}")
    print()

    # Run training
    start_time = time.time()
    trainer.train(num_iterations=args.iterations)
    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total games: {trainer.total_games}")
    print(f"Total samples: {trainer.total_samples}")
    print(f"Samples/second: {trainer.total_samples / elapsed:.1f}")
    print()
    print("If this ran successfully, the training pipeline is working!")
    print("Next: Deploy to cloud with real Forge integration")
    print("=" * 70)


if __name__ == "__main__":
    main()
