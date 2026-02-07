#!/usr/bin/env python3
"""
Network Benchmark

Profiles the actual AlphaZero network forward pass to get real timing numbers.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.training.profiler import get_gpu_info, estimate_batch_memory


def benchmark_forward_pass(
    batch_size: int = 32,
    num_iterations: int = 100,
    device_str: str = "auto",
):
    """Benchmark forward pass with different configurations."""
    from src.training.self_play import AlphaZeroNetwork
    from src.forge.game_state_encoder import GameStateConfig

    # Device selection
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Device: {device}")

    # Create network
    print("Loading network...")
    encoder_config = GameStateConfig(output_dim=512)
    network = AlphaZeroNetwork(encoder_config=encoder_config).to(device)
    network.eval()

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Parameters: {num_params:,}")

    # Memory estimate
    mem = estimate_batch_memory(num_params, batch_size)
    print(f"Estimated memory (batch={batch_size}): {mem['total_mb']:.1f} MB")
    print()

    # Generate dummy inputs matching the encoder's expected format
    # The ForgeGameStateEncoder expects specific zone inputs
    # For benchmarking, we'll bypass the encoder and test the network components

    print("=" * 60)
    print("FORWARD PASS BENCHMARK")
    print("=" * 60)

    # Test with raw state embedding (bypassing encoder)
    state_dim = encoder_config.output_dim
    dummy_state = torch.randn(batch_size, state_dim, device=device)

    # Warmup
    print("Warming up...")
    for _ in range(20):
        with torch.no_grad():
            # Test policy and value heads directly
            _ = network.policy_head(dummy_state, return_logits=True)
            _ = network.value_head(dummy_state)

    # Benchmark heads only (most common operation during MCTS)
    print(f"\nBenchmarking {num_iterations} iterations at batch_size={batch_size}...")

    times = []
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = network.policy_head(dummy_state, return_logits=True)
            _ = network.value_head(dummy_state)

        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to ms

    print(f"\nResults (heads only, batch={batch_size}):")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Std:  {times.std():.2f} ms")
    print(f"  Min:  {times.min():.2f} ms")
    print(f"  Max:  {times.max():.2f} ms")
    print(f"  P95:  {np.percentile(times, 95):.2f} ms")
    print(f"  Throughput: {batch_size / (times.mean() / 1000):.0f} states/sec")

    # Test different batch sizes
    print("\n" + "=" * 60)
    print("BATCH SIZE SCALING")
    print("=" * 60)
    print(f"{'Batch':>8} {'Mean (ms)':>12} {'Throughput':>15}")
    print("-" * 40)

    for bs in [1, 8, 16, 32, 64, 128, 256]:
        dummy = torch.randn(bs, state_dim, device=device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = network.policy_head(dummy, return_logits=True)

        # Benchmark
        times = []
        for _ in range(50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                _ = network.policy_head(dummy, return_logits=True)
                _ = network.value_head(dummy)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_ms = np.mean(times) * 1000
        throughput = bs / np.mean(times)
        print(f"{bs:>8} {mean_ms:>12.2f} {throughput:>15,.0f}")

    # Memory usage
    if device.type == "cuda":
        print("\n" + "=" * 60)
        print("GPU MEMORY")
        print("=" * 60)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")


def main():
    print("=" * 60)
    print("MTG RL NETWORK BENCHMARK")
    print("=" * 60)
    print()

    # GPU info
    gpu_info = get_gpu_info()
    print("GPU Information:")
    print(f"  CUDA available: {gpu_info['cuda_available']}")
    print(f"  MPS available: {gpu_info['mps_available']}")
    if gpu_info['devices']:
        for dev in gpu_info['devices']:
            print(f"  GPU {dev['index']}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
    print()

    # Run benchmark
    benchmark_forward_pass(batch_size=32, num_iterations=100)


if __name__ == "__main__":
    main()
