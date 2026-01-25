"""
Training Profiler for MTG RL

Measures bottlenecks in self-play training:
1. Network forward pass time
2. MCTS simulation time
3. Forge communication latency
4. Data transfer overhead
5. Memory usage

Usage:
    from src.training.profiler import TrainingProfiler, profile_training_step

    profiler = TrainingProfiler()
    with profiler.measure("forward_pass"):
        output = network(state)
    profiler.report()
"""

import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import statistics

import torch


# =============================================================================
# PROFILER CLASSES
# =============================================================================

@dataclass
class TimingStats:
    """Statistics for a single profiling category."""
    times: List[float] = field(default_factory=list)

    def add(self, duration: float):
        self.times.append(duration)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0.0

    @property
    def p95(self) -> float:
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(0.95 * len(sorted_times))
        return sorted_times[min(idx, len(sorted_times) - 1)]


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    measurements: List[Dict[str, float]] = field(default_factory=list)

    def add(self, stats: Dict[str, float]):
        self.measurements.append(stats)

    @property
    def peak_gpu_mb(self) -> float:
        if not self.measurements:
            return 0.0
        return max(m.get("gpu_allocated_mb", 0) for m in self.measurements)

    @property
    def current_gpu_mb(self) -> float:
        if not self.measurements:
            return 0.0
        return self.measurements[-1].get("gpu_allocated_mb", 0)


class TrainingProfiler:
    """
    Comprehensive profiler for self-play training.

    Categories:
    - forward_pass: Neural network inference
    - backward_pass: Gradient computation
    - mcts_simulation: MCTS tree search
    - mcts_expansion: Node expansion
    - forge_comm: Forge daemon communication
    - data_transfer: CPU<->GPU transfers
    - batch_assembly: Collecting and batching data
    """

    def __init__(self, enabled: bool = True, sync_cuda: bool = True):
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.memory = MemoryStats()
        self.counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()

    @contextmanager
    def measure(self, category: str):
        """Context manager to measure time for a category."""
        if not self.enabled:
            yield
            return

        # Sync CUDA if available and requested
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            duration = time.perf_counter() - start
            with self._lock:
                self.timings[category].add(duration)

    def record_memory(self):
        """Record current memory usage."""
        if not self.enabled:
            return

        stats = {}

        # GPU memory (CUDA)
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
            stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6

        # GPU memory (MPS)
        elif torch.backends.mps.is_available():
            # MPS doesn't have detailed memory APIs yet
            stats["mps_available"] = True

        with self._lock:
            self.memory.add(stats)

    def increment(self, counter: str, amount: int = 1):
        """Increment a counter."""
        if not self.enabled:
            return
        with self._lock:
            self.counters[counter] += amount

    def get_stats(self) -> Dict[str, Any]:
        """Get all profiling statistics."""
        elapsed = time.perf_counter() - self._start_time

        stats = {
            "elapsed_seconds": elapsed,
            "timings": {},
            "counters": dict(self.counters),
            "memory": {
                "peak_gpu_mb": self.memory.peak_gpu_mb,
                "current_gpu_mb": self.memory.current_gpu_mb,
            }
        }

        for category, timing in self.timings.items():
            stats["timings"][category] = {
                "count": timing.count,
                "total_s": timing.total,
                "mean_ms": timing.mean * 1000,
                "median_ms": timing.median * 1000,
                "std_ms": timing.std * 1000,
                "min_ms": timing.min * 1000,
                "max_ms": timing.max * 1000,
                "p95_ms": timing.p95 * 1000,
                "pct_of_total": (timing.total / elapsed * 100) if elapsed > 0 else 0,
            }

        return stats

    def report(self, show_all: bool = False) -> str:
        """Generate a human-readable profiling report."""
        stats = self.get_stats()

        lines = [
            "=" * 70,
            "TRAINING PROFILER REPORT",
            "=" * 70,
            f"Total elapsed: {stats['elapsed_seconds']:.2f}s",
            "",
            "TIMING BREAKDOWN:",
            "-" * 70,
            f"{'Category':<25} {'Count':>8} {'Total':>10} {'Mean':>10} {'P95':>10} {'%':>6}",
            "-" * 70,
        ]

        # Sort by total time descending
        sorted_timings = sorted(
            stats["timings"].items(),
            key=lambda x: x[1]["total_s"],
            reverse=True
        )

        for category, timing in sorted_timings:
            if not show_all and timing["count"] < 5:
                continue
            lines.append(
                f"{category:<25} {timing['count']:>8} "
                f"{timing['total_s']:>9.2f}s "
                f"{timing['mean_ms']:>9.2f}ms "
                f"{timing['p95_ms']:>9.2f}ms "
                f"{timing['pct_of_total']:>5.1f}%"
            )

        lines.extend([
            "-" * 70,
            "",
            "COUNTERS:",
        ])
        for counter, value in sorted(stats["counters"].items()):
            lines.append(f"  {counter}: {value:,}")

        lines.extend([
            "",
            "MEMORY:",
            f"  Peak GPU: {stats['memory']['peak_gpu_mb']:.1f} MB",
            f"  Current GPU: {stats['memory']['current_gpu_mb']:.1f} MB",
            "=" * 70,
        ])

        return "\n".join(lines)

    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self.timings.clear()
            self.memory = MemoryStats()
            self.counters.clear()
            self._start_time = time.perf_counter()


# =============================================================================
# GPU UTILITIES
# =============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_count": 0,
        "devices": [],
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    return info


def estimate_batch_memory(
    model_params: int,
    batch_size: int,
    state_dim: int = 512,
    dtype_bytes: int = 4,  # float32
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        state_dim: State embedding dimension
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)

    Returns:
        Memory estimates in MB
    """
    # Model weights
    model_mb = model_params * dtype_bytes / 1e6

    # Gradients (same size as model)
    gradients_mb = model_mb

    # Optimizer state (Adam: 2x model for momentum + variance)
    optimizer_mb = 2 * model_mb

    # Activations (rough estimate: 2-4x batch * state_dim per layer)
    # Assume ~20 layers with intermediate sizes
    activations_mb = batch_size * state_dim * 20 * dtype_bytes / 1e6

    # Input/output tensors
    io_mb = batch_size * state_dim * 2 * dtype_bytes / 1e6

    total_mb = model_mb + gradients_mb + optimizer_mb + activations_mb + io_mb

    return {
        "model_mb": model_mb,
        "gradients_mb": gradients_mb,
        "optimizer_mb": optimizer_mb,
        "activations_mb": activations_mb,
        "io_mb": io_mb,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
    }


# =============================================================================
# TRAINING TIME ESTIMATOR
# =============================================================================

@dataclass
class TrainingEstimate:
    """Training time and resource estimates."""
    games_per_hour: float
    samples_per_hour: float
    time_to_1m_samples_hours: float
    time_to_1m_samples_days: float
    gpu_memory_required_gb: float
    estimated_cost_usd: float  # AWS spot pricing

    def __str__(self) -> str:
        return (
            f"Games/hour: {self.games_per_hour:,.0f}\n"
            f"Samples/hour: {self.samples_per_hour:,.0f}\n"
            f"Time to 1M samples: {self.time_to_1m_samples_hours:.1f}h ({self.time_to_1m_samples_days:.1f} days)\n"
            f"GPU memory required: {self.gpu_memory_required_gb:.1f} GB\n"
            f"Estimated cost: ${self.estimated_cost_usd:.2f}"
        )


def estimate_training_time(
    num_actors: int = 1,
    mcts_simulations: int = 100,
    avg_game_length: int = 40,  # moves
    forward_pass_ms: float = 5.0,  # per inference
    forge_latency_ms: float = 50.0,  # per game action
    batch_size: int = 32,
    model_params: int = 6_000_000,
    gpu_type: str = "g4dn.xlarge",
) -> TrainingEstimate:
    """
    Estimate training time for different configurations.

    Key assumptions:
    - Each MCTS simulation requires 1 forward pass
    - Each game move requires MCTS + Forge communication
    - Parallel actors share GPU for batched inference
    """

    # Time per move (MCTS + Forge)
    mcts_time_ms = mcts_simulations * forward_pass_ms
    move_time_ms = mcts_time_ms + forge_latency_ms

    # Time per game
    game_time_ms = avg_game_length * move_time_ms
    game_time_s = game_time_ms / 1000

    # With batching, N actors can share forward passes
    # Effective speedup is sqrt(N) due to synchronization overhead
    effective_parallelism = num_actors ** 0.7
    effective_game_time_s = game_time_s / effective_parallelism

    # Games per hour
    games_per_hour = 3600 / effective_game_time_s

    # Samples per game (both players' perspectives for each move)
    samples_per_game = avg_game_length * 2
    samples_per_hour = games_per_hour * samples_per_game

    # Time to 1M samples
    time_to_1m_hours = 1_000_000 / samples_per_hour
    time_to_1m_days = time_to_1m_hours / 24

    # Memory estimate
    mem = estimate_batch_memory(model_params, batch_size * num_actors)

    # Cost estimate (AWS spot pricing)
    spot_prices = {
        "g4dn.xlarge": 0.16,    # 1x T4, 16GB
        "g4dn.2xlarge": 0.23,   # 1x T4, 32GB
        "g4dn.12xlarge": 1.20,  # 4x T4, 192GB
        "p3.2xlarge": 0.92,     # 1x V100, 16GB
        "p3.8xlarge": 3.67,     # 4x V100, 64GB
    }
    hourly_cost = spot_prices.get(gpu_type, 0.20)
    total_cost = time_to_1m_hours * hourly_cost

    return TrainingEstimate(
        games_per_hour=games_per_hour,
        samples_per_hour=samples_per_hour,
        time_to_1m_samples_hours=time_to_1m_hours,
        time_to_1m_samples_days=time_to_1m_days,
        gpu_memory_required_gb=mem["total_gb"],
        estimated_cost_usd=total_cost,
    )


def compare_configurations() -> str:
    """Compare different training configurations."""
    configs = [
        ("1 actor, 50 MCTS sims (baseline)", dict(num_actors=1, mcts_simulations=50)),
        ("1 actor, 100 MCTS sims", dict(num_actors=1, mcts_simulations=100)),
        ("4 actors, 50 MCTS sims", dict(num_actors=4, mcts_simulations=50)),
        ("8 actors, 50 MCTS sims", dict(num_actors=8, mcts_simulations=50)),
        ("8 actors, 100 MCTS sims", dict(num_actors=8, mcts_simulations=100)),
        ("16 actors, 50 MCTS sims (g4dn.12xlarge)", dict(num_actors=16, mcts_simulations=50, gpu_type="g4dn.12xlarge")),
    ]

    lines = [
        "=" * 90,
        "TRAINING CONFIGURATION COMPARISON",
        "=" * 90,
        f"{'Configuration':<40} {'Games/hr':>10} {'Samples/hr':>12} {'Days to 1M':>12} {'Cost':>10}",
        "-" * 90,
    ]

    for name, kwargs in configs:
        est = estimate_training_time(**kwargs)
        lines.append(
            f"{name:<40} {est.games_per_hour:>10,.0f} {est.samples_per_hour:>12,.0f} "
            f"{est.time_to_1m_samples_days:>12.1f} ${est.estimated_cost_usd:>9.2f}"
        )

    lines.extend([
        "-" * 90,
        "",
        "Notes:",
        "- Assumes 5ms forward pass, 50ms Forge latency, 40 moves/game",
        "- Parallelism efficiency: N^0.7 (accounts for synchronization)",
        "- Costs are AWS spot prices (70% cheaper than on-demand)",
        "=" * 90,
    ])

    return "\n".join(lines)


# =============================================================================
# PROFILED TRAINING STEP
# =============================================================================

def profile_training_step(
    network: torch.nn.Module,
    batch_size: int = 32,
    state_dim: int = 512,
    num_actions: int = 153,
    device: str = "cuda",
    num_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Profile a training step to measure actual performance.

    Returns detailed timing breakdown.
    """
    profiler = TrainingProfiler()

    # Setup
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    network = network.to(device)
    network.eval()

    # Create dummy data
    dummy_state = torch.randn(batch_size, state_dim, device=device)
    dummy_action_mask = torch.ones(batch_size, num_actions, device=device)
    dummy_target_policy = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    dummy_target_value = torch.randn(batch_size, 1, device=device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = network(dummy_state, dummy_action_mask)

    # Profile inference
    network.eval()
    for _ in range(num_iterations):
        with profiler.measure("inference"):
            with torch.no_grad():
                _ = network(dummy_state, dummy_action_mask)

    # Profile training step
    network.train()
    for _ in range(num_iterations):
        with profiler.measure("forward_train"):
            policy, value = network(dummy_state, dummy_action_mask)

        with profiler.measure("loss_compute"):
            policy_loss = -torch.sum(dummy_target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = torch.nn.functional.mse_loss(value, dummy_target_value)
            loss = policy_loss + value_loss

        with profiler.measure("backward"):
            optimizer.zero_grad()
            loss.backward()

        with profiler.measure("optimizer_step"):
            optimizer.step()

    # Profile data transfer
    cpu_tensor = torch.randn(batch_size, state_dim)
    for _ in range(num_iterations):
        with profiler.measure("cpu_to_gpu"):
            gpu_tensor = cpu_tensor.to(device)

        with profiler.measure("gpu_to_cpu"):
            _ = gpu_tensor.cpu()

    profiler.record_memory()

    return {
        "report": profiler.report(),
        "stats": profiler.get_stats(),
        "device": str(device),
        "batch_size": batch_size,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(compare_configurations())
    print()

    # Get GPU info
    gpu_info = get_gpu_info()
    print("GPU Information:")
    print(f"  CUDA available: {gpu_info['cuda_available']}")
    print(f"  MPS available: {gpu_info['mps_available']}")
    if gpu_info["devices"]:
        for dev in gpu_info["devices"]:
            print(f"  GPU {dev['index']}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
    print()

    # Memory estimate for our model
    mem = estimate_batch_memory(
        model_params=6_200_000,  # GameStateEncoder + PolicyValue
        batch_size=64,
    )
    print("Memory Estimate (batch=64, 6.2M params):")
    for k, v in mem.items():
        if k.endswith("_mb"):
            print(f"  {k}: {v:.1f} MB")
        elif k.endswith("_gb"):
            print(f"  {k}: {v:.2f} GB")
