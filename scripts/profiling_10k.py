#!/usr/bin/env python3
"""
Comprehensive Profiling for 10,000 Games

Collects detailed timing data at function level:
- Per-game timing breakdown
- Network latency vs computation
- Game phases timing
- Decision complexity metrics
"""

import socket
import time
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Configuration
DAEMON_HOST = "localhost"
DAEMON_PORT = 17171
NUM_GAMES = 10000
MAX_PARALLEL = 10
GAME_TIMEOUT = 120

@dataclass
class GameProfile:
    """Detailed profile for a single game."""
    game_id: int
    success: bool
    total_duration_ms: float
    connect_time_ms: float = 0
    send_time_ms: float = 0
    first_byte_time_ms: float = 0
    recv_time_ms: float = 0
    parse_time_ms: float = 0
    turns: int = 0
    decisions: int = 0
    winner: str = ""
    error: str = ""
    response_size_bytes: int = 0


@dataclass
class AggregateStats:
    """Aggregate profiling statistics."""
    games: List[GameProfile] = field(default_factory=list)

    def add(self, profile: GameProfile):
        self.games.append(profile)

    def compute_stats(self) -> Dict:
        successful = [g for g in self.games if g.success]
        failed = [g for g in self.games if not g.success]

        if not successful:
            return {"error": "No successful games"}

        # Duration stats
        durations = [g.total_duration_ms for g in successful]
        connect_times = [g.connect_time_ms for g in successful]
        send_times = [g.send_time_ms for g in successful]
        first_byte_times = [g.first_byte_time_ms for g in successful]
        recv_times = [g.recv_time_ms for g in successful]

        def percentile(data, p):
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data)-1)]

        stats = {
            "total_games": len(self.games),
            "successful_games": len(successful),
            "failed_games": len(failed),
            "success_rate": len(successful) / len(self.games),

            # Total duration
            "duration_min_ms": min(durations),
            "duration_max_ms": max(durations),
            "duration_mean_ms": statistics.mean(durations),
            "duration_median_ms": statistics.median(durations),
            "duration_stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "duration_p50_ms": percentile(durations, 50),
            "duration_p90_ms": percentile(durations, 90),
            "duration_p95_ms": percentile(durations, 95),
            "duration_p99_ms": percentile(durations, 99),

            # Breakdown by phase
            "connect_mean_ms": statistics.mean(connect_times),
            "connect_p95_ms": percentile(connect_times, 95),
            "send_mean_ms": statistics.mean(send_times),
            "first_byte_mean_ms": statistics.mean(first_byte_times),
            "first_byte_p95_ms": percentile(first_byte_times, 95),
            "recv_mean_ms": statistics.mean(recv_times),
            "recv_p95_ms": percentile(recv_times, 95),

            # Percentage breakdown
            "connect_pct": statistics.mean(connect_times) / statistics.mean(durations) * 100,
            "first_byte_pct": statistics.mean(first_byte_times) / statistics.mean(durations) * 100,
            "recv_pct": statistics.mean(recv_times) / statistics.mean(durations) * 100,

            # Game complexity
            "turns_mean": statistics.mean([g.turns for g in successful]),
            "response_size_mean_kb": statistics.mean([g.response_size_bytes for g in successful]) / 1024,
        }

        # Compute server-side time (total - network)
        network_times = [g.connect_time_ms + g.send_time_ms for g in successful]
        server_times = [g.first_byte_time_ms - g.connect_time_ms - g.send_time_ms for g in successful]

        stats["network_mean_ms"] = statistics.mean(network_times)
        stats["server_processing_mean_ms"] = statistics.mean(server_times)
        stats["server_pct"] = statistics.mean(server_times) / statistics.mean(durations) * 100

        return stats


def run_profiled_game(game_id: int) -> GameProfile:
    """Run a single game with detailed timing."""
    profile = GameProfile(
        game_id=game_id,
        success=False,
        total_duration_ms=0
    )

    t_start = time.perf_counter()

    try:
        # Create socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(GAME_TIMEOUT)

        # Connect timing
        t_connect_start = time.perf_counter()
        s.connect((DAEMON_HOST, DAEMON_PORT))
        t_connect_end = time.perf_counter()
        profile.connect_time_ms = (t_connect_end - t_connect_start) * 1000

        # Send command timing
        cmd = "NEWGAME test_red.dck test_blue.dck -q -c 60\n"
        t_send_start = time.perf_counter()
        s.sendall(cmd.encode())
        t_send_end = time.perf_counter()
        profile.send_time_ms = (t_send_end - t_send_start) * 1000

        # Time to first byte
        t_recv_start = time.perf_counter()
        first_chunk = s.recv(4096)
        t_first_byte = time.perf_counter()
        profile.first_byte_time_ms = (t_first_byte - t_start) * 1000

        # Receive rest of response
        response = first_chunk
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"GAME_RESULT:" in response or b"ERROR:" in response:
                    break
            except socket.timeout:
                profile.error = "recv timeout"
                break

        t_recv_end = time.perf_counter()
        profile.recv_time_ms = (t_recv_end - t_recv_start) * 1000
        profile.response_size_bytes = len(response)

        # Parse response
        response_str = response.decode('utf-8', errors='ignore')

        if "GAME_RESULT:" in response_str:
            profile.success = True
            # Extract winner
            if "won" in response_str:
                parts = response_str.strip().split()
                for i, p in enumerate(parts):
                    if p == "won":
                        profile.winner = parts[i-1] if i > 0 else ""
                        break

            # Count turns from response
            profile.turns = response_str.count("Turn ")

        elif "ERROR:" in response_str:
            profile.error = response_str.strip()[:100]
        else:
            profile.error = f"Unexpected response: {response_str[:50]}"

        s.close()

    except Exception as e:
        profile.error = str(e)

    t_end = time.perf_counter()
    profile.total_duration_ms = (t_end - t_start) * 1000

    return profile


def run_benchmark():
    """Run the full benchmark with detailed profiling."""
    print("=" * 70)
    print("MTG FORGE DAEMON PROFILING - 10,000 GAMES")
    print("=" * 70)
    print("Configuration:")
    print(f"  Total games: {NUM_GAMES:,}")
    print(f"  Parallel workers: {MAX_PARALLEL}")
    print(f"  Game timeout: {GAME_TIMEOUT}s")
    print("=" * 70)
    print()

    stats = AggregateStats()
    wins = defaultdict(int)
    start_time = time.time()

    # Progress tracking
    completed = 0
    lock = threading.Lock()

    def update_progress(profile):
        nonlocal completed
        with lock:
            completed += 1
            if profile.success:
                wins[profile.winner] += 1

            if completed % 500 == 0 or completed == NUM_GAMES:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = NUM_GAMES - completed
                eta = remaining / rate if rate > 0 else 0

                # Compute running stats
                successful = [g for g in stats.games if g.success]
                avg_duration = statistics.mean([g.total_duration_ms for g in successful]) if successful else 0

                print(f"Progress: {completed:,}/{NUM_GAMES:,} | "
                      f"Rate: {rate:.1f}/s | "
                      f"Avg game: {avg_duration:.0f}ms | "
                      f"ETA: {eta/60:.1f}min")

    # Run parallel games
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_profiled_game, i): i
                   for i in range(NUM_GAMES)}

        for future in as_completed(futures):
            profile = future.result()
            stats.add(profile)
            update_progress(profile)

    total_time = time.time() - start_time

    # Compute final statistics
    final_stats = stats.compute_stats()

    # Print detailed results
    print()
    print("=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)

    print(f"\n{'='*40}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*40}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Throughput: {NUM_GAMES/total_time:.2f} games/sec")
    print(f"Throughput: {NUM_GAMES/total_time*3600:.0f} games/hour")
    print(f"Success rate: {final_stats['success_rate']*100:.1f}%")

    print(f"\n{'='*40}")
    print("GAME DURATION BREAKDOWN")
    print(f"{'='*40}")
    print(f"  Min:    {final_stats['duration_min_ms']:.0f} ms")
    print(f"  Mean:   {final_stats['duration_mean_ms']:.0f} ms")
    print(f"  Median: {final_stats['duration_median_ms']:.0f} ms")
    print(f"  P90:    {final_stats['duration_p90_ms']:.0f} ms")
    print(f"  P95:    {final_stats['duration_p95_ms']:.0f} ms")
    print(f"  P99:    {final_stats['duration_p99_ms']:.0f} ms")
    print(f"  Max:    {final_stats['duration_max_ms']:.0f} ms")
    print(f"  StdDev: {final_stats['duration_stdev_ms']:.0f} ms")

    print(f"\n{'='*40}")
    print("TIME BREAKDOWN BY PHASE")
    print(f"{'='*40}")
    print(f"  Connect:     {final_stats['connect_mean_ms']:.1f} ms ({final_stats['connect_pct']:.1f}%)")
    print(f"  First byte:  {final_stats['first_byte_mean_ms']:.0f} ms ({final_stats['first_byte_pct']:.1f}%)")
    print(f"  Server proc: {final_stats['server_processing_mean_ms']:.0f} ms ({final_stats['server_pct']:.1f}%)")
    print(f"  Recv data:   {final_stats['recv_mean_ms']:.1f} ms ({final_stats['recv_pct']:.1f}%)")
    print(f"  Network:     {final_stats['network_mean_ms']:.1f} ms")

    print(f"\n{'='*40}")
    print("GAME COMPLEXITY")
    print(f"{'='*40}")
    print(f"  Avg turns: {final_stats['turns_mean']:.1f}")
    print(f"  Avg response size: {final_stats['response_size_mean_kb']:.1f} KB")

    print(f"\n{'='*40}")
    print("WIN DISTRIBUTION")
    print(f"{'='*40}")
    for winner, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = count / final_stats['successful_games'] * 100
        print(f"  {winner}: {count:,} ({pct:.1f}%)")

    # Compute theoretical limits
    print(f"\n{'='*40}")
    print("THEORETICAL SPEEDUP ANALYSIS")
    print(f"{'='*40}")

    server_time = final_stats['server_processing_mean_ms']
    total_time_ms = final_stats['duration_mean_ms']

    print(f"\nCurrent bottleneck: Server processing ({server_time:.0f} ms = {final_stats['server_pct']:.1f}%)")

    # If we reduced server time by 50%
    reduced_server = server_time * 0.5
    new_total = total_time_ms - server_time + reduced_server
    speedup = total_time_ms / new_total
    print("\nIf server time reduced 50%:")
    print(f"  New game time: {new_total:.0f} ms (was {total_time_ms:.0f} ms)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  New throughput: {1000/new_total*MAX_PARALLEL:.1f} games/sec")

    # Network-bound analysis
    network_time = final_stats['network_mean_ms']
    theoretical_min = network_time + 100  # Assume 100ms minimum server time
    max_speedup = total_time_ms / theoretical_min
    print("\nTheoretical maximum speedup (network-bound):")
    print(f"  Min possible game time: ~{theoretical_min:.0f} ms")
    print(f"  Max speedup: {max_speedup:.2f}x")

    # Parallelism analysis
    print(f"\n{'='*40}")
    print("PARALLELISM ANALYSIS")
    print(f"{'='*40}")

    game_time_sec = total_time_ms / 1000
    for workers in [10, 20, 40, 80, 160]:
        # Assuming ~linear scaling up to CPU cores
        effective_workers = min(workers, workers * 0.8)  # 80% efficiency
        games_per_sec = effective_workers / game_time_sec
        games_per_hour = games_per_sec * 3600
        print(f"  {workers:3d} workers: ~{games_per_sec:.1f} games/sec ({games_per_hour:,.0f}/hour)")

    # Save detailed results
    output_file = "/tmp/profiling_10k_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'num_games': NUM_GAMES,
                'max_parallel': MAX_PARALLEL,
                'timeout': GAME_TIMEOUT,
            },
            'summary': final_stats,
            'win_distribution': dict(wins),
            'game_profiles': [
                {
                    'game_id': g.game_id,
                    'success': g.success,
                    'total_ms': g.total_duration_ms,
                    'connect_ms': g.connect_time_ms,
                    'first_byte_ms': g.first_byte_time_ms,
                    'recv_ms': g.recv_time_ms,
                    'turns': g.turns,
                    'winner': g.winner,
                }
                for g in stats.games[:1000]  # Save first 1000 detailed
            ]
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return final_stats


if __name__ == "__main__":
    run_benchmark()
