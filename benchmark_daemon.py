#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/benchmark.py instead.

Migration:
    # Old
    python benchmark_daemon.py

    # New
    python scripts/benchmark.py --games 100 --parallel 10
"""

import warnings
warnings.warn(
    "benchmark_daemon.py is deprecated. Use scripts/benchmark.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Original docstring for reference:
# Benchmark the Forge daemon with parallel game execution.

import socket
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Configuration
DAEMON_HOST = "localhost"
DAEMON_PORT = 17171
NUM_GAMES = 100
MAX_PARALLEL = 10
GAME_TIMEOUT = 120

def run_single_game(game_id: int) -> dict:
    """Run a single game and return timing info."""
    result = {
        "game_id": game_id,
        "success": False,
        "duration_ms": 0,
        "winner": None,
        "error": None
    }

    start_time = time.time()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(GAME_TIMEOUT)
            s.connect((DAEMON_HOST, DAEMON_PORT))

            # Send game command
            cmd = "NEWGAME test_red.dck test_blue.dck -q -c 60\n"
            s.sendall(cmd.encode())

            # Receive response
            response = ""
            while True:
                try:
                    data = s.recv(4096)
                    if not data:
                        break
                    response += data.decode()
                    if "GAME_RESULT:" in response or "ERROR:" in response:
                        break
                except socket.timeout:
                    result["error"] = "Socket timeout"
                    break

            elapsed_ms = (time.time() - start_time) * 1000
            result["duration_ms"] = elapsed_ms

            # Parse response
            if "GAME_RESULT:" in response:
                result["success"] = True
                # Extract winner and game time from response
                # Format: "GAME_RESULT: Ai(1)-Test Red won in 1596ms"
                parts = response.strip().split()
                if "won" in response:
                    result["winner"] = parts[1]  # Ai(1)-Test or Ai(2)-Test
                elif "Draw" in response:
                    result["winner"] = "Draw"
            elif "ERROR:" in response:
                result["error"] = response.strip()
            else:
                result["error"] = f"Unexpected response: {response[:100]}"

    except Exception as e:
        result["error"] = str(e)
        result["duration_ms"] = (time.time() - start_time) * 1000

    return result


def run_benchmark():
    """Run the full benchmark."""
    print("Forge Daemon Benchmark")
    print("=" * 50)
    print(f"Total games: {NUM_GAMES}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print("=" * 50)
    print()

    results = []
    wins = defaultdict(int)
    errors = []

    overall_start = time.time()

    # Run games with thread pool
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_single_game, i): i for i in range(NUM_GAMES)}

        completed = 0
        for future in as_completed(futures):
            game_id = futures[future]
            result = future.result()
            results.append(result)
            completed += 1

            if result["success"]:
                wins[result["winner"]] += 1
                status = f"Game {game_id}: {result['duration_ms']:.0f}ms - {result['winner']}"
            else:
                errors.append(result)
                status = f"Game {game_id}: ERROR - {result['error']}"

            # Progress update every 10 games
            if completed % 10 == 0:
                print(f"Progress: {completed}/{NUM_GAMES} games completed")

    overall_duration = time.time() - overall_start

    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        durations = [r["duration_ms"] for r in successful_results]

        print()
        print("=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Total time: {overall_duration:.1f}s")
        print(f"Successful games: {len(successful_results)}/{NUM_GAMES}")
        print(f"Failed games: {len(errors)}")
        print()
        print("Game Duration Statistics (ms):")
        print(f"  Min:    {min(durations):.0f}")
        print(f"  Max:    {max(durations):.0f}")
        print(f"  Mean:   {statistics.mean(durations):.0f}")
        print(f"  Median: {statistics.median(durations):.0f}")
        print(f"  StdDev: {statistics.stdev(durations):.0f}" if len(durations) > 1 else "  StdDev: N/A")
        print()
        print(f"Throughput: {len(successful_results) / overall_duration:.1f} games/second")
        print(f"            {len(successful_results) / overall_duration * 3600:.0f} games/hour")
        print()
        print("Win Distribution:")
        for winner, count in sorted(wins.items()):
            print(f"  {winner}: {count} ({count/len(successful_results)*100:.1f}%)")

        if errors:
            print()
            print(f"Errors ({len(errors)}):")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  Game {err['game_id']}: {err['error']}")

        # Generate histogram data
        print()
        print("=" * 50)
        print("DURATION HISTOGRAM (ms)")
        print("=" * 50)

        # Create histogram bins
        min_d, max_d = min(durations), max(durations)
        bin_size = max(100, int((max_d - min_d) / 15))  # ~15 bins
        bins = defaultdict(int)
        for d in durations:
            bin_start = int(d // bin_size) * bin_size
            bins[bin_start] += 1

        # Print histogram
        max_count = max(bins.values())
        for bin_start in sorted(bins.keys()):
            count = bins[bin_start]
            bar_len = int(count / max_count * 40)
            bar = "#" * bar_len
            print(f"{bin_start:5d}-{bin_start+bin_size:5d}: {bar} ({count})")

        # Save raw data for later analysis
        with open("/tmp/benchmark_results.txt", "w") as f:
            f.write("game_id,duration_ms,success,winner,error\n")
            for r in results:
                f.write(f"{r['game_id']},{r['duration_ms']:.0f},{r['success']},{r['winner'] or ''},{r['error'] or ''}\n")
        print()
        print("Raw data saved to /tmp/benchmark_results.txt")

    else:
        print("No successful games!")
        if errors:
            print("Errors:")
            for err in errors[:10]:
                print(f"  Game {err['game_id']}: {err['error']}")


if __name__ == "__main__":
    run_benchmark()
