#!/usr/bin/env python3
"""Benchmark 1000 games on the Forge daemon with 10 parallel connections."""

import socket
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json

# Configuration
DAEMON_HOST = "localhost"
DAEMON_PORT = 17171
NUM_GAMES = 1000
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

            cmd = "NEWGAME test_red.dck test_blue.dck -q -c 60\n"
            s.sendall(cmd.encode())

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

            if "GAME_RESULT:" in response:
                result["success"] = True
                parts = response.strip().split()
                if "won" in response:
                    result["winner"] = parts[1]
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
    print(f"=" * 60)
    print(f"FORGE DAEMON BENCHMARK - 1000 GAMES")
    print(f"=" * 60)
    print(f"Total games: {NUM_GAMES}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print(f"=" * 60)
    print()

    results = []
    wins = defaultdict(int)
    errors = []

    overall_start = time.time()

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
            else:
                errors.append(result)

            if completed % 100 == 0:
                elapsed = time.time() - overall_start
                rate = completed / elapsed
                eta = (NUM_GAMES - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{NUM_GAMES} ({rate:.1f} games/sec, ETA: {eta:.0f}s)")

    overall_duration = time.time() - overall_start

    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        durations = [r["duration_ms"] for r in successful_results]

        print()
        print(f"=" * 60)
        print(f"BENCHMARK RESULTS")
        print(f"=" * 60)
        print(f"Total time: {overall_duration:.1f}s")
        print(f"Successful games: {len(successful_results)}/{NUM_GAMES}")
        print(f"Failed games: {len(errors)}")
        print()
        print(f"Game Duration Statistics (ms):")
        print(f"  Min:    {min(durations):.0f}")
        print(f"  Max:    {max(durations):.0f}")
        print(f"  Mean:   {statistics.mean(durations):.0f}")
        print(f"  Median: {statistics.median(durations):.0f}")
        if len(durations) > 1:
            print(f"  StdDev: {statistics.stdev(durations):.0f}")
            print(f"  P95:    {sorted(durations)[int(len(durations)*0.95)]:.0f}")
            print(f"  P99:    {sorted(durations)[int(len(durations)*0.99)]:.0f}")
        print()
        print(f"Throughput:")
        print(f"  {len(successful_results) / overall_duration:.2f} games/second")
        print(f"  {len(successful_results) / overall_duration * 60:.0f} games/minute")
        print(f"  {len(successful_results) / overall_duration * 3600:.0f} games/hour")
        print()
        print(f"Win Distribution:")
        for winner, count in sorted(wins.items()):
            print(f"  {winner}: {count} ({count/len(successful_results)*100:.1f}%)")

        if errors:
            print()
            print(f"Errors ({len(errors)}):")
            for err in errors[:5]:
                print(f"  Game {err['game_id']}: {err['error']}")

        # Save results to JSON for later analysis
        output = {
            "config": {
                "num_games": NUM_GAMES,
                "max_parallel": MAX_PARALLEL,
                "daemon_host": DAEMON_HOST,
                "daemon_port": DAEMON_PORT
            },
            "summary": {
                "total_time_s": overall_duration,
                "successful_games": len(successful_results),
                "failed_games": len(errors),
                "games_per_second": len(successful_results) / overall_duration,
                "games_per_hour": len(successful_results) / overall_duration * 3600,
                "duration_min_ms": min(durations),
                "duration_max_ms": max(durations),
                "duration_mean_ms": statistics.mean(durations),
                "duration_median_ms": statistics.median(durations),
                "duration_stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            },
            "wins": dict(wins),
            "all_durations_ms": durations
        }

        with open("/tmp/benchmark_1000_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print()
        print(f"Full results saved to /tmp/benchmark_1000_results.json")

    else:
        print("No successful games!")


if __name__ == "__main__":
    run_benchmark()
