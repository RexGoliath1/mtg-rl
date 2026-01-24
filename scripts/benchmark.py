#!/usr/bin/env python3
"""
Forge Daemon Benchmark

Benchmark the Forge daemon with configurable parallel game execution.

Usage:
    # Quick benchmark (100 games)
    python scripts/benchmark.py

    # Full benchmark (1000 games)
    python scripts/benchmark.py --games 1000

    # Custom settings
    python scripts/benchmark.py --games 500 --parallel 20 --host daemon.local --port 17171

    # With specific decks
    python scripts/benchmark.py --deck1 decks/mono_red.dck --deck2 decks/control.dck
"""

import argparse
import json
import socket
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_single_game(
    game_id: int,
    daemon_host: str,
    daemon_port: int,
    deck1: str,
    deck2: str,
    timeout: int,
) -> dict:
    """Run a single game and return timing info."""
    result = {
        "game_id": game_id,
        "success": False,
        "duration_ms": 0,
        "winner": None,
        "error": None,
    }

    start_time = time.time()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((daemon_host, daemon_port))

            cmd = f"NEWGAME {deck1} {deck2} -q -c {timeout}\n"
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


def run_benchmark(
    num_games: int,
    max_parallel: int,
    daemon_host: str,
    daemon_port: int,
    deck1: str,
    deck2: str,
    timeout: int,
    output_file: str = None,
    quiet: bool = False,
):
    """Run the full benchmark."""
    print("=" * 60)
    print("FORGE DAEMON BENCHMARK")
    print("=" * 60)
    print(f"Total games: {num_games}")
    print(f"Max parallel: {max_parallel}")
    print(f"Host: {daemon_host}:{daemon_port}")
    print(f"Decks: {deck1} vs {deck2}")
    print("=" * 60)
    print()

    results = []
    wins = defaultdict(int)
    errors = []

    overall_start = time.time()

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                run_single_game,
                i,
                daemon_host,
                daemon_port,
                deck1,
                deck2,
                timeout,
            ): i
            for i in range(num_games)
        }

        completed = 0
        progress_interval = max(1, num_games // 20)  # ~20 progress updates

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if result["success"]:
                wins[result["winner"]] += 1
            else:
                errors.append(result)

            if not quiet and completed % progress_interval == 0:
                elapsed = time.time() - overall_start
                rate = completed / elapsed
                eta = (num_games - completed) / rate if rate > 0 else 0
                print(
                    f"Progress: {completed}/{num_games} "
                    f"({rate:.1f} games/sec, ETA: {eta:.0f}s)"
                )

    overall_duration = time.time() - overall_start

    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        durations = [r["duration_ms"] for r in successful_results]

        print()
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total time: {overall_duration:.1f}s")
        print(f"Successful games: {len(successful_results)}/{num_games}")
        print(f"Failed games: {len(errors)}")
        print()
        print("Game Duration Statistics (ms):")
        print(f"  Min:    {min(durations):.0f}")
        print(f"  Max:    {max(durations):.0f}")
        print(f"  Mean:   {statistics.mean(durations):.0f}")
        print(f"  Median: {statistics.median(durations):.0f}")
        if len(durations) > 1:
            print(f"  StdDev: {statistics.stdev(durations):.0f}")
            print(f"  P95:    {sorted(durations)[int(len(durations)*0.95)]:.0f}")
            print(f"  P99:    {sorted(durations)[int(len(durations)*0.99)]:.0f}")
        print()
        print("Throughput:")
        print(f"  {len(successful_results) / overall_duration:.2f} games/second")
        print(f"  {len(successful_results) / overall_duration * 60:.0f} games/minute")
        print(f"  {len(successful_results) / overall_duration * 3600:.0f} games/hour")
        print()
        print("Win Distribution:")
        for winner, count in sorted(wins.items()):
            pct = count / len(successful_results) * 100
            print(f"  {winner}: {count} ({pct:.1f}%)")

        if errors:
            print()
            print(f"Errors ({len(errors)}):")
            for err in errors[:5]:
                print(f"  Game {err['game_id']}: {err['error']}")

        # Save results
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = Path("/tmp/benchmark_results.json")

        output = {
            "config": {
                "num_games": num_games,
                "max_parallel": max_parallel,
                "daemon_host": daemon_host,
                "daemon_port": daemon_port,
                "deck1": deck1,
                "deck2": deck2,
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
                "duration_stdev_ms": (
                    statistics.stdev(durations) if len(durations) > 1 else 0
                ),
            },
            "wins": dict(wins),
            "all_durations_ms": durations,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print()
        print(f"Full results saved to {output_path}")

        return output

    else:
        print("No successful games!")
        if errors:
            print("Errors:")
            for err in errors[:10]:
                print(f"  Game {err['game_id']}: {err['error']}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Forge daemon")
    parser.add_argument(
        "--games", "-n", type=int, default=100, help="Number of games to run"
    )
    parser.add_argument(
        "--parallel", "-p", type=int, default=10, help="Max parallel connections"
    )
    parser.add_argument("--host", default="localhost", help="Daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Daemon port")
    parser.add_argument("--deck1", default="test_red.dck", help="Deck 1 path")
    parser.add_argument("--deck2", default="test_blue.dck", help="Deck 2 path")
    parser.add_argument("--timeout", type=int, default=120, help="Game timeout (sec)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    run_benchmark(
        num_games=args.games,
        max_parallel=args.parallel,
        daemon_host=args.host,
        daemon_port=args.port,
        deck1=args.deck1,
        deck2=args.deck2,
        timeout=args.timeout,
        output_file=args.output,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
