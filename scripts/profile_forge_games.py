#!/usr/bin/env python3
"""
Forge Game Profiler

Profiles actual Forge game latency to understand where time is spent.
Measures:
- Forge decision latency (time to receive next decision)
- Network/parsing overhead
- Decision complexity
- Overall throughput

Usage:
    python scripts/profile_forge_games.py --games 10 --deck1 deck1.dck --deck2 deck2.dck
"""

import os
import sys
import time
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forge.forge_client import ForgeClient, Decision, DecisionType
from src.training.profiler import TrainingProfiler


@dataclass
class GameStats:
    """Stats for a single game."""
    game_id: int
    turns: int = 0
    decisions: int = 0
    duration_ms: float = 0
    winner: str = ""
    decision_times_ms: list = field(default_factory=list)
    decision_types: dict = field(default_factory=lambda: defaultdict(int))


class RandomAgent:
    """Simple random agent for profiling."""

    def decide(self, decision: Decision) -> str:
        """Make a random valid decision."""
        import random

        if decision.decision_type == DecisionType.CHOOSE_ACTION:
            # Prioritize lands in early game, then random
            actions = decision.actions
            if actions:
                # Find land plays
                land_actions = [a for a in actions if a.is_land]
                if land_actions and decision.turn <= 5:
                    # Prefer playing lands early
                    return str(land_actions[0].index)
                # Random action (including pass)
                valid = [a for a in actions if a.index >= 0]
                if valid:
                    return str(random.choice(valid).index)
            return "-1"  # Pass

        elif decision.decision_type == DecisionType.DECLARE_ATTACKERS:
            # Attack with all creatures
            attackers = decision.attackers
            if attackers:
                return ",".join(str(a["index"]) for a in attackers)
            return ""

        elif decision.decision_type == DecisionType.DECLARE_BLOCKERS:
            # Don't block (for profiling simplicity)
            return ""

        elif decision.decision_type == DecisionType.PLAY_TRIGGER:
            return "y"  # Always play triggers

        elif decision.decision_type == DecisionType.CONFIRM_ACTION:
            return "y"

        elif decision.decision_type == DecisionType.CHOOSE_CARDS:
            cards = decision.cards
            if cards:
                min_cards = decision.raw_data.get("min", 0)
                indices = list(range(min(min_cards, len(cards))))
                if indices:
                    return ",".join(str(i) for i in indices)
            return ""

        elif decision.decision_type == DecisionType.CHOOSE_ENTITY:
            return "0"

        elif decision.decision_type == DecisionType.ANNOUNCE_VALUE:
            return "0"

        else:
            # Default: first option or pass
            return "0"


def profile_game(
    client: ForgeClient,
    deck1: str,
    deck2: str,
    game_id: int,
    profiler: TrainingProfiler,
    agent: RandomAgent,
    timeout: int = 120,
    seed: int = None,
) -> GameStats:
    """Profile a single game."""
    stats = GameStats(game_id=game_id)

    game_start = time.perf_counter()

    # Start game
    with profiler.measure("game_start"):
        success = client.start_game(deck1, deck2, timeout=timeout, seed=seed)
        if not success:
            print(f"Game {game_id}: Failed to start")
            return stats

    current_turn = 0

    while True:
        # Receive decision
        with profiler.measure("receive_decision"):
            decision_start = time.perf_counter()
            decision = client.receive_decision()
            decision_time_ms = (time.perf_counter() - decision_start) * 1000

        if decision is None:
            break

        stats.decisions += 1
        stats.decision_times_ms.append(decision_time_ms)
        stats.decision_types[decision.decision_type.value] += 1

        # Track turns
        if decision.turn > current_turn:
            current_turn = decision.turn
            stats.turns = current_turn

        # Make decision
        with profiler.measure("agent_decide"):
            response = agent.decide(decision)

        # Send response
        with profiler.measure("send_response"):
            client.send_response(response)

        profiler.increment("decisions", 1)

    # Get result
    result = client.get_result()
    if result:
        stats.winner = result.winner or "Draw"
        stats.duration_ms = result.duration_ms
    else:
        stats.duration_ms = (time.perf_counter() - game_start) * 1000

    profiler.increment("games", 1)

    return stats


def run_profiling(
    host: str,
    port: int,
    deck1: str,
    deck2: str,
    num_games: int,
    timeout: int = 120,
    verbose: bool = False,
):
    """Run profiling for multiple games."""
    print("=" * 70)
    print("FORGE GAME PROFILER")
    print("=" * 70)
    print(f"Host: {host}:{port}")
    print(f"Decks: {deck1} vs {deck2}")
    print(f"Games: {num_games}")
    print()

    profiler = TrainingProfiler()
    agent = RandomAgent()
    all_stats = []

    total_start = time.perf_counter()

    for game_id in range(1, num_games + 1):
        # Create fresh connection for each game
        client = ForgeClient(host, port, timeout=timeout)

        try:
            with profiler.measure("connection"):
                client.connect()

            stats = profile_game(
                client, deck1, deck2, game_id, profiler, agent,
                timeout=timeout, seed=game_id  # Deterministic for debugging
            )
            all_stats.append(stats)

            if verbose or game_id <= 3 or game_id == num_games:
                avg_decision_ms = (
                    sum(stats.decision_times_ms) / max(1, len(stats.decision_times_ms))
                )
                print(
                    f"Game {game_id}: {stats.turns} turns, {stats.decisions} decisions, "
                    f"avg {avg_decision_ms:.1f}ms/decision, winner: {stats.winner}"
                )

        except Exception as e:
            print(f"Game {game_id}: Error - {e}")
        finally:
            client.close()

    total_time = time.perf_counter() - total_start

    # Aggregate stats
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not all_stats:
        print("No games completed!")
        return

    total_decisions = sum(s.decisions for s in all_stats)
    total_turns = sum(s.turns for s in all_stats)
    all_decision_times = []
    for s in all_stats:
        all_decision_times.extend(s.decision_times_ms)

    # Decision type breakdown
    decision_counts = defaultdict(int)
    for s in all_stats:
        for dt, count in s.decision_types.items():
            decision_counts[dt] += count

    print(f"\nGames completed: {len(all_stats)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Games/second: {len(all_stats)/total_time:.2f}")
    print(f"Total decisions: {total_decisions}")
    print(f"Decisions/second: {total_decisions/total_time:.1f}")
    print(f"Total turns: {total_turns}")
    print(f"Avg turns/game: {total_turns/len(all_stats):.1f}")
    print(f"Avg decisions/game: {total_decisions/len(all_stats):.1f}")

    print("\n--- Decision Timing ---")
    if all_decision_times:
        import numpy as np
        times = np.array(all_decision_times)
        print(f"Mean: {times.mean():.2f}ms")
        print(f"Median: {np.median(times):.2f}ms")
        print(f"Std: {times.std():.2f}ms")
        print(f"Min: {times.min():.2f}ms")
        print(f"Max: {times.max():.2f}ms")
        print(f"P95: {np.percentile(times, 95):.2f}ms")
        print(f"P99: {np.percentile(times, 99):.2f}ms")

    print("\n--- Decision Type Breakdown ---")
    for dt, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        pct = count / total_decisions * 100
        print(f"  {dt}: {count} ({pct:.1f}%)")

    print("\n--- Profiler Report ---")
    print(profiler.report())

    # Estimate training throughput
    print("\n--- Training Throughput Estimate ---")
    games_per_second = len(all_stats) / total_time

    # Each decision is roughly one training sample
    samples_per_second = total_decisions / total_time
    samples_per_hour = samples_per_second * 3600

    print(f"Games/second: {games_per_second:.2f}")
    print(f"Samples/second: {samples_per_second:.1f}")
    print(f"Samples/hour: {samples_per_hour:,.0f}")
    print(f"Time to 1M samples: {1_000_000 / samples_per_hour:.2f} hours")

    # With GPU forward pass
    # Assume 0.5ms per forward pass (from our benchmark)
    forward_pass_time_per_sample = 0.5  # ms
    total_forward_pass_time = total_decisions * forward_pass_time_per_sample / 1000  # seconds
    forward_pass_pct = total_forward_pass_time / total_time * 100

    print(f"\nWith GPU forward pass ({forward_pass_time_per_sample}ms each):")
    print(f"  Forward pass total: {total_forward_pass_time:.1f}s ({forward_pass_pct:.1f}% of total)")
    print(f"  Forge overhead: {100-forward_pass_pct:.1f}%")

    # Save results
    results = {
        "num_games": len(all_stats),
        "total_time_s": total_time,
        "total_decisions": total_decisions,
        "total_turns": total_turns,
        "games_per_second": games_per_second,
        "decisions_per_second": samples_per_second,
        "samples_per_hour": samples_per_hour,
        "hours_to_1m": 1_000_000 / samples_per_hour,
        "avg_decision_ms": float(np.mean(all_decision_times)) if all_decision_times else 0,
        "median_decision_ms": float(np.median(all_decision_times)) if all_decision_times else 0,
        "p95_decision_ms": float(np.percentile(all_decision_times, 95)) if all_decision_times else 0,
        "decision_types": dict(decision_counts),
    }

    with open("forge_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to forge_profile_results.json")


def main():
    parser = argparse.ArgumentParser(description="Profile Forge game latency")
    parser.add_argument("--host", default="localhost", help="Forge daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Forge daemon port")
    parser.add_argument("--deck1", default="deck1.dck", help="First deck")
    parser.add_argument("--deck2", default="deck2.dck", help="Second deck")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--timeout", type=int, default=120, help="Game timeout")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_profiling(
        host=args.host,
        port=args.port,
        deck1=args.deck1,
        deck2=args.deck2,
        num_games=args.games,
        timeout=args.timeout,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
