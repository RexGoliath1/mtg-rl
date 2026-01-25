#!/usr/bin/env python3
"""
Collect AI Training Data

Run games in observation mode (-o) where Forge AI makes decisions
and all decisions are logged for training data collection.
"""

import os
import sys
import json
import socket
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_training_game(
    deck1: str = "red_aggro.dck",
    deck2: str = "white_weenie.dck",
    seed: int = None,
    timeout: int = 60,
    host: str = "localhost",
    port: int = 17171
) -> dict:
    """Run a single game in observation mode and collect decisions."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout + 30)
    sock.connect((host, port))

    rfile = sock.makefile("r", buffering=1)
    wfile = sock.makefile("w", buffering=1)

    decisions = []
    game_result = None

    try:
        # Start game in observation mode (-o)
        cmd = f"NEWGAME {deck1} {deck2} -o -q -c {timeout}"
        if seed is not None:
            cmd += f" -s {seed}"

        wfile.write(cmd + "\n")
        wfile.flush()

        # Read all decisions until game ends
        while True:
            line = rfile.readline()
            if not line:
                break

            line = line.strip()
            if line.startswith("DECISION:"):
                try:
                    data = json.loads(line[9:])
                    decisions.append(data)
                except json.JSONDecodeError:
                    pass

            elif line.startswith("GAME_RESULT:"):
                game_result = line[12:].strip()
                break

            elif line.startswith("GAME_TIMEOUT:"):
                game_result = "TIMEOUT"
                break

            elif line.startswith("ERROR:"):
                game_result = f"ERROR: {line}"
                break

    except socket.timeout:
        game_result = "SOCKET_TIMEOUT"
    except Exception as e:
        game_result = f"EXCEPTION: {str(e)}"
    finally:
        sock.close()

    return {
        "decisions": decisions,
        "result": game_result,
        "deck1": deck1,
        "deck2": deck2,
        "seed": seed
    }


def collect_training_batch(
    num_games: int = 10,
    output_dir: str = "training_data",
    deck1: str = "red_aggro.dck",
    deck2: str = "white_weenie.dck"
):
    """Collect training data from multiple games."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_decisions = []
    stats = defaultdict(int)

    print(f"Collecting training data from {num_games} games...")
    print(f"Decks: {deck1} vs {deck2}")
    print()

    for game_id in range(1, num_games + 1):
        seed = game_id * 1000
        print(f"Game {game_id}/{num_games}...", end=" ", flush=True)

        result = collect_training_game(
            deck1=deck1,
            deck2=deck2,
            seed=seed
        )

        num_decisions = len(result["decisions"])
        game_result = result["result"]

        if game_result and "won" in game_result.lower():
            stats["completed"] += 1
        elif game_result == "TIMEOUT":
            stats["timeout"] += 1
        else:
            stats["error"] += 1

        # Count decision types
        for d in result["decisions"]:
            dtype = d.get("decision_type", "unknown")
            stats[f"decision_{dtype}"] += 1
            all_decisions.append(d)

        print(f"{num_decisions} decisions, {game_result}")

    # Save all decisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    decisions_file = output_path / f"decisions_{timestamp}.jsonl"

    with open(decisions_file, "w") as f:
        for d in all_decisions:
            f.write(json.dumps(d) + "\n")

    print(f"\nSaved {len(all_decisions)} decisions to {decisions_file}")

    # Print summary
    print(f"\n=== Collection Summary ===")
    print(f"Total games: {num_games}")
    print(f"Completed: {stats['completed']}")
    print(f"Timeout: {stats['timeout']}")
    print(f"Error: {stats['error']}")
    print(f"\nDecisions by type:")
    for key, value in sorted(stats.items()):
        if key.startswith("decision_"):
            print(f"  {key[9:]}: {value}")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "num_games": num_games,
        "total_decisions": len(all_decisions),
        "decks": {"deck1": deck1, "deck2": deck2},
        "stats": dict(stats)
    }

    summary_file = output_path / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_file}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect AI training data")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--output", default="training_data", help="Output directory")
    parser.add_argument("--deck1", default="red_aggro.dck", help="First deck")
    parser.add_argument("--deck2", default="white_weenie.dck", help="Second deck")
    args = parser.parse_args()

    collect_training_batch(
        num_games=args.games,
        output_dir=args.output,
        deck1=args.deck1,
        deck2=args.deck2
    )


if __name__ == "__main__":
    main()
