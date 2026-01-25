#!/usr/bin/env python3
"""
Collect AI Training Data

Run games in observation mode (-o) where Forge AI makes decisions
and all decisions are logged for training data collection.

Storage: HDF5 for efficient numerical data, with JSON metadata.

For imitation learning bootstrapping:
- Focus on card selection and turn flow
- Use diverse decks for embedding space coverage
- Target 50,000+ games for robust training
"""

import os
import sys
import json
import socket
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import itertools
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Default deck pool for diverse coverage
# Covers: mono-color aggro, 2-color aggro, 3-color control, midrange
DEFAULT_DECKS = [
    "red_aggro.dck",           # Mono-red aggro
    "white_weenie.dck",        # Mono-white aggro
    "mono_red_aggro.dck",      # Mono-red competitive
    "boros_aggro.dck",         # R/W aggro
    "dimir_midrange.dck",      # U/B midrange
    "jeskai_control.dck",      # U/W/R control
]


@dataclass
class CollectionStats:
    """Track collection statistics."""
    games_completed: int = 0
    games_timeout: int = 0
    games_error: int = 0
    total_decisions: int = 0
    total_turns: int = 0
    decision_counts: dict = field(default_factory=lambda: defaultdict(int))
    game_durations_ms: list = field(default_factory=list)
    decisions_per_game: list = field(default_factory=list)
    turns_per_game: list = field(default_factory=list)
    winners: list = field(default_factory=list)
    deck_pairs_played: dict = field(default_factory=lambda: defaultdict(int))
    cards_seen: set = field(default_factory=set)


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
    max_turn = 0
    cards_seen = set()

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
                    max_turn = max(max_turn, data.get("turn", 0))

                    # Track cards seen for coverage metrics
                    for action in data.get("actions", []):
                        card = action.get("card", "")
                        if card:
                            cards_seen.add(card)
                    ai_choice = data.get("ai_choice", {})
                    if isinstance(ai_choice, dict) and ai_choice.get("card"):
                        cards_seen.add(ai_choice["card"])

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
        "seed": seed,
        "max_turn": max_turn,
        "cards_seen": cards_seen
    }


def encode_game_state(state: dict) -> np.ndarray:
    """Encode game state as fixed-size numpy array."""
    # Fixed-size encoding: [p1_life, p1_hand, p1_lib, p1_creatures, p1_lands, p1_other, p1_mana,
    #                       p2_life, p2_hand, p2_lib, p2_creatures, p2_lands, p2_other, p2_mana,
    #                       turn, phase_idx, is_game_over]
    players = state.get("players", [{}, {}])
    p1 = players[0] if len(players) > 0 else {}
    p2 = players[1] if len(players) > 1 else {}

    phase_map = {
        "UNTAP": 0, "UPKEEP": 1, "DRAW": 2, "MAIN1": 3,
        "COMBAT_BEGIN": 4, "COMBAT_DECLARE_ATTACKERS": 5,
        "COMBAT_DECLARE_BLOCKERS": 6, "COMBAT_FIRST_STRIKE_DAMAGE": 7,
        "COMBAT_DAMAGE": 8, "COMBAT_END": 9, "MAIN2": 10,
        "END_OF_TURN": 11, "CLEANUP": 12
    }

    return np.array([
        p1.get("life", 20),
        p1.get("hand_size", 0),
        p1.get("library_size", 0),
        p1.get("battlefield_creatures", 0),
        p1.get("battlefield_lands", 0),
        p1.get("battlefield_other", 0),
        p1.get("mana_pool", {}).get("total", 0),
        p2.get("life", 20),
        p2.get("hand_size", 0),
        p2.get("library_size", 0),
        p2.get("battlefield_creatures", 0),
        p2.get("battlefield_lands", 0),
        p2.get("battlefield_other", 0),
        p2.get("mana_pool", {}).get("total", 0),
        state.get("turn", 1) if "turn" not in state else 1,
        phase_map.get(state.get("phase", "MAIN1"), 3),
        1 if state.get("is_game_over", False) else 0
    ], dtype=np.float32)


def encode_decision(decision: dict) -> tuple:
    """Encode a decision into arrays for HDF5 storage."""
    dtype = decision.get("decision_type", "unknown")

    # Encode game state
    game_state = decision.get("game_state", {})
    state_vec = encode_game_state(game_state)

    # Encode decision metadata
    turn = decision.get("turn", 0)

    # Encode AI choice as index
    ai_choice = decision.get("ai_choice", {})
    if dtype == "choose_action":
        if isinstance(ai_choice, dict) and ai_choice.get("action") == "pass":
            choice_idx = -1
        elif isinstance(ai_choice, dict):
            choice_idx = 0  # Played something
        else:
            choice_idx = -1
    elif dtype in ("declare_attackers", "declare_blockers"):
        if isinstance(ai_choice, list):
            choice_idx = len(ai_choice)  # Number of attackers/blockers
        else:
            choice_idx = 0
    else:
        choice_idx = -1

    # Number of available actions
    actions = decision.get("actions", [])
    num_actions = len(actions)

    return state_vec, turn, choice_idx, num_actions, dtype


def save_to_hdf5(decisions: list, output_path: Path, metadata: dict):
    """Save decisions to HDF5 file."""
    if not decisions:
        return

    # Encode all decisions
    state_vecs = []
    turns = []
    choices = []
    num_actions_list = []
    decision_types = []

    type_map = {"choose_action": 0, "declare_attackers": 1, "declare_blockers": 2, "unknown": 3}

    for d in decisions:
        state_vec, turn, choice_idx, num_actions, dtype = encode_decision(d)
        state_vecs.append(state_vec)
        turns.append(turn)
        choices.append(choice_idx)
        num_actions_list.append(num_actions)
        decision_types.append(type_map.get(dtype, 3))

    # Stack into arrays
    states = np.stack(state_vecs)
    turns = np.array(turns, dtype=np.int32)
    choices = np.array(choices, dtype=np.int32)
    num_actions = np.array(num_actions_list, dtype=np.int32)
    dtypes = np.array(decision_types, dtype=np.int8)

    # Save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("states", data=states, compression="gzip", compression_opts=4)
        f.create_dataset("turns", data=turns, compression="gzip")
        f.create_dataset("choices", data=choices, compression="gzip")
        f.create_dataset("num_actions", data=num_actions, compression="gzip")
        f.create_dataset("decision_types", data=dtypes, compression="gzip")

        # Store metadata as attributes
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                f.attrs[k] = v
            elif isinstance(v, dict):
                f.attrs[k] = json.dumps(v)


def generate_deck_pairs(decks: List[str], num_pairs: int) -> List[Tuple[str, str]]:
    """Generate deck pairs for diverse matchup coverage."""
    # Generate all possible pairs (including mirrors)
    all_pairs = list(itertools.combinations_with_replacement(decks, 2))

    # Also add reverse pairs for non-mirrors (player order matters)
    extended_pairs = []
    for d1, d2 in all_pairs:
        extended_pairs.append((d1, d2))
        if d1 != d2:
            extended_pairs.append((d2, d1))

    # Repeat to reach desired count
    pairs = []
    while len(pairs) < num_pairs:
        random.shuffle(extended_pairs)
        pairs.extend(extended_pairs)

    return pairs[:num_pairs]


def generate_report(stats: CollectionStats, output_dir: Path, timestamp: str):
    """Generate a LaTeX training report."""
    report_path = output_dir / f"collection_report_{timestamp}.tex"

    avg_decisions = np.mean(stats.decisions_per_game) if stats.decisions_per_game else 0
    avg_turns = np.mean(stats.turns_per_game) if stats.turns_per_game else 0
    avg_duration = np.mean(stats.game_durations_ms) if stats.game_durations_ms else 0
    total_games = stats.games_completed + stats.games_timeout + stats.games_error

    # Count winners
    winner_counts = defaultdict(int)
    for w in stats.winners:
        winner_counts[w] += 1

    latex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{xcolor}}
\\usepackage{{longtable}}
\\geometry{{margin=1in}}

\\title{{AI Training Data Collection Report}}
\\author{{ForgeRL System}}
\\date{{{datetime.now().strftime("%Y-%m-%d %H:%M")}}}

\\begin{{document}}
\\maketitle

\\section{{Collection Summary}}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Total Games & {total_games:,} \\\\
Completed & {stats.games_completed:,} \\\\
Timeout & {stats.games_timeout:,} \\\\
Error & {stats.games_error:,} \\\\
\\midrule
Total Decisions & {stats.total_decisions:,} \\\\
Total Turns & {stats.total_turns:,} \\\\
Unique Cards Seen & {len(stats.cards_seen):,} \\\\
\\midrule
Avg Decisions/Game & {avg_decisions:.1f} \\\\
Avg Turns/Game & {avg_turns:.1f} \\\\
Avg Duration (ms) & {avg_duration:.0f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Collection Metrics}}
\\end{{table}}

\\section{{Decision Types}}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Type}} & \\textbf{{Count}} \\\\
\\midrule
"""
    for dtype, count in sorted(stats.decision_counts.items()):
        latex += f"{dtype} & {count:,} \\\\\n"

    latex += f"""\\bottomrule
\\end{{tabular}}
\\caption{{Decisions by Type}}
\\end{{table}}

\\section{{Deck Coverage}}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Deck Pair}} & \\textbf{{Games}} \\\\
\\midrule
"""
    for pair, count in sorted(stats.deck_pairs_played.items(), key=lambda x: -x[1])[:15]:
        latex += f"{pair[:40]} & {count:,} \\\\\n"

    latex += f"""\\bottomrule
\\end{{tabular}}
\\caption{{Top Deck Matchups}}
\\end{{table}}

\\section{{Data Quality Notes}}

\\begin{{itemize}}
\\item Forge AI serves as the expert for imitation learning
\\item Focus: card selection, turn flow, combat decisions
\\item NOT optimizing for win rate (self-play handles that)
\\item Storage: HDF5 with gzip compression (\\textasciitilde 20x smaller than JSON)
\\item State encoding: 17-dimensional vector
\\end{{itemize}}

\\section{{Coverage Analysis}}

\\begin{{itemize}}
\\item Unique cards observed: {len(stats.cards_seen):,}
\\item Unique deck pairs: {len(stats.deck_pairs_played)}
\\item Games per deck pair (avg): {total_games / max(len(stats.deck_pairs_played), 1):.1f}
\\end{{itemize}}

\\section{{Recommended Next Steps}}

\\begin{{enumerate}}
\\item {"\\textbf{COMPLETE} - Ready for imitation learning training" if stats.games_completed >= 50000 else f"Collect \\textbf{{{max(50000 - stats.games_completed, 0):,} more games}} to reach 50K target"}
\\item Train imitation learning policy on collected data
\\item Evaluate policy accuracy on held-out decisions
\\item Begin self-play fine-tuning once accuracy $>$ 50\\%
\\end{{enumerate}}

\\end{{document}}
"""

    with open(report_path, "w") as f:
        f.write(latex)

    return report_path


def collect_training_batch(
    num_games: int = 10,
    output_dir: str = "training_data",
    decks: List[str] = None,
    host: str = "localhost",
    port: int = 17171,
    save_interval: int = 1000,
    timeout: int = 60
):
    """Collect training data from multiple games with deck rotation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if decks is None:
        decks = DEFAULT_DECKS

    # Generate deck pairs for full coverage
    deck_pairs = generate_deck_pairs(decks, num_games)

    all_decisions = []
    stats = CollectionStats()

    print(f"=" * 60)
    print("AI TRAINING DATA COLLECTION")
    print(f"=" * 60)
    print(f"Target games: {num_games:,}")
    print(f"Deck pool: {len(decks)} decks")
    print(f"Unique matchups: {len(set(deck_pairs))}")
    print(f"Output: {output_path}")
    print(f"Save interval: every {save_interval} games")
    print(f"=" * 60)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()

    for game_id in range(1, num_games + 1):
        deck1, deck2 = deck_pairs[game_id - 1]
        seed = game_id * 1000 + random.randint(0, 999)

        # Progress indicator
        if game_id % 100 == 0 or game_id == 1:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = game_id / elapsed if elapsed > 0 else 0
            eta = (num_games - game_id) / rate if rate > 0 else 0
            print(f"Game {game_id:,}/{num_games:,} ({game_id*100/num_games:.1f}%) - "
                  f"{rate:.1f} games/sec - ETA: {eta/60:.1f} min")

        result = collect_training_game(
            deck1=deck1,
            deck2=deck2,
            seed=seed,
            host=host,
            port=port,
            timeout=timeout
        )

        num_decisions = len(result["decisions"])
        game_result = result["result"]
        max_turn = result["max_turn"]

        # Track deck pair
        pair_key = f"{deck1} vs {deck2}"
        stats.deck_pairs_played[pair_key] += 1

        # Track cards seen
        stats.cards_seen.update(result.get("cards_seen", set()))

        # Parse result
        if game_result and "won" in game_result.lower():
            stats.games_completed += 1
            parts = game_result.split(" won in ")
            if len(parts) == 2:
                winner = parts[0].strip()
                duration_str = parts[1].replace("ms", "").strip()
                try:
                    duration = int(duration_str)
                    stats.game_durations_ms.append(duration)
                except ValueError:
                    pass
                stats.winners.append(winner)
        elif game_result == "TIMEOUT":
            stats.games_timeout += 1
        else:
            stats.games_error += 1

        # Count decision types
        for d in result["decisions"]:
            dtype = d.get("decision_type", "unknown")
            stats.decision_counts[dtype] += 1
            all_decisions.append(d)

        stats.total_decisions += num_decisions
        stats.total_turns += max_turn
        stats.decisions_per_game.append(num_decisions)
        stats.turns_per_game.append(max_turn)

        # Periodic save
        if game_id % save_interval == 0:
            checkpoint_file = output_path / f"training_data_{timestamp}_checkpoint_{game_id}.h5"
            metadata = {
                "timestamp": timestamp,
                "num_games": game_id,
                "total_decisions": stats.total_decisions,
                "games_completed": stats.games_completed,
                "checkpoint": True
            }
            save_to_hdf5(all_decisions, checkpoint_file, metadata)
            print(f"  Checkpoint saved: {checkpoint_file.name} ({stats.total_decisions:,} decisions)")

    # Final save
    hdf5_file = output_path / f"training_data_{timestamp}_final.h5"
    metadata = {
        "timestamp": timestamp,
        "num_games": num_games,
        "total_decisions": stats.total_decisions,
        "games_completed": stats.games_completed,
        "games_timeout": stats.games_timeout,
        "games_error": stats.games_error,
        "unique_cards": len(stats.cards_seen),
        "unique_matchups": len(stats.deck_pairs_played)
    }
    save_to_hdf5(all_decisions, hdf5_file, metadata)

    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nHDF5 data saved to {hdf5_file}")
    print(f"  States shape: ({stats.total_decisions:,}, 17)")
    print(f"  Compressed size: {hdf5_file.stat().st_size / (1024*1024):.2f} MB")

    # Generate report
    report_path = generate_report(stats, output_path, timestamp)
    print(f"Report saved to {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total games: {num_games:,}")
    print(f"  Completed: {stats.games_completed:,}")
    print(f"  Timeout: {stats.games_timeout:,}")
    print(f"  Error: {stats.games_error:,}")
    print(f"\nTotal decisions: {stats.total_decisions:,}")
    print(f"Unique cards seen: {len(stats.cards_seen):,}")
    print(f"Unique matchups: {len(stats.deck_pairs_played)}")
    print(f"Avg decisions/game: {np.mean(stats.decisions_per_game):.1f}")
    print(f"Avg turns/game: {np.mean(stats.turns_per_game):.1f}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCollection time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Rate: {num_games/elapsed:.1f} games/sec")

    print(f"\nDecisions by type:")
    for dtype, count in sorted(stats.decision_counts.items()):
        print(f"  {dtype}: {count:,}")

    # Save summary JSON
    summary = {
        "timestamp": timestamp,
        "num_games": num_games,
        "total_decisions": stats.total_decisions,
        "unique_cards": len(stats.cards_seen),
        "unique_matchups": len(stats.deck_pairs_played),
        "stats": {
            "completed": stats.games_completed,
            "timeout": stats.games_timeout,
            "error": stats.games_error,
            "avg_decisions_per_game": float(np.mean(stats.decisions_per_game)),
            "avg_turns_per_game": float(np.mean(stats.turns_per_game)),
            "decision_counts": dict(stats.decision_counts)
        },
        "files": {
            "hdf5": str(hdf5_file),
            "report": str(report_path)
        },
        "collection_time_seconds": elapsed
    }

    summary_file = output_path / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect AI training data")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--output", default="training_data", help="Output directory")
    parser.add_argument("--decks", nargs="+", default=None, help="Deck files to use")
    parser.add_argument("--host", default="localhost", help="Daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Daemon port")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint every N games")
    parser.add_argument("--timeout", type=int, default=60, help="Game timeout in seconds")
    args = parser.parse_args()

    collect_training_batch(
        num_games=args.games,
        output_dir=args.output,
        decks=args.decks,
        host=args.host,
        port=args.port,
        save_interval=args.save_interval,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()
