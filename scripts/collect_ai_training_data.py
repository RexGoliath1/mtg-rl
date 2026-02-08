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

import json
import logging
import os
import platform
import resource
import socket
import sys
import traceback
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
logger = logging.getLogger("forgerl.collect")


# Deck directories
MODERN_DECKS_DIR = Path(__file__).parent.parent / "decks" / "modern"
ROOT_DECKS_DIR = Path(__file__).parent.parent / "decks"


def load_deck_pool() -> List[str]:
    """Load all available decks from decks/ directory tree.

    Returns just filenames (not paths) since the Forge daemon searches its
    own configured decks directory. This works both locally and in Docker
    where collector and daemon are separate containers.
    """
    decks = []

    for deck_dir in [MODERN_DECKS_DIR, ROOT_DECKS_DIR]:
        if deck_dir.exists():
            for dck in deck_dir.glob("*.dck"):
                decks.append(dck.name)

    if not decks:
        decks = ["red_aggro.dck", "white_weenie.dck"]

    return sorted(set(decks))


# Load deck pool on import
DEFAULT_DECKS = load_deck_pool()


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


def _get_memory_mb() -> float:
    """Return current process RSS in MB (works on Linux and macOS)."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports in bytes, Linux in KB
        if platform.system() == "Darwin":
            return ru.ru_maxrss / (1024 * 1024)
        return ru.ru_maxrss / 1024
    except Exception:
        return 0.0


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
                    logger.warning("Failed to parse DECISION JSON line")

            elif line.startswith("GAME_RESULT:"):
                game_result = line[12:].strip()
                break

            elif line.startswith("GAME_TIMEOUT:"):
                game_result = "TIMEOUT"
                break

            elif line.startswith("ERROR:"):
                game_result = f"ERROR: {line}"
                logger.warning("Forge returned error: %s", line)
                break

    except socket.timeout:
        game_result = "SOCKET_TIMEOUT"
        logger.warning("Socket timeout for game (seed=%s)", seed)
    except Exception as e:
        game_result = f"EXCEPTION: {str(e)}"
        logger.error("Exception in game (seed=%s): %s\n%s", seed, e, traceback.format_exc())
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


def find_action_index(ai_choice: dict, actions: list) -> int:
    """Find the index of ai_choice in the actions list.

    Matches by comparing action type and card name (if present).
    Returns -1 if not found.
    """
    if not isinstance(ai_choice, dict) or not actions:
        return -1

    choice_action = ai_choice.get("action", "")
    choice_card = ai_choice.get("card", "")

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        # Match by action type and card name
        if action.get("action") == choice_action:
            # If both have cards, they must match
            if choice_card and action.get("card"):
                if action.get("card") == choice_card:
                    return idx
            # If choice has no card, just match action type
            elif not choice_card and not action.get("card"):
                return idx
            # If only action matches and there's no card to compare
            elif not choice_card:
                return idx

    # Fallback: try exact match on action type only (first match)
    for idx, action in enumerate(actions):
        if isinstance(action, dict) and action.get("action") == choice_action:
            return idx

    return -1


def encode_decision(decision: dict) -> tuple:
    """Encode a decision into arrays for HDF5 storage."""
    dtype = decision.get("decision_type", "unknown")

    # Encode game state
    game_state = decision.get("game_state", {})
    state_vec = encode_game_state(game_state)

    # Encode decision metadata
    turn = decision.get("turn", 0)

    # Get actions list first (needed for index lookup)
    actions = decision.get("actions", [])
    num_actions = len(actions)

    # Encode AI choice as index into actions list
    ai_choice = decision.get("ai_choice", {})
    if dtype == "choose_action":
        if isinstance(ai_choice, dict) and ai_choice.get("action") == "pass":
            choice_idx = -1  # Pass is not in actions list
        elif isinstance(ai_choice, dict):
            # Find actual index in actions list
            choice_idx = find_action_index(ai_choice, actions)
            if choice_idx == -1:
                # Couldn't find match - log for debugging but treat as pass
                choice_idx = -1
        else:
            choice_idx = -1
    elif dtype in ("declare_attackers", "declare_blockers"):
        if isinstance(ai_choice, list):
            choice_idx = len(ai_choice)  # Number of attackers/blockers
        else:
            choice_idx = 0
    else:
        choice_idx = -1

    return state_vec, turn, choice_idx, num_actions, dtype


def save_to_hdf5(decisions: list, output_path: Path, metadata: dict):
    """Save decisions to HDF5 file.

    Stores both the legacy 17-dim state encoding AND the raw game state JSON.
    The raw JSON enables re-encoding with ForgeGameStateEncoder at training
    time, producing full 512-dim card-level mechanics embeddings.
    """
    if not decisions:
        return

    # Encode all decisions
    state_vecs = []
    turns = []
    choices = []
    num_actions_list = []
    decision_types = []
    game_state_jsons = []

    type_map = {"choose_action": 0, "declare_attackers": 1, "declare_blockers": 2, "unknown": 3}

    for d in decisions:
        state_vec, turn, choice_idx, num_actions, dtype = encode_decision(d)
        state_vecs.append(state_vec)
        turns.append(turn)
        choices.append(choice_idx)
        num_actions_list.append(num_actions)
        decision_types.append(type_map.get(dtype, 3))

        # Serialize full game state for rich encoding at training time
        game_state = d.get("game_state", {})
        game_state_jsons.append(json.dumps(game_state, separators=(",", ":")))

    # Stack into arrays
    states = np.stack(state_vecs)
    turns = np.array(turns, dtype=np.int32)
    choices = np.array(choices, dtype=np.int32)
    num_actions = np.array(num_actions_list, dtype=np.int32)
    dtypes = np.array(decision_types, dtype=np.int8)

    # Save to HDF5
    logger.info("HDF5 write start: %s (%d rows)", output_path.name, len(decisions))
    with h5py.File(output_path, "w") as f:
        f.create_dataset("states", data=states, compression="gzip", compression_opts=4)
        f.create_dataset("turns", data=turns, compression="gzip")
        f.create_dataset("choices", data=choices, compression="gzip")
        f.create_dataset("num_actions", data=num_actions, compression="gzip")
        f.create_dataset("decision_types", data=dtypes, compression="gzip")

        # Store raw game state JSON for ForgeGameStateEncoder at training time
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("game_state_json", data=game_state_jsons, dtype=dt,
                         compression="gzip", compression_opts=4)

        # Store metadata as attributes
        f.attrs["encoding_version"] = 2  # v1=17-dim only, v2=17-dim + raw JSON
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                f.attrs[k] = v
            elif isinstance(v, dict):
                f.attrs[k] = json.dumps(v)

    file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    logger.info("HDF5 write done: %s (%.2f MB, %d rows)", output_path.name, file_size_mb, len(decisions))


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

    # Compute next steps text (avoid backslashes in f-string expressions for Python 3.10)
    games_needed = max(50000 - stats.games_completed, 0)
    if stats.games_completed >= 50000:
        next_step_text = r"\textbf{COMPLETE} - Ready for imitation learning training"
    else:
        next_step_text = r"Collect \textbf{" + f"{games_needed:,}" + r" more games} to reach 50K target"

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

    latex += """\\bottomrule
\\end{tabular}
\\caption{Decisions by Type}
\\end{table}

\\section{Deck Coverage}

\\begin{table}[h]
\\centering
\\begin{tabular}{lr}
\\toprule
\\textbf{Deck Pair} & \\textbf{Games} \\\\
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
\\item State encoding: 17-dim legacy + raw JSON for 512-dim mechanics encoding
\\end{{itemize}}

\\section{{Coverage Analysis}}

\\begin{{itemize}}
\\item Unique cards observed: {len(stats.cards_seen):,}
\\item Unique deck pairs: {len(stats.deck_pairs_played)}
\\item Games per deck pair (avg): {total_games / max(len(stats.deck_pairs_played), 1):.1f}
\\end{{itemize}}

\\section{{Recommended Next Steps}}

\\begin{{enumerate}}
\\item {next_step_text}
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
    timeout: int = 60,
    workers: int = 8  # Parallel workers (Forge daemon supports up to 10)
):
    """Collect training data from multiple games with deck rotation.

    Uses parallel execution for ~5-8x speedup with 8 workers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if decks is None:
        decks = DEFAULT_DECKS

    # Generate deck pairs for full coverage
    deck_pairs = generate_deck_pairs(decks, num_games)

    all_decisions = []
    stats = CollectionStats()
    import threading
    stats_lock = threading.Lock()

    # --- Startup logging ---
    logger.info("=" * 60)
    logger.info("AI TRAINING DATA COLLECTION")
    logger.info("=" * 60)
    logger.info("Python %s on %s", sys.version.split()[0], platform.platform())
    logger.info("PID: %d, Host: %s:%d", os.getpid(), host, port)
    logger.info("NumPy %s, h5py %s", np.__version__, h5py.__version__)
    logger.info("Target games: %d", num_games)
    logger.info("Deck pool: %d decks", len(decks))
    logger.info("Unique matchups: %d", len(set(deck_pairs)))
    logger.info("Output: %s", output_path)
    logger.info("Save interval: every %d games", save_interval)
    logger.info("Parallel workers: %d", workers)
    logger.info("Game timeout: %ds", timeout)
    logger.info("Initial RSS: %.1f MB", _get_memory_mb())
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()
    games_completed_count = [0]  # Mutable for closure

    def run_single_game(game_id):
        """Run a single game and return results."""
        deck1, deck2 = deck_pairs[game_id - 1]
        seed = game_id * 1000 + random.randint(0, 999)
        return game_id, collect_training_game(
            deck1=deck1,
            deck2=deck2,
            seed=seed,
            host=host,
            port=port,
            timeout=timeout
        )

    def process_result(game_id, result):
        """Process a game result thread-safely."""
        num_decisions = len(result["decisions"])
        game_result = result["result"]
        max_turn = result["max_turn"]
        deck1, deck2 = deck_pairs[game_id - 1]

        with stats_lock:
            # Track deck pair
            pair_key = f"{deck1} vs {deck2}"
            stats.deck_pairs_played[pair_key] += 1
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
            games_completed_count[0] += 1

            # Progress indicator every 50 games (or first game)
            completed = games_completed_count[0]
            if completed % 50 == 0 or completed == 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (num_games - completed) / rate if rate > 0 else 0
                success_rate = stats.games_completed / completed * 100 if completed > 0 else 0
                logger.info(
                    "PROGRESS game=%d/%d (%.1f%%) | success=%.1f%% | "
                    "throughput=%.1f games/s | elapsed=%.0fs | ETA=%.1f min | "
                    "decisions=%d | errors=%d | timeouts=%d",
                    completed, num_games, completed * 100 / num_games,
                    success_rate, rate, elapsed, eta / 60,
                    stats.total_decisions, stats.games_error, stats.games_timeout,
                )

            # Memory logging every 100 games
            if completed % 100 == 0 and completed > 0:
                logger.info("MEMORY RSS=%.1f MB | buffer_decisions=%d", _get_memory_mb(), len(all_decisions))

            # Periodic checkpoint
            if completed % save_interval == 0:
                checkpoint_file = output_path / f"training_data_{timestamp}_checkpoint_{completed}.h5"
                metadata = {
                    "timestamp": timestamp,
                    "num_games": completed,
                    "total_decisions": stats.total_decisions,
                    "games_completed": stats.games_completed,
                    "checkpoint": True
                }
                save_to_hdf5(all_decisions.copy(), checkpoint_file, metadata)
                logger.info("Checkpoint saved: %s (%d decisions)", checkpoint_file.name, stats.total_decisions)

    # Run games in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_game, i): i for i in range(1, num_games + 1)}
        for future in as_completed(futures):
            try:
                game_id, result = future.result()
                process_result(game_id, result)
            except Exception as e:
                logger.error("Worker exception for game %d: %s\n%s",
                             futures[future], e, traceback.format_exc())

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

    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info("HDF5 data saved to %s", hdf5_file)
    logger.info("  States shape: (%d, 17)", stats.total_decisions)
    if hdf5_file.exists():
        size_mb = hdf5_file.stat().st_size / (1024 * 1024)
        logger.info("  Compressed size: %.2f MB", size_mb)
    else:
        logger.warning("No data collected (0 decisions). File not created.")

    # Generate report
    report_path = generate_report(stats, output_path, timestamp)
    logger.info("Report saved to %s", report_path)

    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total games: %d (completed=%d, timeout=%d, error=%d)",
                num_games, stats.games_completed, stats.games_timeout, stats.games_error)
    logger.info("Total decisions: %d", stats.total_decisions)
    logger.info("Unique cards seen: %d", len(stats.cards_seen))
    logger.info("Unique matchups: %d", len(stats.deck_pairs_played))
    logger.info("Avg decisions/game: %.1f", np.mean(stats.decisions_per_game))
    logger.info("Avg turns/game: %.1f", np.mean(stats.turns_per_game))
    logger.info("Collection time: %.1f min (%.2f hours)", elapsed / 60, elapsed / 3600)
    logger.info("Throughput: %.1f games/sec", num_games / elapsed if elapsed > 0 else 0)
    logger.info("Final RSS: %.1f MB", _get_memory_mb())

    logger.info("Decisions by type:")
    for dtype, count in sorted(stats.decision_counts.items()):
        logger.info("  %s: %d", dtype, count)

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


def _configure_logging(level: str = "INFO", log_file: str | None = None):
    """Configure logging to stderr + optional file with structured format."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates
    root.handlers.clear()

    fmt = logging.Formatter(LOG_FORMAT)

    # Always log to stderr (visible in CloudWatch and terminal)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    # Optionally log to a file (for S3 upload)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info("Logging to file: %s", log_file)


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
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (max 10)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file (in addition to stderr)")
    args = parser.parse_args()

    _configure_logging(level=args.log_level, log_file=args.log_file)

    collect_training_batch(
        num_games=args.games,
        output_dir=args.output,
        decks=args.decks,
        host=args.host,
        port=args.port,
        save_interval=args.save_interval,
        timeout=args.timeout,
        workers=min(args.workers, 10)  # Forge daemon max is 10
    )


if __name__ == "__main__":
    main()
