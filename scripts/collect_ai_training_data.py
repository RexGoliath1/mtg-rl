#!/usr/bin/env python3
"""
Collect AI Training Data (v2/v3 format)

Run games in observation mode (-o) where the Forge AI makes all
decisions autonomously. The collector observes and records each
decision point (game state, available actions, AI's choice) for
imitation learning.

Supports two wire formats (auto-detected per line):
- v2 (JSON): DECISION:{json} lines → HDF5 with game_state_json
- v3 (binary): DECISION_BIN:<base64> lines → HDF5 compound dataset (1060 bytes/decision)

The v3 binary format uses 93% less memory than v2 JSON and writes
directly to structured numpy arrays with no string allocation.

For imitation learning bootstrapping:
- Focus on card selection and turn flow
- Use diverse decks for embedding space coverage
- Target 50,000+ games for robust training
"""

import base64
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
from typing import List, Optional, Tuple
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.forge.binary_state import (
    DECISION_DTYPE,
    BinaryDecisionBuffer,
)

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
                if ' ' in dck.name:
                    continue
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
    """Run a single game in observation mode (-o) and collect decisions.

    The Forge AI makes all decisions autonomously. We observe the
    DECISION: stream (game state + ai_choice) without sending responses.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout + 30)
    sock.connect((host, port))

    rfile = sock.makefile("r", buffering=1)
    wfile = sock.makefile("w", buffering=1)

    decisions = []
    binary_decisions = []  # Binary DECISION_DTYPE records (v3 format)
    game_result = None
    max_turn = 0
    cards_seen = set()

    # Map binary decision_type enum → string for filtering
    _BINARY_DTYPE_MAP = {0: "choose_action", 1: "declare_attackers", 2: "declare_blockers"}

    try:
        # Start game in observation mode (-o) where Forge AI makes all
        # decisions autonomously.  The daemon streams DECISION: lines
        # with game_state, actions, and ai_choice (the AI's pick).
        # No responses needed -- purely one-way observation.
        cmd = f"NEWGAME {deck1} {deck2} -o -c {timeout}"
        if seed is not None:
            cmd += f" -s {seed}"

        wfile.write(cmd + "\n")
        wfile.flush()

        # Read all decisions until game ends (no responses needed)
        while True:
            line = rfile.readline()
            if not line:
                break

            line = line.strip()
            logger.debug("DAEMON: %s", line[:200])
            if line.startswith("DECISION:"):
                try:
                    data = json.loads(line[9:])
                    max_turn = max(max_turn, data.get("turn", 0))

                    decision_type = data.get("decision_type", "")

                    # Only keep decisions with real training signal.
                    # Other types (reveal, choose_entity, confirm_action,
                    # play_trigger, etc.) are noise that inflates the
                    # dataset ~15x and causes OOM.
                    if decision_type in TRAINING_DECISION_TYPES:
                        decisions.append(data)

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

            elif line.startswith("DECISION_BIN:"):
                try:
                    payload = line[13:]
                    raw = base64.b64decode(payload)
                    record = np.frombuffer(raw, dtype=DECISION_DTYPE)[0].copy()
                    max_turn = max(max_turn, int(record['turn']))

                    dtype_str = _BINARY_DTYPE_MAP.get(int(record['decision_type']), "")
                    if dtype_str in TRAINING_DECISION_TYPES:
                        binary_decisions.append(record)

                except Exception:
                    logger.warning("Failed to parse DECISION_BIN line")

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
        "binary_decisions": binary_decisions,
        "result": game_result,
        "deck1": deck1,
        "deck2": deck2,
        "seed": seed,
        "max_turn": max_turn,
        "cards_seen": cards_seen,
    }


def _extract_decision_fields(decision: dict) -> tuple:
    """Extract HDF5 fields from a raw decision dict.

    Returns (turn, choice_idx, num_actions, decision_type_str, game_state_json).
    The choice_idx is the index of ai_choice in the actions list as reported
    by Forge (ai_choice_index field), or -1 for pass / unresolved.
    """
    dtype = decision.get("decision_type", "unknown")
    turn = decision.get("turn", 0)
    actions = decision.get("actions", [])
    num_actions = len(actions)

    # Use Forge-reported choice index when available (v2 path).
    # Falls back to -1 (pass / unknown).
    ai_choice = decision.get("ai_choice", {})
    choice_idx = decision.get("ai_choice_index", -1)
    if choice_idx == -1 and isinstance(ai_choice, dict) and ai_choice.get("action") == "pass":
        choice_idx = -1
    elif dtype in ("declare_attackers", "declare_blockers"):
        if isinstance(ai_choice, list):
            choice_idx = len(ai_choice)  # Number of attackers/blockers
        else:
            choice_idx = 0

    game_state = decision.get("game_state", {})
    game_state_json = json.dumps(game_state, separators=(",", ":"))

    return turn, choice_idx, num_actions, dtype, game_state_json


def save_to_hdf5(decisions: list, output_path: Path, metadata: dict):
    """Save decisions to HDF5 file (v2 format).

    Stores raw game_state_json for re-encoding with ForgeGameStateEncoder
    at training time, plus decision metadata (turns, choices, num_actions,
    decision_types).
    """
    if not decisions:
        return

    turns = []
    choices = []
    num_actions_list = []
    decision_types = []
    game_state_jsons = []

    type_map = {"choose_action": 0, "declare_attackers": 1, "declare_blockers": 2, "unknown": 3}

    for d in decisions:
        turn, choice_idx, num_actions, dtype, gs_json = _extract_decision_fields(d)
        turns.append(turn)
        choices.append(choice_idx)
        num_actions_list.append(num_actions)
        decision_types.append(type_map.get(dtype, 3))
        game_state_jsons.append(gs_json)

    turns = np.array(turns, dtype=np.int32)
    choices = np.array(choices, dtype=np.int32)
    num_actions = np.array(num_actions_list, dtype=np.int32)
    dtypes = np.array(decision_types, dtype=np.int8)

    logger.info("HDF5 write start: %s (%d rows)", output_path.name, len(decisions))
    with h5py.File(output_path, "w") as f:
        f.create_dataset("turns", data=turns, compression="gzip")
        f.create_dataset("choices", data=choices, compression="gzip")
        f.create_dataset("num_actions", data=num_actions, compression="gzip")
        f.create_dataset("decision_types", data=dtypes, compression="gzip")

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("game_state_json", data=game_state_jsons, dtype=dt,
                         compression="gzip", compression_opts=4)

        f.attrs["encoding_version"] = 2
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                f.attrs[k] = v
            elif isinstance(v, dict):
                f.attrs[k] = json.dumps(v)

    file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    logger.info("HDF5 write done: %s (%.2f MB, %d rows)", output_path.name, file_size_mb, len(decisions))


# Decision types that contain useful training signal. Other types
# (reveal, choose_entity, choose_ability, confirm_action, play_trigger,
# play_from_effect, announce_value) are noise -- either auto-confirmed or
# contain no meaningful choice the model needs to learn.
TRAINING_DECISION_TYPES = frozenset({
    "choose_action",
    "declare_attackers",
    "declare_blockers",
})


class IncrementalHDF5Writer:
    """Write decisions to HDF5 incrementally, avoiding OOM.

    Opens the HDF5 file once with extensible (resizable) datasets and
    flushes buffered decisions periodically so that only a small window
    of data is ever held in RAM.

    V2 format: stores game_state_json (raw JSON for ForgeGameStateEncoder)
    plus decision metadata (turns, choices, num_actions, decision_types).
    """

    TYPE_MAP = {"choose_action": 0, "declare_attackers": 1, "declare_blockers": 2, "unknown": 3}

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.total_rows = 0
        dt_vlen = h5py.special_dtype(vlen=str)

        self._file = h5py.File(output_path, "w")
        self._file.attrs["encoding_version"] = 2

        # Create extensible datasets with chunked storage for efficient
        # appends and gzip compression.
        self._ds_turns = self._file.create_dataset(
            "turns", shape=(0,), maxshape=(None,),
            dtype="i4", chunks=(1000,), compression="gzip",
        )
        self._ds_choices = self._file.create_dataset(
            "choices", shape=(0,), maxshape=(None,),
            dtype="i4", chunks=(1000,), compression="gzip",
        )
        self._ds_num_actions = self._file.create_dataset(
            "num_actions", shape=(0,), maxshape=(None,),
            dtype="i4", chunks=(1000,), compression="gzip",
        )
        self._ds_decision_types = self._file.create_dataset(
            "decision_types", shape=(0,), maxshape=(None,),
            dtype="i1", chunks=(1000,), compression="gzip",
        )
        self._ds_game_state_json = self._file.create_dataset(
            "game_state_json", shape=(0,), maxshape=(None,),
            dtype=dt_vlen, chunks=(1000,),
            compression="gzip", compression_opts=4,
        )

    def flush(self, decisions: list) -> int:
        """Encode *decisions* and append to the open HDF5 datasets.

        Returns the number of rows written (for logging).
        """
        if not decisions:
            return 0

        turns = []
        choices = []
        num_actions_list = []
        decision_types = []
        game_state_jsons = []

        for d in decisions:
            turn, choice_idx, num_actions, dtype, gs_json = _extract_decision_fields(d)
            turns.append(turn)
            choices.append(choice_idx)
            num_actions_list.append(num_actions)
            decision_types.append(self.TYPE_MAP.get(dtype, 3))
            game_state_jsons.append(gs_json)

        n = len(decisions)
        old = self.total_rows
        new = old + n

        self._ds_turns.resize((new,))
        self._ds_turns[old:new] = np.array(turns, dtype=np.int32)

        self._ds_choices.resize((new,))
        self._ds_choices[old:new] = np.array(choices, dtype=np.int32)

        self._ds_num_actions.resize((new,))
        self._ds_num_actions[old:new] = np.array(num_actions_list, dtype=np.int32)

        self._ds_decision_types.resize((new,))
        self._ds_decision_types[old:new] = np.array(decision_types, dtype=np.int8)

        self._ds_game_state_json.resize((new,))
        self._ds_game_state_json[old:new] = game_state_jsons

        self._file.flush()
        self.total_rows = new
        return n

    def set_metadata(self, metadata: dict):
        """Store metadata as HDF5 file-level attributes."""
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                self._file.attrs[k] = v
            elif isinstance(v, dict):
                self._file.attrs[k] = json.dumps(v)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


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
\\item State encoding: raw game\\_state\\_json for 768-dim ForgeGameStateEncoder at training time
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
    workers: int = 8,  # Parallel workers (Forge daemon supports up to 10)
    flush_interval: int = 50,  # Flush buffered decisions to HDF5 every N games
):
    """Collect training data from multiple games with deck rotation.

    Uses parallel execution for ~5-8x speedup with 8 workers.
    Writes decisions incrementally to HDF5 to avoid OOM on large runs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if decks is None:
        decks = DEFAULT_DECKS

    # Generate deck pairs for full coverage
    deck_pairs = generate_deck_pairs(decks, num_games)

    # Buffer holds decisions between flushes. Cleared after each flush.
    decision_buffer = []
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
    logger.info("Flush interval: every %d games", flush_interval)
    logger.info("Parallel workers: %d", workers)
    logger.info("Game timeout: %ds", timeout)
    logger.info("Decision filter: %s", sorted(TRAINING_DECISION_TYPES))
    logger.info("Initial RSS: %.1f MB", _get_memory_mb())
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()
    games_completed_count = [0]  # Mutable for closure

    # Open incremental HDF5 writer for the final output file
    hdf5_file = output_path / f"training_data_{timestamp}_final.h5"
    writer = IncrementalHDF5Writer(hdf5_file)

    # Binary writer (v3) — initialized lazily when first binary decision arrives
    binary_buffer: Optional[BinaryDecisionBuffer] = None
    binary_decision_count = [0]

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
        num_decisions = len(result["decisions"]) + len(result.get("binary_decisions", []))
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

            # Count decision types and buffer for HDF5
            for d in result["decisions"]:
                dtype = d.get("decision_type", "unknown")
                stats.decision_counts[dtype] += 1
                decision_buffer.append(d)

            # Handle binary decisions (v3 format)
            bin_decs = result.get("binary_decisions", [])
            if bin_decs:
                nonlocal binary_buffer
                if binary_buffer is None:
                    binary_dir = output_path / f"binary_{timestamp}"
                    binary_dir.mkdir(parents=True, exist_ok=True)
                    binary_buffer = BinaryDecisionBuffer(
                        str(binary_dir), save_interval=5000
                    )
                for rec in bin_decs:
                    binary_buffer.add(rec)
                    binary_decision_count[0] += 1

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
                    "decisions=%d (disk=%d buf=%d) | errors=%d | timeouts=%d",
                    completed, num_games, completed * 100 / num_games,
                    success_rate, rate, elapsed, eta / 60,
                    stats.total_decisions, writer.total_rows, len(decision_buffer),
                    stats.games_error, stats.games_timeout,
                )

            # Memory logging every 100 games
            if completed % 100 == 0 and completed > 0:
                logger.info("MEMORY RSS=%.1f MB | buffer=%d | disk=%d",
                            _get_memory_mb(), len(decision_buffer), writer.total_rows)

            # Flush buffer to HDF5 every flush_interval games
            if completed % flush_interval == 0 and decision_buffer:
                n = writer.flush(decision_buffer)
                logger.info("HDF5 flush: wrote %d decisions to disk (total=%d)",
                            n, writer.total_rows)
                decision_buffer.clear()

            # Legacy checkpoint (separate file) at save_interval
            if completed % save_interval == 0:
                # Flush remaining buffer first so checkpoint is consistent
                if decision_buffer:
                    n = writer.flush(decision_buffer)
                    logger.info("HDF5 flush (checkpoint): wrote %d decisions (total=%d)",
                                n, writer.total_rows)
                    decision_buffer.clear()
                logger.info("Checkpoint at game %d: %d total decisions on disk",
                            completed, writer.total_rows)

    # Run games in parallel
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_single_game, i): i for i in range(1, num_games + 1)}
            for future in as_completed(futures):
                try:
                    game_id, result = future.result()
                    process_result(game_id, result)
                except Exception as e:
                    logger.error("Worker exception for game %d: %s\n%s",
                                 futures[future], e, traceback.format_exc())

        # Flush any remaining buffered decisions
        if decision_buffer:
            n = writer.flush(decision_buffer)
            logger.info("Final HDF5 flush: wrote %d decisions (total=%d)",
                        n, writer.total_rows)
            decision_buffer.clear()

        # Write metadata and close
        metadata = {
            "timestamp": timestamp,
            "num_games": num_games,
            "total_decisions": writer.total_rows,
            "games_completed": stats.games_completed,
            "games_timeout": stats.games_timeout,
            "games_error": stats.games_error,
            "unique_cards": len(stats.cards_seen),
            "unique_matchups": len(stats.deck_pairs_played),
            "flush_interval": flush_interval,
        }
        writer.set_metadata(metadata)
    finally:
        writer.close()

        # Finalize binary buffer if used
        if binary_buffer is not None:
            binary_final = binary_buffer.finalize()
            logger.info("Binary HDF5 (v3) finalized: %s (%d decisions)",
                        binary_final, binary_decision_count[0])

    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info("HDF5 data saved to %s", hdf5_file)
    logger.info("  JSON decisions (v2): %d rows", writer.total_rows)
    if binary_decision_count[0] > 0:
        logger.info("  Binary decisions (v3): %d rows", binary_decision_count[0])
    if hdf5_file.exists():
        size_mb = hdf5_file.stat().st_size / (1024 * 1024)
        logger.info("  Compressed size: %.2f MB", size_mb)
    else:
        if binary_decision_count[0] == 0:
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
    parser.add_argument("--flush-interval", type=int, default=50,
                        help="Flush buffered decisions to HDF5 every N games (default 50)")
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
        workers=min(args.workers, 10),  # Forge daemon max is 10
        flush_interval=args.flush_interval,
    )


if __name__ == "__main__":
    main()
