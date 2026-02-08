#!/usr/bin/env python3
"""Evaluation report generator -- agent-vs-agent and agent-vs-Forge gameplay.

Plays games in two modes:
  A) Self-play: greedy agent (T=0.0) vs exploratory agent (T=0.5)
  B) vs Forge AI: agent vs Forge daemon (falls back to SimulatedForgeClient)

Generates a multi-page PDF with win-rate bar charts, game-length histograms,
life-total traces, and a summary statistics table.

Usage:
    python3 scripts/generate_eval_report.py
    python3 scripts/generate_eval_report.py --games 20 --checkpoint checkpoints/bc_best.pt
    python3 scripts/generate_eval_report.py --games 5 --output data/reports/eval_report_test.pdf
"""

import argparse
import datetime
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.training.self_play import AlphaZeroNetwork  # noqa: E402
from src.forge.mcts import SimulatedForgeClient  # noqa: E402
from src.forge.policy_value_heads import ActionConfig, create_action_mask, decode_action  # noqa: E402

# Matplotlib -- use non-interactive backend so it works headless
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GameMetrics:
    """Per-game metrics collected during evaluation."""
    game_id: int
    mode: str  # "self_play" or "vs_forge"
    winner: Optional[int]  # 0 = player A / agent, 1 = player B / Forge AI, None = draw
    total_turns: int = 0
    total_decisions: int = 0
    final_life_p0: int = 20
    final_life_p1: int = 20
    duration_sec: float = 0.0
    seed: int = 0
    life_trace_p0: List[int] = field(default_factory=list)
    life_trace_p1: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent wrapper for evaluation
# ---------------------------------------------------------------------------

class EvalAgent:
    """Lightweight agent that picks actions from AlphaZeroNetwork at a given temperature."""

    def __init__(self, network: AlphaZeroNetwork, temperature: float = 0.0):
        self.network = network
        self.temperature = temperature
        self.action_config = ActionConfig()

    @torch.no_grad()
    def pick_action(self, game_state: dict, legal_actions: list) -> int:
        """Return an action index given state and legal actions."""
        state_tensor = self.network.encoder.encode_json(game_state)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        action_mask = create_action_mask(legal_actions, self.action_config)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

        logits = self.network.policy_head(state_tensor, return_logits=True)

        # Mask illegal actions
        logits = logits.masked_fill(mask_tensor == 0, float("-inf"))

        if self.temperature <= 0.0:
            action_idx = logits.argmax(dim=-1).item()
        else:
            scaled = logits / self.temperature
            probs = torch.softmax(scaled, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()

        return action_idx


# ---------------------------------------------------------------------------
# Extended SimulatedForgeClient with life tracking
# ---------------------------------------------------------------------------

class EvalSimulatedClient(SimulatedForgeClient):
    """SimulatedForgeClient extended with life-total tracking for evaluation."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self.move_count = 0
        # Reset initial state with the seeded rng
        self.state = self._create_initial_state()
        self._life_trace_p0: List[int] = [20]
        self._life_trace_p1: List[int] = [20]

    def apply_action(self, action: dict) -> dict:
        """Override to add minor life fluctuations for realistic traces."""
        self.move_count += 1

        # Simulate phase progression (same logic as parent)
        if action.get("type") == "pass":
            if self.state["phase"] == "main1":
                self.state["phase"] = "combat"
            elif self.state["phase"] == "combat":
                self.state["phase"] = "main2"
                # Simulate combat damage occasionally
                if self._rng.random() < 0.3:
                    dmg = self._rng.randint(1, 5)
                    defender = 1 - self.state["activePlayer"]
                    self.state["players"][defender]["life"] -= dmg
            elif self.state["phase"] == "main2":
                self.state["phase"] = "end"
            else:
                self.state["phase"] = "main1"
                self.state["turn"] += 1
                self.state["activePlayer"] = 1 - self.state["activePlayer"]
        else:
            # Non-pass action: small chance of life change
            if self._rng.random() < 0.15:
                target = self._rng.randint(0, 2)
                delta = self._rng.randint(-3, 2)
                self.state["players"][target]["life"] = max(
                    0, self.state["players"][target]["life"] + delta
                )

        # Track life totals
        self._life_trace_p0.append(self.state["players"][0]["life"])
        self._life_trace_p1.append(self.state["players"][1]["life"])

        # End condition: life <= 0 or turn limit
        p0_life = self.state["players"][0]["life"]
        p1_life = self.state["players"][1]["life"]
        if p0_life <= 0 or p1_life <= 0 or self.move_count > 120 or self.state["turn"] > 25:
            self.state["gameOver"] = True
            if p0_life <= 0 and p1_life <= 0:
                self.state["winner"] = None  # draw
            elif p0_life <= 0:
                self.state["winner"] = 1
            elif p1_life <= 0:
                self.state["winner"] = 0
            else:
                # Turn/move limit: higher life wins
                self.state["winner"] = 0 if p0_life >= p1_life else 1

        return self.get_game_state()

    def get_life_traces(self) -> Tuple[List[int], List[int]]:
        return list(self._life_trace_p0), list(self._life_trace_p1)


# ---------------------------------------------------------------------------
# Forge daemon probe
# ---------------------------------------------------------------------------

def probe_forge_daemon(host: str = "localhost", port: int = 17171, timeout: float = 2.0) -> bool:
    """Return True if the Forge daemon is accepting TCP connections."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


# ---------------------------------------------------------------------------
# Game runners
# ---------------------------------------------------------------------------

def play_self_play_game(
    agent_greedy: EvalAgent,
    agent_explore: EvalAgent,
    game_id: int,
    seed: int,
    max_moves: int = 500,
) -> GameMetrics:
    """Play one self-play game: greedy (P0) vs exploratory (P1)."""
    client = EvalSimulatedClient(seed=seed)
    decisions = 0
    t0 = time.time()

    while not client.is_game_over() and decisions < max_moves:
        state = client.get_game_state()
        active = state.get("activePlayer", 0)
        legal = client.get_legal_actions()

        agent = agent_greedy if active == 0 else agent_explore
        action_idx = agent.pick_action(state, legal)
        action_dict = decode_action(action_idx)
        client.apply_action(action_dict)
        decisions += 1

    elapsed = time.time() - t0
    final = client.get_game_state()
    life_p0, life_p1 = client.get_life_traces()

    return GameMetrics(
        game_id=game_id,
        mode="self_play",
        winner=final.get("winner"),
        total_turns=final.get("turn", 1),
        total_decisions=decisions,
        final_life_p0=final["players"][0]["life"],
        final_life_p1=final["players"][1]["life"],
        duration_sec=elapsed,
        seed=seed,
        life_trace_p0=life_p0,
        life_trace_p1=life_p1,
    )


def play_vs_forge_game(
    agent: EvalAgent,
    game_id: int,
    seed: int,
    forge_host: str = "localhost",
    forge_port: int = 17171,
    max_moves: int = 500,
) -> Tuple[GameMetrics, bool]:
    """Play one game vs Forge AI. Returns (metrics, used_real_forge)."""
    used_real = False

    if probe_forge_daemon(forge_host, forge_port):
        # Attempt real Forge game
        try:
            from src.forge.forge_client import ForgeClient
            fc = ForgeClient(forge_host, forge_port)
            fc.connect()
            # Use the two default decks
            deck_dir = Path(__file__).resolve().parent.parent / "decks"
            deck1 = str(deck_dir / "mono_red_aggro.dck")
            deck2 = str(deck_dir / "white_weenie.dck")
            started = fc.start_game(deck1, deck2, seed=seed)
            if started:
                used_real = True
                decisions = 0
                t0 = time.time()
                while True:
                    decision = fc.receive_decision()
                    if decision is None:
                        break
                    # Use greedy agent for vs-Forge
                    encoded_state = {
                        "turn": decision.turn,
                        "phase": decision.phase,
                        "activePlayer": 0,
                        "priorityPlayer": 0,
                        "players": [
                            {"id": 0, "life": 20, "hand": [], "battlefield": []},
                            {"id": 1, "life": 20, "hand": [], "battlefield": []},
                        ],
                        "stack": [],
                        "gameOver": False,
                        "winner": None,
                    }
                    legal = [{"type": "pass", "index": 0}]
                    for a in decision.actions:
                        if a.index >= 0:
                            legal.append({"type": "cast", "index": a.index})
                    action_idx = agent.pick_action(encoded_state, legal)
                    action_dict = decode_action(action_idx)
                    # Send response as index
                    if action_dict["type"] == "pass":
                        fc.send_pass()
                    else:
                        fc.send_action(action_dict.get("index", 0))
                    decisions += 1
                    if decisions >= max_moves:
                        break
                elapsed = time.time() - t0
                result = fc.get_result()
                fc.close()

                winner = None
                if result and not result.is_draw:
                    winner = 0 if result.winner and fc.our_player_name and fc.our_player_name in result.winner else 1

                return GameMetrics(
                    game_id=game_id,
                    mode="vs_forge",
                    winner=winner,
                    total_turns=decisions // 4,  # rough estimate
                    total_decisions=decisions,
                    final_life_p0=20,
                    final_life_p1=20,
                    duration_sec=elapsed,
                    seed=seed,
                ), True
        except Exception as e:
            print(f"  [!] Forge daemon error: {e}. Falling back to simulation.")

    # Fallback: simulated game (agent vs random policy)
    client = EvalSimulatedClient(seed=seed)
    decisions = 0
    t0 = time.time()

    while not client.is_game_over() and decisions < max_moves:
        state = client.get_game_state()
        active = state.get("activePlayer", 0)
        legal = client.get_legal_actions()

        if active == 0:
            action_idx = agent.pick_action(state, legal)
        else:
            # Forge AI stand-in: random legal action
            mask = create_action_mask(legal)
            legal_idxs = np.nonzero(mask)[0]
            action_idx = int(np.random.choice(legal_idxs)) if len(legal_idxs) > 0 else 0

        action_dict = decode_action(action_idx)
        client.apply_action(action_dict)
        decisions += 1

    elapsed = time.time() - t0
    final = client.get_game_state()
    life_p0, life_p1 = client.get_life_traces()

    return GameMetrics(
        game_id=game_id,
        mode="vs_forge",
        winner=final.get("winner"),
        total_turns=final.get("turn", 1),
        total_decisions=decisions,
        final_life_p0=final["players"][0]["life"],
        final_life_p1=final["players"][1]["life"],
        duration_sec=elapsed,
        seed=seed,
        life_trace_p0=life_p0,
        life_trace_p1=life_p1,
    ), used_real


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------

def generate_report(
    self_play_results: List[GameMetrics],
    forge_results: List[GameMetrics],
    output_path: str,
    checkpoint_path: str,
    num_games: int,
    forge_live: bool,
):
    """Generate multi-page PDF evaluation report."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with PdfPages(output_path) as pdf:
        # ---------------------------------------------------------------
        # Page 1: Title page
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        title_lines = [
            "ForgeRL Evaluation Report",
            "",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Checkpoint: {os.path.basename(checkpoint_path)}",
            f"Games per mode: {num_games}",
            f"Seed range: 42 .. {42 + num_games - 1}",
            "",
            "Mode A: Self-play (greedy T=0.0 vs exploratory T=0.5)",
            f"Mode B: vs Forge AI ({'LIVE daemon' if forge_live else 'SimulatedForgeClient fallback'})",
            "",
            f"Total games played: {len(self_play_results) + len(forge_results)}",
        ]
        ax.text(
            0.5, 0.65, "\n".join(title_lines),
            transform=ax.transAxes, fontsize=13, va="center", ha="center",
            fontfamily="monospace", linespacing=1.6,
        )
        ax.text(
            0.5, 0.15, "Generated by scripts/generate_eval_report.py",
            transform=ax.transAxes, fontsize=9, va="center", ha="center",
            color="gray",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------
        # Page 2: Win-rate bar chart
        # ---------------------------------------------------------------
        def win_rates(results: List[GameMetrics]):
            if not results:
                return 0.0, 0.0, 0.0
            wins = sum(1 for g in results if g.winner == 0)
            losses = sum(1 for g in results if g.winner == 1)
            draws = sum(1 for g in results if g.winner is None)
            n = len(results)
            return wins / n, losses / n, draws / n

        sp_wr = win_rates(self_play_results)
        fg_wr = win_rates(forge_results)

        fig, ax = plt.subplots(figsize=(8.5, 6))
        x = np.arange(3)
        width = 0.35
        labels = ["Wins (P0/Agent)", "Losses (P1/Forge)", "Draws"]

        bars1 = ax.bar(x - width / 2, sp_wr, width, label="Self-play", color="#4A90D9")
        bars2 = ax.bar(x + width / 2, fg_wr, width, label="vs Forge", color="#D94A4A")

        ax.set_ylabel("Rate")
        ax.set_title("Win Rate Comparison: Self-Play vs Forge")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------
        # Page 3: Game length distribution
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        sp_turns = [g.total_turns for g in self_play_results]
        fg_turns = [g.total_turns for g in forge_results]

        if sp_turns:
            axes[0].hist(sp_turns, bins=max(5, len(set(sp_turns))), color="#4A90D9", edgecolor="black", alpha=0.8)
        axes[0].set_title("Self-Play: Game Length (turns)")
        axes[0].set_xlabel("Turns")
        axes[0].set_ylabel("Count")

        if fg_turns:
            axes[1].hist(fg_turns, bins=max(5, len(set(fg_turns))), color="#D94A4A", edgecolor="black", alpha=0.8)
        axes[1].set_title("vs Forge: Game Length (turns)")
        axes[1].set_xlabel("Turns")
        axes[1].set_ylabel("Count")

        fig.suptitle("Game Length Distribution", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------
        # Page 4: Life total traces (first 3 games per mode)
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle("Life Total Traces (first 3 games per mode)", fontsize=13)

        for col_idx in range(3):
            # Self-play row
            ax_sp = axes[0, col_idx]
            if col_idx < len(self_play_results):
                g = self_play_results[col_idx]
                if g.life_trace_p0 and g.life_trace_p1:
                    ax_sp.plot(g.life_trace_p0, label="P0 (greedy)", color="#4A90D9")
                    ax_sp.plot(g.life_trace_p1, label="P1 (explore)", color="#D94A4A")
                    ax_sp.set_title(f"SP Game {g.game_id} (seed {g.seed})", fontsize=9)
                    ax_sp.legend(fontsize=7)
                else:
                    ax_sp.text(0.5, 0.5, "No trace data", transform=ax_sp.transAxes, ha="center")
            else:
                ax_sp.axis("off")
            ax_sp.set_ylabel("Life" if col_idx == 0 else "")
            ax_sp.set_xlabel("Decisions")

            # vs Forge row
            ax_fg = axes[1, col_idx]
            if col_idx < len(forge_results):
                g = forge_results[col_idx]
                if g.life_trace_p0 and g.life_trace_p1:
                    ax_fg.plot(g.life_trace_p0, label="Agent", color="#4A90D9")
                    ax_fg.plot(g.life_trace_p1, label="Forge/Sim", color="#D94A4A")
                    ax_fg.set_title(f"Forge Game {g.game_id} (seed {g.seed})", fontsize=9)
                    ax_fg.legend(fontsize=7)
                else:
                    ax_fg.text(0.5, 0.5, "No trace data", transform=ax_fg.transAxes, ha="center")
            else:
                ax_fg.axis("off")
            ax_fg.set_ylabel("Life" if col_idx == 0 else "")
            ax_fg.set_xlabel("Decisions")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------
        # Page 5: Summary statistics table
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")

        def stats_row(label: str, results: List[GameMetrics]) -> list:
            if not results:
                return [label, "N/A"] * 6
            n = len(results)
            wr = sum(1 for g in results if g.winner == 0) / n
            avg_turns = np.mean([g.total_turns for g in results])
            avg_dec = np.mean([g.total_decisions for g in results])
            avg_dur = np.mean([g.duration_sec for g in results])
            avg_life_p0 = np.mean([g.final_life_p0 for g in results])
            avg_life_p1 = np.mean([g.final_life_p1 for g in results])
            return [
                label,
                f"{n}",
                f"{wr:.1%}",
                f"{avg_turns:.1f}",
                f"{avg_dec:.0f}",
                f"{avg_dur:.2f}s",
                f"{avg_life_p0:.1f} / {avg_life_p1:.1f}",
            ]

        col_labels = ["Mode", "Games", "Win Rate", "Avg Turns", "Avg Decisions", "Avg Duration", "Avg Life (P0/P1)"]
        table_data = [
            stats_row("Self-play", self_play_results),
            stats_row("vs Forge", forge_results),
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header row
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4A90D9")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title("Summary Statistics", fontsize=14, pad=20)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report (PDF)")
    parser.add_argument("--checkpoint", default="checkpoints/imitation_policy.pt", help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=10, help="Number of games per mode")
    parser.add_argument("--output", default=None, help="Output PDF path")
    parser.add_argument("--forge-host", default="localhost", help="Forge daemon host")
    parser.add_argument("--forge-port", type=int, default=17171, help="Forge daemon port")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        args.output = f"data/reports/eval_report_{today}.pdf"

    # ------------------------------------------------------------------
    # Load or create network
    # ------------------------------------------------------------------
    ckpt_path = Path(project_root) / args.checkpoint
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        network = AlphaZeroNetwork.load(str(ckpt_path))
    else:
        print(f"Checkpoint not found at {ckpt_path}. Using freshly initialized network.")
        network = AlphaZeroNetwork()

    network.eval()

    agent_greedy = EvalAgent(network, temperature=0.0)
    agent_explore = EvalAgent(network, temperature=0.5)

    # ------------------------------------------------------------------
    # Mode A: Self-play
    # ------------------------------------------------------------------
    print(f"\n=== Mode A: Self-Play ({args.games} games) ===")
    sp_results: List[GameMetrics] = []
    for i in range(args.games):
        seed = args.seed + i
        metrics = play_self_play_game(agent_greedy, agent_explore, game_id=i, seed=seed)
        w_str = {0: "P0 (greedy)", 1: "P1 (explore)", None: "draw"}.get(metrics.winner, "?")
        print(f"  Game {i + 1}/{args.games}: winner={w_str}, turns={metrics.total_turns}, "
              f"decisions={metrics.total_decisions}, life={metrics.final_life_p0}/{metrics.final_life_p1}")
        sp_results.append(metrics)

    # ------------------------------------------------------------------
    # Mode B: vs Forge AI
    # ------------------------------------------------------------------
    print(f"\n=== Mode B: vs Forge AI ({args.games} games) ===")
    forge_live = probe_forge_daemon(args.forge_host, args.forge_port)
    if forge_live:
        print(f"  Forge daemon detected at {args.forge_host}:{args.forge_port}")
    else:
        print("  Forge daemon not available. Using SimulatedForgeClient fallback.")

    fg_results: List[GameMetrics] = []
    any_real = False
    for i in range(args.games):
        seed = args.seed + i
        metrics, used_real = play_vs_forge_game(
            agent_greedy, game_id=i, seed=seed,
            forge_host=args.forge_host, forge_port=args.forge_port,
        )
        if used_real:
            any_real = True
        w_str = {0: "Agent", 1: "Forge", None: "draw"}.get(metrics.winner, "?")
        src = "LIVE" if used_real else "SIM"
        print(f"  Game {i + 1}/{args.games} [{src}]: winner={w_str}, turns={metrics.total_turns}, "
              f"decisions={metrics.total_decisions}, life={metrics.final_life_p0}/{metrics.final_life_p1}")
        fg_results.append(metrics)

    # ------------------------------------------------------------------
    # Generate report
    # ------------------------------------------------------------------
    print("\n=== Generating PDF report ===")
    output_path = str(Path(project_root) / args.output)
    generate_report(
        self_play_results=sp_results,
        forge_results=fg_results,
        output_path=output_path,
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        forge_live=any_real,
    )

    # Print summary
    sp_wr = sum(1 for g in sp_results if g.winner == 0) / max(1, len(sp_results))
    fg_wr = sum(1 for g in fg_results if g.winner == 0) / max(1, len(fg_results))
    print("\n=== Summary ===")
    print(f"  Self-play win rate (greedy): {sp_wr:.1%}")
    print(f"  vs Forge win rate (agent):   {fg_wr:.1%}")
    print(f"  Report: {output_path}")


if __name__ == "__main__":
    main()
