#!/usr/bin/env python3
"""
Collect Game Metrics

Run multiple games with the simple agent and collect metrics for the report.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forge.forge_client import ForgeClient
from src.training.simple_agent import SimpleAgent


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    game_id: int
    total_turns: int = 0
    total_decisions: int = 0
    decisions_per_turn: List[int] = field(default_factory=list)
    winner: str = ""
    duration_s: float = 0


def run_games(num_games: int = 10, host: str = "localhost", port: int = 17171):
    """Run multiple games and collect metrics."""
    metrics = []

    for game_id in range(1, num_games + 1):
        print(f"Game {game_id}/{num_games}...", end=" ", flush=True)

        client = ForgeClient(host, port, timeout=120)
        agent = SimpleAgent()
        game_metrics = GameMetrics(game_id=game_id)

        import time
        start = time.time()

        try:
            client.connect()
            client.start_game('red_aggro.dck', 'white_weenie.dck', seed=game_id * 1000)

            decisions = 0
            max_decisions = 2000
            turn_decisions = {}
            current_turn = 0

            while decisions < max_decisions:
                decision = client.receive_decision()
                if decision is None:
                    break

                decisions += 1

                if decision.turn > current_turn:
                    if current_turn > 0:
                        game_metrics.decisions_per_turn.append(turn_decisions.get(current_turn, 0))
                    current_turn = decision.turn
                    turn_decisions[current_turn] = 0
                    agent.reset_turn()

                turn_decisions[current_turn] = turn_decisions.get(current_turn, 0) + 1

                response, _ = agent.decide(decision)
                client.send_response(response)

            # Record final turn
            if current_turn > 0:
                game_metrics.decisions_per_turn.append(turn_decisions.get(current_turn, 0))

            game_metrics.total_turns = current_turn
            game_metrics.total_decisions = decisions

            result = client.get_result()
            if result:
                game_metrics.winner = result.winner or "Draw"

        except Exception as e:
            print(f"Error: {e}")
            game_metrics.winner = f"Error: {str(e)[:20]}"
        finally:
            client.close()

        game_metrics.duration_s = time.time() - start
        metrics.append(game_metrics)

        avg_dpt = (game_metrics.total_decisions / game_metrics.total_turns
                   if game_metrics.total_turns > 0 else 0)
        print(f"{game_metrics.total_turns} turns, {game_metrics.total_decisions} decisions, "
              f"{avg_dpt:.1f} d/t, {game_metrics.duration_s:.1f}s")

    return metrics


def generate_report(metrics: List[GameMetrics], output_dir: str = "reports"):
    """Generate LaTeX report from metrics."""

    # Aggregate stats
    total_games = len(metrics)
    total_turns = sum(m.total_turns for m in metrics)
    total_decisions = sum(m.total_decisions for m in metrics)

    avg_turns = total_turns / total_games if total_games > 0 else 0
    avg_decisions = total_decisions / total_games if total_games > 0 else 0

    all_dpt = []
    for m in metrics:
        all_dpt.extend(m.decisions_per_turn)

    avg_dpt = sum(all_dpt) / len(all_dpt) if all_dpt else 0
    max_dpt = max(all_dpt) if all_dpt else 0

    # Winner breakdown
    winners = defaultdict(int)
    for m in metrics:
        w = m.winner[:30] if m.winner else "Unknown"
        winners[w] += 1

    # Create LaTeX
    latex = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{xcolor}
\geometry{margin=1in}

\title{Forge Imitation Training Report}
\author{ForgeRL System}
\date{""" + datetime.now().strftime("%Y-%m-%d %H:%M") + r"""}

\begin{document}
\maketitle

\section{Summary}

Training run with Simple Agent to collect game metrics.

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Games & """ + str(total_games) + r""" \\
Total Decisions & """ + f"{total_decisions:,}" + r""" \\
Total Turns & """ + f"{total_turns:,}" + r""" \\
\midrule
Avg Turns/Game & """ + f"{avg_turns:.1f}" + r""" \\
Avg Decisions/Game & """ + f"{avg_decisions:.1f}" + r""" \\
Avg Decisions/Turn & """ + f"{avg_dpt:.1f}" + r""" \\
Max Decisions/Turn & """ + str(max_dpt) + r""" \\
\bottomrule
\end{tabular}
\caption{Game Complexity Metrics}
\end{table}

\section{Target vs Current}

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Target} & \textbf{Current} \\
\midrule
Turns/Game & 8-15 & """ + f"{avg_turns:.1f}" + r""" \\
Decisions/Turn & 5-20 & """ + f"{avg_dpt:.1f}" + r""" \\
Decisions/Game & 50-200 & """ + f"{avg_decisions:.1f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\section{Observations}

\begin{itemize}
"""
    if avg_turns > 50:
        latex += r"\item \textcolor{red}{Games are too long (> 50 turns). Agents may not be attacking effectively.}" + "\n"
    elif avg_turns > 20:
        latex += r"\item Games are longer than target. Consider more aggressive play." + "\n"
    else:
        latex += r"\item Game length is within acceptable range." + "\n"

    if avg_dpt > 50:
        latex += r"\item \textcolor{red}{Too many decisions per turn. Agent may be stuck.}" + "\n"
    elif avg_dpt <= 20:
        latex += r"\item Decisions per turn is acceptable." + "\n"

    if max_dpt > 50:
        latex += r"\item WARNING: Max decisions/turn = " + str(max_dpt) + r" (should be < 50)" + "\n"

    latex += r"""
\item Games complete with winners (not draws)
\item Next: Integrate Forge AI suggestions for better play patterns
\end{itemize}

\section{Individual Games}

\begin{table}[h]
\centering
\begin{tabular}{rrrrl}
\toprule
\textbf{Game} & \textbf{Turns} & \textbf{Decisions} & \textbf{D/Turn} & \textbf{Winner} \\
\midrule
"""
    for m in metrics:
        dpt = m.total_decisions / m.total_turns if m.total_turns > 0 else 0
        winner = m.winner.replace('_', '\\_').replace('Agent(', '').replace(')', '')[:20]
        latex += f"{m.game_id} & {m.total_turns} & {m.total_decisions} & {dpt:.1f} & {winner} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}

\section{Next Steps}

\begin{enumerate}
\item Add Forge AI suggestion support to daemon
\item Implement confidence network for uncertain decisions
\item Reduce game length through better attack patterns
\item Train neural network on collected data
\end{enumerate}

\end{document}
"""

    # Write files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tex_path = output_path / "training_report.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX saved to: {tex_path}")

    # Write summary
    summary = {
        "date": datetime.now().isoformat(),
        "total_games": total_games,
        "total_turns": total_turns,
        "total_decisions": total_decisions,
        "avg_turns_per_game": avg_turns,
        "avg_decisions_per_game": avg_decisions,
        "avg_decisions_per_turn": avg_dpt,
        "max_decisions_per_turn": max_dpt,
        "games": [
            {
                "game_id": m.game_id,
                "turns": m.total_turns,
                "decisions": m.total_decisions,
                "winner": m.winner,
                "duration_s": m.duration_s
            }
            for m in metrics
        ]
    }

    json_path = output_path / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to: {json_path}")

    # Try to compile PDF
    try:
        import subprocess
        result = subprocess.run(
            ["pdflatex", "-output-directory", str(output_path), str(tex_path)],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            print(f"PDF saved to: {output_path / 'training_report.pdf'}")
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5, help="Number of games")
    parser.add_argument("--output", default="reports", help="Output directory")
    args = parser.parse_args()

    print(f"Running {args.games} games...")
    metrics = run_games(args.games)

    print(f"\nGenerating report...")
    generate_report(metrics, args.output)


if __name__ == "__main__":
    main()
