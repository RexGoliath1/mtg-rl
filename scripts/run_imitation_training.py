#!/usr/bin/env python3
"""
Run Imitation Training and Generate Report

This script:
1. Runs games with the heuristic agent
2. Collects game metrics (turns, decisions, etc.)
3. Generates a LaTeX report showing progress
4. Saves training data for model development

Usage:
    python scripts/run_imitation_training.py --games 20 --output reports/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.forge_imitation import (
    ImitationDataCollector
)


def generate_latex_report(
    metrics: list,
    stats: dict,
    output_dir: str,
    experiment_name: str = "Imitation Training"
) -> str:
    """Generate a LaTeX report from game metrics."""

    # Calculate additional statistics
    turns_list = [m.total_turns for m in metrics]
    decisions_list = [m.total_decisions for m in metrics]

    all_dpt = []  # decisions per turn
    for m in metrics:
        all_dpt.extend(m.decisions_per_turn)

    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0
    avg_decisions = sum(decisions_list) / len(decisions_list) if decisions_list else 0
    avg_dpt = sum(all_dpt) / len(all_dpt) if all_dpt else 0
    max_dpt = max(all_dpt) if all_dpt else 0
    min_dpt = min(all_dpt) if all_dpt else 0

    # Winners breakdown
    winners = {}
    for m in metrics:
        w = m.winner or "Unknown"
        winners[w] = winners.get(w, 0) + 1

    # Decision type breakdown
    type_counts = {}
    for m in metrics:
        for dt, count in m.decision_type_counts.items():
            type_counts[dt] = type_counts.get(dt, 0) + count

    # Create LaTeX content
    latex = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}

\geometry{margin=1in}
\title{""" + experiment_name + r""" Report}
\author{ForgeRL Training System}
\date{""" + datetime.now().strftime("%Y-%m-%d %H:%M") + r"""}

\begin{document}
\maketitle

\section{Executive Summary}

This report summarizes the game complexity metrics from training runs with the
heuristic agent. The goal is to track progress toward more human-like game lengths
and decision complexity.

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Games & """ + str(len(metrics)) + r""" \\
Total Decisions & """ + f"{stats.get('total_decisions', 0):,}" + r""" \\
Total Turns & """ + f"{stats.get('total_turns', 0):,}" + r""" \\
\midrule
Avg Turns/Game & """ + f"{avg_turns:.1f}" + r""" \\
Avg Decisions/Game & """ + f"{avg_decisions:.1f}" + r""" \\
Avg Decisions/Turn & """ + f"{avg_dpt:.1f}" + r""" \\
Max Decisions/Turn & """ + str(max_dpt) + r""" \\
\bottomrule
\end{tabular}
\caption{Game Complexity Metrics}
\end{table}

\section{Target Benchmarks}

Human-level game complexity benchmarks we're aiming for:

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Target} & \textbf{Current} \\
\midrule
Turns per Game & 8-15 & """ + f"{avg_turns:.1f}" + r""" \\
Decisions per Turn & 5-20 & """ + f"{avg_dpt:.1f}" + r""" \\
Decisions per Game & 50-200 & """ + f"{avg_decisions:.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Target vs Current Performance}
\end{table}

\section{Game Results}

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Winner} & \textbf{Count} \\
\midrule
"""
    for w, count in sorted(winners.items(), key=lambda x: -x[1]):
        latex += f"{w} & {count} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\caption{Win Distribution}
\end{table}

\section{Decision Type Breakdown}

\begin{table}[h]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Decision Type} & \textbf{Count} & \textbf{Percentage} \\
\midrule
"""
    total_decisions = sum(type_counts.values())
    for dt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total_decisions * 100 if total_decisions > 0 else 0
        latex += f"{dt.replace('_', '\\_')} & {count:,} & {pct:.1f}\\% \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\caption{Decision Types}
\end{table}

\section{Individual Game Details}

\begin{table}[h]
\centering
\begin{tabular}{rrrrl}
\toprule
\textbf{Game} & \textbf{Turns} & \textbf{Decisions} & \textbf{Decisions/Turn} & \textbf{Winner} \\
\midrule
"""
    for m in metrics[:20]:  # First 20 games
        dpt = m.total_decisions / m.total_turns if m.total_turns > 0 else 0
        winner = m.winner.replace('_', '\\_') if m.winner else "Unknown"
        latex += f"{m.game_id} & {m.total_turns} & {m.total_decisions} & {dpt:.1f} & {winner} \\\\\n"

    if len(metrics) > 20:
        latex += f"\\multicolumn{{5}}{{c}}{{... and {len(metrics) - 20} more games}} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\caption{Per-Game Statistics}
\end{table}

\section{Progress Tracking}

\textbf{Key Observations:}
\begin{itemize}
"""
    # Add observations based on metrics
    if avg_turns > 0 and avg_turns < 5:
        latex += r"\item Games are completing very quickly (< 5 turns). May indicate aggressive play." + "\n"
    elif avg_turns > 20:
        latex += r"\item Games are running long (> 20 turns). May indicate stalled positions." + "\n"
    else:
        latex += r"\item Game length is within reasonable range." + "\n"

    if avg_dpt > 50:
        latex += r"\item \textcolor{red}{High decisions per turn (> 50). Agent may be stuck in loops.}" + "\n"
    elif avg_dpt > 20:
        latex += r"\item Decisions per turn is above target. Room for optimization." + "\n"
    else:
        latex += r"\item Decisions per turn is within acceptable range." + "\n"

    if max_dpt > 100:
        latex += r"\item \textcolor{red}{WARNING: Max decisions/turn = " + str(max_dpt) + r". Likely loop detected.}" + "\n"

    latex += r"""
\end{itemize}

\section{Next Steps}

\begin{enumerate}
\item Add Forge AI suggestion support to daemon for better imitation targets
\item Implement confidence network to detect when agent is stuck
\item Add reward shaping to incentivize efficient play
\item Track metrics over time to measure improvement
\end{enumerate}

\end{document}
"""

    # Write LaTeX file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tex_path = output_path / "training_report.tex"
    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"LaTeX report saved to: {tex_path}")

    # Try to compile to PDF if pdflatex is available
    try:
        import subprocess
        result = subprocess.run(
            ["pdflatex", "-output-directory", str(output_path), str(tex_path)],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            pdf_path = output_path / "training_report.pdf"
            print(f"PDF report saved to: {pdf_path}")
    except Exception as e:
        print(f"Note: Could not compile PDF ({e}). LaTeX file is available.")

    return str(tex_path)


def save_summary(stats: dict, metrics: list, output_dir: str):
    """Save a plain text summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("IMITATION TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("GAME METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Games: {stats.get('total_games', 0)}\n")
        f.write(f"Total Decisions: {stats.get('total_decisions', 0):,}\n")
        f.write(f"Total Turns: {stats.get('total_turns', 0):,}\n")
        f.write(f"Avg Turns/Game: {stats.get('avg_turns_per_game', 0):.1f}\n")
        f.write(f"Avg Decisions/Game: {stats.get('avg_decisions_per_game', 0):.1f}\n")
        f.write(f"Avg Decisions/Turn: {stats.get('avg_decisions_per_turn', 0):.1f}\n")
        f.write(f"Max Decisions/Turn: {stats.get('max_decisions_per_turn', 0)}\n\n")

        f.write("TARGET BENCHMARKS\n")
        f.write("-" * 40 + "\n")
        f.write("Turns per Game:     8-15  (current: {:.1f})\n".format(
            stats.get('avg_turns_per_game', 0)))
        f.write("Decisions per Turn: 5-20  (current: {:.1f})\n".format(
            stats.get('avg_decisions_per_turn', 0)))
        f.write("Decisions per Game: 50-200 (current: {:.1f})\n\n".format(
            stats.get('avg_decisions_per_game', 0)))

        f.write("DECISION TYPE BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for dt, count in stats.get('decision_type_breakdown', {}).items():
            f.write(f"  {dt}: {count}\n")

    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run imitation training and generate report")
    parser.add_argument("--host", default="localhost", help="Forge daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Forge daemon port")
    parser.add_argument("--deck1", default="red_aggro.dck", help="First deck")
    parser.add_argument("--deck2", default="white_weenie.dck", help="Second deck")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument("--save-samples", action="store_true", help="Save training samples")
    args = parser.parse_args()

    print("=" * 60)
    print("IMITATION TRAINING DATA COLLECTION")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Decks: {args.deck1} vs {args.deck2}")
    print(f"Games: {args.games}")
    print(f"Output: {args.output}")
    print()

    # Collect data
    collector = ImitationDataCollector(
        host=args.host,
        port=args.port,
        deck1=args.deck1,
        deck2=args.deck2,
    )

    print("Collecting games...")
    samples, metrics = collector.collect_games(args.games, verbose=True)

    # Get statistics
    stats = collector.get_statistics()

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total games: {stats['total_games']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Avg turns/game: {stats['avg_turns_per_game']:.1f}")
    print(f"Avg decisions/game: {stats['avg_decisions_per_game']:.1f}")
    print(f"Avg decisions/turn: {stats['avg_decisions_per_turn']:.1f}")
    print(f"Max decisions/turn: {stats['max_decisions_per_turn']}")

    # Generate reports
    print("\nGenerating reports...")
    save_summary(stats, metrics, args.output)
    generate_latex_report(metrics, stats, args.output, "Forge Imitation Training")

    # Save samples if requested
    if args.save_samples and samples:
        samples_path = Path(args.output) / "samples.json"
        sample_dicts = []
        for s in samples[:1000]:  # First 1000 samples
            sample_dicts.append({
                "game_id": s.game_id,
                "turn": s.turn,
                "phase": s.phase,
                "decision_type": s.decision_type,
                "action_response": s.action_response,
                "action_description": s.action_description,
                "num_legal_actions": s.num_legal_actions,
                "decision_reason": s.decision_reason,
            })
        with open(samples_path, "w") as f:
            json.dump(sample_dicts, f, indent=2)
        print(f"Samples saved to: {samples_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
