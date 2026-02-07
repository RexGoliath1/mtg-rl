#!/usr/bin/env python3
"""
Analyze collected imitation learning training data and generate LaTeX report.
"""

import sys
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def analyze_hdf5_file(filepath: Path) -> dict:
    """Analyze a single HDF5 training data file."""
    with h5py.File(filepath, 'r') as f:
        states = f['states'][:]
        turns = f['turns'][:]
        choices = f['choices'][:]
        num_actions = f['num_actions'][:]
        decision_types = f['decision_types'][:]

        # Get metadata
        metadata = {k: f.attrs[k] for k in f.attrs.keys()}

    return {
        'states': states,
        'turns': turns,
        'choices': choices,
        'num_actions': num_actions,
        'decision_types': decision_types,
        'metadata': metadata
    }


def detect_games(turns: np.ndarray) -> list:
    """Detect game boundaries by turn resets."""
    game_boundaries = [0]
    for i in range(1, len(turns)):
        # New game starts when turn decreases (reset to 1)
        if turns[i] < turns[i-1]:
            game_boundaries.append(i)
    game_boundaries.append(len(turns))
    return game_boundaries


def analyze_games(data: dict) -> dict:
    """Perform comprehensive game-level analysis."""
    turns = data['turns']
    states = data['states']
    decision_types = data['decision_types']
    num_actions = data['num_actions']

    # Detect game boundaries
    boundaries = detect_games(turns)
    num_games = len(boundaries) - 1

    # Per-game statistics
    games_stats = []
    for i in range(num_games):
        start, end = boundaries[i], boundaries[i+1]
        game_turns = turns[start:end]
        game_states = states[start:end]
        game_dtypes = decision_types[start:end]
        game_actions = num_actions[start:end]

        # Extract life totals at game end (indices 0 and 7 in state vector)
        final_state = game_states[-1]
        p1_life_end = final_state[0]
        p2_life_end = final_state[7]

        # Determine winner based on life totals
        if p1_life_end <= 0 and p2_life_end > 0:
            winner = 2
        elif p2_life_end <= 0 and p1_life_end > 0:
            winner = 1
        elif p1_life_end > p2_life_end:
            winner = 1  # Player 1 ahead when game ended
        elif p2_life_end > p1_life_end:
            winner = 2
        else:
            winner = 0  # Draw or unknown

        # Count decision types in this game
        dtype_counts = defaultdict(int)
        for dt in game_dtypes:
            dtype_counts[int(dt)] += 1

        games_stats.append({
            'decisions': end - start,
            'max_turn': int(game_turns.max()),
            'avg_actions': float(game_actions.mean()),
            'winner': winner,
            'p1_life_end': float(p1_life_end),
            'p2_life_end': float(p2_life_end),
            'dtype_counts': dict(dtype_counts)
        })

    return {
        'num_games': num_games,
        'games': games_stats,
        'total_decisions': len(turns)
    }


def generate_latex_report(analysis: dict, output_path: Path):
    """Generate comprehensive LaTeX report."""
    games = analysis['games']
    num_games = analysis['num_games']
    total_decisions = analysis['total_decisions']

    # Compute statistics
    decisions_per_game = [g['decisions'] for g in games]
    turns_per_game = [g['max_turn'] for g in games]
    actions_per_game = [g['avg_actions'] for g in games]

    # Winner statistics
    p1_wins = sum(1 for g in games if g['winner'] == 1)
    p2_wins = sum(1 for g in games if g['winner'] == 2)
    draws = sum(1 for g in games if g['winner'] == 0)

    # Life total analysis
    p1_life_ends = [g['p1_life_end'] for g in games]
    p2_life_ends = [g['p2_life_end'] for g in games]

    # Decision type totals
    dtype_totals = defaultdict(int)
    for g in games:
        for dt, count in g['dtype_counts'].items():
            dtype_totals[dt] += count

    dtype_names = {0: "Choose Action", 1: "Declare Attackers", 2: "Declare Blockers", 3: "Unknown"}

    # Turn distribution bins
    turn_bins = [0, 5, 10, 15, 20, 30, 50, 100]
    turn_hist = np.histogram(turns_per_game, bins=turn_bins)[0]

    # Decision distribution bins
    dec_bins = [0, 100, 200, 300, 400, 500, 750, 1000, 2000]
    dec_hist = np.histogram(decisions_per_game, bins=dec_bins)[0]

    # Decisions per turn
    decisions_per_turn = np.array(decisions_per_game) / np.maximum(np.array(turns_per_game), 1)

    # Generate LaTeX
    latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{pgfplots}
\usepackage{float}
\pgfplotsset{compat=1.18}
\geometry{margin=1in}

\title{MTG Imitation Learning Data Collection Report}
\author{ForgeRL System}
\date{""" + datetime.now().strftime("%B %d, %Y") + r"""}

\begin{document}
\maketitle

\begin{abstract}
This report summarizes the data collected from """ + f"{num_games:,}" + r""" Magic: The Gathering games
played by the Forge AI in observation mode. The dataset contains """ + f"{total_decisions:,}" + r"""
decision points suitable for imitation learning. Games were played using Modern format decks
to capture diverse strategic patterns across competitive archetypes.
\end{abstract}

\section{Executive Summary}

\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Games & """ + f"{num_games:,}" + r""" \\
Total Decisions & """ + f"{total_decisions:,}" + r""" \\
Average Decisions per Game & """ + f"{np.mean(decisions_per_game):.1f}" + r""" \\
Average Turns per Game & """ + f"{np.mean(turns_per_game):.1f}" + r""" \\
Average Decisions per Turn & """ + f"{np.mean(decisions_per_turn):.1f}" + r""" \\
Average Actions Available & """ + f"{np.mean(actions_per_game):.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Collection Summary Statistics}
\end{table}

\section{Game Outcomes}

\subsection{Win Distribution}

\begin{table}[H]
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Outcome} & \textbf{Count} & \textbf{Percentage} & \textbf{Avg Turns} \\
\midrule
Player 1 Wins & """ + f"{p1_wins:,}" + r""" & """ + f"{100*p1_wins/num_games:.1f}" + r"""\% & """ + f"{np.mean([g['max_turn'] for g in games if g['winner']==1]):.1f}" + r""" \\
Player 2 Wins & """ + f"{p2_wins:,}" + r""" & """ + f"{100*p2_wins/num_games:.1f}" + r"""\% & """ + f"{np.mean([g['max_turn'] for g in games if g['winner']==2]):.1f}" + r""" \\
Draw/Unknown & """ + f"{draws:,}" + r""" & """ + f"{100*draws/num_games:.1f}" + r"""\% & """ + f"{np.mean([g['max_turn'] for g in games if g['winner']==0]) if draws > 0 else 0:.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Game Outcome Distribution}
\end{table}

\subsection{Life Total Analysis}

\begin{table}[H]
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{Player} & \textbf{Avg Final Life} & \textbf{Min} & \textbf{Max} & \textbf{Std Dev} \\
\midrule
Player 1 & """ + f"{np.mean(p1_life_ends):.1f}" + r""" & """ + f"{np.min(p1_life_ends):.0f}" + r""" & """ + f"{np.max(p1_life_ends):.0f}" + r""" & """ + f"{np.std(p1_life_ends):.1f}" + r""" \\
Player 2 & """ + f"{np.mean(p2_life_ends):.1f}" + r""" & """ + f"{np.min(p2_life_ends):.0f}" + r""" & """ + f"{np.max(p2_life_ends):.0f}" + r""" & """ + f"{np.std(p2_life_ends):.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Life Total Statistics at Game End}
\end{table}

\section{Game Length Analysis}

\subsection{Turn Distribution}

\begin{table}[H]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Turn Range} & \textbf{Games} & \textbf{Percentage} \\
\midrule
"""
    for i in range(len(turn_bins)-1):
        pct = 100 * turn_hist[i] / num_games
        latex += f"{turn_bins[i]+1}--{turn_bins[i+1]} turns & {turn_hist[i]:,} & {pct:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Distribution of Game Length by Turns}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Minimum Turns & """ + f"{np.min(turns_per_game)}" + r""" \\
Maximum Turns & """ + f"{np.max(turns_per_game)}" + r""" \\
Median Turns & """ + f"{np.median(turns_per_game):.0f}" + r""" \\
25th Percentile & """ + f"{np.percentile(turns_per_game, 25):.0f}" + r""" \\
75th Percentile & """ + f"{np.percentile(turns_per_game, 75):.0f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Turn Count Statistics}
\end{table}

\section{Decision Analysis}

\subsection{Decisions per Game}

\begin{table}[H]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Decision Range} & \textbf{Games} & \textbf{Percentage} \\
\midrule
"""
    for i in range(len(dec_bins)-1):
        pct = 100 * dec_hist[i] / num_games
        latex += f"{dec_bins[i]+1}--{dec_bins[i+1]} & {dec_hist[i]:,} & {pct:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Distribution of Decisions per Game}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Minimum Decisions & """ + f"{np.min(decisions_per_game)}" + r""" \\
Maximum Decisions & """ + f"{np.max(decisions_per_game)}" + r""" \\
Median Decisions & """ + f"{np.median(decisions_per_game):.0f}" + r""" \\
Standard Deviation & """ + f"{np.std(decisions_per_game):.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Decisions per Game Statistics}
\end{table}

\subsection{Decisions per Turn}

\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Average Decisions per Turn & """ + f"{np.mean(decisions_per_turn):.1f}" + r""" \\
Minimum & """ + f"{np.min(decisions_per_turn):.1f}" + r""" \\
Maximum & """ + f"{np.max(decisions_per_turn):.1f}" + r""" \\
Median & """ + f"{np.median(decisions_per_turn):.1f}" + r""" \\
Standard Deviation & """ + f"{np.std(decisions_per_turn):.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Decisions per Turn Statistics}
\end{table}

\subsection{Decision Types}

\begin{table}[H]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Decision Type} & \textbf{Count} & \textbf{Percentage} \\
\midrule
"""
    for dt in sorted(dtype_totals.keys()):
        name = dtype_names.get(dt, f"Type {dt}")
        count = dtype_totals[dt]
        pct = 100 * count / total_decisions
        latex += f"{name} & {count:,} & {pct:.1f}\\% \\\\\n"

    latex += r"""\midrule
\textbf{Total} & """ + f"{total_decisions:,}" + r""" & 100.0\% \\
\bottomrule
\end{tabular}
\caption{Decision Type Distribution}
\end{table}

\section{Actions Available Analysis}

\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Average Actions Available & """ + f"{np.mean(actions_per_game):.2f}" + r""" \\
Minimum (avg per game) & """ + f"{np.min(actions_per_game):.2f}" + r""" \\
Maximum (avg per game) & """ + f"{np.max(actions_per_game):.2f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Actions Available per Decision}
\end{table}

\section{Data Quality Assessment}

\begin{itemize}
\item \textbf{Coverage}: Games span """ + f"{np.min(turns_per_game)}" + r""" to """ + f"{np.max(turns_per_game)}" + r""" turns, covering early, mid, and late game scenarios.
\item \textbf{Decision Diversity}: """ + f"{len(dtype_totals)}" + r""" distinct decision types captured.
\item \textbf{Action Space}: Average of """ + f"{np.mean(actions_per_game):.1f}" + r""" legal actions per decision point.
\item \textbf{Balance}: Player 1 win rate """ + f"{100*p1_wins/(p1_wins+p2_wins):.1f}" + r"""\% vs Player 2 """ + f"{100*p2_wins/(p1_wins+p2_wins):.1f}" + r"""\% (slight first-player advantage expected in MTG).
\end{itemize}

\section{Training Recommendations}

Based on the collected data:

\begin{enumerate}
\item \textbf{Dataset Size}: """ + f"{total_decisions:,}" + r""" decisions provides sufficient data for initial imitation learning.
\item \textbf{State Encoding}: 17-dimensional state vectors capture essential game state features.
\item \textbf{Action Distribution}: With average """ + f"{np.mean(actions_per_game):.1f}" + r""" actions per decision, the action space is tractable.
\item \textbf{Recommended Architecture}: Transformer or MLP with softmax output over action space.
\item \textbf{Expected Accuracy}: Target $>$60\% top-1 accuracy, $>$85\% top-3 accuracy for successful imitation.
\end{enumerate}

\section{Appendix: Technical Details}

\subsection{State Vector Encoding}

The 17-dimensional state vector contains:

\begin{table}[H]
\centering
\begin{tabular}{clc}
\toprule
\textbf{Index} & \textbf{Feature} & \textbf{Range} \\
\midrule
0 & Player 1 Life Total & 0--20+ \\
1 & Player 1 Hand Size & 0--7+ \\
2 & Player 1 Library Size & 0--60 \\
3 & Player 1 Creatures & 0--10+ \\
4 & Player 1 Lands & 0--10+ \\
5 & Player 1 Other Permanents & 0--10+ \\
6 & Player 1 Available Mana & 0--10+ \\
7--13 & Player 2 (same features) & -- \\
14 & Current Turn Number & 1--50+ \\
15 & Current Phase Index & 0--12 \\
16 & Game Over Flag & 0 or 1 \\
\bottomrule
\end{tabular}
\caption{State Vector Feature Encoding}
\end{table}

\end{document}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training data and generate report")
    parser.add_argument("--data-dir", default="training_data/imitation_aws", help="Data directory")
    parser.add_argument("--output", default="reports/imitation_data_report.tex", help="Output LaTeX file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find the largest checkpoint (most complete data)
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        print(f"No HDF5 files found in {data_dir}")
        sys.exit(1)

    # Use the largest file (most data)
    largest_file = max(h5_files, key=lambda f: f.stat().st_size)
    print(f"Analyzing: {largest_file}")

    # Load and analyze
    data = analyze_hdf5_file(largest_file)
    print(f"Loaded {len(data['states']):,} decisions")

    analysis = analyze_games(data)
    print(f"Detected {analysis['num_games']:,} games")

    # Generate report
    report_path = generate_latex_report(analysis, output_path)
    print(f"\nLaTeX report saved to: {report_path}")

    # Print summary
    games = analysis['games']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total Games: {analysis['num_games']:,}")
    print(f"Total Decisions: {analysis['total_decisions']:,}")
    print(f"Avg Decisions/Game: {np.mean([g['decisions'] for g in games]):.1f}")
    print(f"Avg Turns/Game: {np.mean([g['max_turn'] for g in games]):.1f}")
    print(f"Avg Decisions/Turn: {np.mean([g['decisions']/g['max_turn'] for g in games]):.1f}")

    p1_wins = sum(1 for g in games if g['winner'] == 1)
    p2_wins = sum(1 for g in games if g['winner'] == 2)
    print(f"\nWin Rate: P1 {100*p1_wins/(p1_wins+p2_wins):.1f}% | P2 {100*p2_wins/(p1_wins+p2_wins):.1f}%")


if __name__ == "__main__":
    main()
