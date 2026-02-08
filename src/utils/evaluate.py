#!/usr/bin/env python3
"""
MTG RL Evaluation Framework

Benchmarks trained agents against Forge's built-in AI at various difficulty levels.
Generates detailed statistics and reports.

NOTE: This module uses the old RL environment which is currently not maintained.
For Forge daemon evaluation, use scripts/evaluate_vs_forge.py instead.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import statistics


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    won: bool
    game_length: int  # Number of turns
    decisions_made: int
    our_final_life: int
    opponent_final_life: int
    duration_seconds: float
    deck_matchup: str  # "deck1 vs deck2"
    opponent_type: str  # "AI", "Random", "Self"


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    total_games: int = 0
    wins: int = 0
    losses: int = 0

    avg_game_length: float = 0.0
    avg_decisions: float = 0.0
    avg_duration: float = 0.0

    avg_final_life_diff: float = 0.0

    win_rate: float = 0.0
    win_rate_std: float = 0.0

    games_by_length: Dict[str, int] = field(default_factory=dict)

    def update(self, results: List[GameResult]):
        """Update metrics from game results."""
        if not results:
            return

        self.total_games = len(results)
        self.wins = sum(1 for r in results if r.won)
        self.losses = self.total_games - self.wins

        self.win_rate = self.wins / self.total_games if self.total_games > 0 else 0.0

        # Calculate win rate standard error using binomial proportion
        if self.total_games > 1:
            p = self.win_rate
            self.win_rate_std = np.sqrt(p * (1 - p) / self.total_games)

        # Averages
        self.avg_game_length = statistics.mean(r.game_length for r in results)
        self.avg_decisions = statistics.mean(r.decisions_made for r in results)
        self.avg_duration = statistics.mean(r.duration_seconds for r in results)

        # Life differential
        life_diffs = [r.our_final_life - r.opponent_final_life for r in results]
        self.avg_final_life_diff = statistics.mean(life_diffs)

        # Games by length buckets
        for r in results:
            if r.game_length <= 5:
                bucket = "1-5"
            elif r.game_length <= 10:
                bucket = "6-10"
            elif r.game_length <= 15:
                bucket = "11-15"
            else:
                bucket = "16+"
            self.games_by_length[bucket] = self.games_by_length.get(bucket, 0) + 1


if __name__ == "__main__":
    print("This module is deprecated. Use scripts/evaluate_vs_forge.py for evaluation.")
