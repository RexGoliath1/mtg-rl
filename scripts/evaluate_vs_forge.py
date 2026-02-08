#!/usr/bin/env python3
"""
Evaluate Trained Agent vs Forge AI

Loads an imitation learning checkpoint and plays games against Forge daemon.
Tracks win rate, game length, decisions made, and other metrics.

Usage:
    python3 scripts/evaluate_vs_forge.py --checkpoint checkpoints/imitation_best.pt --games 10
    python3 scripts/evaluate_vs_forge.py --checkpoint checkpoints/imitation_best.pt --games 50 --deck1 decks/red_aggro.dck --deck2 decks/white_weenie.dck
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch

from src.forge.forge_client import ForgeClient
from src.forge.state_mapper import ForgeNetworkAgent, StateMapper
from src.training.self_play import AlphaZeroNetwork


@dataclass
class GameStats:
    """Statistics for a single game."""
    game_id: int
    won: bool
    total_decisions: int
    total_turns: int
    duration_seconds: float
    our_final_life: int
    opponent_final_life: int
    timeout: bool = False
    error: bool = False
    error_msg: str = ""


class ForgeEvaluator:
    """Evaluates a trained network against Forge AI."""

    def __init__(
        self,
        network: AlphaZeroNetwork,
        host: str = "localhost",
        port: int = 17171,
        timeout: int = 120,
        our_player_name: str = "Player0",
    ):
        self.network = network
        self.host = host
        self.port = port
        self.timeout = timeout
        self.our_player_name = our_player_name

        # Create agent
        self.agent = ForgeNetworkAgent(network, mapper=StateMapper(), temperature=0.5)

        self.stats: list[GameStats] = []

    def play_game(
        self,
        game_id: int,
        deck1: str,
        deck2: str,
        seed: Optional[int] = None,
    ) -> GameStats:
        """
        Play a single game against Forge AI.

        Args:
            game_id: Game number
            deck1: Path to our deck file
            deck2: Path to opponent deck file
            seed: Random seed for reproducibility

        Returns:
            GameStats for this game
        """
        stats = GameStats(
            game_id=game_id,
            won=False,
            total_decisions=0,
            total_turns=0,
            duration_seconds=0.0,
            our_final_life=0,
            opponent_final_life=0,
        )

        client = ForgeClient(self.host, self.port, timeout=self.timeout)

        try:
            client.connect()

            success = client.start_game(deck1, deck2, timeout=self.timeout, seed=seed)
            if not success:
                stats.error = True
                stats.error_msg = "Failed to start game"
                return stats

            game_start = time.perf_counter()
            current_turn = 0

            while True:
                decision = client.receive_decision()
                if decision is None:
                    break

                # Track turn changes
                if decision.turn > current_turn:
                    current_turn = decision.turn

                stats.total_decisions += 1

                # Make decision with neural network
                response = self.agent.decide_greedy(decision, self.our_player_name)

                # Send response
                client.send_response(response)

            stats.total_turns = current_turn
            stats.duration_seconds = time.perf_counter() - game_start

            # Get game result
            result = client.get_result()
            if result:
                stats.won = (result.winner == self.our_player_name)

                # Try to extract final life totals (if available in result)
                # This is optional - Forge may not provide this
                if hasattr(result, 'our_life'):
                    stats.our_final_life = result.our_life
                if hasattr(result, 'opponent_life'):
                    stats.opponent_final_life = result.opponent_life

            client.disconnect()

        except TimeoutError:
            stats.timeout = True
            stats.error = True
            stats.error_msg = "Game timeout"
        except Exception as e:
            stats.error = True
            stats.error_msg = str(e)
        finally:
            try:
                client.disconnect()
            except Exception:
                pass

        return stats

    def evaluate(
        self,
        num_games: int,
        deck1: str,
        deck2: str,
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate network over multiple games.

        Args:
            num_games: Number of games to play
            deck1: Path to our deck
            deck2: Path to opponent deck
            verbose: Print progress

        Returns:
            Dictionary of aggregate statistics
        """
        self.stats = []

        if verbose:
            print(f"\n{'='*60}")
            print("Evaluating network vs Forge AI")
            print(f"Decks: {deck1} vs {deck2}")
            print(f"Games: {num_games}")
            print(f"{'='*60}\n")

        for game_id in range(1, num_games + 1):
            if verbose:
                print(f"Game {game_id}/{num_games}...", end=" ", flush=True)

            stats = self.play_game(game_id, deck1, deck2, seed=game_id)
            self.stats.append(stats)

            if verbose:
                if stats.error:
                    print(f"ERROR ({stats.error_msg})")
                elif stats.timeout:
                    print("TIMEOUT")
                else:
                    result_str = "WIN" if stats.won else "LOSS"
                    print(f"{result_str} ({stats.total_turns} turns, {stats.total_decisions} decisions, {stats.duration_seconds:.1f}s)")

        # Compute aggregate stats
        return self._compute_summary(verbose)

    def _compute_summary(self, verbose: bool = True) -> dict:
        """Compute and print summary statistics."""
        total = len(self.stats)
        if total == 0:
            return {}

        completed = [s for s in self.stats if not s.error]
        wins = sum(1 for s in completed if s.won)
        losses = len(completed) - wins
        timeouts = sum(1 for s in self.stats if s.timeout)
        errors = sum(1 for s in self.stats if s.error and not s.timeout)

        win_rate = wins / len(completed) if completed else 0.0

        avg_turns = sum(s.total_turns for s in completed) / len(completed) if completed else 0.0
        avg_decisions = sum(s.total_decisions for s in completed) / len(completed) if completed else 0.0
        avg_duration = sum(s.duration_seconds for s in completed) / len(completed) if completed else 0.0

        summary = {
            "total_games": total,
            "completed": len(completed),
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "errors": errors,
            "win_rate": win_rate,
            "avg_turns": avg_turns,
            "avg_decisions": avg_decisions,
            "avg_duration": avg_duration,
        }

        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total Games:     {total}")
            print(f"Completed:       {len(completed)}")
            print(f"Wins:            {wins}")
            print(f"Losses:          {losses}")
            print(f"Timeouts:        {timeouts}")
            print(f"Errors:          {errors}")
            print(f"\nWin Rate:        {win_rate*100:.1f}%")
            print(f"Avg Turns:       {avg_turns:.1f}")
            print(f"Avg Decisions:   {avg_decisions:.0f}")
            print(f"Avg Duration:    {avg_duration:.1f}s")
            print(f"{'='*60}\n")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained network vs Forge AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to network checkpoint (.pt file)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Forge daemon host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=17171,
        help="Forge daemon port (default: 17171)",
    )
    parser.add_argument(
        "--deck1",
        type=str,
        default="decks/red_aggro.dck",
        help="Path to our deck (default: decks/red_aggro.dck)",
    )
    parser.add_argument(
        "--deck2",
        type=str,
        default="decks/white_weenie.dck",
        help="Path to opponent deck (default: decks/white_weenie.dck)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Game timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run network on (default: cuda if available, else cpu)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    # Validate decks exist
    deck1_path = Path(args.deck1)
    deck2_path = Path(args.deck2)
    if not deck1_path.exists():
        print(f"Error: Deck not found: {args.deck1}")
        return 1
    if not deck2_path.exists():
        print(f"Error: Deck not found: {args.deck2}")
        return 1

    # Load network
    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device(args.device)
    network = AlphaZeroNetwork.load(str(checkpoint_path), device=device)
    network.eval()
    print(f"Network loaded on {device}")

    # Create evaluator
    evaluator = ForgeEvaluator(
        network=network,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )

    # Run evaluation
    evaluator.evaluate(
        num_games=args.games,
        deck1=str(deck1_path.absolute()),
        deck2=str(deck2_path.absolute()),
        verbose=True,
    )

    return 0


if __name__ == "__main__":
    exit(main())
