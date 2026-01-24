#!/usr/bin/env python3
"""
MTG RL Evaluation Framework

Benchmarks trained agents against Forge's built-in AI at various difficulty levels.
Generates detailed statistics and reports.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
import statistics

import torch

from rl_environment import MTGEnvironment, GameState
from policy_network import TransformerConfig
from ppo_agent import PPOAgent, PPOConfig


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


class AgentPlayer:
    """Wrapper for the RL agent to play in evaluation."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load agent
        self.agent = self._load_agent(checkpoint_path)
        self.agent.policy.eval()

    def _load_agent(self, path: str) -> PPOAgent:
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Reconstruct configs
        ppo_config = checkpoint.get('ppo_config', PPOConfig())
        transformer_config = checkpoint.get('transformer_config', TransformerConfig())

        # Create agent
        agent = PPOAgent(ppo_config, transformer_config, self.device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])

        return agent

    def select_action(
        self,
        game_state: GameState,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select action given game state."""
        action, _ = self.agent.get_action(game_state, action_mask, deterministic=deterministic)
        return action


class RandomPlayer:
    """Random action selection baseline."""

    def select_action(
        self,
        game_state: GameState,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select random valid action."""
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return -1  # Pass
        return int(np.random.choice(valid_actions))


class Evaluator:
    """Main evaluation class."""

    def __init__(
        self,
        agent_path: str = None,
        output_dir: str = "eval_results",
        docker_image: str = "forge-sim:latest",
        game_timeout: int = 120,
        device: torch.device = None
    ):
        self.output_dir = output_dir
        self.docker_image = docker_image
        self.game_timeout = game_timeout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        # Load agent if provided
        self.agent = None
        if agent_path and os.path.exists(agent_path):
            print(f"Loading agent from {agent_path}")
            self.agent = AgentPlayer(agent_path, self.device)

        self.random_player = RandomPlayer()

    def run_game(
        self,
        deck1: str,
        deck2: str,
        player: str = "agent",  # "agent", "random"
        opponent: str = "ai",   # "ai", "random", "agent"
    ) -> GameResult:
        """Run a single game and return result."""

        # Create environment
        env = MTGEnvironment(
            deck1=deck1,
            deck2=deck2,
            docker_image=self.docker_image,
            player_id=1,
            timeout=self.game_timeout,
            reward_shaping=False
        )

        start_time = time.time()
        decisions = 0

        try:
            obs, info = env.reset()

            while not env.game_over:
                action_mask = env.get_action_mask()

                # Select action based on player type
                if player == "agent" and self.agent:
                    if env.current_state:
                        action = self.agent.select_action(
                            env.current_state, action_mask, deterministic=True
                        )
                    else:
                        action = -1
                elif player == "random":
                    action = self.random_player.select_action(
                        env.current_state, action_mask
                    )
                else:
                    action = -1  # Pass (fallback)

                obs, reward, done, truncated, info = env.step(action)
                decisions += 1

                # Safety limit
                if decisions > 1000:
                    break

            duration = time.time() - start_time

            # Get final state info
            our_life = 0
            opp_life = 0
            game_length = info.get('turn', 0) if isinstance(info, dict) else 0

            if env.current_state:
                our_life = env.current_state.our_player.life
                opp_life = env.current_state.opponent.life
                game_length = env.current_state.turn

            return GameResult(
                game_id=0,
                won=env.won,
                game_length=game_length,
                decisions_made=decisions,
                our_final_life=our_life,
                opponent_final_life=opp_life,
                duration_seconds=duration,
                deck_matchup=f"{deck1} vs {deck2}",
                opponent_type=opponent
            )

        finally:
            env._cleanup()

    def evaluate(
        self,
        n_games: int = 100,
        deck1: str = "red_aggro.dck",
        deck2: str = "white_weenie.dck",
        player: str = "agent",
        opponent: str = "ai",
        verbose: bool = True
    ) -> EvalMetrics:
        """
        Run evaluation over multiple games.

        Args:
            n_games: Number of games to play
            deck1: First deck (agent's deck)
            deck2: Second deck (opponent's deck)
            player: "agent" or "random"
            opponent: "ai" or "random"
            verbose: Print progress

        Returns:
            Aggregated evaluation metrics
        """
        results = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation: {player} vs {opponent}")
            print(f"Decks: {deck1} vs {deck2}")
            print(f"Games: {n_games}")
            print(f"{'='*60}\n")

        for i in range(n_games):
            if verbose and (i + 1) % 10 == 0:
                current_wins = sum(1 for r in results if r.won)
                current_wr = current_wins / len(results) if results else 0
                print(f"  Game {i+1}/{n_games} - Current WR: {current_wr*100:.1f}%")

            try:
                result = self.run_game(deck1, deck2, player, opponent)
                result.game_id = i + 1
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Game {i+1} failed: {e}")

        # Calculate metrics
        metrics = EvalMetrics()
        metrics.update(results)

        if verbose:
            self._print_summary(metrics)

        # Save results
        self._save_results(results, metrics, deck1, deck2, player, opponent)

        return metrics

    def benchmark_vs_baselines(
        self,
        n_games_per_matchup: int = 50,
        decks: List[Tuple[str, str]] = None
    ) -> Dict[str, EvalMetrics]:
        """
        Run comprehensive benchmark against different baselines.

        Returns dict mapping matchup name to metrics.
        """
        if decks is None:
            decks = [
                ("red_aggro.dck", "white_weenie.dck"),
                ("white_weenie.dck", "red_aggro.dck"),
            ]

        all_results = {}

        print("\n" + "="*60)
        print("COMPREHENSIVE BENCHMARK")
        print("="*60)

        for deck1, deck2 in decks:
            # Agent vs AI
            if self.agent:
                name = f"Agent vs AI ({deck1} vs {deck2})"
                print(f"\n{name}")
                metrics = self.evaluate(
                    n_games=n_games_per_matchup,
                    deck1=deck1, deck2=deck2,
                    player="agent", opponent="ai",
                    verbose=False
                )
                all_results[name] = metrics
                print(f"  Win Rate: {metrics.win_rate*100:.1f}% +/- {metrics.win_rate_std*100:.1f}%")

            # Random vs AI (baseline)
            name = f"Random vs AI ({deck1} vs {deck2})"
            print(f"\n{name}")
            metrics = self.evaluate(
                n_games=n_games_per_matchup,
                deck1=deck1, deck2=deck2,
                player="random", opponent="ai",
                verbose=False
            )
            all_results[name] = metrics
            print(f"  Win Rate: {metrics.win_rate*100:.1f}% +/- {metrics.win_rate_std*100:.1f}%")

        # Print comparison table
        self._print_comparison_table(all_results)

        return all_results

    def _print_summary(self, metrics: EvalMetrics):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Games: {metrics.total_games}")
        print(f"Wins: {metrics.wins} | Losses: {metrics.losses}")
        print(f"\nWin Rate: {metrics.win_rate*100:.1f}% (+/- {metrics.win_rate_std*100:.1f}%)")
        print(f"\nAvg Game Length: {metrics.avg_game_length:.1f} turns")
        print(f"Avg Decisions: {metrics.avg_decisions:.1f}")
        print(f"Avg Game Duration: {metrics.avg_duration:.1f}s")
        print(f"Avg Life Differential: {metrics.avg_final_life_diff:+.1f}")

        if metrics.games_by_length:
            print("\nGames by Length:")
            for bucket, count in sorted(metrics.games_by_length.items()):
                pct = count / metrics.total_games * 100
                print(f"  {bucket} turns: {count} ({pct:.1f}%)")

        print(f"{'='*60}\n")

    def _print_comparison_table(self, all_results: Dict[str, EvalMetrics]):
        """Print comparison table of all matchups."""
        print(f"\n{'='*80}")
        print("BENCHMARK COMPARISON")
        print(f"{'='*80}")
        print(f"{'Matchup':<45} {'Win Rate':>12} {'Avg Turns':>12} {'Avg Life':>12}")
        print("-"*80)

        for name, metrics in all_results.items():
            wr = f"{metrics.win_rate*100:.1f}%"
            turns = f"{metrics.avg_game_length:.1f}"
            life = f"{metrics.avg_final_life_diff:+.1f}"
            print(f"{name:<45} {wr:>12} {turns:>12} {life:>12}")

        print(f"{'='*80}\n")

    def _save_results(
        self,
        results: List[GameResult],
        metrics: EvalMetrics,
        deck1: str,
        deck2: str,
        player: str,
        opponent: str
    ):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{player}_vs_{opponent}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        data = {
            "config": {
                "deck1": deck1,
                "deck2": deck2,
                "player": player,
                "opponent": opponent,
                "timestamp": timestamp,
            },
            "metrics": {
                "total_games": metrics.total_games,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "win_rate": metrics.win_rate,
                "win_rate_std": metrics.win_rate_std,
                "avg_game_length": metrics.avg_game_length,
                "avg_decisions": metrics.avg_decisions,
                "avg_duration": metrics.avg_duration,
                "avg_final_life_diff": metrics.avg_final_life_diff,
            },
            "games": [asdict(r) for r in results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filepath}")


def test_evaluation_mock():
    """Test evaluation with mock games."""
    print("Testing Evaluation Framework (Mock)")
    print("="*60)

    # Create mock results
    results = []
    for i in range(20):
        results.append(GameResult(
            game_id=i+1,
            won=np.random.random() > 0.4,  # ~60% win rate
            game_length=np.random.randint(5, 20),
            decisions_made=np.random.randint(20, 100),
            our_final_life=np.random.randint(0, 20),
            opponent_final_life=np.random.randint(0, 20),
            duration_seconds=np.random.uniform(10, 60),
            deck_matchup="red_aggro.dck vs white_weenie.dck",
            opponent_type="ai"
        ))

    # Calculate metrics
    metrics = EvalMetrics()
    metrics.update(results)

    print("\nMock Evaluation Results:")
    print(f"  Total Games: {metrics.total_games}")
    print(f"  Win Rate: {metrics.win_rate*100:.1f}% (+/- {metrics.win_rate_std*100:.1f}%)")
    print(f"  Avg Game Length: {metrics.avg_game_length:.1f} turns")
    print(f"  Avg Life Diff: {metrics.avg_final_life_diff:+.1f}")

    print("\nMock evaluation test passed!")


def evaluate_random_baseline():
    """Quick evaluation of random player baseline."""
    print("\nEvaluating Random Baseline")
    print("="*60)

    evaluator = Evaluator(agent_path=None)

    metrics = evaluator.evaluate(
        n_games=10,
        player="random",
        opponent="ai",
        verbose=True
    )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MTG RL Evaluation")
    parser.add_argument("--mode", choices=["test", "random", "agent", "benchmark"],
                       default="test", help="Evaluation mode")
    parser.add_argument("--agent", type=str, default=None,
                       help="Path to agent checkpoint")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of games to play")
    parser.add_argument("--deck1", type=str, default="red_aggro.dck",
                       help="Agent's deck")
    parser.add_argument("--deck2", type=str, default="white_weenie.dck",
                       help="Opponent's deck")
    parser.add_argument("--output", type=str, default="eval_results",
                       help="Output directory")

    args = parser.parse_args()

    if args.mode == "test":
        test_evaluation_mock()

    elif args.mode == "random":
        # Evaluate random baseline
        evaluator = Evaluator(output_dir=args.output)
        evaluator.evaluate(
            n_games=args.games,
            deck1=args.deck1,
            deck2=args.deck2,
            player="random",
            opponent="ai"
        )

    elif args.mode == "agent":
        if not args.agent:
            print("Error: --agent path required for agent evaluation")
            exit(1)

        evaluator = Evaluator(agent_path=args.agent, output_dir=args.output)
        evaluator.evaluate(
            n_games=args.games,
            deck1=args.deck1,
            deck2=args.deck2,
            player="agent",
            opponent="ai"
        )

    elif args.mode == "benchmark":
        evaluator = Evaluator(agent_path=args.agent, output_dir=args.output)
        evaluator.benchmark_vs_baselines(n_games_per_matchup=args.games // 2)
