#!/usr/bin/env python3
"""
Training Dashboard for MTG RL Training

Uses Weights & Biases (wandb) for experiment tracking and visualization.
Install: pip install wandb

Usage:
    # Initialize once
    wandb login

    # Run training with dashboard
    python training_dashboard.py --games 10000 --parallel 10
"""

import argparse
import socket
import time
import statistics
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Try to import wandb, fall back to simple logging if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run: pip install wandb")
    print("Falling back to simple console logging.")


@dataclass
class TrainingConfig:
    """Training configuration."""
    project_name: str = "mtg-rl-training"
    run_name: Optional[str] = None
    total_games: int = 10000
    max_parallel: int = 10
    daemon_host: str = "localhost"
    daemon_port: int = 17171
    checkpoint_interval: int = 1000  # Games between checkpoints
    log_interval: int = 100  # Games between logs
    deck1: str = "test_red.dck"
    deck2: str = "test_blue.dck"


@dataclass
class TrainingMetrics:
    """Accumulated training metrics."""
    games_played: int = 0
    games_successful: int = 0
    games_failed: int = 0
    total_duration_ms: float = 0.0
    wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    durations: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_result(self, success: bool, duration_ms: float, winner: Optional[str], error: Optional[str]):
        self.games_played += 1
        if success:
            self.games_successful += 1
            self.total_duration_ms += duration_ms
            self.durations.append(duration_ms)
            if winner:
                self.wins[winner] += 1
        else:
            self.games_failed += 1
            if error:
                self.errors.append(error)

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        games_per_sec = self.games_played / elapsed if elapsed > 0 else 0

        stats = {
            "games_played": self.games_played,
            "games_successful": self.games_successful,
            "games_failed": self.games_failed,
            "success_rate": self.games_successful / max(1, self.games_played),
            "elapsed_seconds": elapsed,
            "games_per_second": games_per_sec,
            "games_per_hour": games_per_sec * 3600,
        }

        if self.durations:
            stats.update({
                "duration_mean_ms": statistics.mean(self.durations),
                "duration_median_ms": statistics.median(self.durations),
                "duration_min_ms": min(self.durations),
                "duration_max_ms": max(self.durations),
            })
            if len(self.durations) > 1:
                stats["duration_stdev_ms"] = statistics.stdev(self.durations)

        # Win rates
        for winner, count in self.wins.items():
            clean_name = winner.replace("(", "_").replace(")", "_").replace("-", "_")
            stats[f"win_rate_{clean_name}"] = count / max(1, self.games_successful)

        return stats


class ForgeDaemonClient:
    """Client for communicating with Forge daemon."""

    def __init__(self, host: str, port: int, timeout: int = 120):
        self.host = host
        self.port = port
        self.timeout = timeout

    def play_game(self, deck1: str, deck2: str) -> dict:
        """Play a single game and return result."""
        result = {
            "success": False,
            "duration_ms": 0,
            "winner": None,
            "error": None
        }

        start_time = time.time()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.port))

                cmd = f"NEWGAME {deck1} {deck2} -q -c 60\n"
                s.sendall(cmd.encode())

                response = ""
                while True:
                    try:
                        data = s.recv(4096)
                        if not data:
                            break
                        response += data.decode()
                        if "GAME_RESULT:" in response or "ERROR:" in response:
                            break
                    except socket.timeout:
                        result["error"] = "Socket timeout"
                        break

                elapsed_ms = (time.time() - start_time) * 1000
                result["duration_ms"] = elapsed_ms

                if "GAME_RESULT:" in response:
                    result["success"] = True
                    parts = response.strip().split()
                    if "won" in response:
                        result["winner"] = parts[1]
                    elif "Draw" in response:
                        result["winner"] = "Draw"
                elif "ERROR:" in response:
                    result["error"] = response.strip()
                else:
                    result["error"] = f"Unexpected: {response[:50]}"

        except Exception as e:
            result["error"] = str(e)
            result["duration_ms"] = (time.time() - start_time) * 1000

        return result


class TrainingDashboard:
    """Main training dashboard with wandb integration."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        self.client = ForgeDaemonClient(config.daemon_host, config.daemon_port)
        self.wandb_run = None

    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            return

        run_name = self.config.run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.wandb_run = wandb.init(
            project=self.config.project_name,
            name=run_name,
            config={
                "total_games": self.config.total_games,
                "max_parallel": self.config.max_parallel,
                "daemon_host": self.config.daemon_host,
                "daemon_port": self.config.daemon_port,
                "deck1": self.config.deck1,
                "deck2": self.config.deck2,
            }
        )

        # Define custom charts
        wandb.define_metric("games_played")
        wandb.define_metric("*", step_metric="games_played")

    def log_metrics(self, force: bool = False):
        """Log metrics to wandb and console."""
        games = self.metrics.games_played

        # Only log at intervals unless forced
        if not force and games % self.config.log_interval != 0:
            return

        stats = self.metrics.get_stats()

        # Console output
        rate = stats["games_per_second"]
        remaining = self.config.total_games - games
        eta = remaining / rate if rate > 0 else 0

        print(f"[{games:6d}/{self.config.total_games}] "
              f"Rate: {rate:.2f}/s | "
              f"Success: {stats['success_rate']*100:.1f}% | "
              f"ETA: {eta/60:.1f}min")

        # Wandb logging
        if self.wandb_run:
            wandb.log(stats)

    def checkpoint(self):
        """Save checkpoint (placeholder for model saving)."""
        games = self.metrics.games_played

        if games % self.config.checkpoint_interval != 0:
            return

        checkpoint_path = f"/tmp/mtg_checkpoint_{games}.json"
        stats = self.metrics.get_stats()

        with open(checkpoint_path, "w") as f:
            json.dump({
                "games_played": games,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        print(f"  Checkpoint saved: {checkpoint_path}")

        if self.wandb_run:
            wandb.save(checkpoint_path)

    def run_game_worker(self, game_id: int) -> dict:
        """Worker function to run a single game."""
        result = self.client.play_game(self.config.deck1, self.config.deck2)
        result["game_id"] = game_id
        return result

    def run_training(self):
        """Run the full training loop."""
        print("=" * 60)
        print("MTG RL TRAINING DASHBOARD")
        print("=" * 60)
        print(f"Total games: {self.config.total_games}")
        print(f"Parallel workers: {self.config.max_parallel}")
        print(f"Daemon: {self.config.daemon_host}:{self.config.daemon_port}")
        print("=" * 60)
        print()

        self.init_wandb()

        with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
            futures = {
                executor.submit(self.run_game_worker, i): i
                for i in range(self.config.total_games)
            }

            for future in as_completed(futures):
                result = future.result()

                self.metrics.add_result(
                    success=result["success"],
                    duration_ms=result["duration_ms"],
                    winner=result.get("winner"),
                    error=result.get("error")
                )

                self.log_metrics()
                self.checkpoint()

        # Final summary
        self.log_metrics(force=True)
        self.print_summary()

        if self.wandb_run:
            wandb.finish()

    def print_summary(self):
        """Print final training summary."""
        stats = self.metrics.get_stats()

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total games: {stats['games_played']}")
        print(f"Successful: {stats['games_successful']}")
        print(f"Failed: {stats['games_failed']}")
        print(f"Total time: {stats['elapsed_seconds']/60:.1f} minutes")
        print(f"Throughput: {stats['games_per_hour']:.0f} games/hour")
        print()

        if stats.get("duration_mean_ms"):
            print("Game Duration (ms):")
            print(f"  Mean: {stats['duration_mean_ms']:.0f}")
            print(f"  Median: {stats['duration_median_ms']:.0f}")
            print(f"  Range: {stats['duration_min_ms']:.0f} - {stats['duration_max_ms']:.0f}")
            print()

        print("Win Rates:")
        for key, value in stats.items():
            if key.startswith("win_rate_"):
                name = key.replace("win_rate_", "")
                print(f"  {name}: {value*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="MTG RL Training Dashboard")
    parser.add_argument("--games", type=int, default=10000, help="Total games to play")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel workers")
    parser.add_argument("--host", default="localhost", help="Daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Daemon port")
    parser.add_argument("--project", default="mtg-rl-training", help="Wandb project name")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--deck1", default="test_red.dck", help="First deck")
    parser.add_argument("--deck2", default="test_blue.dck", help="Second deck")

    args = parser.parse_args()

    config = TrainingConfig(
        project_name=args.project,
        run_name=args.name,
        total_games=args.games,
        max_parallel=args.parallel,
        daemon_host=args.host,
        daemon_port=args.port,
        deck1=args.deck1,
        deck2=args.deck2,
    )

    dashboard = TrainingDashboard(config)
    dashboard.run_training()


if __name__ == "__main__":
    main()
