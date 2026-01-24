#!/usr/bin/env python3
"""
Forge Performance Profiler

Measures different aspects of Forge performance to identify bottlenecks:
1. Initialization time (JVM startup, model loading)
2. Per-turn timing
3. Per-decision timing
4. Communication overhead (for interactive mode)
5. Game state serialization cost
"""

import subprocess
import time
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TurnTiming:
    turn_number: int
    player: str
    duration_ms: float
    decisions: int
    phase: str = ""

@dataclass
class GameProfile:
    game_id: int
    total_time_ms: float
    init_time_ms: float
    first_decision_time_ms: float
    turns: List[TurnTiming] = field(default_factory=list)
    total_decisions: int = 0
    winner: str = ""

    @property
    def avg_turn_time(self) -> float:
        if not self.turns:
            return 0
        return statistics.mean(t.duration_ms for t in self.turns)

    @property
    def avg_decision_time(self) -> float:
        if self.total_decisions == 0:
            return 0
        gameplay_time = self.total_time_ms - self.init_time_ms
        return gameplay_time / self.total_decisions


def run_ai_vs_ai_game(
    deck1: str = "red_aggro.dck",
    deck2: str = "white_weenie.dck",
    timeout: int = 120,
    use_docker: bool = True,
    forge_jar: str = None
) -> Optional[GameProfile]:
    """
    Run a single AI vs AI game and profile it.
    """
    profile = GameProfile(game_id=0, total_time_ms=0, init_time_ms=0, first_decision_time_ms=0)

    if use_docker:
        cmd = [
            "docker", "run", "--rm", "-i",
            "--entrypoint", "/bin/bash",
            "forge-sim:latest",
            "-c",
            f"cd /forge && timeout {timeout} xvfb-run -a java -Xmx2048m "
            f"--add-opens java.base/java.lang=ALL-UNNAMED "
            f"--add-opens java.base/java.util=ALL-UNNAMED "
            f"-Dsentry.dsn= -jar forge.jar sim "
            f"-d {deck1} {deck2} -n 1 -q -c {timeout}"
        ]
    else:
        if not forge_jar:
            print("Error: forge_jar path required when not using docker")
            return None
        cmd = [
            "java", "-Xmx2048m",
            "--add-opens", "java.base/java.lang=ALL-UNNAMED",
            "--add-opens", "java.base/java.util=ALL-UNNAMED",
            "-jar", forge_jar, "sim",
            "-d", deck1, deck2, "-n", "1", "-q", "-c", str(timeout)
        ]

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30
        )

        end_time = time.perf_counter()
        profile.total_time_ms = (end_time - start_time) * 1000

        # Parse output for timing info
        output = result.stdout + result.stderr

        # Look for "Game Result: Game X ended in Y ms"
        match = re.search(r"Game Result: Game \d+ ended in (\d+) ms", output)
        if match:
            game_time_ms = int(match.group(1))
            # Init time is total - game time
            profile.init_time_ms = profile.total_time_ms - game_time_ms

        # Look for winner
        match = re.search(r"(\w+\([^)]+\)[^\s]*) has won", output)
        if match:
            profile.winner = match.group(1)

        return profile

    except subprocess.TimeoutExpired:
        print("Game timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def run_interactive_game_profile(
    deck1: str = "red_aggro.dck",
    deck2: str = "white_weenie.dck",
    timeout: int = 120,
    max_decisions: int = 500
) -> Optional[GameProfile]:
    """
    Run interactive mode game and profile decision timing.
    """
    profile = GameProfile(game_id=0, total_time_ms=0, init_time_ms=0, first_decision_time_ms=0)

    cmd = [
        "docker", "run", "--rm", "-i",
        "--entrypoint", "/bin/bash",
        "forge-sim:latest",
        "-c",
        f"cd /forge && timeout {timeout} xvfb-run -a java -Xmx2048m "
        f"--add-opens java.base/java.lang=ALL-UNNAMED "
        f"--add-opens java.base/java.util=ALL-UNNAMED "
        f"-Dsentry.dsn= -jar forge.jar sim "
        f"-d {deck1} {deck2} -n 1 -i -q -c {timeout}"
    ]

    start_time = time.perf_counter()
    init_end_time = None
    first_decision_time = None
    decision_times = []
    current_turn = 0
    decisions_this_turn = 0
    turn_start_time = None

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        decision_count = 0

        while process.poll() is None and decision_count < max_decisions:
            line = process.stdout.readline()
            if not line:
                continue

            line = line.strip()

            # Check for game end
            if "has won" in line.lower():
                match = re.search(r"(\w+\([^)]+\)[^\s]*) has won", line)
                if match:
                    profile.winner = match.group(1)
                break

            # Parse decision
            if line.startswith("DECISION:"):
                decision_time = time.perf_counter()

                if init_end_time is None:
                    init_end_time = decision_time
                    profile.init_time_ms = (init_end_time - start_time) * 1000
                    turn_start_time = decision_time

                if first_decision_time is None:
                    first_decision_time = decision_time
                    profile.first_decision_time_ms = (first_decision_time - start_time) * 1000

                try:
                    data = json.loads(line[9:])
                    turn = data.get('turn', 0)

                    if turn != current_turn:
                        if current_turn > 0 and turn_start_time:
                            turn_duration = (decision_time - turn_start_time) * 1000
                            profile.turns.append(TurnTiming(
                                turn_number=current_turn,
                                player="",
                                duration_ms=turn_duration,
                                decisions=decisions_this_turn
                            ))
                        current_turn = turn
                        decisions_this_turn = 0
                        turn_start_time = decision_time

                    decisions_this_turn += 1

                except json.JSONDecodeError:
                    pass

                # Measure response time
                response_start = time.perf_counter()
                process.stdin.write("-1\n")  # Always pass
                process.stdin.flush()
                response_end = time.perf_counter()

                decision_times.append((response_end - response_start) * 1000)
                decision_count += 1
                profile.total_decisions = decision_count

        process.terminate()
        process.wait(timeout=5)

    except Exception as e:
        print(f"Error: {e}")
        return None

    end_time = time.perf_counter()
    profile.total_time_ms = (end_time - start_time) * 1000

    # Add communication overhead stats
    if decision_times:
        print("\nCommunication overhead per decision:")
        print(f"  Mean: {statistics.mean(decision_times):.3f} ms")
        print(f"  Median: {statistics.median(decision_times):.3f} ms")
        print(f"  Max: {max(decision_times):.3f} ms")

    return profile


def profile_initialization():
    """
    Profile just the initialization time by starting and immediately killing.
    """
    print("\n" + "="*60)
    print("INITIALIZATION PROFILING")
    print("="*60)

    times = []

    for i in range(3):
        cmd = [
            "docker", "run", "--rm", "-i",
            "--entrypoint", "/bin/bash",
            "forge-sim:latest",
            "-c",
            "cd /forge && timeout 60 xvfb-run -a java -Xmx2048m "
            "--add-opens java.base/java.lang=ALL-UNNAMED "
            "-Dsentry.dsn= -jar forge.jar sim "
            "-d red_aggro.dck white_weenie.dck -n 1 -q -c 60"
        ]

        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end = time.perf_counter()

        total_ms = (end - start) * 1000

        # Extract game time from output
        match = re.search(r"ended in (\d+) ms", result.stdout)
        if match:
            game_ms = int(match.group(1))
            init_ms = total_ms - game_ms
            times.append({
                'total': total_ms,
                'game': game_ms,
                'init': init_ms
            })
            print(f"  Run {i+1}: Total={total_ms:.0f}ms, Game={game_ms}ms, Init={init_ms:.0f}ms")

    if times:
        avg_init = statistics.mean(t['init'] for t in times)
        avg_game = statistics.mean(t['game'] for t in times)
        print(f"\n  Average init time: {avg_init:.0f}ms ({avg_init/1000:.1f}s)")
        print(f"  Average game time: {avg_game:.0f}ms ({avg_game/1000:.1f}s)")
        print(f"  Init overhead: {avg_init/(avg_init+avg_game)*100:.1f}%")


def profile_ai_games(n_games: int = 5):
    """
    Profile multiple AI vs AI games.
    """
    print("\n" + "="*60)
    print(f"AI VS AI PROFILING ({n_games} games)")
    print("="*60)

    profiles = []

    for i in range(n_games):
        print(f"\n  Running game {i+1}/{n_games}...", end=" ", flush=True)
        profile = run_ai_vs_ai_game()
        if profile:
            profiles.append(profile)
            print(f"Done in {profile.total_time_ms:.0f}ms (game: {profile.total_time_ms - profile.init_time_ms:.0f}ms)")

    if profiles:
        total_times = [p.total_time_ms for p in profiles]
        game_times = [p.total_time_ms - p.init_time_ms for p in profiles]
        init_times = [p.init_time_ms for p in profiles]

        print(f"\n  Results ({len(profiles)} successful games):")
        print(f"  Total time:  {statistics.mean(total_times):.0f}ms avg ({statistics.stdev(total_times):.0f}ms std)")
        print(f"  Game time:   {statistics.mean(game_times):.0f}ms avg")
        print(f"  Init time:   {statistics.mean(init_times):.0f}ms avg")
        print(f"\n  Games per hour: {3600000 / statistics.mean(total_times):.1f}")
        print(f"  Games per hour (no init): {3600000 / statistics.mean(game_times):.1f}")


def profile_interactive_mode():
    """
    Profile interactive mode with decision-level timing.
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE PROFILING")
    print("="*60)

    print("\n  Running interactive game (passing all decisions)...")
    profile = run_interactive_game_profile(max_decisions=200)

    if profile:
        print("\n  Results:")
        print(f"  Total time: {profile.total_time_ms:.0f}ms")
        print(f"  Init time: {profile.init_time_ms:.0f}ms")
        print(f"  First decision at: {profile.first_decision_time_ms:.0f}ms")
        print(f"  Total decisions: {profile.total_decisions}")
        print(f"  Turns completed: {len(profile.turns)}")

        if profile.total_decisions > 0:
            gameplay_time = profile.total_time_ms - profile.init_time_ms
            print(f"\n  Avg time per decision: {gameplay_time / profile.total_decisions:.1f}ms")

        if profile.turns:
            turn_times = [t.duration_ms for t in profile.turns]
            print(f"  Avg time per turn: {statistics.mean(turn_times):.0f}ms")
            print(f"  Avg decisions per turn: {profile.total_decisions / len(profile.turns):.1f}")


def check_docker_available() -> bool:
    """Check if Docker and forge-sim image are available."""
    try:
        result = subprocess.run(
            ["docker", "images", "forge-sim", "--format", "{{.Repository}}"],
            capture_output=True, text=True, timeout=10
        )
        return "forge-sim" in result.stdout
    except:
        return False


def estimate_scaling():
    """
    Estimate what's needed to hit target games/hour.
    """
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)

    # Assumptions based on profiling
    init_time_s = 8.0  # Typical JVM + model init
    game_time_s = 5.0  # Typical AI vs AI game
    decision_time_ms = 50  # Time per decision in interactive mode
    decisions_per_game = 100  # Rough average

    print("\n  Assumptions (adjust based on profiling):")
    print(f"    Init time: {init_time_s}s")
    print(f"    Game time (AI vs AI): {game_time_s}s")
    print(f"    Decision time: {decision_time_ms}ms")
    print(f"    Decisions per game: {decisions_per_game}")

    # Current rate
    total_time_s = init_time_s + game_time_s
    current_rate = 3600 / total_time_s

    print(f"\n  Current estimated rate: {current_rate:.0f} games/hour")

    # If we reuse JVM (no init per game)
    reuse_rate = 3600 / game_time_s
    print(f"  With JVM reuse: {reuse_rate:.0f} games/hour")

    # Parallel instances
    for n in [4, 8, 16, 32]:
        parallel_rate = current_rate * n
        print(f"  With {n} parallel instances: {parallel_rate:.0f} games/hour")

    # Target analysis
    target = 10000
    instances_needed = target / current_rate
    instances_needed_reuse = target / reuse_rate

    print(f"\n  To hit {target} games/hour:")
    print(f"    Current approach: {instances_needed:.0f} parallel instances")
    print(f"    With JVM reuse: {instances_needed_reuse:.0f} parallel instances")

    # Interactive mode overhead
    interactive_game_time = (decisions_per_game * decision_time_ms) / 1000 + game_time_s
    interactive_rate = 3600 / (init_time_s + interactive_game_time)

    print(f"\n  Interactive mode estimate: {interactive_rate:.0f} games/hour")
    print(f"    (Communication adds ~{decisions_per_game * decision_time_ms / 1000:.1f}s per game)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile Forge performance")
    parser.add_argument("--mode", choices=["init", "ai", "interactive", "scaling", "all"],
                       default="scaling", help="Profiling mode")
    parser.add_argument("--games", type=int, default=5, help="Number of games for AI profiling")

    args = parser.parse_args()

    if args.mode in ["init", "ai", "interactive", "all"]:
        if not check_docker_available():
            print("Error: Docker image 'forge-sim' not found.")
            print("Build it first with: docker build -t forge-sim .")
            exit(1)

    if args.mode == "init" or args.mode == "all":
        profile_initialization()

    if args.mode == "ai" or args.mode == "all":
        profile_ai_games(args.games)

    if args.mode == "interactive" or args.mode == "all":
        profile_interactive_mode()

    if args.mode == "scaling" or args.mode == "all":
        estimate_scaling()
