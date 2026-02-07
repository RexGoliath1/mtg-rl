#!/usr/bin/env python3
"""
Watch a game replay in Forge's actual GUI.

This launches Forge in simulation mode with the same seed,
decks, and settings as the original game, so you can watch
the AI play out the same game deterministically.

Usage:
    python3 watch_replay.py 0          # Watch replay #0
    python3 watch_replay.py abc123     # Watch replay by ID
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.replay_recorder import ReplayRecorder


def watch_replay(run: str, replay_dir: str = "replays", quiet: bool = False):
    """Launch Forge to watch a replay."""
    recorder = ReplayRecorder(replay_dir)

    # Try as number first
    try:
        run_num = int(run)
        replay = recorder.get_replay_by_index(run_num)
    except ValueError:
        replay = recorder.load_replay(run)

    if not replay:
        print(f"Replay '{run}' not found!")
        return False

    print(f"Replay: {replay.replay_id}")
    print(f"  Seed: {replay.seed}")
    print(f"  Decks: {replay.deck1} vs {replay.deck2}")
    print(f"  Original winner: {replay.winner}")
    print(f"  Turns: {replay.turns}")
    print()

    # Build Forge command - use absolute path
    script_dir = Path(__file__).parent.absolute()
    forge_dir = script_dir / "forge-repo" / "forge-gui-desktop"
    forge_jar = forge_dir / "target" / "forge-gui-desktop-2.0.09-SNAPSHOT-jar-with-dependencies.jar"

    if not forge_jar.exists():
        print(f"Forge JAR not found at {forge_jar}")
        print("Build it with: cd forge-repo && mvn package -DskipTests -pl forge-gui-desktop -am")
        return False

    # Launch Forge sim mode with seed
    cmd = [
        "java", "-Xmx2g",
        "-jar", str(forge_jar),
        "sim",
        "-d", replay.deck1, replay.deck2,
        "-s", str(replay.seed),
        "-n", "1"
    ]

    if quiet:
        cmd.append("-q")

    print(f"Launching Forge simulation with seed {replay.seed}...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("=" * 60)
    print("GAME OUTPUT")
    print("=" * 60)

    # Run and stream output from forge-gui-desktop directory
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(forge_dir)
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()
        return process.returncode == 0

    except FileNotFoundError:
        print("Java not found! Make sure Java is installed.")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted.")
        process.kill()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Watch a game replay using Forge's simulation mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s 0              # Watch replay #0
    %(prog)s abc123         # Watch by replay ID
    %(prog)s 0 -q           # Quiet mode (result only)
"""
    )

    parser.add_argument("run", help="Run number or replay ID")
    parser.add_argument("-d", "--replay-dir", default="replays",
                       help="Replay directory (default: replays)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Quiet mode - only show result")

    args = parser.parse_args()

    success = watch_replay(args.run, args.replay_dir, args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
