#!/usr/bin/env python3
"""
Replay CLI - Easy interface for managing and viewing game replays.

Commands:
    list                    List recent replays
    show <run>             Show replay details
    video <run>            Generate video for a run
    video-batch <start> <end>  Generate videos for a range of runs
"""

import argparse
import sys
from pathlib import Path

from replay_recorder import ReplayRecorder
from replay_video import VideoGenerator, RenderConfig


def cmd_list(args):
    """List recent replays."""
    recorder = ReplayRecorder(args.replay_dir)
    replays = recorder.list_replays(limit=args.limit, offset=args.offset)

    if not replays:
        print("No replays found.")
        return

    print(f"{'#':<6} {'ID':<14} {'Winner':<10} {'Turns':<6} {'Actions':<8} {'Duration':<10}")
    print("-" * 60)

    for r in replays:
        duration_s = r.get('duration_ms', 0) / 1000
        print(f"{r['index']:<6} {r['replay_id']:<14} {r.get('winner', '?'):<10} "
              f"{r.get('turns', 0):<6} {r.get('num_actions', 0):<8} {duration_s:.1f}s")


def cmd_show(args):
    """Show details of a specific replay."""
    recorder = ReplayRecorder(args.replay_dir)

    # Try as number first
    try:
        run_num = int(args.run)
        replay = recorder.get_replay_by_index(run_num)
    except ValueError:
        replay = recorder.load_replay(args.run)

    if not replay:
        print(f"Replay '{args.run}' not found!")
        return 1

    print(f"Replay: {replay.replay_id}")
    print(f"  Seed: {replay.seed}")
    print(f"  Decks: {replay.deck1} vs {replay.deck2}")
    print(f"  Winner: {replay.winner}")
    print(f"  Turns: {replay.turns}")
    print(f"  Duration: {replay.duration_ms/1000:.1f}s")
    print(f"  Actions: {len(replay.actions)}")
    print(f"  Snapshots: {len(replay.snapshots)}")

    if args.verbose and replay.actions:
        print("\nActions:")
        for i, action in enumerate(replay.actions[:20]):
            action_type = action.get('type', '?')
            player = action.get('player', '?')
            turn = action.get('turn', 0)
            print(f"  T{turn}: {player} - {action_type}")
            if 'card' in action:
                print(f"       Card: {action['card']}")
        if len(replay.actions) > 20:
            print(f"  ... and {len(replay.actions) - 20} more actions")

    return 0


def cmd_video(args):
    """Generate video for a replay."""
    recorder = ReplayRecorder(args.replay_dir)

    # Try as number first
    try:
        run_num = int(args.run)
        replay = recorder.get_replay_by_index(run_num)
        default_output = f"videos/run_{run_num}.mp4"
    except ValueError:
        replay = recorder.load_replay(args.run)
        default_output = f"videos/{args.run}.mp4"

    if not replay:
        print(f"Replay '{args.run}' not found!")
        return 1

    output = args.output or default_output

    config = RenderConfig(
        fps=args.fps,
        width=args.width,
        height=args.height
    )

    generator = VideoGenerator(config)
    success = generator.generate_video(replay, output, verbose=True)

    return 0 if success else 1


def cmd_video_batch(args):
    """Generate videos for a range of runs."""
    recorder = ReplayRecorder(args.replay_dir)

    Path("videos").mkdir(exist_ok=True)

    config = RenderConfig(fps=args.fps)
    generator = VideoGenerator(config)

    success_count = 0
    fail_count = 0

    for run_num in range(args.start, args.end + 1):
        replay = recorder.get_replay_by_index(run_num)
        if not replay:
            print(f"Run {run_num}: Not found, skipping")
            fail_count += 1
            continue

        output = f"videos/run_{run_num}.mp4"
        success = generator.generate_video(replay, output, verbose=False)

        if success:
            print(f"Run {run_num}: Generated {output}")
            success_count += 1
        else:
            print(f"Run {run_num}: Failed to generate video")
            fail_count += 1

    print(f"\nGenerated {success_count} videos, {fail_count} failed")
    return 0 if fail_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="MTG Game Replay CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s list                      # List recent replays
    %(prog)s show 42                   # Show details of run 42
    %(prog)s video 100                 # Generate video for run 100
    %(prog)s video 100 -o my_game.mp4  # Custom output path
    %(prog)s video-batch 0 9           # Generate videos for runs 0-9
        """
    )

    parser.add_argument("-d", "--replay-dir", default="replays",
                       help="Replay directory (default: replays)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent replays")
    list_parser.add_argument("-n", "--limit", type=int, default=20,
                            help="Number of replays to show")
    list_parser.add_argument("--offset", type=int, default=0,
                            help="Offset for pagination")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show replay details")
    show_parser.add_argument("run", help="Run number or replay ID")
    show_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Show action details")

    # Video command
    video_parser = subparsers.add_parser("video", help="Generate video")
    video_parser.add_argument("run", help="Run number or replay ID")
    video_parser.add_argument("-o", "--output", help="Output video path")
    video_parser.add_argument("--fps", type=int, default=1,
                             help="Frames per second (default: 1)")
    video_parser.add_argument("--width", type=int, default=1920,
                             help="Video width (default: 1920)")
    video_parser.add_argument("--height", type=int, default=1080,
                             help="Video height (default: 1080)")

    # Video batch command
    batch_parser = subparsers.add_parser("video-batch",
                                         help="Generate videos for range")
    batch_parser.add_argument("start", type=int, help="Start run number")
    batch_parser.add_argument("end", type=int, help="End run number")
    batch_parser.add_argument("--fps", type=int, default=1,
                             help="Frames per second")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "video":
        return cmd_video(args)
    elif args.command == "video-batch":
        return cmd_video_batch(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
