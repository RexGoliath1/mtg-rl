#!/usr/bin/env python3
"""
Headless Video Generator for MTG Game Replays

Renders game states to frames using PIL/matplotlib and stitches into video.
Works completely headless - no display required.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil

# Use non-interactive backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from replay_recorder import ReplayRecorder, GameReplay


@dataclass
class RenderConfig:
    """Configuration for video rendering."""
    width: int = 1920
    height: int = 1080
    fps: int = 1  # 1 frame per second = 1 state per second
    seconds_per_action: float = 2.0  # How long to show each action
    font_size: int = 12
    card_width: int = 80
    card_height: int = 112
    background_color: str = '#1a1a2e'
    player1_color: str = '#e94560'  # Red
    player2_color: str = '#0f3460'  # Blue
    text_color: str = '#ffffff'
    card_color: str = '#16213e'
    tapped_color: str = '#4a4a6a'


class GameStateRenderer:
    """Renders game states to images."""

    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()

    def render_state(self, state: Dict[str, Any], action_text: str = "") -> np.ndarray:
        """
        Render a game state to an image array.

        Args:
            state: Game state dictionary
            action_text: Text describing current action

        Returns:
            numpy array of RGB image
        """
        fig, ax = plt.subplots(1, 1, figsize=(
            self.config.width / 100,
            self.config.height / 100
        ), dpi=100)

        # Set background
        ax.set_facecolor(self.config.background_color)
        fig.patch.set_facecolor(self.config.background_color)

        # Remove axes
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        # Get players
        players = state.get('players', [])
        if len(players) >= 2:
            p1, p2 = players[0], players[1]
        else:
            p1 = players[0] if players else {}
            p2 = {}

        # Draw opponent area (top)
        self._draw_player_area(ax, p2, is_opponent=True, y_base=55)

        # Draw your area (bottom)
        self._draw_player_area(ax, p1, is_opponent=False, y_base=5)

        # Draw stack in middle
        self._draw_stack(ax, state.get('stack', []))

        # Draw turn/phase info
        turn = state.get('turn', 0)
        phase = state.get('phase', 'Unknown')
        active = state.get('active_player', '')
        ax.text(50, 98, f"Turn {turn} - {phase} ({active})",
                ha='center', va='top', fontsize=14,
                color=self.config.text_color, fontweight='bold')

        # Draw action text
        if action_text:
            ax.text(50, 52, action_text,
                    ha='center', va='center', fontsize=16,
                    color='#ffd700', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#333355', alpha=0.8))

        # Convert to array
        fig.canvas.draw()
        # Get the RGBA buffer and convert to RGB
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = img[:, :, :3]  # Remove alpha channel

        plt.close(fig)
        return img

    def _draw_player_area(self, ax, player: Dict, is_opponent: bool, y_base: float):
        """Draw a player's area (life, hand, battlefield)."""
        name = player.get('name', 'Unknown')
        life = player.get('life', 0)
        hand_size = player.get('hand_size', 0)
        library_size = player.get('library_size', 0)

        color = self.config.player2_color if is_opponent else self.config.player1_color

        # Player info box
        info_y = y_base + 38 if is_opponent else y_base
        ax.add_patch(FancyBboxPatch(
            (2, info_y), 18, 8,
            boxstyle="round,pad=0.3",
            facecolor=color, edgecolor='white', linewidth=2
        ))
        ax.text(11, info_y + 4, f"{name}\nLife: {life}",
                ha='center', va='center', fontsize=11,
                color=self.config.text_color, fontweight='bold')

        # Hand/Library info
        ax.text(11, info_y - 2 if is_opponent else info_y + 10,
                f"Hand: {hand_size} | Library: {library_size}",
                ha='center', va='center', fontsize=9,
                color=self.config.text_color)

        # Battlefield
        battlefield = player.get('battlefield', [])
        self._draw_battlefield(ax, battlefield, y_base + 10, is_opponent)

        # Hand (only show for non-opponent or if visible)
        hand = player.get('hand', [])
        if hand and not is_opponent:
            self._draw_hand(ax, hand, y_base)

    def _draw_battlefield(self, ax, cards: List, y_base: float, is_opponent: bool):
        """Draw cards on battlefield."""
        if not cards:
            ax.text(50, y_base + 12, "(empty battlefield)",
                    ha='center', va='center', fontsize=10,
                    color='#666666', style='italic')
            return

        # Separate lands and non-lands
        lands = [c for c in cards if c.get('is_land', False)]
        nonlands = [c for c in cards if not c.get('is_land', False)]

        # Draw non-lands in top row
        x_start = 22
        for i, card in enumerate(nonlands[:8]):  # Max 8 per row
            x = x_start + i * 9
            self._draw_card(ax, card, x, y_base + 15)

        # Draw lands in bottom row
        for i, card in enumerate(lands[:10]):  # Max 10 lands
            x = x_start + i * 8
            self._draw_card(ax, card, x, y_base + 3, is_land=True)

    def _draw_card(self, ax, card: Dict, x: float, y: float, is_land: bool = False):
        """Draw a single card."""
        name = card.get('name', '?')
        tapped = card.get('tapped', False)
        is_creature = card.get('is_creature', False)

        # Card dimensions
        w, h = (6, 4) if is_land else (7, 10)
        if tapped:
            w, h = h, w

        color = self.config.tapped_color if tapped else self.config.card_color

        # Draw card
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='white', linewidth=1
        ))

        # Card name (truncated)
        display_name = name[:8] + '..' if len(name) > 10 else name
        ax.text(x + w/2, y + h/2 + 1, display_name,
                ha='center', va='center', fontsize=7,
                color=self.config.text_color, rotation=90 if tapped else 0)

        # Power/Toughness for creatures
        if is_creature and not is_land:
            power = card.get('power', 0)
            toughness = card.get('toughness', 0)
            ax.text(x + w - 0.5, y + 0.5, f"{power}/{toughness}",
                    ha='right', va='bottom', fontsize=6,
                    color='#ffd700', fontweight='bold')

    def _draw_hand(self, ax, hand: List[str], y_base: float):
        """Draw hand of cards."""
        x_start = 25
        for i, card_name in enumerate(hand[:7]):  # Max 7 cards shown
            x = x_start + i * 8
            ax.add_patch(FancyBboxPatch(
                (x, y_base - 8), 7, 9,
                boxstyle="round,pad=0.1",
                facecolor=self.config.card_color, edgecolor='#ffd700', linewidth=1
            ))
            display_name = card_name[:8] if len(card_name) > 8 else card_name
            ax.text(x + 3.5, y_base - 3.5, display_name,
                    ha='center', va='center', fontsize=7,
                    color=self.config.text_color)

    def _draw_stack(self, ax, stack: List[Dict]):
        """Draw the stack."""
        if not stack:
            return

        ax.text(85, 50, "Stack:", ha='left', va='center', fontsize=10,
                color=self.config.text_color, fontweight='bold')

        for i, item in enumerate(stack[:5]):  # Max 5 items
            desc = item.get('description', '?')[:20]
            ax.text(85, 47 - i*3, f"- {desc}",
                    ha='left', va='center', fontsize=8,
                    color='#ffa500')


class VideoGenerator:
    """Generates videos from game replays."""

    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        self.renderer = GameStateRenderer(config)

    def generate_video(self, replay: GameReplay, output_path: str,
                       verbose: bool = True) -> bool:
        """
        Generate a video from a replay.

        Args:
            replay: GameReplay object with states/actions
            output_path: Path for output video file
            verbose: Print progress

        Returns:
            True if successful
        """
        # Create temp directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)

            if verbose:
                print(f"Generating video for replay {replay.replay_id}")
                print(f"  Seed: {replay.seed}")
                print(f"  {len(replay.actions)} actions, {len(replay.snapshots)} snapshots")

            # Generate frames from snapshots
            frame_idx = 0

            if replay.snapshots:
                # Use snapshots for state
                for i, snapshot in enumerate(replay.snapshots):
                    state = snapshot.get('state', {})
                    turn = snapshot.get('turn', 0)

                    # Find action for this turn
                    action_text = ""
                    for action in replay.actions:
                        if action.get('turn', 0) == turn:
                            action_text = self._format_action(action)
                            break

                    # Render frame
                    img = self.renderer.render_state(state, action_text)
                    frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
                    plt.imsave(str(frame_path), img)
                    frame_idx += 1

                    if verbose and (i + 1) % 10 == 0:
                        print(f"  Rendered {i + 1}/{len(replay.snapshots)} snapshots")

            else:
                # Generate synthetic states from actions
                frame_idx = self._generate_frames_from_actions(
                    replay, frames_dir, verbose
                )

            if frame_idx == 0:
                if verbose:
                    print("  No frames generated!")
                return False

            # Use ffmpeg to create video
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build ffmpeg command
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-framerate', str(self.config.fps),
                '-i', str(frames_dir / 'frame_%05d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',  # Quality (lower = better, 18-28 is good)
                str(output_path)
            ]

            if verbose:
                print(f"  Creating video with {frame_idx} frames...")

            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    if verbose:
                        print(f"  ffmpeg error: {result.stderr}")
                    return False
            except FileNotFoundError:
                if verbose:
                    print("  ffmpeg not found! Install with: brew install ffmpeg")
                return False

            if verbose:
                file_size = output_path.stat().st_size / 1024
                print(f"  Video saved: {output_path} ({file_size:.1f} KB)")

            return True

    def _generate_frames_from_actions(self, replay: GameReplay,
                                       frames_dir: Path, verbose: bool) -> int:
        """Generate frames from action sequence (synthetic state reconstruction)."""
        # Build synthetic state from actions
        state = {
            'turn': 0,
            'phase': 'Main',
            'active_player': replay.deck1.replace('.dck', ''),
            'players': [
                {
                    'name': replay.deck1.replace('.dck', ''),
                    'life': 20,
                    'hand_size': 7,
                    'library_size': 53,
                    'hand': [],
                    'battlefield': []
                },
                {
                    'name': replay.deck2.replace('.dck', ''),
                    'life': 20,
                    'hand_size': 7,
                    'library_size': 53,
                    'hand': [],
                    'battlefield': []
                }
            ],
            'stack': []
        }

        frame_idx = 0

        # Render initial state
        img = self.renderer.render_state(state, "Game Start")
        plt.imsave(str(frames_dir / f"frame_{frame_idx:05d}.png"), img)
        frame_idx += 1

        # Process actions
        for action in replay.actions:
            action_text = self._format_action(action)
            state['turn'] = action.get('turn', state['turn'])

            # Update synthetic state based on action
            self._apply_action_to_state(state, action)

            # Render frame
            img = self.renderer.render_state(state, action_text)
            frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
            plt.imsave(str(frame_path), img)
            frame_idx += 1

        # Final frame with result
        result_text = f"{replay.winner} wins!" if replay.winner else "Game Over"
        img = self.renderer.render_state(state, result_text)
        plt.imsave(str(frames_dir / f"frame_{frame_idx:05d}.png"), img)
        frame_idx += 1

        return frame_idx

    def _apply_action_to_state(self, state: Dict, action: Dict):
        """Apply an action to update synthetic state."""
        action_type = action.get('type', '')
        player_name = action.get('player', '')

        # Find player
        player_idx = 0
        for i, p in enumerate(state['players']):
            if p['name'] == player_name or player_name in p['name']:
                player_idx = i
                break

        player = state['players'][player_idx]

        if action_type == 'play_land':
            card = action.get('card', 'Land')
            player['battlefield'].append({
                'name': card,
                'is_land': True,
                'tapped': False
            })
            player['hand_size'] = max(0, player['hand_size'] - 1)

        elif action_type == 'cast_spell':
            card = action.get('card', 'Spell')
            # Add to stack briefly (would resolve)
            state['stack'] = [{'description': f"Cast {card}"}]

        elif action_type == 'play_creature':
            card = action.get('card', 'Creature')
            power = action.get('power', 2)
            toughness = action.get('toughness', 2)
            player['battlefield'].append({
                'name': card,
                'is_creature': True,
                'is_land': False,
                'tapped': False,
                'power': power,
                'toughness': toughness
            })
            player['hand_size'] = max(0, player['hand_size'] - 1)

        elif action_type == 'attack':
            # Mark attackers as tapped
            attackers = action.get('attackers', [])
            for card in player['battlefield']:
                if card.get('is_creature') and card['name'] in attackers:
                    card['tapped'] = True

        elif action_type == 'damage':
            target = action.get('target', '')
            amount = action.get('amount', 0)
            # Find target player and reduce life
            for p in state['players']:
                if target in p['name']:
                    p['life'] -= amount

        elif action_type == 'pass_turn':
            # Untap, new turn
            state['turn'] += 1
            for card in player['battlefield']:
                card['tapped'] = False

        # Clear stack after each action
        if action_type != 'cast_spell':
            state['stack'] = []

    def _format_action(self, action: Dict) -> str:
        """Format an action for display."""
        action_type = action.get('type', '')
        player = action.get('player', '')

        if action_type == 'play_land':
            card = action.get('card', 'Land')
            return f"{player} plays {card}"

        elif action_type == 'cast_spell':
            card = action.get('card', 'Spell')
            target = action.get('target', '')
            if target:
                return f"{player} casts {card} targeting {target}"
            return f"{player} casts {card}"

        elif action_type == 'play_creature':
            card = action.get('card', 'Creature')
            return f"{player} plays {card}"

        elif action_type == 'attack':
            attackers = action.get('attackers', [])
            return f"{player} attacks with {', '.join(attackers)}"

        elif action_type == 'block':
            return f"{player} blocks"

        elif action_type == 'pass_turn':
            return f"{player} passes turn"

        elif action_type == 'damage':
            target = action.get('target', '')
            amount = action.get('amount', 0)
            return f"{target} takes {amount} damage"

        return f"{player}: {action_type}"


def generate_video_for_run(run_number: int, output_path: str = None,
                          replay_dir: str = "replays") -> bool:
    """
    Convenience function to generate video for a specific run.

    Args:
        run_number: The run/game index (0-based)
        output_path: Output video path (default: videos/run_{n}.mp4)
        replay_dir: Directory containing replays

    Returns:
        True if successful
    """
    recorder = ReplayRecorder(replay_dir)
    replay = recorder.get_replay_by_index(run_number)

    if not replay:
        print(f"Run {run_number} not found!")
        return False

    if output_path is None:
        output_path = f"videos/run_{run_number}.mp4"

    generator = VideoGenerator()
    return generator.generate_video(replay, output_path)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate video from MTG game replay")
    parser.add_argument("run", type=str, help="Run number or replay ID")
    parser.add_argument("-o", "--output", type=str, help="Output video path")
    parser.add_argument("-d", "--replay-dir", type=str, default="replays",
                       help="Replay directory")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")

    args = parser.parse_args()

    # Try to parse as number first
    try:
        run_num = int(args.run)
        output = args.output or f"videos/run_{run_num}.mp4"
        success = generate_video_for_run(run_num, output, args.replay_dir)
    except ValueError:
        # Treat as replay ID
        recorder = ReplayRecorder(args.replay_dir)
        replay = recorder.load_replay(args.run)
        if replay:
            output = args.output or f"videos/{args.run}.mp4"
            config = RenderConfig(fps=args.fps)
            generator = VideoGenerator(config)
            success = generator.generate_video(replay, output)
        else:
            print(f"Replay '{args.run}' not found!")
            success = False

    exit(0 if success else 1)
