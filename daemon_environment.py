#!/usr/bin/env python3
"""
MTG Environment using Forge Daemon

Connects to a running Forge daemon via TCP socket.
Much faster than spawning new processes since card DB is already loaded.

Memory Architecture:
- Daemon holds card database (~500MB) ONCE
- Each game connection uses ~20MB for game state
- Can run 8-32 games concurrently on a single daemon

Usage:
    # Start daemon (in Docker or locally)
    java -jar forge.jar daemon -p 17171

    # Use in Python
    env = DaemonMTGEnvironment(host='localhost', port=17171)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)

    # With replay recording
    env = DaemonMTGEnvironment(host='localhost', port=17171, record_replays=True)
    obs, info = env.reset()
    # ... play game ...
    # Replay is automatically saved at game end
    # Generate video later: python replay_cli.py video <run_number>
"""

import socket
import json
import random
import os
import numpy as np
from typing import Tuple, Dict, List, Optional
import time

from rl_environment import (
    GameState, Action, ActionType, RewardShaper
)
from replay_recorder import ReplayRecorder, GameReplay


class DaemonMTGEnvironment:
    """
    OpenAI Gym-style environment that connects to Forge daemon.

    This is faster than MTGEnvironment because:
    1. No JVM startup per game
    2. No card database loading per game
    3. Multiple games can share one daemon

    Protocol:
    - Connect to daemon
    - Send: NEWGAME deck1.dck deck2.dck -i
    - Receive/Send: Same JSON protocol as interactive mode
    - Connection closes when game ends
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 17171,
        deck1: str = "decks/red_aggro.dck",
        deck2: str = "decks/white_weenie.dck",
        player_id: int = 1,
        timeout: int = 120,
        reward_shaping: bool = True,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        record_replays: bool = False,
        replay_dir: str = "replays",
        seed: Optional[int] = None,
        snapshot_interval: int = 1,  # Snapshot every N decisions (1=every decision)
    ):
        self.host = host
        self.port = port
        # Resolve deck paths to absolute paths for daemon compatibility
        self.deck1 = self._resolve_deck_path(deck1)
        self.deck2 = self._resolve_deck_path(deck2)
        self.player_id = player_id
        self.timeout = timeout
        self.reward_shaping = reward_shaping
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        # Replay recording
        self.record_replays = record_replays
        self.replay_dir = replay_dir
        self.seed = seed
        self.snapshot_interval = snapshot_interval
        self.replay_recorder: Optional[ReplayRecorder] = None
        self.current_replay: Optional[GameReplay] = None
        self.game_start_time: float = 0

        if record_replays:
            self.replay_recorder = ReplayRecorder(replay_dir)

        self.socket: Optional[socket.socket] = None
        self.reader = None
        self.reward_shaper = RewardShaper()
        self.current_state: Optional[GameState] = None
        self.available_actions: List[Dict] = []
        self.game_over = False
        self.won = False
        self.decision_count = 0

        # Observation and action space dimensions
        self.observation_dim = 38
        self.max_actions = 50

        # Track last replay ID for external access
        self.last_replay_id: Optional[str] = None

    def _resolve_deck_path(self, deck_path: str) -> str:
        """Resolve deck path to absolute path for daemon compatibility.

        The daemon runs from a different directory, so we need absolute paths.
        """
        if os.path.isabs(deck_path):
            return deck_path
        # Try relative to current working directory
        abs_path = os.path.abspath(deck_path)
        if os.path.exists(abs_path):
            return abs_path
        # Return as-is if not found (let daemon report error)
        return deck_path

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and start a new game via daemon."""
        self._cleanup()
        self.reward_shaper.reset()
        self.game_over = False
        self.won = False
        self.decision_count = 0
        self.game_start_time = time.time()

        # Generate seed for this game (for replay determinism)
        game_seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)

        # Start replay recording if enabled
        if self.record_replays and self.replay_recorder:
            self.current_replay = self.replay_recorder.create_replay(
                seed=game_seed,
                deck1=self.deck1,
                deck2=self.deck2
            )

        # Connect to daemon
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.connect_timeout)

        try:
            self.socket.connect((self.host, self.port))
        except (socket.timeout, ConnectionRefusedError) as e:
            raise RuntimeError(f"Cannot connect to Forge daemon at {self.host}:{self.port}: {e}")

        self.socket.settimeout(self.read_timeout)
        self.reader = self.socket.makefile('r')

        # Send new game command
        # TODO: Add seed parameter when daemon supports it: -s {game_seed}
        cmd = f"NEWGAME {self.deck1} {self.deck2} -i -c {self.timeout}\n"
        self.socket.sendall(cmd.encode())

        # Wait for first decision
        obs, info = self._wait_for_decision()

        # Record initial state
        if self.current_replay and self.current_state:
            self._record_snapshot()

        info['seed'] = game_seed
        info['replay_id'] = self.current_replay.replay_id if self.current_replay else None
        return obs, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an action in the environment.

        Args:
            action_idx: Index of action in available_actions (or -1 to pass)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {}

        # Track state before action to detect failed actions

        prev_turn = self.current_state.turn if self.current_state else 0
        prev_phase = getattr(self.current_state, 'phase', '') if self.current_state else ''

        # Convert action index to response
        if action_idx < 0 or action_idx >= len(self.available_actions):
            response = "-1"  # Pass
            action = Action(ActionType.PASS, -1)
            action_data = {"type": "pass", "index": -1}
        else:
            action_data = self.available_actions[action_idx]
            response = str(action_data.get('index', -1))
            action = Action(
                ActionType.PLAY_SPELL if action_data.get('index', -1) >= 0 else ActionType.PASS,
                action_data.get('index', -1),
                action_data
            )

        # Record action for replay
        if self.current_replay:
            self._record_action(action_data)

        # Send response
        try:
            self.socket.sendall((response + "\n").encode())
        except (BrokenPipeError, OSError):
            self.game_over = True
            self._finalize_replay()
            return self._get_observation(), -1.0, True, False, {"error": "connection_lost"}

        # Wait for next decision or game end
        obs, info = self._wait_for_decision()

        # Detect if action failed (game didn't progress)
        # If turn/phase didn't change and we didn't just pass, the action likely failed
        current_turn = self.current_state.turn if self.current_state else 0
        current_phase = getattr(self.current_state, 'phase', '') if self.current_state else ''

        if (not self.game_over and
            action_data.get('index', -1) >= 0 and  # Not a pass
            current_turn == prev_turn and
            current_phase == prev_phase):
            # Action failed to progress game - force a pass
            info['action_failed'] = True
            try:
                self.socket.sendall(b"-1\n")  # Send pass
                obs, info = self._wait_for_decision()
                info['forced_pass'] = True
            except (BrokenPipeError, OSError):
                self.game_over = True

        # Record snapshot if enabled
        if self.current_replay and self.current_state:
            if self.decision_count % self.snapshot_interval == 0:
                self._record_snapshot()

        # Check if game ended
        if self.game_over:
            self._finalize_replay()

        # Compute reward (small penalty for failed actions to discourage them)
        reward = self.reward_shaper.compute_reward(
            self.current_state, action, self.game_over, self.won
        ) if self.reward_shaping and self.current_state else 0.0

        if info.get('action_failed'):
            reward -= 0.01  # Small penalty for trying invalid actions

        return obs, reward, self.game_over, False, info

    def _wait_for_decision(self) -> Tuple[np.ndarray, Dict]:
        """Wait for next decision point from the daemon."""
        our_player_name = f"Agent({self.player_id})"

        while True:
            try:
                line = self.reader.readline()
            except socket.timeout:
                self.game_over = True
                return self._get_observation(), {"error": "timeout"}

            if not line:
                self.game_over = True
                return self._get_observation(), {"game_over": True, "won": self.won}

            line = line.strip()

            # Check for game end
            if line.startswith("GAME_RESULT:"):
                self.game_over = True
                self.won = our_player_name.lower() in line.lower() and "won" in line.lower()
                return self._get_observation(), {"game_over": True, "won": self.won, "result": line}

            if line.startswith("GAME_ERROR:") or line.startswith("GAME_TIMEOUT:"):
                self.game_over = True
                return self._get_observation(), {"error": line}

            # Parse decision
            if not line.startswith("DECISION:"):
                continue

            try:
                data = json.loads(line[9:])
            except json.JSONDecodeError:
                continue

            # Only respond to our player's decisions
            player_name = data.get('player', '')
            if our_player_name not in player_name:
                # Pass for opponent
                self.socket.sendall(b"-1\n")
                continue

            # Update state
            self.current_state = self._parse_state(data)
            self.available_actions = data.get('actions', [])
            self.decision_count += 1

            return self._get_observation(), {
                "turn": data.get('turn', 0),
                "phase": data.get('phase', ''),
                "actions": self.available_actions,
                "decision_count": self.decision_count
            }

    def _parse_state(self, data: Dict) -> GameState:
        """Parse decision data into GameState."""
        our_player_name = f"Agent({self.player_id})"
        return GameState.from_dict(data, our_player_name)

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        if self.current_state is None:
            return np.zeros(self.observation_dim, dtype=np.float32)
        return self.current_state.to_observation()

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions (1=valid, 0=invalid)."""
        mask = np.zeros(self.max_actions, dtype=np.float32)
        for i, action in enumerate(self.available_actions[:self.max_actions]):
            mask[i] = 1.0
        return mask

    # =========================================================================
    # Replay Recording Methods
    # =========================================================================

    def _record_action(self, action_data: Dict):
        """Record an action for replay."""
        if not self.current_replay:
            return

        # Determine action type from data
        action_type = action_data.get('type', 'unknown')
        if action_type == 'unknown':
            # Try to infer from action description
            desc = action_data.get('description', '').lower()
            if 'play' in desc and 'land' in desc:
                action_type = 'play_land'
            elif 'cast' in desc:
                action_type = 'cast_spell'
            elif 'attack' in desc:
                action_type = 'attack'
            elif 'block' in desc:
                action_type = 'block'
            elif 'activate' in desc:
                action_type = 'activate_ability'
            elif action_data.get('index', -1) < 0:
                action_type = 'pass'
            else:
                action_type = 'action'

        # Extract card name if present
        card_name = action_data.get('card', action_data.get('description', ''))

        # Get current turn from state
        turn = 0
        if self.current_state:
            turn = getattr(self.current_state, 'turn', 0)

        self.current_replay.turns = turn
        self.current_replay.add_action(action_type, {
            'card': card_name,
            'player': f"Agent({self.player_id})",
            'index': action_data.get('index', -1),
            'description': action_data.get('description', ''),
        })

    def _record_snapshot(self):
        """Record a game state snapshot for replay."""
        if not self.current_replay or not self.current_state:
            return

        state = self.current_state

        # Build snapshot dict from GameState
        snapshot = {
            'turn': getattr(state, 'turn', 0),
            'phase': getattr(state, 'phase', 'Unknown'),
            'active_player': getattr(state, 'active_player', ''),
            'players': []
        }

        # Add player states
        for player_state in [state.our_player, state.opponent]:
            if player_state:
                player_dict = {
                    'name': getattr(player_state, 'name', 'Unknown'),
                    'life': getattr(player_state, 'life', 20),
                    'hand_size': len(getattr(player_state, 'hand', [])),
                    'library_size': getattr(player_state, 'library_size', 0),
                    'hand': [getattr(c, 'name', str(c)) for c in getattr(player_state, 'hand', [])],
                    'battlefield': []
                }

                # Add battlefield cards
                for card in getattr(player_state, 'battlefield', []):
                    card_dict = {
                        'name': getattr(card, 'name', 'Unknown'),
                        'is_creature': getattr(card, 'is_creature', False),
                        'is_land': getattr(card, 'is_land', False),
                        'tapped': getattr(card, 'tapped', False),
                    }
                    if card_dict['is_creature']:
                        card_dict['power'] = getattr(card, 'power', 0)
                        card_dict['toughness'] = getattr(card, 'toughness', 0)
                    player_dict['battlefield'].append(card_dict)

                snapshot['players'].append(player_dict)

        # Add stack
        snapshot['stack'] = []
        for entry in getattr(state, 'stack', []):
            snapshot['stack'].append({
                'description': getattr(entry, 'description', str(entry))
            })

        self.current_replay.add_snapshot(snapshot)

    def _finalize_replay(self):
        """Finalize and save the current replay."""
        if not self.current_replay or not self.replay_recorder:
            return

        # Calculate duration
        duration_ms = (time.time() - self.game_start_time) * 1000

        # Determine winner
        winner = ""
        if self.won:
            winner = f"Agent({self.player_id})"
        elif self.game_over:
            winner = "Opponent"

        self.current_replay.winner = winner
        self.current_replay.duration_ms = duration_ms

        # Save replay
        replay_id = self.replay_recorder.save_replay(self.current_replay)
        self.last_replay_id = replay_id
        self.current_replay = None

    def _cleanup(self):
        """Clean up socket connection."""
        if self.reader:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None

        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def __del__(self):
        self._cleanup()


class DaemonPool:
    """
    Pool of daemon connections for parallel training.

    Manages multiple connections to one or more daemons.
    """

    def __init__(
        self,
        n_envs: int = 8,
        hosts: List[str] = None,
        port: int = 17171,
        deck1: str = "red_aggro.dck",
        deck2: str = "white_weenie.dck",
        record_replays: bool = False,
        replay_dir: str = "replays",
        snapshot_interval: int = 5,  # Snapshot every N decisions (lower = more data)
    ):
        if hosts is None:
            hosts = ['localhost']

        self.n_envs = n_envs
        self.hosts = hosts
        self.port = port
        self.deck1 = deck1
        self.deck2 = deck2
        self.record_replays = record_replays
        self.replay_dir = replay_dir

        self.envs: List[DaemonMTGEnvironment] = []

        # Create environments, distributing across hosts
        for i in range(n_envs):
            host = hosts[i % len(hosts)]
            env = DaemonMTGEnvironment(
                host=host,
                port=port,
                deck1=deck1,
                deck2=deck2,
                record_replays=record_replays,
                replay_dir=replay_dir,
                snapshot_interval=snapshot_interval,
            )
            self.envs.append(env)

    def reset_all(self) -> List[Tuple[np.ndarray, Dict]]:
        """Reset all environments."""
        results = []
        for env in self.envs:
            try:
                result = env.reset()
            except Exception as e:
                print(f"Warning: Failed to reset env: {e}")
                result = (np.zeros(38, dtype=np.float32), {"error": str(e)})
            results.append(result)
        return results

    def step_all(self, actions: List[int]) -> List[Tuple[np.ndarray, float, bool, bool, Dict]]:
        """Step all environments with given actions."""
        results = []
        for env, action in zip(self.envs, actions):
            try:
                result = env.step(action)
            except Exception as e:
                result = (np.zeros(38, dtype=np.float32), -1.0, True, False, {"error": str(e)})
            results.append(result)
        return results

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env._cleanup()


def check_daemon_status(host: str = 'localhost', port: int = 17171) -> Dict:
    """Check if daemon is running and get its status."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        sock.sendall(b"STATUS\n")

        response = []
        reader = sock.makefile('r')
        while True:
            line = reader.readline()
            if not line:
                break
            response.append(line.strip())

        sock.close()
        return {"status": "running", "info": response}
    except Exception as e:
        return {"status": "not_running", "error": str(e)}


def benchmark_daemon(
    host: str = 'localhost',
    port: int = 17171,
    n_games: int = 10,
    deck1: str = "decks/red_aggro.dck",
    deck2: str = "decks/white_weenie.dck",
    record_replays: bool = False,
    replay_dir: str = "replays",
) -> Dict:
    """Benchmark daemon performance with optional replay recording."""
    print(f"Benchmarking Forge daemon at {host}:{port}")
    print(f"Playing {n_games} games...")
    if record_replays:
        print(f"Recording replays to: {replay_dir}/")

    times = []
    wins = 0
    replay_ids = []

    for i in range(n_games):
        env = DaemonMTGEnvironment(
            host=host, port=port,
            deck1=deck1, deck2=deck2,
            record_replays=record_replays,
            replay_dir=replay_dir,
            snapshot_interval=5,  # Snapshot every 5 decisions
        )

        start = time.perf_counter()
        try:
            obs, info = env.reset()
            decisions = 0

            while not env.game_over and decisions < 1000:
                action_mask = env.get_action_mask()
                # Random valid action
                valid_actions = np.where(action_mask > 0)[0]
                action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else -1

                obs, reward, done, truncated, info = env.step(action)
                decisions += 1

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Finalize replay if truncated (hit decision limit before game end)
            if record_replays and not env.game_over:
                env._finalize_replay()

            if env.won:
                wins += 1

            replay_info = f" [replay: {env.last_replay_id}]" if env.last_replay_id else ""
            print(f"  Game {i+1}: {elapsed:.2f}s, {decisions} decisions, {'Won' if env.won else 'Lost'}{replay_info}")

            if env.last_replay_id:
                replay_ids.append(env.last_replay_id)

        except Exception as e:
            print(f"  Game {i+1}: Error - {e}")
        finally:
            env._cleanup()

    if times:
        avg_time = sum(times) / len(times)
        games_per_hour = 3600 / avg_time if avg_time > 0 else 0

        print("\nResults:")
        print(f"  Average game time: {avg_time:.2f}s")
        print(f"  Games per hour: {games_per_hour:.0f}")
        print(f"  Win rate: {wins/len(times)*100:.1f}%")

        if replay_ids:
            print(f"\nReplays saved: {len(replay_ids)}")
            print("  Generate video: python replay_cli.py video 0")
            print("  List replays:   python replay_cli.py list")

        return {
            "avg_time": avg_time,
            "games_per_hour": games_per_hour,
            "win_rate": wins / len(times),
            "replay_ids": replay_ids,
        }

    return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Forge daemon environment",
        epilog="""
Examples:
    %(prog)s --mode status                     # Check daemon status
    %(prog)s --mode benchmark --games 10       # Benchmark without replays
    %(prog)s --mode benchmark --games 5 --record-replays  # Benchmark with replays
    %(prog)s --mode test --record-replays      # Test with replay recording
        """
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=17171)
    parser.add_argument("--mode", choices=["status", "benchmark", "test"], default="status")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--record-replays", action="store_true",
                       help="Record game replays for video generation")
    parser.add_argument("--replay-dir", type=str, default="replays",
                       help="Directory to store replays")

    args = parser.parse_args()

    if args.mode == "status":
        status = check_daemon_status(args.host, args.port)
        print(f"Daemon status: {status['status']}")
        if 'info' in status:
            for line in status['info']:
                print(f"  {line}")
        if 'error' in status:
            print(f"  Error: {status['error']}")

    elif args.mode == "benchmark":
        benchmark_daemon(
            args.host, args.port, args.games,
            record_replays=args.record_replays,
            replay_dir=args.replay_dir
        )

    elif args.mode == "test":
        print("Testing daemon environment...")
        if args.record_replays:
            print(f"Recording replays to: {args.replay_dir}/")

        env = DaemonMTGEnvironment(
            host=args.host, port=args.port,
            record_replays=args.record_replays,
            replay_dir=args.replay_dir,
        )
        try:
            obs, info = env.reset()
            print(f"Reset successful. Observation shape: {obs.shape}")
            print(f"Available actions: {len(env.available_actions)}")
            if args.record_replays:
                print(f"Replay ID: {info.get('replay_id')}")

            # Play a few moves
            for i in range(5):
                action_mask = env.get_action_mask()
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) == 0:
                    break
                action = int(valid_actions[0])  # Take first valid action
                obs, reward, done, truncated, info = env.step(action)
                print(f"Step {i+1}: reward={reward:.4f}, done={done}")
                if done:
                    break

            print("Test successful!")
            if env.last_replay_id:
                print(f"\nReplay saved: {env.last_replay_id}")
                print(f"Generate video: python replay_cli.py video 0 -d {args.replay_dir}")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env._cleanup()
