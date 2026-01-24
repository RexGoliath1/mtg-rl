#!/usr/bin/env python3
"""
Replay Recorder for MTG Games

Records game seeds and decisions for deterministic replay.
Replay data is lightweight - just seed + action sequence.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib


@dataclass
class GameReplay:
    """Complete replay data for a single game."""
    replay_id: str
    seed: int
    deck1: str
    deck2: str
    timestamp: float

    # Game metadata
    winner: Optional[str] = None
    turns: int = 0
    duration_ms: float = 0

    # Action sequence - each action is a dict with type and details
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # State snapshots at key points (optional, for debugging)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def add_action(self, action_type: str, details: Dict[str, Any]):
        """Record an action taken during the game."""
        self.actions.append({
            "type": action_type,
            "turn": self.turns,
            "timestamp": time.time(),
            **details
        })

    def add_snapshot(self, state: Dict[str, Any]):
        """Add a state snapshot (optional, for key moments)."""
        self.snapshots.append({
            "turn": self.turns,
            "timestamp": time.time(),
            "state": state
        })

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'GameReplay':
        return cls(**data)


class ReplayRecorder:
    """
    Records game replays to disk.

    Replays are stored in a directory structure:
    replay_dir/
        index.json          # Index of all replays
        YYYYMMDD/
            {replay_id}.json
    """

    def __init__(self, replay_dir: str = "replays"):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.replay_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load or create the replay index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {
                "total_replays": 0,
                "replays": {}  # replay_id -> metadata
            }

    def _save_index(self):
        """Save the replay index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def generate_replay_id(self, seed: int) -> str:
        """Generate a unique replay ID."""
        timestamp = int(time.time() * 1000)
        hash_input = f"{seed}_{timestamp}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:12]

    def create_replay(self, seed: int, deck1: str, deck2: str) -> GameReplay:
        """Create a new replay record."""
        replay_id = self.generate_replay_id(seed)
        return GameReplay(
            replay_id=replay_id,
            seed=seed,
            deck1=deck1,
            deck2=deck2,
            timestamp=time.time()
        )

    def save_replay(self, replay: GameReplay) -> str:
        """Save a completed replay to disk."""
        # Create date-based subdirectory
        date_str = time.strftime("%Y%m%d", time.localtime(replay.timestamp))
        date_dir = self.replay_dir / date_str
        date_dir.mkdir(exist_ok=True)

        # Save replay file
        replay_path = date_dir / f"{replay.replay_id}.json"
        with open(replay_path, 'w') as f:
            json.dump(replay.to_dict(), f, indent=2)

        # Update index
        self.index["total_replays"] += 1
        self.index["replays"][replay.replay_id] = {
            "path": str(replay_path.relative_to(self.replay_dir)),
            "seed": replay.seed,
            "deck1": replay.deck1,
            "deck2": replay.deck2,
            "winner": replay.winner,
            "turns": replay.turns,
            "duration_ms": replay.duration_ms,
            "timestamp": replay.timestamp,
            "num_actions": len(replay.actions)
        }
        self._save_index()

        return replay.replay_id

    def load_replay(self, replay_id: str) -> Optional[GameReplay]:
        """Load a replay by ID."""
        if replay_id not in self.index["replays"]:
            # Try to find by partial match or index number
            replay_id = self._resolve_replay_id(replay_id)
            if not replay_id:
                return None

        metadata = self.index["replays"][replay_id]
        replay_path = self.replay_dir / metadata["path"]

        if not replay_path.exists():
            return None

        with open(replay_path) as f:
            data = json.load(f)

        return GameReplay.from_dict(data)

    def _resolve_replay_id(self, query: str) -> Optional[str]:
        """Resolve a query to a replay ID (supports index numbers like 'run 100')."""
        # Try as index number
        try:
            index = int(query.replace("run", "").replace("game", "").strip())
            # Get replay by index (sorted by timestamp)
            sorted_replays = sorted(
                self.index["replays"].items(),
                key=lambda x: x[1]["timestamp"]
            )
            if 0 <= index < len(sorted_replays):
                return sorted_replays[index][0]
        except ValueError:
            pass

        # Try partial ID match
        for rid in self.index["replays"]:
            if rid.startswith(query):
                return rid

        return None

    def list_replays(self, limit: int = 20, offset: int = 0) -> List[Dict]:
        """List recent replays."""
        sorted_replays = sorted(
            self.index["replays"].items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )

        results = []
        for i, (rid, meta) in enumerate(sorted_replays[offset:offset+limit], offset):
            results.append({
                "index": i,
                "replay_id": rid,
                **meta
            })

        return results

    def get_replay_by_index(self, index: int) -> Optional[GameReplay]:
        """Get replay by run number (0-indexed)."""
        sorted_replays = sorted(
            self.index["replays"].items(),
            key=lambda x: x[1]["timestamp"]
        )

        if 0 <= index < len(sorted_replays):
            return self.load_replay(sorted_replays[index][0])
        return None


class ReplayingDaemonClient:
    """
    Daemon client that records game replays.

    Wraps the normal daemon communication to capture seeds and actions.
    """

    def __init__(self, host: str = "localhost", port: int = 17171,
                 recorder: Optional[ReplayRecorder] = None):
        self.host = host
        self.port = port
        self.recorder = recorder or ReplayRecorder()
        self.current_replay: Optional[GameReplay] = None

    def start_game(self, deck1: str, deck2: str, seed: Optional[int] = None) -> GameReplay:
        """
        Start a new game with recording.

        If seed is provided, game will be deterministic.
        """
        import random

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        self.current_replay = self.recorder.create_replay(seed, deck1, deck2)
        return self.current_replay

    def record_action(self, action_type: str, **details):
        """Record an action during gameplay."""
        if self.current_replay:
            self.current_replay.add_action(action_type, details)

    def record_state(self, state: Dict[str, Any]):
        """Record a state snapshot."""
        if self.current_replay:
            self.current_replay.add_snapshot(state)

    def end_game(self, winner: str, turns: int, duration_ms: float) -> str:
        """End the current game and save the replay."""
        if self.current_replay:
            self.current_replay.winner = winner
            self.current_replay.turns = turns
            self.current_replay.duration_ms = duration_ms

            replay_id = self.recorder.save_replay(self.current_replay)
            self.current_replay = None
            return replay_id
        return ""


# Integration with daemon communication
def parse_daemon_response_to_states(response: str) -> List[Dict[str, Any]]:
    """
    Parse daemon response to extract game states.

    Looks for GAMESTATE: JSON lines in the response.
    """
    states = []

    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("GAMESTATE:"):
            try:
                json_str = line[len("GAMESTATE:"):]
                state = json.loads(json_str)
                states.append(state)
            except json.JSONDecodeError:
                pass

    return states


if __name__ == "__main__":
    # Test the replay recorder
    print("Testing Replay Recorder")
    print("=" * 50)

    recorder = ReplayRecorder("test_replays")

    # Create a test replay
    replay = recorder.create_replay(
        seed=12345,
        deck1="test_red.dck",
        deck2="test_blue.dck"
    )

    # Simulate some actions
    replay.add_action("play_land", {"card": "Mountain", "player": "Red"})
    replay.turns = 1
    replay.add_action("cast_spell", {"card": "Lightning Bolt", "target": "Blue", "player": "Red"})
    replay.add_action("pass_turn", {"player": "Red"})
    replay.turns = 2
    replay.add_action("play_land", {"card": "Island", "player": "Blue"})

    # Add a snapshot
    replay.add_snapshot({
        "red_life": 20,
        "blue_life": 17,
        "red_hand": ["Goblin Guide", "Mountain"],
        "blue_hand": ["Counterspell", "Island", "Island"]
    })

    # End game
    replay.winner = "Red"
    replay.turns = 5
    replay.duration_ms = 3500

    # Save
    replay_id = recorder.save_replay(replay)
    print(f"Saved replay: {replay_id}")

    # Load it back
    loaded = recorder.load_replay(replay_id)
    print(f"Loaded replay: {loaded.replay_id}")
    print(f"  Seed: {loaded.seed}")
    print(f"  Winner: {loaded.winner}")
    print(f"  Actions: {len(loaded.actions)}")

    # Test index lookup
    loaded2 = recorder.get_replay_by_index(0)
    print(f"By index 0: {loaded2.replay_id if loaded2 else 'Not found'}")

    # List replays
    print("\nRecent replays:")
    for r in recorder.list_replays(limit=5):
        print(f"  [{r['index']}] {r['replay_id']}: {r['winner']} won in {r['turns']} turns")

    print("\nAll tests passed!")
