#!/usr/bin/env python3
"""
Draft Environment for MTG RL Training

Connects to the Forge Draft Daemon to simulate drafts.
One agent drafts against 7 AI opponents.

Usage:
    # Start the draft daemon first:
    # cd forge-repo/forge-gui-desktop && mvn exec:java -Dexec.args="draft -p 17272"

    # Then run drafts:
    python draft_environment.py --mode test
    python draft_environment.py --mode benchmark --drafts 10
"""

import socket
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import argparse
import time


@dataclass
class DraftState:
    """Represents the current state of a draft."""
    pack_num: int = 1
    pick_num: int = 1
    cards_in_pack: List[Dict] = field(default_factory=list)
    pool: List[str] = field(default_factory=list)
    complete: bool = False
    set_code: str = ""


@dataclass
class Card:
    """Represents a card in the draft."""
    index: int
    name: str
    mana_cost: str
    card_type: str
    rarity: str
    set_code: str
    power: Optional[int] = None
    toughness: Optional[int] = None
    oracle_text: str = ""
    colors: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> 'Card':
        return cls(
            index=data.get('index', 0),
            name=data.get('name', ''),
            mana_cost=data.get('mana_cost', ''),
            card_type=data.get('type', ''),
            rarity=data.get('rarity', ''),
            set_code=data.get('set', ''),
            power=data.get('power'),
            toughness=data.get('toughness'),
            oracle_text=data.get('oracle_text', ''),
            colors=data.get('colors', ''),
        )

    def cmc(self) -> int:
        """Calculate converted mana cost from mana_cost string."""
        if not self.mana_cost:
            return 0
        cmc = 0
        i = 0
        while i < len(self.mana_cost):
            c = self.mana_cost[i]
            if c == '{':
                # Find closing brace
                end = self.mana_cost.find('}', i)
                if end != -1:
                    symbol = self.mana_cost[i+1:end]
                    if symbol.isdigit():
                        cmc += int(symbol)
                    elif symbol in ['W', 'U', 'B', 'R', 'G']:
                        cmc += 1
                    elif '/' in symbol:  # Hybrid mana
                        cmc += 1
                    elif symbol == 'X':
                        pass  # X costs 0 for CMC calculation
                    else:
                        cmc += 1  # Other symbols count as 1
                    i = end + 1
                    continue
            i += 1
        return cmc


class DraftEnvironment:
    """
    Gymnasium-style environment for MTG draft.

    Observation space: Current pack contents + drafted pool
    Action space: Index of card to pick (0 to len(pack)-1)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 17272,
        set_code: str = "NEO",
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.default_set = set_code
        self.timeout = timeout

        self.socket: Optional[socket.socket] = None
        self.state: Optional[DraftState] = None
        self.current_pack: List[Card] = []

    def connect(self) -> bool:
        """Connect to the draft daemon."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            return True
        except (socket.error, socket.timeout) as e:
            print(f"Failed to connect to draft daemon: {e}")
            return False

    def disconnect(self):
        """Disconnect from the daemon."""
        if self.socket:
            try:
                self.socket.sendall(b"QUIT\n")
                self.socket.close()
            except:
                pass
            self.socket = None

    def _send(self, message: str) -> str:
        """Send a message and receive response."""
        if not self.socket:
            raise RuntimeError("Not connected to daemon")

        self.socket.sendall((message + "\n").encode())

        # Read response (may be multiple lines)
        response_lines = []
        buffer = ""

        while True:
            try:
                data = self.socket.recv(4096).decode()
                if not data:
                    break
                buffer += data

                # Check for complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        response_lines.append(line)

                    # Check if we have a complete response
                    # For NEWDRAFT: we get DRAFT_STARTED then PACK
                    # For picks: we get PICKED then PACK (or DRAFT_COMPLETE)
                    if line.startswith("DRAFT_STARTED "):
                        # Continue reading to get PACK
                        continue
                    if line.startswith("PICKED "):
                        # Continue reading to get next PACK or DRAFT_COMPLETE
                        continue
                    if any(line.startswith(prefix) for prefix in
                           ["PACK ", "DRAFT_COMPLETE ", "ERROR",
                            "STATUS ", "POOL ", "GOODBYE"]):
                        return "\n".join(response_lines)

            except socket.timeout:
                break

        return "\n".join(response_lines)

    def reset(self, set_code: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Start a new draft.

        Args:
            set_code: MTG set code (e.g., "NEO", "MKM", "BRO")

        Returns:
            observation: Dict with current pack and pool
            info: Dict with additional info
        """
        if not self.socket:
            if not self.connect():
                raise RuntimeError("Could not connect to draft daemon")

        set_code = set_code or self.default_set

        # Start new draft
        response = self._send(f"NEWDRAFT {set_code}")

        # Parse response
        lines = response.split("\n")
        draft_info = None
        pack_data = None

        for line in lines:
            if line.startswith("DRAFT_STARTED "):
                draft_info = json.loads(line[len("DRAFT_STARTED "):])
            elif line.startswith("PACK "):
                pack_data = json.loads(line[len("PACK "):])
            elif line.startswith("ERROR"):
                raise RuntimeError(f"Draft error: {line}")

        if not pack_data:
            raise RuntimeError(f"No pack data received: {response}")

        # Initialize state
        self.state = DraftState(
            pack_num=pack_data.get('pack_num', 1),
            pick_num=pack_data.get('pick_num', 1),
            cards_in_pack=pack_data.get('cards', []),
            pool=pack_data.get('pool', []),
            complete=False,
            set_code=set_code,
        )

        self.current_pack = [Card.from_dict(c) for c in self.state.cards_in_pack]

        obs = self._get_observation()
        info = {
            'set': set_code,
            'pack_num': self.state.pack_num,
            'pick_num': self.state.pick_num,
            'cards_in_pack': len(self.current_pack),
            'pool_size': len(self.state.pool),
        }

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Make a pick.

        Args:
            action: Index of card to pick (0 to len(pack)-1)

        Returns:
            observation: Dict with new pack and updated pool
            reward: 0.0 (rewards come from playing games with drafted deck)
            terminated: True if draft is complete
            truncated: False (drafts don't get truncated)
            info: Dict with pick info
        """
        if not self.state or self.state.complete:
            return self._get_observation(), 0.0, True, False, {'error': 'Draft complete'}

        if action < 0 or action >= len(self.current_pack):
            return self._get_observation(), -0.1, False, False, {'error': f'Invalid action {action}'}

        picked_card = self.current_pack[action]

        # Send pick
        response = self._send(str(action))

        # Parse response
        lines = response.split("\n")
        pick_result = None
        pack_data = None
        draft_complete = False

        for line in lines:
            if line.startswith("PICKED "):
                pick_result = json.loads(line[len("PICKED "):])
            elif line.startswith("PACK "):
                pack_data = json.loads(line[len("PACK "):])
            elif line.startswith("DRAFT_COMPLETE "):
                draft_complete = True
                complete_data = json.loads(line[len("DRAFT_COMPLETE "):])
                self.state.pool = complete_data.get('pool', [])
            elif line.startswith("ERROR"):
                return self._get_observation(), -0.1, False, False, {'error': line}

        # Update state
        if draft_complete:
            self.state.complete = True
            self.current_pack = []
        elif pack_data:
            self.state.pack_num = pack_data.get('pack_num', self.state.pack_num)
            self.state.pick_num = pack_data.get('pick_num', self.state.pick_num)
            self.state.cards_in_pack = pack_data.get('cards', [])
            self.state.pool = pack_data.get('pool', [])
            self.current_pack = [Card.from_dict(c) for c in self.state.cards_in_pack]

        obs = self._get_observation()
        info = {
            'picked': picked_card.name,
            'pack_num': self.state.pack_num,
            'pick_num': self.state.pick_num,
            'cards_in_pack': len(self.current_pack),
            'pool_size': len(self.state.pool),
            'complete': self.state.complete,
        }

        return obs, 0.0, self.state.complete, False, info

    def _get_observation(self) -> Dict:
        """Get current observation."""
        return {
            'pack': [self._card_to_features(c) for c in self.current_pack],
            'pool': list(self.state.pool) if self.state else [],
            'pack_num': self.state.pack_num if self.state else 0,
            'pick_num': self.state.pick_num if self.state else 0,
            'complete': self.state.complete if self.state else True,
        }

    def _card_to_features(self, card: Card) -> Dict:
        """Convert card to feature dict for ML."""
        return {
            'index': card.index,
            'name': card.name,
            'cmc': card.cmc(),
            'mana_cost': card.mana_cost,
            'type': card.card_type,
            'rarity': card.rarity,
            'power': card.power,
            'toughness': card.toughness,
            'colors': card.colors,
            'oracle_text': card.oracle_text,
        }

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions (all cards in pack are valid)."""
        mask = np.zeros(15, dtype=np.float32)  # Max pack size
        for i in range(len(self.current_pack)):
            mask[i] = 1.0
        return mask

    def get_pool(self) -> List[str]:
        """Get list of cards in drafted pool."""
        return list(self.state.pool) if self.state else []

    def is_complete(self) -> bool:
        """Check if draft is complete."""
        return self.state.complete if self.state else True


def check_draft_daemon(host: str = "localhost", port: int = 17272) -> Dict:
    """Check if draft daemon is running."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((host, port))
        sock.sendall(b"STATUS\n")
        response = sock.recv(1024).decode()
        sock.close()
        return {'status': 'running', 'response': response}
    except Exception as e:
        return {'status': 'not_running', 'error': str(e)}


def test_draft():
    """Test a single draft with random picks."""
    print("Testing draft environment...")

    env = DraftEnvironment()

    try:
        obs, info = env.reset("NEO")
        print(f"\nDraft started: {info}")
        print(f"First pack ({len(obs['pack'])} cards):")

        for card in obs['pack'][:5]:
            print(f"  [{card['index']}] {card['name']} - {card['mana_cost']} ({card['rarity']})")
        if len(obs['pack']) > 5:
            print(f"  ... and {len(obs['pack']) - 5} more")

        # Make picks (random for testing)
        pick_count = 0
        while not env.is_complete():
            # Random pick
            action = np.random.randint(0, len(env.current_pack))
            obs, reward, done, truncated, info = env.step(action)

            pick_count += 1
            print(f"Pick {pick_count}: {info.get('picked', '?')} "
                  f"(Pack {info['pack_num']}, Pick {info['pick_num']})")

            if done:
                break

        print(f"\nDraft complete! Pool size: {len(env.get_pool())}")
        print("Pool:")
        for card in env.get_pool():
            print(f"  - {card}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.disconnect()


def benchmark_drafts(n_drafts: int = 10):
    """Benchmark draft speed."""
    print(f"Benchmarking {n_drafts} drafts...")

    env = DraftEnvironment()
    times = []
    picks_list = []

    for i in range(n_drafts):
        start = time.time()

        try:
            obs, info = env.reset()
            picks = 0

            while not env.is_complete():
                action = np.random.randint(0, len(env.current_pack))
                obs, reward, done, truncated, info = env.step(action)
                picks += 1
                if done:
                    break

            elapsed = time.time() - start
            times.append(elapsed)
            picks_list.append(picks)

            print(f"  Draft {i+1}: {elapsed:.2f}s, {picks} picks")

        except Exception as e:
            print(f"  Draft {i+1}: ERROR - {e}")

    env.disconnect()

    if times:
        print(f"\nResults:")
        print(f"  Average time: {np.mean(times):.2f}s")
        print(f"  Average picks: {np.mean(picks_list):.1f}")
        print(f"  Drafts per hour: {3600 / np.mean(times):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Draft Environment")
    parser.add_argument("--mode", choices=["test", "benchmark", "status"],
                        default="status", help="Mode to run")
    parser.add_argument("--host", default="localhost", help="Daemon host")
    parser.add_argument("--port", type=int, default=17272, help="Daemon port")
    parser.add_argument("--drafts", type=int, default=10, help="Number of drafts for benchmark")
    parser.add_argument("--set", default="NEO", help="Set code for draft")

    args = parser.parse_args()

    if args.mode == "status":
        status = check_draft_daemon(args.host, args.port)
        print(f"Draft daemon status: {status['status']}")
        if status['status'] == 'running':
            print(f"Response: {status.get('response', 'N/A')}")
        else:
            print(f"Error: {status.get('error', 'Unknown')}")
            print("\nTo start the draft daemon:")
            print("  cd forge-repo/forge-gui-desktop")
            print("  mvn exec:java -Dexec.args=\"draft -p 17272\"")

    elif args.mode == "test":
        test_draft()

    elif args.mode == "benchmark":
        benchmark_drafts(args.drafts)


if __name__ == "__main__":
    main()
