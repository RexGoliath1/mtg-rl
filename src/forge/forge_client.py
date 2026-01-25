"""
Forge Game Client

Python client that connects to the Forge daemon for RL training.
Communicates via TCP socket with JSON protocol.

Protocol:
- Connect to TCP port 17171
- Send: NEWGAME deck1.dck deck2.dck -i
- Receive: DECISION:{"decision_type":"...", "game_state":{...}, ...}
- Send: Response (integer index, comma-separated list, etc.)
- Repeat until GAME_RESULT: ...
"""

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class DecisionType(Enum):
    """Types of decisions the agent can make."""
    CHOOSE_ACTION = "choose_action"
    DECLARE_ATTACKERS = "declare_attackers"
    DECLARE_BLOCKERS = "declare_blockers"
    CHOOSE_ABILITY = "choose_ability"
    PLAY_TRIGGER = "play_trigger"
    CONFIRM_ACTION = "confirm_action"
    CHOOSE_CARDS = "choose_cards"
    CHOOSE_ENTITY = "choose_entity"
    ANNOUNCE_VALUE = "announce_value"
    PLAY_FROM_EFFECT = "play_from_effect"
    REVEAL = "reveal"


@dataclass
class ManaPool:
    """Current mana available to a player."""
    total: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0
    colorless: int = 0

    @classmethod
    def from_json(cls, data: dict) -> "ManaPool":
        return cls(
            total=data.get("total", 0),
            white=data.get("white", 0),
            blue=data.get("blue", 0),
            black=data.get("black", 0),
            red=data.get("red", 0),
            green=data.get("green", 0),
            colorless=data.get("colorless", 0),
        )


@dataclass
class CardInfo:
    """Information about a card."""
    id: int
    name: str
    mana_cost: str = ""
    cmc: int = 0
    types: str = ""
    oracle_text: str = ""
    power: Optional[int] = None
    toughness: Optional[int] = None
    keywords: list[str] = field(default_factory=list)
    # Battlefield-specific
    tapped: bool = False
    summoning_sick: bool = False
    damage: int = 0
    loyalty: Optional[int] = None
    counters: str = ""
    is_creature: bool = False
    is_land: bool = False
    is_artifact: bool = False
    is_enchantment: bool = False
    is_planeswalker: bool = False

    @classmethod
    def from_json(cls, data: dict) -> "CardInfo":
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            mana_cost=data.get("mana_cost", ""),
            cmc=data.get("cmc", 0),
            types=data.get("types", ""),
            oracle_text=data.get("oracle_text", ""),
            power=data.get("power"),
            toughness=data.get("toughness"),
            keywords=data.get("keywords", []),
            tapped=data.get("tapped", False),
            summoning_sick=data.get("summoning_sick", False),
            damage=data.get("damage", 0),
            loyalty=data.get("loyalty"),
            counters=data.get("counters", ""),
            is_creature=data.get("is_creature", False),
            is_land=data.get("is_land", False),
            is_artifact=data.get("is_artifact", False),
            is_enchantment=data.get("is_enchantment", False),
            is_planeswalker=data.get("is_planeswalker", False),
        )


@dataclass
class PlayerState:
    """State of a player in the game."""
    name: str
    life: int
    poison: int = 0
    has_lost: bool = False
    lands_played_this_turn: int = 0
    max_land_plays: int = 1
    hand_size: int = 0
    hand: list[CardInfo] = field(default_factory=list)
    library_size: int = 0
    graveyard: list[CardInfo] = field(default_factory=list)
    battlefield: list[CardInfo] = field(default_factory=list)
    exile: list[CardInfo] = field(default_factory=list)
    mana_pool: ManaPool = field(default_factory=ManaPool)

    @classmethod
    def from_json(cls, data: dict) -> "PlayerState":
        return cls(
            name=data.get("name", ""),
            life=data.get("life", 20),
            poison=data.get("poison", 0),
            has_lost=data.get("has_lost", False),
            lands_played_this_turn=data.get("lands_played_this_turn", 0),
            max_land_plays=data.get("max_land_plays", 1),
            hand_size=data.get("hand_size", 0),
            hand=[CardInfo.from_json(c) for c in data.get("hand", [])],
            library_size=data.get("library_size", 0),
            graveyard=[CardInfo.from_json(c) for c in data.get("graveyard", [])],
            battlefield=[CardInfo.from_json(c) for c in data.get("battlefield", [])],
            exile=[CardInfo.from_json(c) for c in data.get("exile", [])],
            mana_pool=ManaPool.from_json(data.get("mana_pool", {})),
        )


@dataclass
class StackItem:
    """An item on the stack."""
    id: int
    description: str
    controller: str
    source_card: Optional[str] = None
    source_card_id: Optional[int] = None

    @classmethod
    def from_json(cls, data: dict) -> "StackItem":
        return cls(
            id=data.get("id", 0),
            description=data.get("description", ""),
            controller=data.get("controller", ""),
            source_card=data.get("source_card"),
            source_card_id=data.get("source_card_id"),
        )


@dataclass
class CombatState:
    """State of combat if in combat phase."""
    attacking_player: str = ""
    attackers: list[dict] = field(default_factory=list)  # Keep raw for flexibility

    @classmethod
    def from_json(cls, data: dict) -> "CombatState":
        return cls(
            attacking_player=data.get("attacking_player", ""),
            attackers=data.get("attackers", []),
        )


@dataclass
class GameState:
    """Complete game state from Forge."""
    is_game_over: bool
    active_player: str
    priority_player: str
    players: list[PlayerState]
    stack: list[StackItem]
    combat: Optional[CombatState] = None
    turn: int = 1
    phase: str = "MAIN1"

    @classmethod
    def from_json(cls, data: dict) -> "GameState":
        combat = None
        if "combat" in data:
            combat = CombatState.from_json(data["combat"])

        return cls(
            is_game_over=data.get("is_game_over", False),
            active_player=data.get("active_player", ""),
            priority_player=data.get("priority_player", ""),
            players=[PlayerState.from_json(p) for p in data.get("players", [])],
            stack=[StackItem.from_json(s) for s in data.get("stack", [])],
            combat=combat,
            turn=data.get("turn", 1),
            phase=data.get("phase", "MAIN1"),
        )

    def get_player(self, name: str) -> Optional[PlayerState]:
        """Get player by name."""
        for p in self.players:
            if p.name == name:
                return p
        return None

    def get_our_player(self, our_name: str) -> Optional[PlayerState]:
        """Get our player's state."""
        return self.get_player(our_name)

    def get_opponent(self, our_name: str) -> Optional[PlayerState]:
        """Get opponent's state."""
        for p in self.players:
            if p.name != our_name:
                return p
        return None


@dataclass
class ActionOption:
    """A single action option."""
    index: int
    description: str
    card: str = ""
    card_id: int = -1
    mana_cost: str = ""
    is_land: bool = False

    @classmethod
    def from_json(cls, data: dict) -> "ActionOption":
        return cls(
            index=data.get("index", 0),
            description=data.get("description", ""),
            card=data.get("card", ""),
            card_id=data.get("card_id", -1),
            mana_cost=data.get("mana_cost", ""),
            is_land=data.get("is_land", False),
        )


@dataclass
class Decision:
    """A decision request from Forge."""
    decision_type: DecisionType
    decision_id: int
    player: str
    turn: int
    phase: str
    game_state: GameState
    message: str = ""
    # Type-specific data
    actions: list[ActionOption] = field(default_factory=list)  # For choose_action
    attackers: list[dict] = field(default_factory=list)  # For declare_attackers
    blockers: list[dict] = field(default_factory=list)  # For declare_blockers
    cards: list[dict] = field(default_factory=list)  # For choose_cards
    raw_data: dict = field(default_factory=dict)  # Original JSON

    @classmethod
    def from_json(cls, data: dict) -> "Decision":
        decision_type = DecisionType(data.get("decision_type", "choose_action"))

        actions = []
        if "actions" in data:
            actions = [ActionOption.from_json(a) for a in data["actions"]]

        return cls(
            decision_type=decision_type,
            decision_id=data.get("decision_id", 0),
            player=data.get("player", ""),
            turn=data.get("turn", 1),
            phase=data.get("phase", ""),
            game_state=GameState.from_json(data.get("game_state", {})),
            message=data.get("message", ""),
            actions=actions,
            attackers=data.get("attackers", []),
            blockers=data.get("blockers", []),
            cards=data.get("cards", []),
            raw_data=data,
        )


@dataclass
class GameResult:
    """Result of a completed game."""
    winner: Optional[str]
    is_draw: bool
    duration_ms: int
    reason: str = ""

    @classmethod
    def from_line(cls, line: str) -> "GameResult":
        """Parse GAME_RESULT: line."""
        # Format: "GAME_RESULT: Winner won in Xms" or "GAME_RESULT: Draw in Xms"
        parts = line.replace("GAME_RESULT:", "").strip()

        if parts.startswith("Draw"):
            is_draw = True
            winner = None
            # Extract time
            if " in " in parts:
                time_part = parts.split(" in ")[1]
                duration_ms = int(time_part.replace("ms", ""))
            else:
                duration_ms = 0
        else:
            is_draw = False
            # Extract winner and time
            if " won in " in parts:
                winner = parts.split(" won in ")[0]
                time_part = parts.split(" won in ")[1]
                duration_ms = int(time_part.replace("ms", ""))
            else:
                winner = parts
                duration_ms = 0

        return cls(winner=winner, is_draw=is_draw, duration_ms=duration_ms)


class ForgeClient:
    """
    Client for connecting to Forge daemon.

    Usage:
        client = ForgeClient("localhost", 17171)
        client.connect()
        client.start_game("deck1.dck", "deck2.dck")

        while True:
            decision = client.receive_decision()
            if decision is None:
                break
            response = agent.decide(decision)
            client.send_response(response)

        result = client.get_result()
        client.close()
    """

    def __init__(self, host: str = "localhost", port: int = 17171, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.reader: Optional[Any] = None
        self.writer: Optional[Any] = None
        self.our_player_name: Optional[str] = None
        self._game_result: Optional[GameResult] = None
        self._connected = False

    def connect(self):
        """Connect to the Forge daemon."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect((self.host, self.port))
        self.reader = self.socket.makefile("r", encoding="utf-8")
        self.writer = self.socket.makefile("w", encoding="utf-8")
        self._connected = True

    def close(self):
        """Close the connection."""
        if self.reader:
            self.reader.close()
        if self.writer:
            self.writer.close()
        if self.socket:
            self.socket.close()
        self._connected = False

    def start_game(
        self,
        deck1: str,
        deck2: str,
        timeout: int = 120,
        seed: Optional[int] = None,
        quiet: bool = True,
    ) -> bool:
        """
        Start a new game.

        Args:
            deck1: Path or name of first deck
            deck2: Path or name of second deck
            timeout: Game timeout in seconds
            seed: Random seed for reproducibility
            quiet: Suppress verbose output

        Returns:
            True if game started successfully
        """
        if not self._connected:
            raise RuntimeError("Not connected to Forge daemon")

        cmd = f"NEWGAME {deck1} {deck2} -i"
        if quiet:
            cmd += " -q"
        if timeout:
            cmd += f" -c {timeout}"
        if seed is not None:
            cmd += f" -s {seed}"

        self._send_line(cmd)

        # Read until we get GAME_START or ERROR
        while True:
            line = self._read_line()
            if line is None:
                return False
            if line.startswith("ERROR:"):
                print(f"Game start error: {line}")
                return False
            if line.startswith("GAME_START:"):
                return True
            if line.startswith("DECISION:"):
                # First decision - extract player name and reprocess
                json_str = line[9:]  # Remove "DECISION:" prefix
                data = json.loads(json_str)
                self.our_player_name = data.get("player", "")
                # Put this decision back for receive_decision to get
                self._pending_decision = Decision.from_json(data)
                return True

    def receive_decision(self) -> Optional[Decision]:
        """
        Receive the next decision from Forge.

        Returns:
            Decision object, or None if game ended
        """
        # Check for pending decision from start_game
        if hasattr(self, "_pending_decision") and self._pending_decision:
            decision = self._pending_decision
            self._pending_decision = None
            return decision

        while True:
            line = self._read_line()
            if line is None:
                return None

            if line.startswith("DECISION:"):
                json_str = line[9:]  # Remove "DECISION:" prefix
                try:
                    data = json.loads(json_str)
                    decision = Decision.from_json(data)
                    if self.our_player_name is None:
                        self.our_player_name = decision.player
                    return decision
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    print(f"Line: {line[:200]}...")
                    continue

            if line.startswith("GAME_RESULT:"):
                self._game_result = GameResult.from_line(line)
                return None

            if line.startswith("GAME_TIMEOUT:"):
                self._game_result = GameResult(
                    winner=None, is_draw=True, duration_ms=0, reason="timeout"
                )
                return None

            if line.startswith("GAME_ERROR:"):
                self._game_result = GameResult(
                    winner=None, is_draw=True, duration_ms=0, reason=line
                )
                return None

            # Other lines (like reveals) - skip
            continue

    def send_response(self, response: str):
        """Send a response to the current decision."""
        self._send_line(response)

    def send_action(self, index: int):
        """Send an action index (for choose_action decisions)."""
        self.send_response(str(index))

    def send_pass(self):
        """Pass priority."""
        self.send_response("-1")

    def send_attackers(self, indices: list[int]):
        """Send attacker indices."""
        if not indices:
            self.send_response("")
        else:
            self.send_response(",".join(str(i) for i in indices))

    def send_blockers(self, pairs: list[tuple[int, int]]):
        """Send blocker assignments as (blocker_idx, attacker_idx) pairs."""
        if not pairs:
            self.send_response("")
        else:
            self.send_response(",".join(f"{b}:{a}" for b, a in pairs))

    def send_yes(self):
        """Send yes/confirm response."""
        self.send_response("y")

    def send_no(self):
        """Send no/decline response."""
        self.send_response("n")

    def get_result(self) -> Optional[GameResult]:
        """Get the game result after the game ends."""
        return self._game_result

    def get_status(self) -> str:
        """Get daemon status."""
        if not self._connected:
            raise RuntimeError("Not connected to Forge daemon")
        self._send_line("STATUS")
        lines = []
        while True:
            line = self._read_line()
            if line is None:
                break
            lines.append(line)
            if "Max concurrent" in line:
                break
        return "\n".join(lines)

    def _send_line(self, line: str):
        """Send a line to the daemon."""
        self.writer.write(line + "\n")
        self.writer.flush()

    def _read_line(self) -> Optional[str]:
        """Read a line from the daemon."""
        try:
            line = self.reader.readline()
            if not line:
                return None
            return line.strip()
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Read error: {e}")
            return None


class ForgeGameEnv:
    """
    Gym-like environment wrapper around ForgeClient.

    Provides a cleaner interface for RL training with step/reset semantics.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 17171,
        deck1: str = "deck1.dck",
        deck2: str = "deck2.dck",
        timeout: int = 120,
    ):
        self.host = host
        self.port = port
        self.deck1 = deck1
        self.deck2 = deck2
        self.timeout = timeout
        self.client: Optional[ForgeClient] = None
        self.current_decision: Optional[Decision] = None
        self.game_active = False
        self.steps = 0

        # Metrics
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_steps = 0

    def reset(self, seed: Optional[int] = None) -> Optional[Decision]:
        """
        Start a new game.

        Returns:
            First decision, or None if game failed to start
        """
        # Close existing connection
        if self.client:
            self.client.close()

        self.client = ForgeClient(self.host, self.port, timeout=self.timeout)
        self.client.connect()

        if not self.client.start_game(
            self.deck1, self.deck2, timeout=self.timeout, seed=seed
        ):
            return None

        self.game_active = True
        self.steps = 0
        self.current_decision = self.client.receive_decision()
        return self.current_decision

    def step(self, action: str) -> tuple[Optional[Decision], float, bool, dict]:
        """
        Take an action and get the next state.

        Args:
            action: Response string to send

        Returns:
            (next_decision, reward, done, info)
        """
        if not self.game_active:
            return None, 0.0, True, {"error": "Game not active"}

        self.client.send_response(action)
        self.steps += 1
        self.total_steps += 1

        self.current_decision = self.client.receive_decision()

        if self.current_decision is None:
            # Game ended
            self.game_active = False
            self.total_games += 1

            result = self.client.get_result()
            if result:
                if result.is_draw:
                    reward = 0.0
                    self.draws += 1
                elif result.winner and self.client.our_player_name in result.winner:
                    reward = 1.0
                    self.wins += 1
                else:
                    reward = -1.0
                    self.losses += 1
            else:
                reward = 0.0
                self.draws += 1

            info = {
                "result": result,
                "steps": self.steps,
                "our_player": self.client.our_player_name,
            }
            return None, reward, True, info

        return self.current_decision, 0.0, False, {"steps": self.steps}

    def close(self):
        """Close the environment."""
        if self.client:
            self.client.close()
        self.game_active = False

    def get_metrics(self) -> dict:
        """Get training metrics."""
        return {
            "total_games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / max(1, self.total_games),
            "total_steps": self.total_steps,
            "avg_steps_per_game": self.total_steps / max(1, self.total_games),
        }


def test_connection():
    """Test connection to local Forge daemon."""
    print("Testing Forge daemon connection...")

    client = ForgeClient("localhost", 17171)
    try:
        client.connect()
        print("Connected!")

        # Get status
        status = client.get_status()
        print(f"Status:\n{status}")

    except ConnectionRefusedError:
        print("Could not connect - is Forge daemon running?")
        print("Start it with: java -jar forge.jar --daemon")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    test_connection()
