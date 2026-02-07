#!/usr/bin/env python3
"""
MTG Reinforcement Learning Environment

Architecture designed for the massive state/action space of Magic: The Gathering.

Key Design Decisions:
1. Hierarchical Action Space - Break down decisions into sub-decisions
2. Card Embeddings - Learn representations for cards rather than one-hot encoding
3. Attention-based State Encoding - Handle variable-length game states
4. Self-Play Training - Train against copies of itself
5. Curriculum Learning - Start with simplified scenarios

State Representation:
- Player state (life, mana, cards in zones)
- Opponent visible state (life, board, graveyard)
- Hidden state estimation (opponent hand size, library size)
- Game phase/turn information
- Available actions encoded

Action Representation:
- Action type (pass, play_card, activate_ability, attack, block)
- Target selection (which card, which target)
- Uses action masking for legal moves only
"""

import json
import subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ActionType(Enum):
    PASS = 0
    PLAY_SPELL = 1
    PLAY_LAND = 2
    ACTIVATE_ABILITY = 3
    DECLARE_ATTACKERS = 4
    DECLARE_BLOCKERS = 5


@dataclass
class ManaPool:
    """Representation of a player's mana pool."""
    total: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0
    colorless: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> 'ManaPool':
        if isinstance(data, dict):
            return cls(
                total=data.get('total', 0),
                white=data.get('white', 0),
                blue=data.get('blue', 0),
                black=data.get('black', 0),
                red=data.get('red', 0),
                green=data.get('green', 0),
                colorless=data.get('colorless', 0)
            )
        return cls(total=data if isinstance(data, int) else 0)


@dataclass
class CardState:
    """Representation of a card in the game."""
    card_id: int
    name: str
    card_type: str  # creature, instant, sorcery, land, etc.
    mana_cost: str = ""
    cmc: int = 0
    power: int = 0
    toughness: int = 0
    damage: int = 0
    is_tapped: bool = False
    is_creature: bool = False
    is_land: bool = False
    is_artifact: bool = False
    is_enchantment: bool = False
    is_planeswalker: bool = False
    summoning_sick: bool = False
    loyalty: int = 0
    counters: str = ""
    keywords: List[str] = field(default_factory=list)
    oracle_text: str = ""

    # Shared embedder instance (lazy loaded)
    _embedder = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'CardState':
        """Parse card from JSON dict."""
        return cls(
            card_id=data.get('id', 0),
            name=data.get('name', ''),
            card_type=data.get('types', ''),
            mana_cost=data.get('mana_cost', ''),
            cmc=data.get('cmc', 0),
            power=data.get('power', 0),
            toughness=data.get('toughness', 0),
            damage=data.get('damage', 0),
            is_tapped=data.get('tapped', False),
            is_creature=data.get('is_creature', False),
            is_land=data.get('is_land', False),
            is_artifact=data.get('is_artifact', False),
            is_enchantment=data.get('is_enchantment', False),
            is_planeswalker=data.get('is_planeswalker', False),
            summoning_sick=data.get('summoning_sick', False),
            loyalty=data.get('loyalty', 0),
            counters=data.get('counters', ''),
            keywords=data.get('keywords', []),
            oracle_text=data.get('oracle_text', '')
        )

    @classmethod
    def get_embedder(cls):
        """Get or create shared embedder instance."""
        if cls._embedder is None:
            try:
                from src.models.card_embeddings import CardEmbedding
                cls._embedder = CardEmbedding(use_text_embeddings=False)
            except ImportError:
                cls._embedder = None
        return cls._embedder

    def to_vector(self, card_embedding_dim: int = 89) -> np.ndarray:
        """Convert card to fixed-size vector representation.

        Uses the CardEmbedding system for structured feature extraction.
        Falls back to simple features if card_embeddings module unavailable.
        """
        embedder = self.get_embedder()

        if embedder is not None:
            # Use the full embedding system
            try:
                embedding = embedder.embed_card(
                    name=self.name,
                    mana_cost=self.mana_cost,
                    type_line=self.card_type,
                    oracle_text=self.oracle_text,
                    power=self.power if self.is_creature else None,
                    toughness=self.toughness if self.is_creature else None,
                    loyalty=self.loyalty if self.is_planeswalker else None,
                    keywords=self.keywords,
                    cmc=self.cmc
                )
                # Add game context features
                context = np.array([
                    1.0 if self.is_tapped else 0.0,
                    1.0 if self.summoning_sick else 0.0,
                    self.damage / 10.0,
                ], dtype=np.float32)
                return np.concatenate([embedding, context])
            except Exception:
                pass  # Fall through to simple features

        # Fallback: simple feature vector
        features = [
            self.cmc / 10.0,
            self.power / 10.0,
            self.toughness / 10.0,
            self.damage / 10.0,
            1.0 if self.is_tapped else 0.0,
            1.0 if self.is_creature else 0.0,
            1.0 if self.is_land else 0.0,
            1.0 if self.is_artifact else 0.0,
            1.0 if self.is_enchantment else 0.0,
            1.0 if self.is_planeswalker else 0.0,
            1.0 if self.summoning_sick else 0.0,
            self.loyalty / 10.0,
            # Common keywords
            1.0 if any('Flying' in kw for kw in self.keywords) else 0.0,
            1.0 if any('Trample' in kw for kw in self.keywords) else 0.0,
            1.0 if any('First strike' in kw for kw in self.keywords) else 0.0,
            1.0 if any('Haste' in kw for kw in self.keywords) else 0.0,
            1.0 if any('Lifelink' in kw for kw in self.keywords) else 0.0,
            1.0 if any('Deathtouch' in kw for kw in self.keywords) else 0.0,
            1.0 if any('Vigilance' in kw for kw in self.keywords) else 0.0,
            1.0 if any('Reach' in kw for kw in self.keywords) else 0.0,
        ]
        # Pad to match embedding dim
        return np.array(features + [0.0] * (card_embedding_dim - len(features)), dtype=np.float32)


@dataclass
class PlayerState:
    """State of a player in the game."""
    name: str
    life: int
    poison: int = 0
    mana_pool: ManaPool = field(default_factory=ManaPool)
    hand_size: int = 0
    library_size: int = 0
    lands_played_this_turn: int = 0
    max_land_plays: int = 1
    has_lost: bool = False
    graveyard: List[CardState] = field(default_factory=list)
    battlefield: List[CardState] = field(default_factory=list)
    exile: List[CardState] = field(default_factory=list)
    hand: List[CardState] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PlayerState':
        """Parse player state from JSON dict."""
        return cls(
            name=data.get('name', ''),
            life=data.get('life', 20),
            poison=data.get('poison', 0),
            mana_pool=ManaPool.from_dict(data.get('mana_pool', {})),
            hand_size=data.get('hand_size', 0),
            library_size=data.get('library_size', 0),
            lands_played_this_turn=data.get('lands_played_this_turn', 0),
            max_land_plays=data.get('max_land_plays', 1),
            has_lost=data.get('has_lost', False),
            graveyard=[CardState.from_dict(c) for c in data.get('graveyard', [])],
            battlefield=[CardState.from_dict(c) for c in data.get('battlefield', [])],
            exile=[CardState.from_dict(c) for c in data.get('exile', [])],
            hand=[CardState.from_dict(c) for c in data.get('hand', [])]
        )


@dataclass
class CombatState:
    """Combat state representation."""
    attacking_player: str = ""
    attackers: List[Dict] = field(default_factory=list)  # card_id, name, power, toughness, attacking, blocked_by

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'CombatState':
        if not data:
            return cls()
        return cls(
            attacking_player=data.get('attacking_player', ''),
            attackers=data.get('attackers', [])
        )


@dataclass
class StackEntry:
    """Stack entry representation."""
    entry_id: int
    description: str
    controller: str
    source_card: str = ""
    source_card_id: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> 'StackEntry':
        return cls(
            entry_id=data.get('id', 0),
            description=data.get('description', ''),
            controller=data.get('controller', ''),
            source_card=data.get('source_card', ''),
            source_card_id=data.get('source_card_id', 0)
        )


@dataclass
class GameState:
    """Full game state representation."""
    turn: int
    phase: str
    active_player: str
    priority_player: str
    is_game_over: bool
    our_player: PlayerState
    opponent: PlayerState
    stack: List[StackEntry] = field(default_factory=list)
    combat: Optional[CombatState] = None

    @classmethod
    def from_dict(cls, data: Dict, our_player_name: str) -> 'GameState':
        """Parse game state from JSON dict."""
        game_state = data.get('game_state', {})
        players = game_state.get('players', [])

        # Find our player and opponent
        our_player_data = None
        opponent_data = None
        for p in players:
            if our_player_name in p.get('name', ''):
                our_player_data = p
            else:
                opponent_data = p

        # Fallback if players not found
        if not our_player_data:
            our_player_data = players[0] if players else {}
        if not opponent_data:
            opponent_data = players[1] if len(players) > 1 else {}

        return cls(
            turn=data.get('turn', 1),
            phase=data.get('phase', 'MAIN1'),
            active_player=game_state.get('active_player', ''),
            priority_player=game_state.get('priority_player', ''),
            is_game_over=game_state.get('is_game_over', False),
            our_player=PlayerState.from_dict(our_player_data),
            opponent=PlayerState.from_dict(opponent_data),
            stack=[StackEntry.from_dict(s) for s in game_state.get('stack', [])],
            combat=CombatState.from_dict(game_state.get('combat'))
        )

    def to_observation(self, max_cards: int = 20) -> np.ndarray:
        """Convert game state to fixed-size observation vector.

        This is a simplified representation. A full implementation would use:
        - Transformer/attention for variable-length card sequences
        - Graph neural networks for card interactions
        - Separate embeddings for different zones
        """
        obs = []

        # Global state
        obs.extend([
            self.turn / 100.0,  # Normalize turn number
            self._phase_to_int(self.phase) / 15.0,
            1.0 if self.active_player == self.our_player.name else 0.0,
            1.0 if self.priority_player == self.our_player.name else 0.0,
            1.0 if self.is_game_over else 0.0,
        ])

        # Our player state
        obs.extend([
            self.our_player.life / 40.0,  # Normalize to typical range
            self.our_player.poison / 10.0,
            self.our_player.mana_pool.total / 20.0,
            self.our_player.mana_pool.white / 10.0,
            self.our_player.mana_pool.blue / 10.0,
            self.our_player.mana_pool.black / 10.0,
            self.our_player.mana_pool.red / 10.0,
            self.our_player.mana_pool.green / 10.0,
            self.our_player.mana_pool.colorless / 10.0,
            self.our_player.hand_size / 10.0,
            self.our_player.library_size / 60.0,
            len(self.our_player.battlefield) / max_cards,
            len(self.our_player.graveyard) / max_cards,
            len(self.our_player.exile) / max_cards,
            self.our_player.lands_played_this_turn / 2.0,
            1.0 if self.our_player.has_lost else 0.0,
        ])

        # Count creature stats on our battlefield
        our_creature_power = sum(c.power for c in self.our_player.battlefield if c.is_creature)
        our_creature_count = sum(1 for c in self.our_player.battlefield if c.is_creature)
        our_land_count = sum(1 for c in self.our_player.battlefield if c.is_land)
        obs.extend([
            our_creature_power / 50.0,
            our_creature_count / 10.0,
            our_land_count / 10.0,
        ])

        # Opponent state
        obs.extend([
            self.opponent.life / 40.0,
            self.opponent.poison / 10.0,
            self.opponent.mana_pool.total / 20.0,
            self.opponent.hand_size / 10.0,
            self.opponent.library_size / 60.0,
            len(self.opponent.battlefield) / max_cards,
            len(self.opponent.graveyard) / max_cards,
            1.0 if self.opponent.has_lost else 0.0,
        ])

        # Count creature stats on opponent battlefield
        opp_creature_power = sum(c.power for c in self.opponent.battlefield if c.is_creature)
        opp_creature_count = sum(1 for c in self.opponent.battlefield if c.is_creature)
        opp_land_count = sum(1 for c in self.opponent.battlefield if c.is_land)
        obs.extend([
            opp_creature_power / 50.0,
            opp_creature_count / 10.0,
            opp_land_count / 10.0,
        ])

        # Stack state
        obs.append(len(self.stack) / 10.0)

        # Combat state
        in_combat = self.combat is not None and len(self.combat.attackers) > 0
        obs.extend([
            1.0 if in_combat else 0.0,
            len(self.combat.attackers) / 10.0 if self.combat else 0.0,
        ])

        return np.array(obs, dtype=np.float32)

    def _phase_to_int(self, phase: str) -> int:
        phases = {
            'UNTAP': 0, 'UPKEEP': 1, 'DRAW': 2,
            'MAIN1': 3, 'COMBAT_BEGIN': 4, 'COMBAT_DECLARE_ATTACKERS': 5,
            'COMBAT_DECLARE_BLOCKERS': 6, 'COMBAT_FIRST_STRIKE_DAMAGE': 7,
            'COMBAT_DAMAGE': 8, 'COMBAT_END': 9,
            'MAIN2': 10, 'END_OF_TURN': 11, 'CLEANUP': 12
        }
        return phases.get(phase, 0)


@dataclass
class Action:
    """Represents an action the agent can take."""
    action_type: ActionType
    action_index: int  # Index in the available actions list
    raw_data: Dict = field(default_factory=dict)

    def to_response(self) -> str:
        """Convert action to string response for the game."""
        if self.action_type == ActionType.PASS:
            return "-1"
        return str(self.action_index)


class RewardShaper:
    """
    Shapes rewards to make learning more tractable.

    Raw MTG rewards are very sparse (win=1, lose=-1).
    We add intermediate rewards to guide learning:
    - Life differential
    - Board presence
    - Card advantage
    - Mana efficiency
    """

    def __init__(self):
        self.prev_state: Optional[GameState] = None
        self.cumulative_reward = 0.0

    def compute_reward(
        self,
        state: GameState,
        action: Action,
        game_over: bool,
        won: bool
    ) -> float:
        """Compute shaped reward for a state transition."""

        # Terminal reward
        if game_over:
            return 1.0 if won else -1.0

        reward = 0.0

        if self.prev_state is not None:
            # Life differential change
            our_life_delta = state.our_player.life - self.prev_state.our_player.life
            opp_life_delta = state.opponent.life - self.prev_state.opponent.life
            reward += (our_life_delta - opp_life_delta) * 0.01

            # Board presence change (using is_creature attribute)
            our_creatures = sum(1 for c in state.our_player.battlefield if c.is_creature)
            prev_creatures = sum(1 for c in self.prev_state.our_player.battlefield if c.is_creature)
            reward += (our_creatures - prev_creatures) * 0.02

            # Creature power advantage
            our_power = sum(c.power for c in state.our_player.battlefield if c.is_creature)
            prev_power = sum(c.power for c in self.prev_state.our_player.battlefield if c.is_creature)
            opp_power = sum(c.power for c in state.opponent.battlefield if c.is_creature)
            prev_opp_power = sum(c.power for c in self.prev_state.opponent.battlefield if c.is_creature)
            power_delta = (our_power - prev_power) - (opp_power - prev_opp_power)
            reward += power_delta * 0.005

            # Card advantage (hand + battlefield vs opponent)
            our_cards = state.our_player.hand_size + len(state.our_player.battlefield)
            opp_cards = state.opponent.hand_size + len(state.opponent.battlefield)
            prev_our = self.prev_state.our_player.hand_size + len(self.prev_state.our_player.battlefield)
            prev_opp = self.prev_state.opponent.hand_size + len(self.prev_state.opponent.battlefield)

            card_adv_delta = (our_cards - opp_cards) - (prev_our - prev_opp)
            reward += card_adv_delta * 0.01

            # Land development reward
            our_lands = sum(1 for c in state.our_player.battlefield if c.is_land)
            prev_lands = sum(1 for c in self.prev_state.our_player.battlefield if c.is_land)
            reward += (our_lands - prev_lands) * 0.01

        # Small negative reward for passing (encourage action)
        if action.action_type == ActionType.PASS:
            reward -= 0.001

        self.prev_state = state
        self.cumulative_reward += reward

        return reward

    def reset(self):
        self.prev_state = None
        self.cumulative_reward = 0.0


class MTGEnvironment:
    """
    OpenAI Gym-style environment for MTG.

    Connects to the Forge game engine via Docker and translates
    between game states and RL observations/actions.
    """

    def __init__(
        self,
        deck1: str = "red_aggro.dck",
        deck2: str = "white_weenie.dck",
        docker_image: str = "forge-sim:latest",
        player_id: int = 1,  # Which player the agent controls (1 or 2)
        timeout: int = 120,
        reward_shaping: bool = True
    ):
        self.deck1 = deck1
        self.deck2 = deck2
        self.docker_image = docker_image
        self.player_id = player_id
        self.timeout = timeout
        self.reward_shaping = reward_shaping

        self.process: Optional[subprocess.Popen] = None
        self.reward_shaper = RewardShaper()
        self.current_state: Optional[GameState] = None
        self.available_actions: List[Dict] = []
        self.game_over = False
        self.won = False
        self.decision_count = 0

        # Observation and action space dimensions
        # Global: 5, Our player: 16+3=19, Opponent: 8+3=11, Stack: 1, Combat: 2 = 38
        self.observation_dim = 38
        self.max_actions = 50  # Maximum number of actions to consider

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and start a new game."""
        self._cleanup()
        self.reward_shaper.reset()
        self.game_over = False
        self.won = False
        self.decision_count = 0

        # Start game process
        cmd = [
            "docker", "run", "--rm", "-i",
            "--entrypoint", "/bin/bash",
            self.docker_image,
            "-c",
            f"cd /forge && timeout {self.timeout} xvfb-run -a java -Xmx2048m "
            f"--add-opens java.base/java.lang=ALL-UNNAMED "
            f"--add-opens java.base/java.util=ALL-UNNAMED "
            f"--add-opens java.base/java.text=ALL-UNNAMED "
            f"--add-opens java.base/java.lang.reflect=ALL-UNNAMED "
            f"--add-opens java.desktop/java.beans=ALL-UNNAMED "
            f"-Dsentry.dsn= -jar forge.jar sim "
            f"-d {self.deck1} {self.deck2} -n 1 -i -q -c {self.timeout}"
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Wait for first decision
        obs, info = self._wait_for_decision()
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

        # Convert action index to response
        if action_idx < 0 or action_idx >= len(self.available_actions):
            response = "-1"  # Pass
            action = Action(ActionType.PASS, -1)
        else:
            action_data = self.available_actions[action_idx]
            response = str(action_data.get('index', -1))
            action = Action(
                ActionType.PLAY_SPELL if action_data.get('index', -1) >= 0 else ActionType.PASS,
                action_data.get('index', -1),
                action_data
            )

        # Send response
        try:
            self.process.stdin.write(response + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.game_over = True
            return self._get_observation(), -1.0, True, False, {"error": "pipe_broken"}

        # Wait for next decision or game end
        obs, info = self._wait_for_decision()

        # Compute reward
        reward = self.reward_shaper.compute_reward(
            self.current_state, action, self.game_over, self.won
        ) if self.reward_shaping and self.current_state else 0.0

        return obs, reward, self.game_over, False, info

    def _wait_for_decision(self) -> Tuple[np.ndarray, Dict]:
        """Wait for next decision point from the game."""
        our_player_name = f"Agent({self.player_id})"

        while self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
            except Exception:
                break

            if not line:
                continue

            line = line.strip()

            # Check for game end
            if "has won" in line.lower():
                self.game_over = True
                self.won = our_player_name.lower() in line.lower()
                break

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
                # Pass for opponent (in single-agent mode)
                self.process.stdin.write("-1\n")
                self.process.stdin.flush()
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

        # Game ended
        self.game_over = True
        return self._get_observation(), {"game_over": True, "won": self.won}

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

    def _cleanup(self):
        """Clean up subprocess."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def close(self):
        """Close the environment."""
        self._cleanup()


class SimpleAgent:
    """
    Simple rule-based agent for testing and as a baseline.
    """

    def select_action(self, obs: np.ndarray, actions: List[Dict], mask: np.ndarray) -> int:
        """Select an action based on simple heuristics."""

        # Priority 1: Play lands
        for i, action in enumerate(actions):
            if action.get('is_land', False) and action.get('index', -1) >= 0:
                return i

        # Priority 2: Play creatures/spells if we have mana
        # (simplified - would check mana properly in real implementation)
        for i, action in enumerate(actions):
            idx = action.get('index', -1)
            if idx >= 0 and not action.get('is_land', False):
                cost_str = action.get('mana_cost', '')
                if not cost_str or cost_str == 'no cost':
                    return i

        # Default: pass
        return -1


def train_episode(env: MTGEnvironment, agent: SimpleAgent) -> Dict:
    """Run a single training episode."""
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while not env.game_over and steps < 10000:
        mask = env.get_action_mask()
        action = agent.select_action(obs, env.available_actions, mask)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "won": env.won,
        "turns": info.get("turn", 0) if isinstance(info, dict) else 0
    }


def main():
    """Test the RL environment."""
    print("MTG RL Environment Test")
    print("=" * 50)

    env = MTGEnvironment(
        deck1="red_aggro.dck",
        deck2="white_weenie.dck",
        player_id=1,
        reward_shaping=True
    )

    agent = SimpleAgent()

    print("\nRunning test episode...")
    result = train_episode(env, agent)

    print("\nEpisode Results:")
    print(f"  Total Reward: {result['total_reward']:.4f}")
    print(f"  Steps: {result['steps']}")
    print(f"  Won: {result['won']}")

    env.close()
    print("\nEnvironment closed successfully.")


if __name__ == "__main__":
    main()
