#!/usr/bin/env python3
"""
Forge AI Imitation Training

Trains our neural network to imitate a heuristic agent that plays reasonably.
This produces sensible game lengths and decision counts while building
the foundation for more sophisticated RL.

Key features:
- HeuristicAgent: Makes reasonable decisions based on game state heuristics
- ImitationTrainer: Collects (state, action) pairs and trains via behavioral cloning
- Game metrics tracking: turns, decisions, game length
- Fallback: Our agent can defer to heuristic when uncertain

Usage:
    python src/training/forge_imitation.py --games 100 --epochs 10
"""

import os
import sys
import time
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.forge.forge_client import (
    ForgeClient, Decision, DecisionType, ActionOption, GameState
)


# ============================================================================
# Heuristic Agent - Makes reasonable decisions like Forge AI
# ============================================================================

class HeuristicAgent:
    """
    A rule-based agent that makes sensible MTG decisions.

    This approximates Forge AI's decision-making process using heuristics.
    Used to generate training data for imitation learning.

    Key heuristics:
    - Land drops: Always play lands when possible
    - Curve out: Cast creatures on curve when able
    - Attack: Attack when favorable or when opponent is low
    - Block: Block to survive or trade up
    - Removal: Use removal on significant threats
    - Pass: Pass when no good options
    """

    def __init__(self, aggression: float = 0.6, greed: float = 0.4):
        """
        Args:
            aggression: How aggressively to attack (0-1)
            greed: How much to hold resources vs use them (0-1)
        """
        self.aggression = aggression
        self.greed = greed
        self.decisions_this_turn = 0
        self.current_turn = 0

    def reset_turn(self):
        """Reset per-turn tracking."""
        self.decisions_this_turn = 0

    def decide(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """
        Make a decision and return (response, metadata).

        Returns:
            Tuple of (response_string, decision_metadata)
        """
        # Track decisions per turn
        if decision.turn > self.current_turn:
            self.current_turn = decision.turn
            self.decisions_this_turn = 0
        self.decisions_this_turn += 1

        # Safety valve: if too many decisions, just pass
        if self.decisions_this_turn > 100:
            return self._pass_response(decision), {"reason": "decision_limit"}

        dtype = decision.decision_type

        if dtype == DecisionType.CHOOSE_ACTION:
            return self._choose_action(decision)
        elif dtype == DecisionType.DECLARE_ATTACKERS:
            return self._declare_attackers(decision)
        elif dtype == DecisionType.DECLARE_BLOCKERS:
            return self._declare_blockers(decision)
        elif dtype == DecisionType.PLAY_TRIGGER:
            return "y", {"reason": "always_trigger"}
        elif dtype == DecisionType.CONFIRM_ACTION:
            return "y", {"reason": "always_confirm"}
        elif dtype == DecisionType.CHOOSE_CARDS:
            return self._choose_cards(decision)
        elif dtype == DecisionType.CHOOSE_ENTITY:
            return self._choose_entity(decision)
        elif dtype == DecisionType.ANNOUNCE_VALUE:
            return "0", {"reason": "announce_min"}
        else:
            return "0", {"reason": "unknown_decision_type"}

    def _pass_response(self, decision: Decision) -> str:
        """Get the pass response for this decision type."""
        if decision.decision_type == DecisionType.CHOOSE_ACTION:
            return "-1"
        elif decision.decision_type == DecisionType.DECLARE_ATTACKERS:
            return ""
        elif decision.decision_type == DecisionType.DECLARE_BLOCKERS:
            return ""
        else:
            return "0"

    def _choose_action(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose an action to take (main decision type)."""
        actions = decision.actions
        if not actions:
            return "-1", {"reason": "no_actions"}

        gs = decision.game_state

        # Get our player
        our_player = None
        opp_player = None
        for p in gs.players:
            if p.name == decision.player:
                our_player = p
            else:
                opp_player = p

        if not our_player:
            return "-1", {"reason": "no_player"}

        # Calculate available mana from battlefield lands
        mana_available = sum(1 for c in our_player.battlefield if c.is_land and not c.tapped)

        # 1. LAND DROPS - Always prioritize lands first
        land_actions = [a for a in actions if a.is_land]
        if land_actions and our_player.lands_played_this_turn < our_player.max_land_plays:
            best_land = self._best_land(land_actions, our_player.hand)
            return str(best_land.index), {"reason": "play_land", "card": best_land.description}

        # If no mana available and no lands to play, just pass
        if mana_available == 0:
            return "-1", {"reason": "no_mana"}

        # 2. Filter actions by mana cost we can afford
        affordable = []
        for a in actions:
            if a.is_land or a.index < 0:
                continue
            # Parse mana cost from the action
            cmc = self._estimate_cmc(a)
            if cmc <= mana_available:
                affordable.append(a)

        if not affordable:
            return "-1", {"reason": "nothing_affordable"}

        # 3. CAST CREATURES ON CURVE (in main phases only)
        if gs.phase in ["MAIN1", "MAIN2"]:
            creature_actions = [a for a in affordable if self._is_creature_action(a)]
            if creature_actions:
                # Prefer cheapest creature that fits mana
                creature_actions.sort(key=lambda a: self._estimate_cmc(a))
                best = creature_actions[0]
                return str(best.index), {"reason": "cast_creature", "card": best.description}

        # 4. USE REMOVAL ON BIG THREATS (check for opponent threats first)
        if opp_player and gs.phase in ["MAIN1", "MAIN2"]:
            removal_actions = [a for a in affordable if self._is_removal_action(a)]
            if removal_actions:
                opp_threats = [c for c in opp_player.battlefield
                              if c.is_creature and (c.power or 0) >= 3]
                if opp_threats:
                    return str(removal_actions[0].index), {
                        "reason": "use_removal",
                        "card": removal_actions[0].description
                    }

        # 5. CAST OTHER AFFORDABLE SPELLS (with some randomness)
        if affordable and gs.phase in ["MAIN1", "MAIN2"]:
            if random.random() > self.greed:
                spell = random.choice(affordable)
                return str(spell.index), {"reason": "cast_spell", "card": spell.description}

        # 6. PASS - Default action
        return "-1", {"reason": "pass_priority"}

    def _estimate_cmc(self, action: ActionOption) -> int:
        """Estimate mana cost from action description or card info."""
        # Try to parse from mana_cost field if available
        # For now, use simple heuristics
        desc = action.description.lower()

        # Check for specific patterns
        if "play" in desc and "land" in desc:
            return 0
        if "{1}" in desc or "{r}" in desc or "{w}" in desc:
            return 1
        if "{2}" in desc or "{1}{r}" in desc or "{1}{w}" in desc:
            return 2
        if "{3}" in desc or "{2}{r}" in desc or "{2}{w}" in desc:
            return 3

        # Default: assume 1 mana
        return 1

    def _declare_attackers(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Declare which creatures to attack with."""
        attackers = decision.attackers
        if not attackers:
            return "", {"reason": "no_attackers"}

        gs = decision.game_state

        # Get opponent's blockers
        opp_player = None
        for p in gs.players:
            if p.name != decision.player:
                opp_player = p
                break

        if not opp_player:
            return "", {"reason": "no_opponent"}

        opp_blockers = [c for c in opp_player.battlefield
                       if c.is_creature and not c.tapped]
        opp_toughness = sum(c.toughness or 0 for c in opp_blockers)

        # Decide which creatures to attack with
        attack_with = []

        for attacker in attackers:
            power = attacker.get("power", 0)
            toughness = attacker.get("toughness", 0)
            has_evasion = any(k in attacker.get("keywords", [])
                             for k in ["Flying", "Trample", "Unblockable"])

            # Attack if:
            # - We have evasion
            # - Opponent has no blockers
            # - We're aggressive and creature survives
            # - Opponent is low on life
            should_attack = (
                has_evasion or
                not opp_blockers or
                (random.random() < self.aggression and toughness > 1) or
                (opp_player.life <= power * 2)
            )

            if should_attack:
                attack_with.append(str(attacker["index"]))

        if attack_with:
            return ",".join(attack_with), {"reason": "attacking",
                                            "num_attackers": len(attack_with)}
        return "", {"reason": "no_attack"}

    def _declare_blockers(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Declare blockers - try to survive."""
        blockers = decision.blockers
        if not blockers:
            return "", {"reason": "no_blockers"}

        # Get attacking creatures
        gs = decision.game_state
        if not gs.combat or not gs.combat.attackers:
            return "", {"reason": "no_combat"}

        attackers = gs.combat.attackers

        # Simple blocking strategy: block biggest threats we can kill
        assignments = []
        used_blockers = set()

        # Sort attackers by power (block biggest first)
        sorted_attackers = sorted(attackers, key=lambda a: a.get("power", 0), reverse=True)

        for attacker in sorted_attackers:
            att_power = attacker.get("power", 0)
            att_toughness = attacker.get("toughness", 0)
            att_id = attacker.get("id")

            # Find a blocker that can kill it and survive (or trade)
            best_blocker = None
            for blocker in blockers:
                if blocker.get("id") in used_blockers:
                    continue

                block_power = blocker.get("power", 0)
                block_toughness = blocker.get("toughness", 0)

                # Can we kill the attacker?
                can_kill = block_power >= att_toughness
                # Do we survive?
                survives = block_toughness > att_power

                if can_kill and survives:
                    best_blocker = blocker
                    break
                elif can_kill and not best_blocker:
                    best_blocker = blocker

            if best_blocker:
                assignments.append(f"{best_blocker['index']}:{att_id}")
                used_blockers.add(best_blocker.get("id"))

        if assignments:
            return ",".join(assignments), {"reason": "blocking",
                                            "num_blocks": len(assignments)}
        return "", {"reason": "no_blocks"}

    def _choose_cards(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose cards (e.g., for scry, discard, etc.)."""
        cards = decision.cards
        if not cards:
            return "", {"reason": "no_cards"}

        min_cards = decision.raw_data.get("min", 0)
        max_cards = decision.raw_data.get("max", len(cards))

        # For discard: discard highest CMC (lands first)
        # For other choices: pick lowest CMC (cheapest)
        is_discard = "discard" in decision.raw_data.get("message", "").lower()

        if is_discard:
            # Prefer discarding lands, then highest CMC
            sorted_cards = sorted(enumerate(cards),
                                 key=lambda x: (not x[1].get("is_land", False),
                                               -x[1].get("cmc", 0)))
        else:
            # Prefer best cards (creatures, then spells, by CMC)
            sorted_cards = sorted(enumerate(cards),
                                 key=lambda x: (not x[1].get("is_creature", False),
                                               x[1].get("cmc", 0)))

        indices = [str(idx) for idx, _ in sorted_cards[:max(min_cards, 1)]]
        return ",".join(indices), {"reason": "choose_cards", "count": len(indices)}

    def _choose_entity(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose a target entity."""
        # Default: first option
        return "0", {"reason": "first_entity"}

    def _best_land(self, lands: List[ActionOption], hand: List) -> ActionOption:
        """Choose the best land to play based on spells in hand."""
        # For now, just play first land
        # TODO: Match colors to spells in hand
        return lands[0]

    def _best_creature(self, creatures: List[ActionOption], player) -> Optional[ActionOption]:
        """Choose the best creature to cast."""
        # Sort by CMC descending - play biggest affordable
        # Extract CMC from description or use default
        def get_cmc(action: ActionOption) -> int:
            # Try to extract from card info if available
            return 0  # Default - actions don't have CMC directly

        # For now, just cast the first creature
        return creatures[0] if creatures else None

    def _is_creature_action(self, action: ActionOption) -> bool:
        """Check if this is casting a creature."""
        desc = action.description.lower()
        return "creature" in desc or "cast" in desc

    def _is_removal_action(self, action: ActionOption) -> bool:
        """Check if this is a removal spell."""
        desc = action.description.lower()
        removal_words = ["destroy", "exile", "damage", "kill", "murder",
                        "bolt", "shock", "burn", "doom"]
        return any(word in desc for word in removal_words)


# ============================================================================
# Training Data Collection
# ============================================================================

@dataclass
class TrainingSample:
    """A single training sample for imitation learning."""
    game_id: int
    decision_id: int
    turn: int
    phase: str
    decision_type: str

    # The game state (raw JSON for flexibility)
    state_json: dict

    # The action taken
    action_response: str
    action_index: int  # Converted to our action space
    action_description: str

    # Decision metadata
    num_legal_actions: int
    decision_reason: str

    # For action mask
    legal_action_indices: List[int] = field(default_factory=list)


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    game_id: int
    winner: str = ""
    is_draw: bool = False
    total_turns: int = 0
    total_decisions: int = 0
    decisions_per_turn: List[int] = field(default_factory=list)
    duration_ms: float = 0

    # Per-player stats
    player1_life_final: int = 0
    player2_life_final: int = 0

    # Decision type breakdown
    decision_type_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ImitationDataCollector:
    """
    Collects training data by playing games with a heuristic agent.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 17171,
        deck1: str = "red_aggro.dck",
        deck2: str = "white_weenie.dck",
        timeout: int = 120,
    ):
        self.host = host
        self.port = port
        self.deck1 = deck1
        self.deck2 = deck2
        self.timeout = timeout

        self.agent = HeuristicAgent()
        self.samples: List[TrainingSample] = []
        self.game_metrics: List[GameMetrics] = []

        # Action space mapping
        self.action_to_index = {}

    def collect_game(self, game_id: int, seed: Optional[int] = None) -> GameMetrics:
        """
        Play one game and collect all decision samples.

        Returns:
            GameMetrics for this game
        """
        metrics = GameMetrics(game_id=game_id)
        client = ForgeClient(self.host, self.port, timeout=self.timeout)

        try:
            client.connect()

            success = client.start_game(self.deck1, self.deck2, timeout=self.timeout, seed=seed)
            if not success:
                print(f"Game {game_id}: Failed to start")
                return metrics

            game_start = time.perf_counter()
            current_turn = 0
            decisions_this_turn = 0
            decision_id = 0

            self.agent.reset_turn()

            while True:
                decision = client.receive_decision()
                if decision is None:
                    break

                # Track turn changes
                if decision.turn > current_turn:
                    if current_turn > 0:
                        metrics.decisions_per_turn.append(decisions_this_turn)
                    current_turn = decision.turn
                    decisions_this_turn = 0
                    self.agent.reset_turn()

                decisions_this_turn += 1
                metrics.total_decisions += 1
                metrics.decision_type_counts[decision.decision_type.value] += 1

                # Make decision with heuristic agent
                response, metadata = self.agent.decide(decision)

                # Create training sample
                sample = TrainingSample(
                    game_id=game_id,
                    decision_id=decision_id,
                    turn=decision.turn,
                    phase=decision.game_state.phase,
                    decision_type=decision.decision_type.value,
                    state_json=decision.raw_data.get("game_state", {}),
                    action_response=response,
                    action_index=self._response_to_index(response, decision),
                    action_description=self._get_action_description(response, decision),
                    num_legal_actions=len(decision.actions) if decision.actions else 0,
                    decision_reason=metadata.get("reason", "unknown"),
                    legal_action_indices=self._get_legal_indices(decision),
                )
                self.samples.append(sample)
                decision_id += 1

                # Send response
                client.send_response(response)

            # Record final turn decisions
            if decisions_this_turn > 0:
                metrics.decisions_per_turn.append(decisions_this_turn)

            metrics.total_turns = current_turn

            # Get game result
            result = client.get_result()
            if result:
                metrics.winner = result.winner or "Draw"
                metrics.is_draw = result.is_draw
                metrics.duration_ms = result.duration_ms
            else:
                metrics.duration_ms = (time.perf_counter() - game_start) * 1000

            # Get final life totals from last decision
            if self.samples:
                last_state = self.samples[-1].state_json
                players = last_state.get("players", [])
                if len(players) >= 2:
                    metrics.player1_life_final = players[0].get("life", 0)
                    metrics.player2_life_final = players[1].get("life", 0)

        except Exception as e:
            print(f"Game {game_id}: Error - {e}")
        finally:
            client.close()

        self.game_metrics.append(metrics)
        return metrics

    def _response_to_index(self, response: str, decision: Decision) -> int:
        """Convert Forge response string to action index."""
        if response == "-1" or response == "":
            return 0  # Pass action

        try:
            # For single indices
            if "," not in response and ":" not in response:
                return int(response) + 1  # Offset by 1 for pass
        except ValueError:
            pass

        return 0

    def _get_action_description(self, response: str, decision: Decision) -> str:
        """Get human-readable description of the action."""
        if response == "-1":
            return "Pass"
        if response == "":
            return "No action"

        try:
            idx = int(response)
            if decision.actions:
                for action in decision.actions:
                    if action.index == idx:
                        return action.description
        except ValueError:
            pass

        return response

    def _get_legal_indices(self, decision: Decision) -> List[int]:
        """Get list of legal action indices."""
        if decision.actions:
            return [a.index for a in decision.actions]
        return []

    def collect_games(
        self,
        num_games: int,
        verbose: bool = True
    ) -> Tuple[List[TrainingSample], List[GameMetrics]]:
        """
        Collect training data from multiple games.

        Returns:
            Tuple of (samples, game_metrics)
        """
        self.samples = []
        self.game_metrics = []

        for game_id in range(1, num_games + 1):
            metrics = self.collect_game(game_id, seed=game_id)

            if verbose:
                print(f"Game {game_id}: {metrics.total_turns} turns, "
                      f"{metrics.total_decisions} decisions, "
                      f"winner: {metrics.winner}")

        return self.samples, self.game_metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from collected games."""
        if not self.game_metrics:
            return {}

        total_games = len(self.game_metrics)
        total_decisions = sum(m.total_decisions for m in self.game_metrics)
        total_turns = sum(m.total_turns for m in self.game_metrics)

        # Decisions per turn
        all_dpt = []
        for m in self.game_metrics:
            all_dpt.extend(m.decisions_per_turn)

        avg_dpt = sum(all_dpt) / len(all_dpt) if all_dpt else 0
        max_dpt = max(all_dpt) if all_dpt else 0

        # Decision type breakdown
        type_counts = defaultdict(int)
        for m in self.game_metrics:
            for dt, count in m.decision_type_counts.items():
                type_counts[dt] += count

        return {
            "total_games": total_games,
            "total_samples": len(self.samples),
            "total_decisions": total_decisions,
            "total_turns": total_turns,
            "avg_turns_per_game": total_turns / total_games,
            "avg_decisions_per_game": total_decisions / total_games,
            "avg_decisions_per_turn": avg_dpt,
            "max_decisions_per_turn": max_dpt,
            "decision_type_breakdown": dict(type_counts),
            "samples_ready": len(self.samples),
        }


# ============================================================================
# Neural Network and Training
# ============================================================================

class SimpleImitationNetwork(nn.Module):
    """
    Simple network for imitation learning.

    Takes flattened game state features and outputs action probabilities.
    This is a placeholder - will be replaced with proper GameStateEncoder.
    """

    def __init__(self, state_dim: int = 512, action_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            state: [B, state_dim] flattened state features
            action_mask: [B, action_dim] 1=legal, 0=illegal

        Returns:
            policy: [B, action_dim] action probabilities
            value: [B, 1] state value estimate
        """
        features = self.encoder(state)

        logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features))

        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        policy = F.softmax(logits, dim=-1)

        return policy, value


class ImitationDataset(Dataset):
    """Dataset for imitation learning from collected samples."""

    def __init__(self, samples: List[TrainingSample], state_dim: int = 512, action_dim: int = 64):
        self.samples = samples
        self.state_dim = state_dim
        self.action_dim = action_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Convert state to features (placeholder - will use proper encoder)
        state = self._encode_state(sample.state_json)

        # Action target
        action = min(sample.action_index, self.action_dim - 1)

        # Action mask (all ones for now - will be proper mask later)
        mask = torch.ones(self.action_dim)

        return state, torch.tensor(action, dtype=torch.long), mask

    def _encode_state(self, state_json: dict) -> torch.Tensor:
        """
        Encode game state to tensor.

        This is a simplified encoding. Will be replaced with proper
        GameStateEncoder for production.
        """
        features = []

        # Player features
        for player in state_json.get("players", []):
            features.extend([
                player.get("life", 20) / 20.0,
                len(player.get("hand", [])) / 10.0,
                len(player.get("battlefield", [])) / 20.0,
                len(player.get("graveyard", [])) / 20.0,
                player.get("library_size", 40) / 60.0,
            ])

        # Pad to fixed size
        while len(features) < self.state_dim:
            features.append(0.0)

        return torch.tensor(features[:self.state_dim], dtype=torch.float32)


class ImitationTrainer:
    """
    Trains neural network to imitate heuristic agent decisions.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.network = network.to(device)
        self.device = device
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

        self.training_history: List[Dict] = []

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()

        total_loss = 0
        total_policy_loss = 0
        correct = 0
        total = 0

        for states, actions, masks in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            masks = masks.to(self.device)

            # Forward
            policy, value = self.network(states, masks)

            # Policy loss (cross-entropy)
            policy_loss = F.cross_entropy(policy.log().clamp(-100, 0), actions)

            # Total loss (just policy for now)
            loss = policy_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()

            predictions = policy.argmax(dim=-1)
            correct += (predictions == actions).sum().item()
            total += actions.size(0)

        self.scheduler.step()

        metrics = {
            "loss": total_loss / len(dataloader),
            "policy_loss": total_policy_loss / len(dataloader),
            "accuracy": correct / total if total > 0 else 0,
            "lr": self.scheduler.get_last_lr()[0],
        }

        self.training_history.append(metrics)
        return metrics

    def train(
        self,
        samples: List[TrainingSample],
        epochs: int = 10,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Train on collected samples.

        Args:
            samples: Training samples from data collection
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            Training history
        """
        dataset = ImitationDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        if verbose:
            print(f"\nTraining on {len(samples)} samples for {epochs} epochs")
            print("-" * 50)

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)

            if verbose:
                print(f"Epoch {epoch+1:3d}: "
                      f"loss={metrics['loss']:.4f}, "
                      f"accuracy={metrics['accuracy']:.3f}, "
                      f"lr={metrics['lr']:.2e}")

        return self.training_history

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Loaded checkpoint from {path}")


# ============================================================================
# Imitation Agent - Uses trained network with fallback
# ============================================================================

class ImitationAgent:
    """
    Agent that uses the trained network for decisions.
    Falls back to heuristic agent when confidence is low.
    """

    def __init__(
        self,
        network: nn.Module,
        fallback: HeuristicAgent,
        confidence_threshold: float = 0.3,
        max_decisions_per_turn: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.network = network.to(device)
        self.network.eval()
        self.device = device

        self.fallback = fallback
        self.confidence_threshold = confidence_threshold
        self.max_decisions_per_turn = max_decisions_per_turn

        self.decisions_this_turn = 0
        self.current_turn = 0

        # Stats
        self.total_decisions = 0
        self.fallback_decisions = 0

    def decide(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """
        Make a decision using the network or fallback.

        Returns:
            Tuple of (response_string, metadata)
        """
        # Track turn
        if decision.turn > self.current_turn:
            self.current_turn = decision.turn
            self.decisions_this_turn = 0
        self.decisions_this_turn += 1
        self.total_decisions += 1

        # Safety: fall back if too many decisions
        if self.decisions_this_turn > self.max_decisions_per_turn:
            self.fallback_decisions += 1
            response, meta = self.fallback.decide(decision)
            meta["used_fallback"] = True
            meta["fallback_reason"] = "decision_limit"
            return response, meta

        # Try network decision
        try:
            with torch.no_grad():
                # Encode state
                state = self._encode_state(decision)
                mask = self._get_action_mask(decision)

                policy, value = self.network(state, mask)

                # Get confidence (max probability)
                confidence = policy.max().item()

                if confidence < self.confidence_threshold:
                    # Low confidence - use fallback
                    self.fallback_decisions += 1
                    response, meta = self.fallback.decide(decision)
                    meta["used_fallback"] = True
                    meta["fallback_reason"] = "low_confidence"
                    meta["network_confidence"] = confidence
                    return response, meta

                # Use network decision
                action_idx = policy.argmax().item()
                response = self._index_to_response(action_idx, decision)

                return response, {
                    "used_fallback": False,
                    "network_confidence": confidence,
                    "value_estimate": value.item(),
                }

        except Exception as e:
            # Error - use fallback
            self.fallback_decisions += 1
            response, meta = self.fallback.decide(decision)
            meta["used_fallback"] = True
            meta["fallback_reason"] = f"error: {str(e)}"
            return response, meta

    def _encode_state(self, decision: Decision) -> torch.Tensor:
        """Encode decision state for network input."""
        # Simplified encoding - will use proper encoder later
        state_json = decision.raw_data.get("game_state", {})

        features = []
        for player in state_json.get("players", []):
            features.extend([
                player.get("life", 20) / 20.0,
                len(player.get("hand", [])) / 10.0,
                len(player.get("battlefield", [])) / 20.0,
                len(player.get("graveyard", [])) / 20.0,
                player.get("library_size", 40) / 60.0,
            ])

        while len(features) < 512:
            features.append(0.0)

        return torch.tensor([features[:512]], dtype=torch.float32, device=self.device)

    def _get_action_mask(self, decision: Decision) -> torch.Tensor:
        """Get action mask for legal actions."""
        mask = torch.zeros(1, 64, device=self.device)

        # Set legal actions
        if decision.actions:
            for action in decision.actions:
                idx = min(action.index + 1, 63)  # +1 for pass at 0
                if idx >= 0:
                    mask[0, idx] = 1.0

        # Always allow pass
        mask[0, 0] = 1.0

        return mask

    def _index_to_response(self, idx: int, decision: Decision) -> str:
        """Convert network action index to Forge response."""
        if idx == 0:
            return "-1"  # Pass

        # Convert to Forge action index
        forge_idx = idx - 1

        if decision.actions:
            for action in decision.actions:
                if action.index == forge_idx:
                    return str(forge_idx)

        return "-1"  # Default to pass

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        fallback_rate = (
            self.fallback_decisions / self.total_decisions
            if self.total_decisions > 0 else 0
        )
        return {
            "total_decisions": self.total_decisions,
            "fallback_decisions": self.fallback_decisions,
            "fallback_rate": fallback_rate,
        }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Forge AI Imitation Training")
    parser.add_argument("--host", default="localhost", help="Forge daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Forge daemon port")
    parser.add_argument("--deck1", default="red_aggro.dck", help="First deck")
    parser.add_argument("--deck2", default="white_weenie.dck", help="Second deck")
    parser.add_argument("--games", type=int, default=10, help="Number of games for data collection")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--eval-games", type=int, default=5, help="Evaluation games")
    parser.add_argument("--checkpoint", default="checkpoints/imitation.pt", help="Checkpoint path")
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train on existing data")
    args = parser.parse_args()

    print("=" * 70)
    print("FORGE AI IMITATION TRAINING")
    print("=" * 70)
    print(f"Host: {args.host}:{args.port}")
    print(f"Decks: {args.deck1} vs {args.deck2}")
    print(f"Games: {args.games}, Epochs: {args.epochs}")
    print()

    # Create checkpoint directory
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: Data Collection
    print("=" * 70)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 70)

    collector = ImitationDataCollector(
        host=args.host,
        port=args.port,
        deck1=args.deck1,
        deck2=args.deck2,
    )

    samples, metrics = collector.collect_games(args.games, verbose=True)
    stats = collector.get_statistics()

    print("\n--- Collection Statistics ---")
    print(f"Total games: {stats['total_games']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Avg turns/game: {stats['avg_turns_per_game']:.1f}")
    print(f"Avg decisions/game: {stats['avg_decisions_per_game']:.1f}")
    print(f"Avg decisions/turn: {stats['avg_decisions_per_turn']:.1f}")
    print(f"Max decisions/turn: {stats['max_decisions_per_turn']}")
    print("\nDecision types:")
    for dt, count in stats['decision_type_breakdown'].items():
        print(f"  {dt}: {count}")

    if args.collect_only:
        print("\n--collect-only flag set, skipping training.")
        return

    # Phase 2: Training
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING")
    print("=" * 70)

    network = SimpleImitationNetwork(state_dim=512, action_dim=64)
    trainer = ImitationTrainer(network, lr=1e-3, batch_size=64)

    trainer.train(samples, epochs=args.epochs, verbose=True)
    trainer.save(args.checkpoint)

    # Phase 3: Evaluation
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION")
    print("=" * 70)

    fallback = HeuristicAgent()
    agent = ImitationAgent(
        network=network,
        fallback=fallback,
        confidence_threshold=0.3,
        max_decisions_per_turn=50,
    )

    print(f"\nRunning {args.eval_games} evaluation games with trained agent...")

    eval_metrics = []
    for game_id in range(1, args.eval_games + 1):
        client = ForgeClient(args.host, args.port)

        try:
            client.connect()
            client.start_game(args.deck1, args.deck2, seed=game_id + 1000)

            decisions = 0
            turns = 0
            current_turn = 0

            while True:
                decision = client.receive_decision()
                if decision is None:
                    break

                if decision.turn > current_turn:
                    current_turn = decision.turn
                    turns = current_turn

                response, meta = agent.decide(decision)
                client.send_response(response)
                decisions += 1

            result = client.get_result()
            winner = result.winner if result else "Unknown"

            print(f"Eval Game {game_id}: {turns} turns, {decisions} decisions, "
                  f"winner: {winner}")

            eval_metrics.append({
                "game_id": game_id,
                "turns": turns,
                "decisions": decisions,
                "winner": winner,
            })

        except Exception as e:
            print(f"Eval Game {game_id}: Error - {e}")
        finally:
            client.close()

    # Final stats
    agent_stats = agent.get_stats()
    print("\n--- Agent Statistics ---")
    print(f"Total decisions: {agent_stats['total_decisions']}")
    print(f"Fallback decisions: {agent_stats['fallback_decisions']}")
    print(f"Fallback rate: {agent_stats['fallback_rate']:.1%}")

    if eval_metrics:
        avg_turns = sum(m['turns'] for m in eval_metrics) / len(eval_metrics)
        avg_decisions = sum(m['decisions'] for m in eval_metrics) / len(eval_metrics)
        print(f"\nAvg turns/game: {avg_turns:.1f}")
        print(f"Avg decisions/game: {avg_decisions:.1f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
