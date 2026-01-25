#!/usr/bin/env python3
"""
Simple Agent for Forge Games

A minimal agent that makes reasonable decisions to produce proper game flow.
This is used to collect training data while we implement proper Forge AI suggestions.

Key insight: The game wasn't advancing because our agent wasn't making proper plays.
This agent focuses on:
1. Playing lands to establish mana
2. Casting affordable creatures/spells
3. Attacking when we have creatures
4. Passing when nothing useful to do

The goal is to produce games that complete in reasonable time (< 20 turns,
< 100 decisions per turn) for training data collection.
"""

import random
from typing import Tuple, Dict, Any, Optional

from src.forge.forge_client import Decision, DecisionType, ActionOption


class SimpleAgent:
    """
    A simple rule-based agent that produces reasonable game flow.

    This agent is NOT trying to play optimally - it's trying to produce
    games that complete in reasonable time for training data collection.
    """

    def __init__(self):
        self.decisions_this_turn = 0
        self.current_turn = 0
        self.played_land_this_turn = False

    def reset_turn(self):
        """Reset per-turn tracking."""
        self.decisions_this_turn = 0
        self.played_land_this_turn = False

    def decide(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Make a decision and return (response, metadata)."""

        # Track turn changes
        if decision.turn > self.current_turn:
            self.current_turn = decision.turn
            self.reset_turn()
        self.decisions_this_turn += 1

        # Safety: if too many decisions, force pass
        if self.decisions_this_turn > 50:
            return self._get_pass(decision), {"reason": "decision_limit"}

        dtype = decision.decision_type

        if dtype == DecisionType.CHOOSE_ACTION:
            return self._choose_action(decision)
        elif dtype == DecisionType.DECLARE_ATTACKERS:
            return self._declare_attackers(decision)
        elif dtype == DecisionType.DECLARE_BLOCKERS:
            return self._declare_blockers(decision)
        elif dtype == DecisionType.PLAY_TRIGGER:
            return "y", {"reason": "trigger_yes"}
        elif dtype == DecisionType.CONFIRM_ACTION:
            return "y", {"reason": "confirm_yes"}
        elif dtype == DecisionType.CHOOSE_CARDS:
            return self._choose_cards(decision)
        elif dtype == DecisionType.CHOOSE_ENTITY:
            return "0", {"reason": "first_entity"}
        elif dtype == DecisionType.ANNOUNCE_VALUE:
            return "0", {"reason": "min_value"}
        else:
            return "0", {"reason": "default"}

    def _get_pass(self, decision: Decision) -> str:
        """Get the appropriate pass response."""
        if decision.decision_type == DecisionType.CHOOSE_ACTION:
            return "-1"
        elif decision.decision_type in [DecisionType.DECLARE_ATTACKERS,
                                         DecisionType.DECLARE_BLOCKERS]:
            return ""
        return "0"

    def _choose_action(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose an action (main decision type)."""
        actions = decision.actions
        if not actions:
            return "-1", {"reason": "no_actions"}

        gs = decision.game_state

        # Find our player
        our_player = None
        for p in gs.players:
            if p.name == decision.player:
                our_player = p
                break

        if not our_player:
            return "-1", {"reason": "no_player"}

        # Count untapped lands for mana estimate
        untapped_lands = sum(1 for c in our_player.battlefield
                            if c.is_land and not c.tapped)

        # 1. PLAY LAND (if we haven't this turn)
        if not self.played_land_this_turn:
            land_actions = [a for a in actions if a.is_land]
            if land_actions:
                land = land_actions[0]  # Just pick the first one
                self.played_land_this_turn = True
                return str(land.index), {"reason": "play_land", "card": land.card}

        # 2. CAST CREATURE (in main phase, if we have mana)
        if gs.phase in ["MAIN1", "MAIN2"] and untapped_lands > 0:
            creature_actions = self._find_creature_actions(actions)
            if creature_actions:
                # Sort by estimated cost, pick cheapest we can afford
                creature_actions.sort(key=lambda a: self._guess_cmc(a))
                for creature in creature_actions:
                    cmc = self._guess_cmc(creature)
                    if cmc <= untapped_lands:
                        return str(creature.index), {
                            "reason": "cast_creature",
                            "card": creature.card
                        }

        # 3. PASS - Default action
        return "-1", {"reason": "pass"}

    def _declare_attackers(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Declare attackers - attack with all untapped creatures."""
        attackers = decision.attackers
        if not attackers:
            return "", {"reason": "no_attackers"}

        # Attack with all creatures
        indices = [str(a["index"]) for a in attackers]
        return ",".join(indices), {"reason": "attack_all", "count": len(indices)}

    def _declare_blockers(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Declare blockers - block to survive if possible."""
        blockers = decision.blockers
        if not blockers:
            return "", {"reason": "no_blockers"}

        # Don't block for simplicity (faster games)
        return "", {"reason": "no_blocks"}

    def _choose_cards(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose cards for effects."""
        cards = decision.cards
        if not cards:
            return "", {"reason": "no_cards"}

        # Pick minimum required
        min_cards = decision.raw_data.get("min", 0)
        if min_cards > 0 and cards:
            indices = [str(i) for i in range(min(min_cards, len(cards)))]
            return ",".join(indices), {"reason": "choose_min"}
        return "", {"reason": "choose_none"}

    def _find_creature_actions(self, actions) -> list:
        """Find actions that cast creatures."""
        creatures = []
        for a in actions:
            if a.is_land or a.index < 0:
                continue
            # Check if description mentions creature type
            desc = a.description.lower()
            if "creature" in desc or "- creature" in a.card.lower():
                creatures.append(a)
                continue
            # Check card name for common creature patterns
            card_lower = a.card.lower()
            if any(word in card_lower for word in ["goblin", "soldier", "knight", "lion"]):
                creatures.append(a)
        return creatures

    def _guess_cmc(self, action: ActionOption) -> int:
        """Guess converted mana cost from action."""
        # Try to extract from description
        # Look for patterns like "{1}{R}" or "{2}{W}{W}"
        desc = action.description

        count = 0
        i = 0
        while i < len(desc):
            if desc[i] == '{':
                end = desc.find('}', i)
                if end > i:
                    mana = desc[i+1:end]
                    if mana.isdigit():
                        count += int(mana)
                    elif mana in 'WUBRG':
                        count += 1
                    i = end
            i += 1

        return max(1, count)  # At least 1


def run_test_game():
    """Test the simple agent with a game."""
    from src.forge.forge_client import ForgeClient

    client = ForgeClient('localhost', 17171, timeout=30)
    agent = SimpleAgent()

    try:
        client.connect()
        print("Connected to Forge daemon")

        success = client.start_game('red_aggro.dck', 'white_weenie.dck', seed=12345)
        if not success:
            print("Failed to start game")
            return

        print("Game started!")

        decisions = 0
        max_decisions = 500

        turn_decisions = {}
        current_turn = 0

        while decisions < max_decisions:
            decision = client.receive_decision()
            if decision is None:
                print("Game ended")
                break

            decisions += 1

            if decision.turn > current_turn:
                if current_turn > 0:
                    print(f"Turn {current_turn}: {turn_decisions.get(current_turn, 0)} decisions")
                current_turn = decision.turn
                turn_decisions[current_turn] = 0
                agent.reset_turn()

            turn_decisions[current_turn] = turn_decisions.get(current_turn, 0) + 1

            response, meta = agent.decide(decision)

            # Debug first few decisions
            if decisions <= 5:
                print(f"  Decision {decisions}: {meta.get('reason', '?')} -> {response}")

            client.send_response(response)

        # Print last turn
        if current_turn > 0:
            print(f"Turn {current_turn}: {turn_decisions.get(current_turn, 0)} decisions")

        result = client.get_result()
        print(f"\nResult: {decisions} decisions, winner: {result.winner if result else 'unknown'}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    run_test_game()
