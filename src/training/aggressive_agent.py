#!/usr/bin/env python3
"""
Aggressive Agent for Forge Games

An agent that actively plays cards to produce faster, more interactive games.
Key changes from SimpleAgent:
1. Always play lands
2. Cast spells as soon as affordable
3. Use burn spells on opponent
4. Attack with all creatures every turn
"""

import random
from typing import Tuple, Dict, Any, List

from src.forge.forge_client import Decision, DecisionType, ActionOption


class AggressiveAgent:
    """
    An aggressive agent that plays fast games.
    """

    def __init__(self):
        self.decisions_this_turn = 0
        self.current_turn = 0
        self.played_land_this_turn = False
        self.actions_taken_this_phase = []

    def reset_turn(self):
        self.decisions_this_turn = 0
        self.played_land_this_turn = False
        self.actions_taken_this_phase = []

    def decide(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Make an aggressive decision."""

        if decision.turn > self.current_turn:
            self.current_turn = decision.turn
            self.reset_turn()
        self.decisions_this_turn += 1

        # Safety limit
        if self.decisions_this_turn > 30:
            return self._pass(decision), {"reason": "decision_limit"}

        dtype = decision.decision_type

        if dtype == DecisionType.CHOOSE_ACTION:
            return self._choose_action(decision)
        elif dtype == DecisionType.DECLARE_ATTACKERS:
            return self._attack_all(decision)
        elif dtype == DecisionType.DECLARE_BLOCKERS:
            return self._block_best(decision)
        elif dtype == DecisionType.PLAY_TRIGGER:
            return "y", {"reason": "trigger"}
        elif dtype == DecisionType.CONFIRM_ACTION:
            return "y", {"reason": "confirm"}
        elif dtype == DecisionType.CHOOSE_CARDS:
            return self._choose_cards(decision)
        elif dtype == DecisionType.CHOOSE_ENTITY:
            # For targeting - prefer opponent for damage
            return self._choose_target(decision)
        elif dtype == DecisionType.ANNOUNCE_VALUE:
            return "0", {"reason": "announce"}
        else:
            return "0", {"reason": "default"}

    def _pass(self, decision: Decision) -> str:
        if decision.decision_type == DecisionType.CHOOSE_ACTION:
            return "-1"
        return ""

    def _choose_action(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose action aggressively."""
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

        # Check lands played from game state (more reliable than our tracking)
        can_play_land = our_player.lands_played_this_turn < our_player.max_land_plays

        # Count available mana from untapped lands
        mana = sum(1 for c in our_player.battlefield if c.is_land and not c.tapped)

        # 1. PLAY LAND FIRST (in main phases)
        if can_play_land and gs.phase in ["MAIN1", "MAIN2"]:
            for a in actions:
                if a.is_land:
                    return str(a.index), {"reason": "play_land", "card": a.card}

        # 2. CAST CREATURES (main phases, with mana)
        if mana > 0 and gs.phase in ["MAIN1", "MAIN2"]:
            # Sort by CMC (cast cheapest first to maximize board presence)
            creature_actions = []
            for a in actions:
                if a.index < 0 or a.is_land:
                    continue
                desc = a.description.lower()
                # Check for creatures - description contains "Creature X / Y" pattern
                # Avoid instants/sorceries that say "creature" in effect text
                if " - creature " in desc or "creature " in desc[:50]:
                    cmc = self._parse_mana_cost(a.mana_cost)
                    if cmc <= mana:
                        creature_actions.append((cmc, a))

            if creature_actions:
                creature_actions.sort(key=lambda x: x[0])
                _, best = creature_actions[0]
                return str(best.index), {"reason": "creature", "card": best.card}

        # 3. CAST BURN SPELLS (any time with mana)
        if mana > 0:
            for a in actions:
                if a.index < 0 or a.is_land:
                    continue
                desc = a.description.lower()
                # Check for damage spells
                if any(word in desc for word in ["damage", "deal", "bolt", "shock"]):
                    cmc = self._parse_mana_cost(a.mana_cost)
                    if cmc <= mana:
                        return str(a.index), {"reason": "burn", "card": a.card}

        # 4. PASS
        return "-1", {"reason": "pass"}

    def _parse_mana_cost(self, mana_cost: str) -> int:
        """Parse mana cost string like '{1}{R}{R}' into CMC."""
        if not mana_cost or mana_cost == "no cost":
            return 0
        count = 0
        i = 0
        while i < len(mana_cost):
            if mana_cost[i] == '{':
                end = mana_cost.find('}', i)
                if end > i:
                    symbol = mana_cost[i+1:end]
                    if symbol.isdigit():
                        count += int(symbol)
                    elif len(symbol) == 1 and symbol in 'WUBRG':
                        count += 1
                    elif '/' in symbol:  # Hybrid like G/W
                        count += 1
                    i = end
            i += 1
        return count

    def _attack_all(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Attack with all creatures."""
        attackers = decision.attackers
        if not attackers:
            return "", {"reason": "no_attackers"}
        indices = [str(a["index"]) for a in attackers]
        return ",".join(indices), {"reason": "attack_all", "count": len(indices)}

    def _block_best(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Block to trade favorably."""
        blockers = decision.blockers
        if not blockers:
            return "", {"reason": "no_blockers"}

        gs = decision.game_state
        if not gs.combat or not gs.combat.attackers:
            return "", {"reason": "no_combat"}

        # Block biggest attacker with best blocker
        attackers = gs.combat.attackers
        if not attackers:
            return "", {"reason": "no_attackers"}

        # Sort attackers by power
        sorted_attackers = sorted(attackers, key=lambda x: x.get("power", 0), reverse=True)
        biggest = sorted_attackers[0]

        # Get attacker index (try different key names)
        attacker_idx = biggest.get("index", biggest.get("card_id", 0))

        # Find best blocker - one that can kill the attacker
        for b in blockers:
            if b.get("power", 0) >= biggest.get("toughness", 0):
                return f"{b['index']}:{attacker_idx}", {"reason": "block"}

        return "", {"reason": "no_good_blocks"}

    def _choose_cards(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose cards for effects."""
        cards = decision.cards
        if not cards:
            return "", {"reason": "no_cards"}

        min_cards = decision.raw_data.get("min", 0)
        if min_cards > 0:
            indices = list(range(min(min_cards, len(cards))))
            return ",".join(str(i) for i in indices), {"reason": "choose_min"}
        return "", {"reason": "choose_none"}

    def _choose_target(self, decision: Decision) -> Tuple[str, Dict[str, Any]]:
        """Choose target - prefer opponent for damage."""
        # For entity selection, 0 is usually the first option
        # Check raw_data for entity list
        entities = decision.raw_data.get("entities", [])

        # Prefer opponent player for damage spells
        for i, e in enumerate(entities):
            if isinstance(e, dict):
                name = e.get("name", "").lower()
                if "opponent" in name or "agent(2)" in name.lower() or "agent(1)" in name.lower():
                    # Check if it's the opponent
                    if decision.player not in name:
                        return str(i), {"reason": "target_opponent"}

        return "0", {"reason": "first_target"}

    def _guess_cmc(self, action: ActionOption) -> int:
        """Estimate mana cost."""
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
                    elif len(mana) == 1 and mana in 'WUBRG':
                        count += 1
                    i = end
            i += 1
        return max(1, count)


def test_agent():
    """Test the aggressive agent."""
    from src.forge.forge_client import ForgeClient
    from collections import defaultdict

    client = ForgeClient('localhost', 17171, timeout=60)
    agent = AggressiveAgent()

    try:
        client.connect()
        client.start_game('red_aggro.dck', 'white_weenie.dck', seed=99999)

        decisions = 0
        max_decisions = 500
        turn_decisions = defaultdict(int)
        reasons = defaultdict(int)

        while decisions < max_decisions:
            decision = client.receive_decision()
            if decision is None:
                break

            decisions += 1
            turn_decisions[decision.turn] += 1

            if decision.turn > agent.current_turn:
                agent.current_turn = decision.turn
                agent.reset_turn()

            response, meta = agent.decide(decision)
            reasons[meta.get("reason", "unknown")] += 1

            client.send_response(response)

        result = client.get_result()

        print(f"Game completed: {decisions} decisions")
        print(f"Turns: {max(turn_decisions.keys()) if turn_decisions else 0}")
        print(f"Winner: {result.winner if result else 'unknown'}")
        print(f"\nDecisions per turn:")
        for t in sorted(turn_decisions.keys())[:10]:
            print(f"  Turn {t}: {turn_decisions[t]}")
        print(f"\nAction reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {r}: {c}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    test_agent()
