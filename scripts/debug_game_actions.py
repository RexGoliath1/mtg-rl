#!/usr/bin/env python3
"""
Debug Game Actions

Run a game and dump all available actions to understand what Forge is offering.
"""

import os
import json
from collections import defaultdict

from src.forge.forge_client import ForgeClient, DecisionType


def run_debug_game(host: str = "localhost", port: int = 17171, max_decisions: int = 100):
    """Run a game and dump action details."""
    client = ForgeClient(host, port, timeout=60)

    action_examples = []
    decision_counts = defaultdict(int)
    action_stats = defaultdict(int)

    try:
        client.connect()
        print("Connected to Forge daemon")

        client.start_game('red_aggro.dck', 'white_weenie.dck', seed=12345)
        print("Game started!\n")

        decisions = 0
        current_turn = 0

        while decisions < max_decisions:
            decision = client.receive_decision()
            if decision is None:
                print("\nGame ended")
                break

            decisions += 1
            decision_counts[decision.decision_type.value] += 1

            if decision.turn > current_turn:
                current_turn = decision.turn
                print(f"\n{'='*60}")
                print(f"TURN {current_turn}")
                print(f"{'='*60}")

            # Find our player
            our_player = None
            for p in decision.game_state.players:
                if p.name == decision.player:
                    our_player = p
                    break

            if decision.decision_type == DecisionType.CHOOSE_ACTION:
                print(f"\n--- Decision {decisions}: CHOOSE_ACTION ({decision.phase}) ---")
                print(f"Player: {decision.player}")

                if our_player:
                    print(f"Life: {our_player.life}")
                    print(f"Lands played this turn: {our_player.lands_played_this_turn}/{our_player.max_land_plays}")
                    print(f"Mana pool: W={our_player.mana_pool.white} U={our_player.mana_pool.blue} "
                          f"B={our_player.mana_pool.black} R={our_player.mana_pool.red} "
                          f"G={our_player.mana_pool.green} C={our_player.mana_pool.colorless} "
                          f"Total={our_player.mana_pool.total}")
                    print(f"Hand size: {our_player.hand_size}")
                    print(f"Battlefield: {len(our_player.battlefield)} permanents")

                    # Count lands
                    lands = [c for c in our_player.battlefield if c.is_land]
                    untapped_lands = [c for c in lands if not c.tapped]
                    print(f"Lands: {len(lands)} total, {len(untapped_lands)} untapped")

                print(f"\nAvailable actions ({len(decision.actions)}):")
                for a in decision.actions:
                    action_stats[f"{a.is_land}|{a.mana_cost}"] += 1

                    # Print action details
                    print(f"  [{a.index}] {a.card or 'N/A'}")
                    print(f"       desc: {a.description[:80]}...")
                    print(f"       is_land: {a.is_land}, mana_cost: '{a.mana_cost}', card_id: {a.card_id}")

                    # Collect examples
                    if len(action_examples) < 50:
                        action_examples.append({
                            "index": a.index,
                            "card": a.card,
                            "description": a.description,
                            "is_land": a.is_land,
                            "mana_cost": a.mana_cost,
                            "card_id": a.card_id,
                        })

                # Simple response: play first land or pass
                response = "-1"
                reason = "pass"
                for a in decision.actions:
                    if a.is_land and our_player and our_player.lands_played_this_turn < our_player.max_land_plays:
                        response = str(a.index)
                        reason = "play_land"
                        break

                print(f"\n  -> Response: {response} ({reason})")
                client.send_response(response)

            elif decision.decision_type == DecisionType.DECLARE_ATTACKERS:
                print(f"\n--- Decision {decisions}: DECLARE_ATTACKERS ---")
                print(f"Available attackers: {len(decision.attackers)}")
                for a in decision.attackers:
                    print(f"  [{a['index']}] {a.get('name', 'Unknown')} - P/T: {a.get('power', '?')}/{a.get('toughness', '?')}")

                # Attack with all
                if decision.attackers:
                    indices = [str(a["index"]) for a in decision.attackers]
                    response = ",".join(indices)
                    print(f"  -> Attacking with: {response}")
                else:
                    response = ""
                    print("  -> No attackers")
                client.send_response(response)

            elif decision.decision_type == DecisionType.DECLARE_BLOCKERS:
                print(f"\n--- Decision {decisions}: DECLARE_BLOCKERS ---")
                print(f"Available blockers: {len(decision.blockers)}")
                client.send_response("")

            else:
                print(f"\n--- Decision {decisions}: {decision.decision_type.value} ---")
                # Default responses
                if decision.decision_type == DecisionType.PLAY_TRIGGER:
                    client.send_response("y")
                elif decision.decision_type == DecisionType.CONFIRM_ACTION:
                    client.send_response("y")
                elif decision.decision_type == DecisionType.CHOOSE_ENTITY:
                    client.send_response("0")
                elif decision.decision_type == DecisionType.CHOOSE_CARDS:
                    client.send_response("")
                else:
                    client.send_response("0")

        result = client.get_result()
        print(f"\n{'='*60}")
        print("GAME RESULT")
        print(f"{'='*60}")
        print(f"Winner: {result.winner if result else 'unknown'}")
        print(f"Total decisions: {decisions}")

        print("\nDecision type counts:")
        for dt, count in sorted(decision_counts.items()):
            print(f"  {dt}: {count}")

        # Save action examples
        output_path = "reports/action_examples.json"
        os.makedirs("reports", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(action_examples, f, indent=2)
        print(f"\nSaved {len(action_examples)} action examples to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    run_debug_game()
