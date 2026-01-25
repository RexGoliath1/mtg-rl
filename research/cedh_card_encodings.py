"""
cEDH Card Mechanics Encoding Examples

Three complex Commander staples to demonstrate the encoding approach:
1. Saw in Half - Token/copy mechanics
2. Deflecting Swat - Interaction/redirection
3. Rhystic Study - Triggered card advantage

These represent different categories of powerful effects that need
proper mechanical representation.
"""

# =============================================================================
# MECHANICS VOCABULARY (Extended for these cards)
# =============================================================================

MECHANICS = {
    # Targeting
    "TARGET_CREATURE": 1,
    "TARGET_SPELL": 2,
    "TARGET_ABILITY": 3,
    "TARGET_SPELL_OR_ABILITY": 4,
    "TARGET_PLAYER": 5,
    "TARGET_ANY": 6,

    # Removal/Interaction
    "DESTROY": 10,
    "EXILE": 11,
    "COUNTER": 12,
    "REDIRECT": 13,
    "CHANGE_TARGETS": 14,

    # Creation
    "CREATE_TOKEN": 20,
    "COPY": 21,
    "TOKEN_COPY": 22,

    # Card advantage
    "DRAW": 30,
    "DRAW_OPTIONAL": 31,  # "may draw"
    "DISCARD": 32,
    "MILL": 33,

    # Mana/Costs
    "FREE_CAST_CONDITION": 40,  # Can cast for free if condition
    "ALTERNATIVE_COST": 41,
    "TAX_OPPONENT": 42,  # Opponent must pay or X happens

    # Triggers
    "ETB_TRIGGER": 50,
    "DEATH_TRIGGER": 51,
    "CAST_TRIGGER": 52,
    "OPPONENT_CASTS": 53,
    "IF_TARGET_DIES": 54,

    # Conditions
    "IF_COMMANDER": 60,  # "If you control a commander"
    "UNLESS_PAYS": 61,   # "unless that player pays {X}"

    # Timing
    "INSTANT_SPEED": 70,
    "SORCERY_SPEED": 71,
    "STATIC": 72,  # Always-on effect

    # Stats
    "HALF_STATS": 80,
    "MODIFY_POWER": 81,
    "MODIFY_TOUGHNESS": 82,

    # Special
    "CHOOSE_NEW_TARGETS": 90,  # Redirect effect
}


# =============================================================================
# CARD 1: Saw in Half {2}{B}
# =============================================================================
# "Destroy target creature. If that creature dies this way, its controller
#  creates two tokens that are copies of that creature, except their power
#  is half that creature's power and their toughness is half that creature's
#  toughness. Round up each time."

saw_in_half = {
    "name": "Saw in Half",
    "cost": {"generic": 2, "B": 1},
    "types": ["instant"],

    # Mechanics sequence
    "mechanics": [
        MECHANICS["INSTANT_SPEED"],
        MECHANICS["TARGET_CREATURE"],
        MECHANICS["TARGET_ANY"],  # Can target yours or opponent's
        MECHANICS["DESTROY"],
        MECHANICS["IF_TARGET_DIES"],
        MECHANICS["CREATE_TOKEN"],
        MECHANICS["TOKEN_COPY"],
        MECHANICS["HALF_STATS"],
    ],

    # Quantitative parameters
    "params": {
        "token_count": 2,
        "stat_modifier": 0.5,  # Half
        "rounding": "up",
    },

    # Derived strategic properties (could be learned)
    "strategy": {
        "removal": True,
        "etb_doubler": True,  # Key insight: use on YOUR creatures
        "death_synergy": True,
        "token_synergy": True,
    }
}


# =============================================================================
# CARD 2: Deflecting Swat {2}{R}
# =============================================================================
# "If you control a commander, you may cast this spell without paying its
#  mana cost. You may choose new targets for target spell or ability."

deflecting_swat = {
    "name": "Deflecting Swat",
    "cost": {"generic": 2, "R": 1},
    "types": ["instant"],

    # Mechanics sequence
    "mechanics": [
        MECHANICS["INSTANT_SPEED"],
        MECHANICS["FREE_CAST_CONDITION"],
        MECHANICS["IF_COMMANDER"],  # The condition for free cast
        MECHANICS["TARGET_SPELL_OR_ABILITY"],
        MECHANICS["CHOOSE_NEW_TARGETS"],
    ],

    "params": {
        "alternative_cost": 0,  # Free if condition met
        "condition": "control_commander",
    },

    "strategy": {
        "protection": True,      # Redirect removal away from your stuff
        "disruption": True,      # Redirect opponent's spells to bad targets
        "free_spell": True,      # Often costs 0 mana
        "instant_interaction": True,
    }
}


# =============================================================================
# CARD 3: Rhystic Study {2}{U}
# =============================================================================
# "Whenever an opponent casts a spell, you may draw a card unless that
#  player pays {1}."

rhystic_study = {
    "name": "Rhystic Study",
    "cost": {"generic": 2, "U": 1},
    "types": ["enchantment"],

    # Mechanics sequence
    "mechanics": [
        MECHANICS["STATIC"],  # Permanent effect
        MECHANICS["OPPONENT_CASTS"],  # Trigger condition
        MECHANICS["DRAW_OPTIONAL"],
        MECHANICS["UNLESS_PAYS"],
        MECHANICS["TAX_OPPONENT"],
    ],

    "params": {
        "tax_amount": 1,  # {1} to avoid trigger
        "draw_count": 1,
        "trigger_scope": "each_opponent",  # In multiplayer, triggers per opponent
    },

    "strategy": {
        "card_advantage": True,
        "tax_effect": True,  # Slows opponents
        "multiplayer_scaling": True,  # Better with more opponents
        "passive": True,  # No mana investment after cast
    }
}


# =============================================================================
# ENCODING COMPARISON
# =============================================================================

def encode_card(card):
    """Convert card to fixed-size feature vector for network input."""
    # This would be the input to the mechanics encoder

    # Mechanics as multi-hot (which mechanics present)
    mechanics_vector = [0] * 100  # Assuming 100 vocab size
    for m in card["mechanics"]:
        mechanics_vector[m] = 1

    # Basic attributes
    cmc = sum(card["cost"].values()) if isinstance(card["cost"], dict) else card["cost"]
    colors = {
        "W": 1 if "W" in card.get("cost", {}) else 0,
        "U": 1 if "U" in card.get("cost", {}) else 0,
        "B": 1 if "B" in card.get("cost", {}) else 0,
        "R": 1 if "R" in card.get("cost", {}) else 0,
        "G": 1 if "G" in card.get("cost", {}) else 0,
    }

    return {
        "mechanics_vector": mechanics_vector,
        "cmc": cmc,
        "colors": colors,
        "params": card.get("params", {}),
    }


# =============================================================================
# WHY THIS APPROACH WORKS FOR ALPHAZERO
# =============================================================================

"""
The key insight: We don't need to pre-compute "Saw in Half + Mulldrifter = draw 4"

Instead:
1. Encode cards by their mechanics
2. Network learns mechanic interactions through self-play
3. MCTS explores game trees where these combos naturally emerge

Example MCTS trace:
- State: I have Mulldrifter (2/2 flyer, drew 2 on ETB), 3 mana, Saw in Half in hand
- Action: Cast Saw in Half targeting Mulldrifter
- Forge simulates: Mulldrifter dies, 2x 1/1 flying tokens enter, draw 2 each = 4 cards
- Network observes: This action led to +4 cards, very high value
- Learns: DESTROY + TOKEN_COPY + (creature with ETB_DRAW) = high value

The network generalizes:
- Never saw Mulldrifter specifically
- But learns: TOKEN_COPY effects multiply ETB triggers
- Applies to ANY ETB creature: Snapcaster, Solemn Simulacrum, etc.
"""


# =============================================================================
# COMPARISON: What the network needs to learn
# =============================================================================

interactions_to_discover = """
Card Combo Discovery (via self-play, not pre-coded):

1. Saw in Half + ETB creature
   - Input: [DESTROY, TOKEN_COPY, HALF_STATS] + [ETB_TRIGGER, DRAW]
   - Learned: Token copies trigger ETB again = 2x value
   - No explicit "Saw in Half + Mulldrifter" rule needed

2. Deflecting Swat + Opponent's removal
   - Input: [CHANGE_TARGETS] + opponent plays [DESTROY] targeting my creature
   - Learned: Can redirect to opponent's creature instead
   - Discovered through MCTS: "What if I redirect their Swords to their commander?"

3. Rhystic Study + Multiplayer
   - Input: [OPPONENT_CASTS, DRAW, UNLESS_PAYS] + 3 opponents
   - Learned: 3x more triggers than 1v1
   - Value scales with opponent count

4. Emergent: Deflecting Swat + Saw in Half interaction
   - Opponent targets my Mulldrifter with removal
   - I cast Deflecting Swat, redirect to a different target
   - Then cast Saw in Half on Mulldrifter for value
   - MCTS discovers this multi-card line through tree search
"""


if __name__ == "__main__":
    print("=" * 60)
    print("cEDH Card Mechanics Encoding")
    print("=" * 60)

    for card in [saw_in_half, deflecting_swat, rhystic_study]:
        print(f"\n{card['name']}:")
        print(f"  Cost: {card['cost']}")
        print(f"  Mechanics IDs: {card['mechanics']}")
        encoded = encode_card(card)
        print(f"  CMC: {encoded['cmc']}")
        print(f"  Mechanics count: {sum(encoded['mechanics_vector'])}")

    print("\n" + "=" * 60)
    print("Key insight: Network learns interactions through MCTS + self-play")
    print("No need to pre-code 'Saw in Half + Mulldrifter = draw 4'")
    print("=" * 60)
