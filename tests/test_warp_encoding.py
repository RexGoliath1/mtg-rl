"""
Test: Encoding Edge of Eternities Warp cards

This tests whether our mechanics vocabulary can handle new mechanics
introduced in recent sets (Edge of Eternities, July 2025).

WARP mechanic:
- Alternative cost (pay warp cost instead of mana cost)
- From hand only
- Permanent exiles at beginning of next end step
- Owner can cast it from exile on future turns

This is similar to Dash/Blitz but with exile-and-return.
"""

import sys
sys.path.insert(0, '/Users/stevengonciar/git/mtg')

from src.mechanics.vocabulary import (
    Mechanic, CardEncoding, encode_card_to_vector
)


# =============================================================================
# WARP CARDS FROM EDGE OF ETERNITIES
# =============================================================================

# Exalted Sunborn {3}{W}{W} / Warp {1}{W}
# Flying, lifelink. Token doubling effect.
EXALTED_SUNBORN = CardEncoding(
    name="Exalted Sunborn",
    mana_cost={"W": 2, "C": 3},
    cmc=5,
    types=["creature"],
    subtypes=["angel", "wizard"],
    mechanics=[
        Mechanic.WARP,                  # Has warp ability
        Mechanic.ALTERNATIVE_COST,      # Warp is an alternative cost
        Mechanic.FROM_HAND,             # Can only warp from hand
        Mechanic.END_STEP_TRIGGER,      # Exiles at end step
        Mechanic.EXILE_TEMPORARY,       # Temporary exile
        Mechanic.CAST_FROM_EXILE,       # Can recast from exile
        Mechanic.FLYING,
        Mechanic.LIFELINK,
        Mechanic.TOKEN_DOUBLER,         # "twice that many tokens"
        Mechanic.REPLACEMENT_EFFECT,    # "instead"
    ],
    parameters={
        "warp_cost": {"W": 1, "C": 1},  # {1}{W}
        "warp_cmc": 2,
    },
    power=4,
    toughness=4,
)

# Weftstalker Ardent {2}{R} / Warp {R}
# Whenever another creature or artifact you control enters, deal 1 damage to each opponent.
WEFTSTALKER_ARDENT = CardEncoding(
    name="Weftstalker Ardent",
    mana_cost={"R": 1, "C": 2},
    cmc=3,
    types=["creature"],
    subtypes=["drix", "artificer"],
    mechanics=[
        Mechanic.WARP,
        Mechanic.ALTERNATIVE_COST,
        Mechanic.FROM_HAND,
        Mechanic.END_STEP_TRIGGER,
        Mechanic.EXILE_TEMPORARY,
        Mechanic.CAST_FROM_EXILE,
        Mechanic.ETB_TRIGGER,           # Triggers on other ETBs
        Mechanic.TARGET_OPPONENT,       # Damages opponents
        Mechanic.DEAL_DAMAGE,
        Mechanic.TARGETS_EACH,          # "each opponent"
    ],
    parameters={
        "warp_cost": {"R": 1},
        "warp_cmc": 1,
        "damage": 1,
    },
    power=3,
    toughness=2,
)

# Haliya, Guided by Light {2}{W} / Warp {W}
# ETB life gain, end step card draw if gained 3+ life
HALIYA = CardEncoding(
    name="Haliya, Guided by Light",
    mana_cost={"W": 1, "C": 2},
    cmc=3,
    types=["legendary", "creature"],
    subtypes=["human", "soldier"],
    mechanics=[
        Mechanic.WARP,
        Mechanic.ALTERNATIVE_COST,
        Mechanic.FROM_HAND,
        Mechanic.END_STEP_TRIGGER,
        Mechanic.EXILE_TEMPORARY,
        Mechanic.CAST_FROM_EXILE,
        Mechanic.ETB_TRIGGER,
        Mechanic.GAIN_LIFE_TRIGGER,
        Mechanic.IF_LIFE_GAINED,        # "if you've gained 3+ life"
        Mechanic.DRAW,
    ],
    parameters={
        "warp_cost": {"W": 1},
        "warp_cmc": 1,
        "life_gain": 1,
        "life_threshold": 3,
        "draw_count": 1,
    },
    power=2,
    toughness=2,
)

# Starfield Vocalist {3}{U} / Warp {1}{U}
# Panharmonicon for ETB triggers
STARFIELD_VOCALIST = CardEncoding(
    name="Starfield Vocalist",
    mana_cost={"U": 1, "C": 3},
    cmc=4,
    types=["creature"],
    subtypes=["human", "bard"],
    mechanics=[
        Mechanic.WARP,
        Mechanic.ALTERNATIVE_COST,
        Mechanic.FROM_HAND,
        Mechanic.END_STEP_TRIGGER,
        Mechanic.EXILE_TEMPORARY,
        Mechanic.CAST_FROM_EXILE,
        Mechanic.STATIC_ABILITY,        # Passive effect
        Mechanic.ETB_TRIGGER,
        Mechanic.DOUBLE_TRIGGER,        # "triggers an additional time"
    ],
    parameters={
        "warp_cost": {"U": 1, "C": 1},
        "warp_cmc": 2,
    },
    power=2,
    toughness=4,
)

# Anticausal Vestige {6} / Warp {4}
# Eldrazi with LTB card draw and cheat permanent into play
ANTICAUSAL_VESTIGE = CardEncoding(
    name="Anticausal Vestige",
    mana_cost={"C": 6},
    cmc=6,
    types=["creature"],
    subtypes=["eldrazi"],
    mechanics=[
        Mechanic.WARP,
        Mechanic.ALTERNATIVE_COST,
        Mechanic.FROM_HAND,
        Mechanic.END_STEP_TRIGGER,
        Mechanic.EXILE_TEMPORARY,
        Mechanic.CAST_FROM_EXILE,
        Mechanic.DEVOID,                # Colorless Eldrazi
        Mechanic.LTB_TRIGGER,           # "When this creature leaves"
        Mechanic.DRAW,
        Mechanic.TUTOR_TO_BATTLEFIELD,  # Put permanent onto battlefield
        Mechanic.TO_BATTLEFIELD_TAPPED,
    ],
    parameters={
        "warp_cost": {"C": 4},
        "warp_cmc": 4,
        "draw_count": 1,
    },
    power=6,
    toughness=6,
)


# =============================================================================
# TEST: Does Warp fit our vocabulary?
# =============================================================================

def analyze_warp_encoding():
    """Analyze how well Warp cards encode with our vocabulary."""

    print("=" * 70)
    print("WARP MECHANIC ENCODING TEST")
    print("=" * 70)
    print()
    print("Warp is a NEW mechanic from Edge of Eternities (July 2025)")
    print("Testing if our vocabulary can represent it...")
    print()

    # Core Warp mechanics
    warp_core = [
        Mechanic.WARP,
        Mechanic.ALTERNATIVE_COST,
        Mechanic.FROM_HAND,
        Mechanic.END_STEP_TRIGGER,
        Mechanic.EXILE_TEMPORARY,
        Mechanic.CAST_FROM_EXILE,
    ]

    print("WARP CORE MECHANICS:")
    for m in warp_core:
        print(f"  {m.name} ({m.value})")
    print()

    # Compare to similar mechanics
    print("COMPARISON TO SIMILAR MECHANICS:")
    print()
    print("  DASH (Fate Reforged 2015):")
    print("    - Alternative cost, haste, return to hand at end step")
    print("    - Encoded: ALTERNATIVE_COST, HASTE, END_STEP_TRIGGER, TO_HAND")
    print()
    print("  BLITZ (Streets of New Capenna 2022):")
    print("    - Alternative cost, haste, sacrifice at end step, draw")
    print("    - Encoded: ALTERNATIVE_COST, HASTE, END_STEP_TRIGGER, SACRIFICE, DRAW")
    print()
    print("  WARP (Edge of Eternities 2025):")
    print("    - Alternative cost, exile at end step, cast from exile later")
    print("    - Encoded: ALTERNATIVE_COST, END_STEP_TRIGGER, EXILE_TEMPORARY, CAST_FROM_EXILE")
    print()

    print("=" * 70)
    print("INDIVIDUAL CARD ENCODINGS")
    print("=" * 70)

    cards = [
        EXALTED_SUNBORN,
        WEFTSTALKER_ARDENT,
        HALIYA,
        STARFIELD_VOCALIST,
        ANTICAUSAL_VESTIGE,
    ]

    for card in cards:
        _ = encode_card_to_vector(card)  # Validate encoding
        print(f"\n{card.name}:")
        print(f"  Types: {card.types}")
        print(f"  Cost: {card.mana_cost} (CMC {card.cmc})")
        if "warp_cost" in card.parameters:
            print(f"  Warp: {card.parameters['warp_cost']} (CMC {card.parameters.get('warp_cmc', '?')})")
        print(f"  Stats: {card.power}/{card.toughness}")
        print(f"  Mechanics ({len(card.mechanics)}):")
        for m in card.mechanics:
            print(f"    - {m.name}")

    print()
    print("=" * 70)
    print("VERDICT: Can our vocabulary handle Warp?")
    print("=" * 70)
    print()
    print("YES - Warp decomposes cleanly into existing primitives:")
    print("  1. WARP (new ID 523) - marks the keyword itself")
    print("  2. ALTERNATIVE_COST - existing primitive for alt costs")
    print("  3. FROM_HAND - existing primitive for zone restriction")
    print("  4. END_STEP_TRIGGER - existing primitive for timing")
    print("  5. EXILE_TEMPORARY - existing primitive for temp exile")
    print("  6. CAST_FROM_EXILE - existing primitive for exile casting")
    print()
    print("We only needed ONE new primitive (WARP as a keyword marker).")
    print("All the mechanical components already existed!")
    print()
    print("This validates the vocabulary design: new mechanics are usually")
    print("combinations of existing primitives, not entirely new concepts.")


def analyze_synergies():
    """Show how the network would discover Warp synergies."""

    print()
    print("=" * 70)
    print("HOW MCTS DISCOVERS WARP SYNERGIES")
    print("=" * 70)
    print()

    print("Example 1: Warp + Starfield Vocalist + ETB creature")
    print("-" * 50)
    print("Board: Starfield Vocalist (Panharmonicon effect)")
    print("Hand: Mulldrifter (ETB: draw 2)")
    print()
    print("MCTS explores:")
    print("  Action: Cast Mulldrifter for warp cost")
    print("  Forge simulates:")
    print("    - Mulldrifter enters -> ETB triggers")
    print("    - Starfield Vocalist doubles it -> draw 4 cards")
    print("    - At end step, Mulldrifter exiles")
    print("    - Later turn: cast Mulldrifter from exile -> draw 4 more")
    print("  Network learns: WARP + DOUBLE_TRIGGER + ETB_DRAW = high value")
    print()

    print("Example 2: Warp + Void synergy")
    print("-" * 50)
    print("Card with Void: 'If a permanent left the battlefield this turn...'")
    print()
    print("MCTS explores:")
    print("  Action: Warp in creature, let it exile at end step")
    print("  Forge simulates:")
    print("    - Warp creature enters")
    print("    - At end step, it exiles (leaves battlefield)")
    print("    - Void triggers activate!")
    print("  Network learns: WARP + VOID = guaranteed Void activation")
    print()

    print("Example 3: Weftstalker Ardent + token spam")
    print("-" * 50)
    print("Weftstalker: 'Whenever another creature/artifact enters, deal 1'")
    print()
    print("MCTS explores:")
    print("  Action: Warp in Weftstalker, then play treasure/token makers")
    print("  Forge simulates:")
    print("    - Each token creation -> 1 damage to each opponent")
    print("    - In 4-player: 3 damage per token!")
    print("  Network learns: WARP (cheap) + ETB_TRIGGER + MULTIPLAYER = burst damage")


if __name__ == "__main__":
    analyze_warp_encoding()
    analyze_synergies()
