"""
Analysis: How to encode "Saw in Half" for a neural network

Card: Saw in Half {2}{B}
Instant
"Destroy target creature. If that creature dies this way, its controller
creates two tokens that are copies of that creature, except their power
is half that creature's power and their toughness is half that creature's
toughness. Round up each time."

This card is complex because it involves:
1. Targeting (any creature - yours or opponent's)
2. Destruction effect
3. Death-conditional trigger
4. Token creation (quantity: 2)
5. Copy effect (copies target with modifications)
6. Stat formula (P/T = ceil(original/2))
7. Controller-relative effect (tokens go to original controller)

Why it's powerful:
- On opponent: Remove threat, leave two smaller bodies
- On YOUR OWN creature: Double ETB triggers! (e.g., Mulldrifter draws 4)
- Synergies: Death triggers, sacrifice, tokens, ETB effects
"""

# =============================================================================
# APPROACH 1: Naive Feature Vector (What we currently do)
# =============================================================================

naive_embedding = {
    # Basic attributes
    "cmc": 3,
    "color_B": 1,
    "color_W": 0, "color_U": 0, "color_R": 0, "color_G": 0,
    "type_instant": 1,
    "type_creature": 0,
    "type_enchantment": 0,
    # ... etc

    # Keywords (none apply)
    "flying": 0,
    "trample": 0,
    # ... etc

    # Text embedding from sentence-transformers
    "text_embedding": "[384-dim vector from MiniLM]",
}

# PROBLEM: The text embedding treats this as English, not MTG mechanics.
# It doesn't understand:
# - "copies" = duplicate ETB triggers
# - "half power" = specific formula
# - Self-targeting is often BETTER than opponent-targeting


# =============================================================================
# APPROACH 2: Structured Effect Decomposition
# =============================================================================

# Parse card into atomic "effects" that the game engine understands

structured_effects = {
    "card_name": "Saw in Half",
    "cost": {"generic": 2, "B": 1},
    "speed": "instant",

    "effects": [
        {
            "type": "DESTROY",
            "target": {
                "type": "creature",
                "controller": "any",  # yours or opponent's
                "count": 1,
            },
            "condition": None,  # unconditional
        },
        {
            "type": "CREATE_TOKEN",
            "trigger": "IF_TARGET_DIES",  # conditional on first effect
            "count": 2,
            "token_properties": {
                "copy_of": "TARGET",  # copies the destroyed creature
                "modifications": [
                    {"stat": "power", "formula": "ceil(original/2)"},
                    {"stat": "toughness", "formula": "ceil(original/2)"},
                ]
            },
            "controller": "TARGET_CONTROLLER",  # tokens go to creature's owner
        }
    ],

    # Derived synergy flags (could be learned or computed)
    "synergies": {
        "etb_doubler": True,      # Creates copies that re-trigger ETB
        "death_trigger": True,    # Original dies
        "token_generator": True,  # Creates tokens
        "removal": True,          # Can destroy opponent's creatures
        "self_target_value": True,  # Often better on your own creatures
    }
}


# =============================================================================
# APPROACH 3: Effect Graph / Mechanics Interaction Model
# =============================================================================

"""
Instead of flat features, represent mechanics as NODES in a graph.
Edges represent how mechanics INTERACT.

For Saw in Half:

    [INSTANT SPEED] -----> [TARGETING: Creature]
                                  |
                                  v
                           [DESTROY Effect]
                                  |
                           (if dies)
                                  v
                    [CREATE TOKEN: Copy x2]
                           /         \
                   [P/T Modified]  [ETB Triggers]
                         |              |
                    [ceil(P/2)]   [Doubled!]

The network learns:
- How effect chains work
- Which mechanics synergize
- Value of self vs opponent targeting
"""

# Graph representation
effect_graph = {
    "nodes": [
        {"id": 0, "type": "CAST", "speed": "instant", "cost": 3},
        {"id": 1, "type": "TARGET", "what": "creature", "whose": "any"},
        {"id": 2, "type": "DESTROY", "target_ref": 1},
        {"id": 3, "type": "CONDITIONAL", "condition": "target_dies"},
        {"id": 4, "type": "CREATE_TOKEN", "count": 2, "copy_of": 1},
        {"id": 5, "type": "MODIFY_STAT", "stat": "power", "formula": "half_round_up"},
        {"id": 6, "type": "MODIFY_STAT", "stat": "toughness", "formula": "half_round_up"},
    ],
    "edges": [
        (0, 1),  # cast -> target
        (1, 2),  # target -> destroy
        (2, 3),  # destroy -> conditional
        (3, 4),  # conditional -> create token
        (4, 5),  # token -> modify power
        (4, 6),  # token -> modify toughness
    ]
}


# =============================================================================
# APPROACH 4: AlphaZero-Style Action Encoding
# =============================================================================

"""
AlphaZero doesn't encode "what the piece is" - it encodes:
1. Current board state
2. Legal actions
3. Learns value(state) and policy(state->action) through self-play

For MTG, we need:
- State: Cards in all zones, life totals, stack, phase, etc.
- Actions: All legal plays (cast spell, activate ability, attack, block, etc.)

The KEY INSIGHT: We don't need to pre-encode card interactions.
The network LEARNS them through self-play.

But we DO need:
- A way to represent "this card can create these effects"
- A way to enumerate legal actions
- A game engine (Forge) to simulate outcomes
"""

# State representation for AlphaZero
game_state = {
    # Zone information
    "battlefield": {
        "my_creatures": [...],  # List of permanent representations
        "opp_creatures": [...],
        "my_other_permanents": [...],
        "opp_other_permanents": [...],
    },
    "hand": [...],  # Cards I can play
    "graveyard": {"mine": [...], "opp": [...]},
    "exile": [...],
    "stack": [...],  # Currently resolving

    # Game state
    "life": {"me": 40, "opp": 40},
    "phase": "MAIN_1",
    "priority": "me",
    "mana_available": {"W": 0, "U": 0, "B": 3, "R": 0, "G": 0, "C": 2},

    # For MCTS: What actions are legal?
    "legal_actions": [
        {"type": "CAST", "card": "Saw in Half", "target": "my_creature_0"},
        {"type": "CAST", "card": "Saw in Half", "target": "opp_creature_0"},
        {"type": "PASS_PRIORITY"},
        # ... etc
    ]
}

# The action space is dynamic - depends on game state
# MCTS explores: "If I cast Saw in Half on my Mulldrifter..."
# Forge simulates the outcome: 2 Mulldrifters with 1/1, draw 4 cards
# Network learns: This action has high value


# =============================================================================
# APPROACH 5: Hybrid - Mechanics Primitives + Learned Combinations
# =============================================================================

"""
PROPOSED ARCHITECTURE:

1. MECHANICS VOCABULARY (finite, hand-coded)
   - ~50-100 primitive mechanics: DESTROY, CREATE_TOKEN, COPY, DRAW, etc.
   - Each card is parsed into a sequence of mechanics
   - This is tractable to hand-code or parse from rules text

2. MECHANIC ENCODER (learned)
   - Each primitive gets a learned embedding
   - Transformer or GNN learns how mechanics compose
   - "DESTROY + CREATE_TOKEN(COPY)" -> combined representation

3. GAME STATE ENCODER (learned)
   - Encodes current board, hand, life, etc.
   - Attention over all cards in play

4. POLICY + VALUE HEADS (AlphaZero style)
   - Policy: P(action | state) for all legal actions
   - Value: Expected win probability from this state

5. MCTS for LOOKAHEAD
   - Use Forge to simulate game trees
   - Network guides tree search
   - Self-play generates training data
"""

# Mechanics vocabulary (partial)
MECHANICS_VOCABULARY = {
    # Targeting
    "TARGET_CREATURE": 1,
    "TARGET_PLAYER": 2,
    "TARGET_PERMANENT": 3,
    "TARGET_SPELL": 4,
    "TARGET_SELF_ONLY": 5,
    "TARGET_OPPONENT_ONLY": 6,
    "TARGET_ANY": 7,

    # Removal
    "DESTROY": 10,
    "EXILE": 11,
    "SACRIFICE": 12,
    "BOUNCE": 13,
    "DAMAGE": 14,

    # Creation
    "CREATE_TOKEN": 20,
    "COPY": 21,
    "TOKEN_COPY": 22,

    # Modification
    "MODIFY_POWER": 30,
    "MODIFY_TOUGHNESS": 31,
    "ADD_COUNTER": 32,
    "REMOVE_COUNTER": 33,
    "HALF_STATS": 34,
    "DOUBLE_STATS": 35,

    # Card advantage
    "DRAW": 40,
    "DISCARD": 41,
    "MILL": 42,
    "TUTOR": 43,

    # Triggers
    "ETB_TRIGGER": 50,
    "DEATH_TRIGGER": 51,
    "CAST_TRIGGER": 52,
    "ATTACK_TRIGGER": 53,

    # Conditions
    "IF_TARGET_DIES": 60,
    "IF_CREATURE_ENTERS": 61,
    "IF_SPELL_CAST": 62,

    # Timing
    "INSTANT_SPEED": 70,
    "SORCERY_SPEED": 71,
    "FLASH": 72,

    # ... etc (probably 100-200 total primitives)
}

# Saw in Half encoded as mechanics sequence
saw_in_half_mechanics = [
    MECHANICS_VOCABULARY["INSTANT_SPEED"],
    MECHANICS_VOCABULARY["TARGET_CREATURE"],
    MECHANICS_VOCABULARY["TARGET_ANY"],
    MECHANICS_VOCABULARY["DESTROY"],
    MECHANICS_VOCABULARY["IF_TARGET_DIES"],
    MECHANICS_VOCABULARY["CREATE_TOKEN"],
    MECHANICS_VOCABULARY["TOKEN_COPY"],
    MECHANICS_VOCABULARY["HALF_STATS"],
]

# The network learns:
# - This sequence = "removal that doubles ETBs on self-target"
# - High value when you have ETB creatures
# - Moderate value as pure removal


# =============================================================================
# COMPARISON: Saw in Half vs Similar Cards
# =============================================================================

"""
Cards with similar mechanics (blink/copy effects):

1. Saw in Half:      DESTROY -> TOKEN_COPY x2 (half stats)
2. Twinflame:        TARGET_CREATURE -> TOKEN_COPY (exile at end)
3. Cackling Counterpart: TARGET_CREATURE -> TOKEN_COPY (instant)
4. Panharmonicon:    PERMANENT -> DOUBLE_ETB_TRIGGERS

The network should learn:
- Saw in Half ≈ Twinflame for ETB abuse
- But Saw in Half also works as removal
- Panharmonicon is passive, others are active

This is what we want MCTS + self-play to discover.
"""


# =============================================================================
# WHAT THIS MEANS FOR IMPLEMENTATION
# =============================================================================

"""
RECOMMENDED PATH:

Phase 1: Build Mechanics Vocabulary
- Parse Scryfall rules text for all cards
- Map to ~100-200 primitive mechanics
- May need LLM assistance for complex parsing

Phase 2: Card Encoder
- Input: Sequence of mechanics + basic attributes (cost, type)
- Architecture: Small transformer or GNN
- Pre-train on: Predicting card interactions? Or skip to Phase 3.

Phase 3: AlphaZero Training Loop
- State encoder for full game state
- Policy head for action selection
- Value head for win probability
- MCTS with Forge as simulator
- Self-play generates training data

Phase 4: Scaling
- Start with simple format (e.g., Pauper, limited card pool)
- Scale to larger formats as network improves
- Transfer learning across formats

KEY DECISION: Do we need human game data at all?
- AlphaZero succeeded with ZERO human games
- But MTG is more complex than Go
- Might need some human games for warm-start
- Or use draft model for initial card understanding
"""

if __name__ == "__main__":
    print("Saw in Half - Mechanics Encoding Analysis")
    print("=" * 50)
    print(f"\nCard mechanics sequence: {saw_in_half_mechanics}")
    print(f"Vocabulary size: {len(MECHANICS_VOCABULARY)} primitives")
    print("\nThis would feed into a transformer/GNN card encoder")
    print("Combined with game state for AlphaZero policy/value heads")
