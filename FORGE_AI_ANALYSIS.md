# Forge AI Analysis

## Purpose

Before we can build something better than Forge AI, we need to understand what it does well and where it falls short. This document outlines a methodology for analyzing Forge AI and proposes improvements.

---

## What Forge AI Does

Forge AI is a **rule-based expert system** with some evaluation heuristics. It's implemented in Java and uses:

1. **Legal move generation**: Forge's rules engine provides all legal actions
2. **Static evaluation**: Board state scoring based on heuristics
3. **Limited lookahead**: Minimax-style search for combat and some decisions
4. **Pattern matching**: Recognizes some common situations

### Key Files in Forge Codebase

```
forge-ai/src/main/java/forge/ai/
├── AiController.java         # Main decision logic
├── AiCombatUtil.java         # Combat math
├── AiAttackController.java   # Attack decisions
├── AiBlockController.java    # Block decisions
├── ComputerUtilCard.java     # Card evaluation
├── ComputerUtilMana.java     # Mana management
├── ComputerUtilCost.java     # Cost payment
└── ability/                  # Per-ability AI handlers
```

---

## Forge AI Strengths

### 1. Rule Correctness
- Never makes illegal plays
- Correctly handles complex interactions
- Understands timing (instant speed, sorcery speed, etc.)

### 2. Combat Math
- Calculates favorable attacks
- Understands trading
- Considers combat tricks (to some extent)

### 3. Mana Management
- Efficient mana spending
- Understands color requirements
- Taps lands intelligently

### 4. Basic Threat Assessment
- Removes large threats
- Values card advantage
- Understands evasion (flying, unblockable)

---

## Forge AI Weaknesses

### 1. No Long-Term Planning

```
Example: Combo Setup
─────────────────────
Forge AI has: 2x combo pieces in hand, missing 1
Correct play: Hold mana, dig for third piece
Forge AI does: Plays combo pieces without payoff

Why: No concept of "incomplete combos" or setup
```

### 2. No Opponent Modeling

```
Example: Reading the Opponent
─────────────────────────────
Opponent: Holds 2 cards, has WW open
Correct play: They might have counterspell or removal
Forge AI does: Plays into it anyway

Why: No prediction of opponent's likely plays
```

### 3. No Bluffing / Information Hiding

```
Example: Representing Tricks
────────────────────────────
Situation: Can attack, have no tricks, opponent has bigger blocker
Correct play: Attack anyway (they might not block, thinking we have trick)
Forge AI does: Never attacks into bad trades

Why: No concept of information asymmetry
```

### 4. Myopic Removal Usage

```
Example: Saving Removal
───────────────────────
Situation: Opponent plays 2/2, we have Murder in hand
Correct play: Save Murder for bigger threats
Forge AI does: Often uses removal immediately

Why: Evaluates current board, not future threats
```

### 5. No Meta-Game Awareness

```
Example: Deck Archetype Recognition
──────────────────────────────────
Opponent: Playing Islands, hasn't cast spells
Correct inference: Likely control, play carefully
Forge AI does: Plays normally, walks into counters

Why: No archetype detection or adaptation
```

### 6. Poor Sequencing Sometimes

```
Example: Play Order
───────────────────
Hand: Land, 2-drop, Cantrip
Correct play: Cantrip first (might find better 2-drop)
Forge AI does: Sometimes plays 2-drop first

Why: Limited search depth for sequencing
```

---

## Methodology: Analyzing Forge AI Decisions

### Step 1: Collect Decision Logs

```python
class ForgeDecisionLogger:
    def log_decision(self, game_state, chosen_action, all_legal_actions):
        self.decisions.append({
            "state": encode_state(game_state),
            "action": chosen_action,
            "alternatives": all_legal_actions,
            "turn": game_state.turn,
            "phase": game_state.phase,
            "life_advantage": game_state.our_life - game_state.opp_life,
            "cards_in_hand": len(game_state.hand),
            "board_complexity": count_permanents(game_state),
        })
```

### Step 2: Categorize Decisions

```python
DECISION_CATEGORIES = {
    "removal_usage": lambda d: is_removal_spell(d.action),
    "combat_attack": lambda d: d.phase == "DECLARE_ATTACKERS",
    "combat_block": lambda d: d.phase == "DECLARE_BLOCKERS",
    "mana_usage": lambda d: d.action.uses_mana,
    "pass_priority": lambda d: d.action == "PASS",
    "card_selection": lambda d: "choose" in d.decision_type,
}
```

### Step 3: Find Questionable Decisions

```python
def find_questionable_plays(decisions, game_outcomes):
    """
    Find decisions that might have been wrong.
    """
    questionable = []

    for game in games:
        if game.outcome == "LOSS" and game.was_ahead_at_turn_5:
            # We lost from a winning position - find the turning point
            for decision in game.decisions:
                if decision.led_to_disadvantage:
                    questionable.append(decision)

    return questionable
```

### Step 4: Pattern Analysis

```python
def analyze_removal_patterns(decisions):
    """
    When does Forge use removal?
    """
    removal_decisions = [d for d in decisions if is_removal(d)]

    stats = {
        "avg_target_power": mean([d.target.power for d in removal_decisions]),
        "avg_target_toughness": mean([d.target.toughness for d in removal_decisions]),
        "used_on_evasion_pct": pct([d for d in removal_decisions if d.target.has_evasion]),
        "held_for_later_pct": pct([d for d in removal_decisions if d.turn > 5]),
    }

    return stats
```

---

## Proposed: Heuristic-Enhanced Agent

A middle ground between Forge AI and our neural agent.

### Architecture

```python
class EnhancedForgeAgent:
    """
    Wraps Forge AI with strategic heuristics learned from analysis.
    """

    def __init__(self):
        self.forge_ai = ForgeAIWrapper()
        self.heuristics = [
            RemovalConservation(),
            ComboRecognition(),
            OpponentModeling(),
            BluffingHeuristic(),
            SequencingOptimizer(),
        ]

    def decide(self, state, legal_actions):
        # Get Forge AI's recommendation
        forge_choice = self.forge_ai.recommend(state)

        # Check if any heuristic wants to override
        for heuristic in self.heuristics:
            override = heuristic.check(state, forge_choice, legal_actions)
            if override is not None:
                return override, heuristic.name

        return forge_choice, "forge_ai"
```

### Heuristic 1: Removal Conservation

```python
class RemovalConservation:
    """
    Don't use premium removal on small threats.
    """

    PREMIUM_REMOVAL = {"Murder", "Path to Exile", "Swords to Plowshares", ...}
    POWER_THRESHOLD = 4

    def check(self, state, forge_choice, legal_actions):
        if forge_choice.card.name not in self.PREMIUM_REMOVAL:
            return None

        target = forge_choice.target
        if target.power < self.POWER_THRESHOLD and not target.has_dangerous_ability:
            # Find alternative play
            alternatives = [a for a in legal_actions
                           if a.card.name not in self.PREMIUM_REMOVAL]
            if alternatives:
                return self.best_alternative(alternatives)

        return None
```

### Heuristic 2: Combo Recognition

```python
class ComboRecognition:
    """
    Recognize and prioritize combo setups.
    """

    KNOWN_COMBOS = [
        {"pieces": ["Splinter Twin", "Pestermite"], "priority": 10},
        {"pieces": ["Thassa's Oracle", "Demonic Consultation"], "priority": 10},
        # ... more combos
    ]

    def check(self, state, forge_choice, legal_actions):
        for combo in self.KNOWN_COMBOS:
            pieces_in_hand = [p for p in combo["pieces"] if p in state.hand]
            pieces_in_play = [p for p in combo["pieces"] if p in state.battlefield]

            if len(pieces_in_hand) + len(pieces_in_play) >= len(combo["pieces"]) - 1:
                # We're one piece away - prioritize combo
                combo_actions = [a for a in legal_actions
                                if a.card.name in combo["pieces"]]
                if combo_actions:
                    return combo_actions[0]

        return None
```

### Heuristic 3: Represent Tricks

```python
class BluffingHeuristic:
    """
    Leave mana open to represent combat tricks.
    """

    TRICK_MANA_COSTS = {
        "W": ["Gods Willing", "Brave the Elements"],
        "G": ["Giant Growth", "Blossoming Defense"],
        "R": ["Lightning Bolt", "Shock"],
    }

    def check(self, state, forge_choice, legal_actions):
        if state.phase != "MAIN_1":
            return None

        # If we're attacking, consider leaving trick mana open
        if self.planning_to_attack(state):
            trick_mana = self.get_trick_mana_for_deck(state.deck_colors)
            available_after_forge = state.mana - forge_choice.mana_cost

            if not self.can_pay(available_after_forge, trick_mana):
                # Forge choice uses too much mana, can't bluff
                cheaper = [a for a in legal_actions
                          if self.can_pay(state.mana - a.mana_cost, trick_mana)]
                if cheaper:
                    return self.best_alternative(cheaper)

        return None
```

### Heuristic 4: Opponent Modeling

```python
class OpponentModeling:
    """
    Infer opponent's deck/hand and adjust play.
    """

    def check(self, state, forge_choice, legal_actions):
        opponent_profile = self.infer_deck_type(state.opponent)

        if opponent_profile == "CONTROL":
            # Don't walk into counterspells
            if self.is_must_resolve(forge_choice) and state.opponent.mana_open >= 2:
                # Maybe bait first
                bait = self.find_bait_spell(legal_actions)
                if bait:
                    return bait

        elif opponent_profile == "AGGRO":
            # Prioritize blockers and removal
            if not self.is_defensive(forge_choice):
                defensive = self.find_defensive_play(legal_actions)
                if defensive:
                    return defensive

        return None

    def infer_deck_type(self, opponent):
        """
        Guess opponent's archetype from observed plays.
        """
        if opponent.avg_cmc < 2.5 and opponent.creature_count > opponent.spell_count:
            return "AGGRO"
        elif opponent.counterspells_seen > 0 or opponent.avg_cmc > 3.5:
            return "CONTROL"
        else:
            return "MIDRANGE"
```

---

## Evaluation: Comparing Agents

### Test Suite

```python
TEST_POSITIONS = [
    {
        "name": "save_removal",
        "state": "opponent has 2/2, we have Murder",
        "correct": "don't use Murder yet",
        "forge_does": "uses Murder",
    },
    {
        "name": "combo_setup",
        "state": "have 2/3 combo pieces",
        "correct": "dig for third piece",
        "forge_does": "plays pieces without payoff",
    },
    {
        "name": "represent_trick",
        "state": "can attack, want to leave W open",
        "correct": "pass with mana open",
        "forge_does": "taps out",
    },
    # ... more test cases
]

def evaluate_agent(agent, test_positions):
    results = {}
    for test in test_positions:
        state = parse_state(test["state"])
        action = agent.decide(state)
        results[test["name"]] = action == test["correct"]
    return results
```

### Matchup Testing

```
Forge AI vs Forge AI:         50% win rate (baseline)
Enhanced Agent vs Forge AI:   Target 55%+ win rate
Neural Agent vs Forge AI:     Target 60%+ after training
Neural Agent vs Enhanced:     Target 55%+ (shows we beat heuristics too)
```

---

## Data Collection Plan

1. **Run 10K Forge AI vs Forge AI games**
   - Collect all decisions with full context
   - Tag game outcomes

2. **Identify failure patterns**
   - Games where Forge lost from ahead
   - Questionable removal usage
   - Missed lethal
   - Poor sequencing

3. **Build test suite from failures**
   - Each failure mode becomes a test case
   - Encode correct behavior

4. **Train enhanced agent to pass tests**
   - Start with Forge AI
   - Add heuristics to fix failure modes
   - Validate on test suite

5. **Use enhanced agent as imitation target**
   - Better than Forge AI
   - Captures strategic patterns
   - Provides richer training signal

---

## Next Steps

1. [ ] Instrument Forge to log all decisions with context
2. [ ] Run 10K games, collect decision data
3. [ ] Analyze removal patterns, combat decisions, sequencing
4. [ ] Identify top 10 failure modes
5. [ ] Implement heuristic overrides for each
6. [ ] Validate enhanced agent beats Forge AI
7. [ ] Use enhanced agent games for imitation learning
