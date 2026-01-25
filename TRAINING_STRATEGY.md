# MTG RL Training Strategy

## Overview

This document outlines our phased approach to training an MTG agent, from imitation learning through self-play RL, with a focus on handling the unique challenges of MTG's variable-length decision sequences.

---

## The Core Challenge

MTG is uniquely difficult for RL because:

1. **Variable decision density**: A turn can have 1 decision or 100+
2. **Triggered ability chains**: One action can cascade into many decisions
3. **Sequencing matters**: Order of plays often determines outcomes
4. **Partial observability**: Unknown cards in opponent's hand/library
5. **Long-horizon credit assignment**: Early decisions affect late-game outcomes

Unlike Chess/Go (atomic moves) or even Dota (continuous but bounded), MTG has **recursive decision structures** that can explode combinatorially.

---

## Phase 1: Forge AI Imitation

**Goal**: Learn basic game mechanics and "sensible" play patterns.

### Data Collection

```python
# Run Forge AI vs Forge AI games
for game in range(100_000):
    states, actions = run_forge_game(ai_vs_ai=True)
    save_trajectory(states, actions)
```

**What we collect**:
- Full game state at each decision point
- Action taken by Forge AI
- Game outcome (for value head training)
- Decision metadata (turn, phase, decision type)

### Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Game State Encoder                        │
│  [Hand] [Battlefield] [Graveyard] [Stack] [Opponent State]  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    Transformer Encoder
                    (Self-attention over all entities)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
    │ Policy  │     │  Value  │     │  Conf.  │
    │  Head   │     │  Head   │     │  Head   │
    └─────────┘     └─────────┘     └─────────┘
    P(action)       V(state)        C(state)
```

### Training Objective

```python
loss = (
    cross_entropy(predicted_action, forge_action) +      # Imitation
    mse(predicted_value, game_outcome) +                 # Value
    0.1 * auxiliary_losses                               # Regularization
)
```

### Success Criteria

- Policy matches Forge AI actions >70% of the time
- Value predictions correlate with game outcomes (r > 0.6)
- Games complete without getting stuck

---

## Phase 2: Confidence Networks

**Goal**: Learn to recognize when the agent is uncertain or stuck.

### The Confidence Head

The confidence head predicts: "How certain am I that my action is correct?"

```python
class ConfidenceHead(nn.Module):
    def forward(self, encoded_state):
        # Output: scalar in [0, 1]
        # High = confident, Low = uncertain
        return torch.sigmoid(self.mlp(encoded_state))
```

### Training Signal for Confidence

**Option A: Hindsight labeling**
```python
# After game completes, label decisions:
confidence_target = 1.0 if action_led_to_good_outcome else 0.0
```

**Option B: Ensemble disagreement**
```python
# Train N copies of policy, measure agreement
actions = [policy_i(state) for i in range(N)]
confidence_target = agreement_score(actions)
```

**Option C: Temporal difference**
```python
# If value estimate changes dramatically after action, we were wrong
confidence_target = 1.0 - abs(V(s') - V(s) - reward)
```

### Using Confidence for Fallback

```python
def decide(state, turn_context):
    action_probs, value, confidence = model(state)

    # TODO: Replace hardcoded thresholds with learned behavior
    # This is a temporary guardrail until confidence head is trained
    if confidence < 0.3 or turn_context.decisions_this_turn > 50:
        return fallback_to_expert(state)

    return sample_action(action_probs)
```

> **NOTE**: The hardcoded thresholds (0.3 confidence, 50 decisions/turn) are
> temporary. Future work should make these learnable, either through:
> - Meta-learning the threshold as a hyperparameter
> - Training a separate "should I defer?" classifier
> - Reward shaping that penalizes excessive fallback

---

## Phase 3: Reward Shaping

**Goal**: Encourage autonomous, efficient play.

### Base Rewards

| Event | Reward |
|-------|--------|
| Win game | +1.0 |
| Lose game | -1.0 |
| Draw | 0.0 |

### Shaped Rewards (Intermediate)

```python
def compute_shaped_reward(state, action, next_state):
    reward = 0.0

    # Progress rewards (encourage game advancement)
    reward += 0.01 * (opponent_life_lost)
    reward += 0.005 * (creatures_played)
    reward += 0.002 * (lands_played)

    # Efficiency rewards (encourage clean turns)
    reward += 0.01 * (mana_spent / mana_available)  # Mana efficiency

    # Autonomy reward (encourage NOT using fallback)
    if not used_fallback:
        reward += 0.005

    # Penalize excessive decisions (encourage efficient sequencing)
    if decisions_this_turn > 20:
        reward -= 0.001 * (decisions_this_turn - 20)

    return reward
```

### Autonomy Shaping

This is key: we want to reward the agent for taking **independent turns that yield good results**.

```python
def autonomy_bonus(turn_stats, turn_outcome):
    """
    Reward taking full turns without fallback, scaled by outcome quality.
    """
    if turn_stats.used_fallback:
        return 0.0

    # How good was this turn?
    outcome_quality = (
        turn_stats.damage_dealt * 0.1 +
        turn_stats.cards_drawn * 0.05 +
        turn_stats.threats_answered * 0.1
    )

    # Bonus for autonomous good turns
    return 0.02 * outcome_quality
```

---

## Phase 4: Self-Play RL

**Goal**: Surpass Forge AI through self-improvement.

### Training Loop

```python
for iteration in range(1000):
    # Self-play: current policy vs historical versions
    games = play_games(
        player1=current_policy,
        player2=sample_from_policy_pool(),
        num_games=1000
    )

    # Collect trajectories (excluding fallback decisions)
    trajectories = [
        (s, a, r) for (s, a, r, is_fallback) in games
        if not is_fallback
    ]

    # PPO update
    policy_loss, value_loss = ppo_update(trajectories)

    # Add to policy pool periodically
    if iteration % 10 == 0:
        policy_pool.add(current_policy.clone())
```

### Curriculum: Gradually Reduce Fallback

```python
# Start with high fallback threshold, gradually lower it
fallback_threshold = max(0.1, 0.5 - 0.01 * iteration)

# Or: start with many allowed decisions, gradually reduce
max_decisions_per_turn = max(30, 100 - iteration)
```

---

## Phase 5: Better Imitation Targets

### Why We Need Something Better Than Forge AI

Forge AI is good at:
- Rule-correct play (never makes illegal moves)
- Basic threat assessment
- Mana curve awareness
- Combat math

Forge AI is weak at:
- Long-term planning (doesn't set up combos)
- Meta-game awareness (doesn't know what opponent might have)
- Bluffing / information hiding
- Complex stack interactions
- Sideboarding / adaptation

### Proposed: Heuristic-Enhanced Policy

A middle ground between Forge AI and our full neural agent:

```python
class HeuristicPolicy:
    """
    Combines Forge AI's rule knowledge with strategic heuristics.
    Used as an improved imitation target.
    """

    def decide(self, state, legal_actions):
        # Get Forge AI's recommendation
        forge_action = forge_ai.recommend(state)

        # Apply strategic overrides
        if self.should_hold_removal(state):
            # Don't use removal on small threats
            action = self.find_non_removal_play(legal_actions)

        elif self.detect_combo_opportunity(state):
            # Prioritize combo pieces
            action = self.find_combo_play(legal_actions)

        elif self.should_represent_trick(state):
            # Leave mana open to bluff
            action = self.pass_with_mana_open()

        else:
            action = forge_action

        return action

    def should_hold_removal(self, state):
        """Don't waste removal on small threats."""
        # Heuristic: if opponent has <4 power on board, save removal
        ...

    def detect_combo_opportunity(self, state):
        """Recognize when we're close to a combo."""
        # Pattern matching on hand + battlefield
        ...

    def should_represent_trick(self, state):
        """Bluff having combat tricks."""
        # Leave 1-2 mana open when attacking
        ...
```

### Analyzing Forge AI Behavior

To understand Forge AI's strengths and weaknesses, we should:

1. **Collect decision logs**:
   ```python
   for game in forge_games:
       for decision in game:
           log(decision.state, decision.action, decision.alternatives)
   ```

2. **Identify patterns**:
   - When does Forge use removal? (threshold analysis)
   - How does it value card advantage vs tempo?
   - Does it recognize synergies?
   - How does it handle combat math edge cases?

3. **Find failure modes**:
   - Games where Forge loses from winning position
   - Turns where Forge makes clearly suboptimal plays
   - Matchups where Forge performs poorly

4. **Create test suite**:
   ```python
   # Specific board states where we know the right play
   test_cases = [
       {"state": combo_setup_state, "correct": "play_combo_piece"},
       {"state": bluff_opportunity, "correct": "hold_mana_open"},
       {"state": removal_decision, "correct": "save_for_bigger_threat"},
   ]
   ```

---

## Architecture: Internal State Description

### Game State Embedding

The network should build an internal representation that captures:

```
Layer 1: Entity Embeddings
├── Each card → 256-dim vector
├── Positional encoding (zone, controller, tapped state)
├── Temporal encoding (when played, when last activated)

Layer 2: Relational Understanding
├── Cross-attention between cards
├── Which cards can target which?
├── Which cards synergize?

Layer 3: Strategic Summary
├── Board advantage estimate
├── Tempo assessment
├── Card advantage assessment
├── Threat/answer balance

Layer 4: Decision Context
├── What phase are we in?
├── What's on the stack?
├── What decisions remain this turn?
├── How complex is this turn so far?
```

### Confidence as Internal State

The confidence head should learn to recognize:

```python
# High confidence situations:
- Clear best play (only one good option)
- Simple board states
- Familiar patterns from training

# Low confidence situations:
- Multiple viable options
- Complex stack interactions
- Unfamiliar card combinations
- High-stakes decisions (game-deciding)
```

---

## Metrics to Track

### Training Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Policy accuracy vs Forge | >70% | Imitation quality |
| Value prediction MSE | <0.1 | State evaluation |
| Confidence calibration | - | Uncertainty quality |
| Fallback rate | <10% | Autonomy |
| Avg decisions/turn | <20 | Efficiency |

### Evaluation Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Win rate vs Forge AI | >55% | Surpassing teacher |
| Win rate vs self (t-100) | >52% | Continued improvement |
| Game completion rate | >99% | Stability |
| Avg game length (turns) | 8-15 | Reasonable games |

---

## TODO / Future Work

1. **Learnable Fallback Thresholds**
   - Current: hardcoded `confidence < 0.3` and `decisions > 50`
   - Future: meta-learn these as hyperparameters or train classifier

2. **Complexity Prediction Head**
   - Predict "how many more decisions will this turn require?"
   - Use for proactive complexity management

3. **Hierarchical Decisions**
   - High-level: "What's my plan this turn?"
   - Low-level: "How do I execute each step?"

4. **Better Imitation Targets**
   - Analyze MTGO/Arena data if available
   - Build heuristic-enhanced Forge wrapper
   - Consider tournament game logs

5. **Curriculum Learning**
   - Start with simple decks (mono-color aggro)
   - Gradually introduce complexity (control, combo, multicolor)

---

## References

- AlphaGo/AlphaZero: Self-play RL with MCTS
- OpenAI Five: Reward shaping at scale
- Decision Transformer: Sequence modeling for RL
- Ensemble methods for uncertainty estimation
