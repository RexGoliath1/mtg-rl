# AlphaZero Architecture for MTG

## The Core Insight

AlphaZero doesn't pre-encode "piece values" or "opening theory" - it learns everything through self-play. The key components:

1. **State representation** - Current board position
2. **Policy network** - P(action | state) for legal moves
3. **Value network** - Expected win probability
4. **MCTS** - Tree search guided by networks

For MTG, we adapt this but face unique challenges.

## MTG vs Chess/Go Complexity

| Aspect | Chess | Go | MTG |
|--------|-------|-----|-----|
| State space | ~10^44 | ~10^170 | ~10^1000+ |
| Branching factor | ~35 | ~250 | Variable (1-1000+) |
| Hidden information | None | None | Hands, library |
| Stochasticity | None | None | Draw, effects |
| State changes per turn | 1 | 1 | Many (stack resolution) |

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GAME STATE ENCODER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Battlefield │  │    Hand     │  │  Graveyard  │   ...    │
│  │  Encoder    │  │   Encoder   │  │   Encoder   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          ▼                                   │
│                   ┌──────────────┐                           │
│                   │  Attention   │                           │
│                   │   Pooling    │                           │
│                   └──────┬───────┘                           │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    COMBINED STATE      │
              │      EMBEDDING         │
              └───────────┬────────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
    ┌──────────────┐              ┌──────────────┐
    │ POLICY HEAD  │              │ VALUE HEAD   │
    │ P(a|s) for   │              │ V(s) = E[win]│
    │ legal actions│              │              │
    └──────────────┘              └──────────────┘
           │                             │
           ▼                             ▼
    ┌──────────────────────────────────────────┐
    │              MCTS SEARCH                  │
    │  - Expand promising nodes                │
    │  - Forge simulates transitions           │
    │  - Network guides exploration            │
    └──────────────────────────────────────────┘
```

## Mechanics Encoder (Key Innovation)

Instead of encoding cards as opaque embeddings, parse into **mechanics primitives**:

```python
# Saw in Half -> Mechanics Sequence
[INSTANT_SPEED, TARGET_CREATURE, DESTROY, IF_DIES, CREATE_TOKEN, COPY, HALF_STATS]

# Mulldrifter -> Mechanics Sequence
[CREATURE, FLYING, ETB_TRIGGER, DRAW(2), EVOKE]

# Network learns: Saw in Half + Mulldrifter = draw 4 cards
# Without ever seeing this combo in training data
```

### Why This Scales

1. **Finite vocabulary** - ~100-200 mechanics covers all of MTG
2. **Composable** - New cards = new combinations of known primitives
3. **Transferable** - Mechanics work the same in Draft, Standard, Commander
4. **MCTS discovers** - Interactions learned through simulation, not pre-coded

## Handling MTG-Specific Challenges

### Hidden Information

```
┌─────────────────────────────────────────────┐
│           BELIEF STATE                       │
│                                             │
│  Known: My hand, battlefield, graveyards    │
│                                             │
│  Unknown: Opponent's hand, library order    │
│           -> Model as distribution          │
│           -> Sample for MCTS rollouts       │
└─────────────────────────────────────────────┘
```

Use **Information Set MCTS (IS-MCTS)** or **Perfect Information Monte Carlo (PIMC)**:
- Sample possible opponent hands consistent with observed play
- Run MCTS on sampled states
- Average across samples

### Variable Action Space

Unlike chess (always ~35 legal moves), MTG actions vary wildly:
- Sometimes 1 legal action (forced block)
- Sometimes 100+ (complex combo turn)

Solution: **Action masking**
- Policy network outputs over all possible action types
- Mask illegal actions before softmax
- Legal actions determined by Forge rules engine

### Stack and Priority

MTG has a LIFO stack where spells resolve:

```
Stack: [Lightning Bolt targeting my creature]
My options:
  - Pass (let it resolve, creature dies)
  - Cast Saw in Half on my creature (creature dies, I get tokens + ETB)
  - Cast counterspell (bolt fizzles)

MCTS must evaluate each line
```

Represent stack as part of state:
```python
state = {
    "battlefield": {...},
    "stack": [
        {"spell": "Lightning Bolt", "target": "my_creature_3", "controller": "opp"}
    ],
    "priority": "me",
}
```

## Training Loop

```
1. Initialize networks randomly (or warm-start from draft model)

2. Self-play game:
   a. From current state, run MCTS for N simulations
   b. Select action from MCTS visit counts (with temperature)
   c. Execute action in Forge
   d. Store (state, policy_target, value_target)

3. Train on collected data:
   - Policy loss: Cross-entropy with MCTS policy
   - Value loss: MSE with game outcome

4. Repeat 2-3 for many games

5. Periodically: Evaluate vs previous checkpoints
```

## Format Progression Strategy

Start simple, scale up:

| Phase | Format | Card Pool | Why |
|-------|--------|-----------|-----|
| 1 | Simplified | ~50 cards | Validate architecture |
| 2 | Pauper | Commons only | Simpler interactions |
| 3 | Standard | ~2000 cards | Competitive format |
| 4 | Modern | ~10000 cards | Complex interactions |
| 5 | Commander | 20000+ | Full complexity |

Each phase transfers knowledge to next.

## Open Questions

1. **How much human data for warm-start?**
   - AlphaZero: Zero
   - MTG complexity suggests some warm-start helps
   - Draft model provides card understanding

2. **How to handle 4-player Commander?**
   - Multiplayer MCTS is harder
   - May need opponent modeling

3. **Deck construction + gameplay jointly?**
   - Or separate models?

4. **Compute requirements?**
   - AlphaZero: 5000 TPUs for 40 days
   - We need efficient approximations

## Next Steps

1. [ ] Define mechanics vocabulary (~100-200 primitives)
2. [ ] Build parser: Card text -> mechanics sequence
3. [ ] Implement state encoder for Forge game states
4. [ ] Build policy/value network architecture
5. [ ] Implement MCTS with Forge integration
6. [ ] Start self-play on simplified format
