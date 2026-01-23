# MTG RL Architecture

## Overview

This document outlines the architectural decisions for training RL agents to play Magic: The Gathering.

## The Challenge

MTG presents several extreme challenges for RL:

| Challenge | Scale | Impact |
|-----------|-------|--------|
| Card Pool | ~30,000 unique cards | Massive action space |
| Hidden Information | Opponent's hand, library order | Partial observability |
| Variable Actions | 0-100+ legal moves per decision | Dynamic action space |
| Game Length | 10-100+ turns | Long credit assignment |
| Sparse Rewards | Win/lose at end | Delayed feedback |
| Combinatorial Complexity | Card interactions | Exponential state space |

## Architectural Decisions

### 1. State Representation

**Decision: Hierarchical + Attention-based encoding**

```
Raw State → Zone Encodings → Attention Aggregation → Fixed Vector
```

**Components:**
- **Card Embeddings**: Learn 64-128 dim vectors per card (not one-hot)
- **Zone Encoders**: Separate encoders for hand, battlefield, graveyard, stack
- **Cross-attention**: Model card interactions across zones
- **Global Features**: Life, mana, turn, phase as scalar features

**Why not one-hot cards?**
- 30,000 cards = 30,000 dim vectors = intractable
- Similar cards should have similar embeddings (Lightning Bolt ≈ Shock)
- Pre-train embeddings on card text using language models

### 2. Action Representation

**Decision: Hierarchical action space with masking**

```
Level 1: Action Type (pass, spell, land, ability, attack, block)
Level 2: Card Selection (which card to use)
Level 3: Target Selection (what to target)
```

**Why hierarchical?**
- Flat action space of all legal moves is too large
- Hierarchical decomposition reduces effective branching factor
- Shared representations across similar actions

**Action Masking:**
- Only allow legal actions at each level
- Mask computed from game rules
- Invalid actions get -∞ logits

### 3. Model Architecture

**Recommended: Transformer + PPO**

```
┌─────────────────────────────────────────────────┐
│                  Input Layer                     │
│  [Card Embeddings] [Zone Positions] [Scalars]   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│            Transformer Encoder                   │
│  - Self-attention over all cards                │
│  - Cross-attention between zones                │
│  - 4-6 layers, 8 heads, 256 dim                │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌────────▼────────┐
│   Policy Head   │    │   Value Head    │
│ (Action logits) │    │ (State value)   │
└─────────────────┘    └─────────────────┘
```

**Why Transformers?**
- Handle variable-length card sequences naturally
- Attention captures card interactions
- Pre-trained card embeddings transfer well

**Why PPO?**
- Stable training with large networks
- Works well with discrete action spaces
- Easy action masking integration

### 4. Training Strategy

**Phase 1: Imitation Learning (Optional but Recommended)**
- Collect expert games from strong AI or human play
- Pre-train policy via behavioral cloning
- Provides good initialization

**Phase 2: Self-Play**
```
┌─────────────┐     ┌─────────────┐
│  Agent v1   │ ──► │  Agent v2   │  (train against v1)
└─────────────┘     └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Agent v3   │  (train against v1, v2)
                    └──────┬──────┘
                           │
                          ...
```
- Train against pool of past versions
- Prevents overfitting to single opponent
- Gradually increasing difficulty

**Phase 3: Population-Based Training**
- Multiple agents with different hyperparameters
- Evolve hyperparameters based on performance
- Handles diverse strategies

### 5. Reward Shaping

**Terminal Rewards:**
- Win: +1.0
- Lose: -1.0

**Intermediate Rewards (shaped):**
```python
reward = 0.0

# Life differential (small)
reward += (our_life_change - opp_life_change) * 0.01

# Board presence (creatures matter)
reward += creature_count_change * 0.02

# Card advantage
reward += card_advantage_change * 0.01

# Mana efficiency (using mana is good)
reward += mana_spent * 0.005

# Tempo (doing things on curve)
reward += on_curve_play * 0.01
```

**Why shaped rewards?**
- Pure win/lose is too sparse for complex games
- Intermediate rewards guide early learning
- Can be annealed toward pure terminal rewards later

### 6. Curriculum Learning

**Stage 1: Simplified Games**
- Limited card pool (basic lands + simple creatures)
- No instants/interrupts (simpler timing)
- No complex abilities

**Stage 2: Add Complexity**
- Introduce instants and stack interactions
- Add removal and combat tricks
- Expand card pool gradually

**Stage 3: Full Games**
- Complete rule set
- Full card pool
- Multiple formats (Standard, Modern, Legacy)

### 7. Infrastructure

```
┌─────────────────────────────────────────────────────────┐
│                    Training Cluster                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Parameter Server                     │  │
│  │         (Stores current policy)                  │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      │                                  │
│    ┌─────────────────┼─────────────────┐               │
│    │                 │                 │               │
│  ┌─▼───┐          ┌─▼───┐          ┌─▼───┐            │
│  │Game │          │Game │          │Game │  ...       │
│  │Env 1│          │Env 2│          │Env N│            │
│  │     │          │     │          │     │            │
│  │Forge│          │Forge│          │Forge│            │
│  │Docker│         │Docker│         │Docker│           │
│  └─────┘          └─────┘          └─────┘            │
│                                                        │
│  Each environment runs parallel games                  │
│  Collects experience → sends to parameter server       │
│  Receives updated policy periodically                  │
└────────────────────────────────────────────────────────┘
```

**Parallelization:**
- Run N game environments in parallel (N=16-64)
- Each environment runs in Docker container
- Batch observations for efficient GPU inference
- Async experience collection

## Implementation Roadmap

### Phase 1: Foundation (Current)
- [x] Forge game engine integration
- [x] Interactive mode with stdin/stdout JSON
- [x] Basic Python environment wrapper
- [x] Simple reward shaping

### Phase 2: State/Action Encoding
- [ ] Implement card embedding system
- [ ] Build zone encoders (hand, battlefield, etc.)
- [ ] Create action masking logic
- [ ] Test with simple scenarios

### Phase 3: Model Training
- [ ] Implement Transformer policy network
- [ ] Set up PPO training loop
- [ ] Add curriculum learning stages
- [ ] Benchmark against rule-based AI

### Phase 4: Scaling
- [ ] Parallelize environment collection
- [ ] Implement self-play training
- [ ] Add population-based training
- [ ] Train on full card pool

## Key Metrics to Track

1. **Win Rate** vs baseline AI
2. **Game Length** (shorter = more efficient play)
3. **Decision Time** (inference latency)
4. **Policy Entropy** (exploration vs exploitation)
5. **Value Loss** (state estimation accuracy)

## Estimated Resources

| Component | Requirement |
|-----------|-------------|
| Training Time | 1-4 weeks (depending on scope) |
| GPUs | 1-4 (V100 or better) |
| RAM | 32GB+ |
| Storage | 100GB+ (for checkpoints, logs) |
| Docker Containers | 16-64 parallel |

## References

- AlphaGo/AlphaZero (self-play, MCTS)
- OpenAI Five (PPO, curriculum learning)
- DeepStack (imperfect information games)
- Hearthstone bots (similar card game work)

## Files

- `rl_environment.py` - Main RL environment
- `agent_wrapper.py` - Low-level game communication
- `test_interactive.py` - Testing script
