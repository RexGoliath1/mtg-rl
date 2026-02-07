# ForgeRL Architecture

## Project Goals

**Goal 1: Gameplay Agent** -- Train one agent that plays Standard/Modern competitively, with Commander as a long-term target. The agent observes the full game state (hand, battlefield, graveyard, exile, stack) and selects actions across all phases of a turn.

**Goal 2: Deck Recommendations** -- Use simulated play to evaluate which decks and card combinations produce the highest win rates. Discover synergies that are hard for human deckbuilders to identify by running thousands of matchups in Forge.

**Goal 3: Turn Recommendations** -- Given a game in progress, recommend the best line of play for the current turn and future turns. This is the policy network applied as an advisory tool rather than an autonomous player.

All three goals share the same core: a neural network that understands MTG game states deeply enough to evaluate positions and select actions. The difference is how that network is deployed.

---

## Complexity Assessment

### How MTG Compares to Solved Games

| Dimension | AlphaStar (SC2) | OpenAI Five (Dota 2) | Pluribus (Poker) | MTG (Modern 1v1) | MTG (Commander 4p) |
|-----------|----------------|---------------------|-----------------|-------------------|---------------------|
| Players | 2 | 2 teams of 5 | 2-6 | 2 | 4 |
| Entity types | ~100 units | ~120 heroes | 52 cards | ~5,000 played | ~27,000 legal |
| Entities per game | ~200 | ~100 | 2-7 cards | ~60-100 | ~200-400 |
| Hidden info | Fog of war (~70%) | Fog of war | Hands (~90%) | Hands + libraries (~60%) | Hands + libraries (~60%) |
| Action space | ~10^26 | ~10^20 | ~10^4 per hand | ~10^5 per turn | ~10^8 per turn |
| Timing | Real-time | Real-time | Turn-based | Turn-based with priority | Turn-based with priority |
| Game length | ~20 min, ~10K actions | ~45 min, ~20K actions | ~5 min, ~200 actions | ~20 min, ~500 decisions | ~60 min, ~2000 decisions |
| Training data | Millions of replays | Millions of games | Self-play only | ~50K games (17lands/Forge) | Very scarce |
| Model size | 139M (55M inference) | 159M | ~1M (tabular) | **Target: 6-15M** | **Target: 10-30M** |

**Key insight:** MTG Modern 1v1 is comparable in complexity to a simplified StarCraft -- fewer real-time decisions but more entity types and deeper interaction rules. Commander 4-player is genuinely harder than anything solved to date in the card game domain, but tractable as an extension of the 1v1 system.

### What Makes MTG Uniquely Hard

1. **Arbitrary card text.** Unlike StarCraft where units have fixed abilities, MTG cards can say anything. The interaction space is combinatorial over card text, not just position and type.

2. **Stack-based priority.** The stack is a LIFO queue where spells and abilities wait to resolve, and any player can respond at instant speed. This creates deeply branching game trees at every priority pass. No other game AI has had to handle this exact structure, though it is analogous to response timing in real-time games.

3. **Phase structure.** Each turn has 12+ phases, each with priority windows. A single turn can involve dozens of decision points. The agent must learn "when to act" as well as "what to do."

4. **Card pool size.** 27,000+ Commander-legal cards means the representation must generalize -- the agent will encounter cards in testing that it rarely or never saw in training.

---

## State Representation

### Per-Card Encoding: Mechanics Primitives (Not Per-Card ID)

The sparsity problem with 27,000+ unique cards is solved by encoding cards by their **mechanical properties**, not their identity. This is analogous to AlphaStar encoding units by (type, health, shield, abilities) rather than a unit name.

```
Lightning Bolt → [INSTANT_SPEED, DEAL_DAMAGE, TARGET_ANY]     + params: {damage: 3}
Shock          → [INSTANT_SPEED, DEAL_DAMAGE, TARGET_ANY]     + params: {damage: 2}
Grapeshot      → [SORCERY_SPEED, DEAL_DAMAGE, TARGET_ANY, STORM] + params: {damage: 1}
```

The network learns what `DEAL_DAMAGE + TARGET_ANY` means. New burn spells work automatically because they're combinations of known primitives.

**Current vocabulary:** 214 mechanics primitives, VOCAB_SIZE embedding dimension of 1,311. Covers ~70% of MTG mechanics well.

**Known gaps requiring work:**
- **Type filters:** No way to encode "noncreature spell" vs "any spell" (these parse identically today)
- **Magnitude storage:** `+2/+0` loses the `0` because the parser only stores the first number found per pattern
- **Conditional scoping:** Delirium is listed as a flat mechanic, not linked to the specific abilities it gates
- **Modal choices:** No primitives for "choose one" or "choose X"
- **Duration:** No distinction between "until end of turn" and "as long as"

**Validation plan:** Run the parser against every Standard/Modern/Commander-legal card via Scryfall API. Measure confidence distribution. Target: >90% of cards parse with confidence >0.7.

### Per-Card Feature Vector

Each card is encoded as **1,363 dimensions**:

```
Mechanics embedding: 1,311 dims  (binary presence of primitives, pre-computed in HDF5)
Numeric parameters:     20 dims  (CMC, power, toughness, damage values, counts)
Dynamic game state:     32 dims  (zone, tapped, attacking, counters, damage, modifiers)
                     ─────────
Total per card:      1,363 dims
```

Dynamic state captures what's happening to the card right now (tapped, has counters, is attacking), while mechanics and parameters capture what the card *is*.

### Zone Encoding: Hierarchical Attention

The game state is organized by zones. Each zone contains a variable number of cards.

```
Per-Card Features (1,363 dims each)
         │
    ┌────┴────┐
    │ Zone    │  Self-attention (4 heads, 256 dim)
    │ Encoder │  Learns: card synergies within this zone
    └────┬────┘  Output: 256-dim zone embedding + per-card embeddings
         │
    Repeat for: Hand, Battlefield, Graveyard, Exile (shared weights)
         │
    ┌────┴────┐
    │ Stack   │  Self-attention with positional encoding (LIFO order)
    │ Encoder │  Learns: spell/ability interaction on stack
    └────┬────┘  Output: 256-dim stack embedding
         │
    ┌────┴────────────────────────────────────────┐
    │ Cross-Zone Attention (4 heads)               │
    │ Learns: inter-zone relationships             │
    │ (graveyard targets for reanimation,          │
    │  hand cards that interact with battlefield)  │
    └────┬────────────────────────────────────────┘
         │
    ┌────┴────┐
    │ Global  │  Life totals (binary), mana pools (per color),
    │ Encoder │  turn number, phase, active/priority player
    └────┬────┘  Output: 128-dim global embedding
         │
    ┌────┴────┐
    │ Combine │  Concatenate all → MLP → 512-dim state embedding
    └────┬────┘
         │
    512-dim game state vector
```

This is **one network that sees everything** -- all zones, all players. The hierarchy is for parameter efficiency (6.3M params vs ~87M for naive flat encoding), not for modularity. Every zone embedding attends to every other zone through cross-zone attention.

**Why not separate networks per zone?** The cross-zone attention layer already handles this. A card in hand only matters in context of the battlefield; a graveyard card only matters if something can reanimate it. These relationships are captured by attention, not by separate networks.

### Hidden Information Strategy: CTDE

During training, use **Centralized Training, Decentralized Execution** (proven in PerfectDou, NeurIPS 2022):

- **Value network (critic):** Sees ALL hands, including opponents'. This is possible because Forge provides complete game state. The critic learns better position evaluation from perfect information.
- **Policy network (actor):** Only sees what a legal player would see -- own hand, all battlefields, graveyards, exile, stack, but NOT opponents' hands or library contents.

The critic "teaches" the policy through the advantage function during training. At inference (actual gameplay), only the policy is used, and it operates on legal observations only.

This approach requires **an order of magnitude fewer training samples** vs training blind (per PerfectDou results on DouDizhu).

**Implementation:** The ForgeGameStateEncoder receives a `visibility_mask` parameter. During training, the critic encodes all zones for all players. The policy encodes all zones for self but only public zones (battlefield, graveyard, exile) for opponents.

---

## Action Representation

### Current: Flat Action Space (153 actions, masked)

```
PASS: 1, MULLIGAN: 1, CONCEDE: 1
CAST_SPELL: 15 (by hand slot)
ACTIVATE: 50 (by battlefield slot)
ATTACK: 50 (by battlefield slot)
BLOCK: 50 (by battlefield slot)
CHOOSE_TARGET: 20, CHOOSE_MODE: 5, PAY_COST: 10
```

This works for small games but does not scale to Commander (100+ permanents, 20+ cards in hand with effects like card draw).

### Target: Auto-Regressive Decomposition (AlphaStar-style)

```
Step 1: Action Type (softmax over ~8 options)
  → Pass priority
  → Play land
  → Cast spell from hand
  → Activate ability on battlefield
  → Declare attackers
  → Declare blockers
  → Special action (mulligan, concede)

Step 2: Source Selection (pointer network into zone)
  Conditioned on Step 1.
  → If "Cast spell": pointer over hand card embeddings
  → If "Activate ability": pointer over battlefield card embeddings
  → If "Declare attackers": pointer over untapped creatures
  Masked by legal options.

Step 3: Mode/Additional Choice
  Conditioned on Steps 1+2.
  → If modal spell: which mode (softmax over modes)
  → If X spell: what X value
  → If kicked: pay kicker?

Step 4: Target Selection (pointer network into legal targets)
  Conditioned on Steps 1+2+3.
  → Pointer over all legal target entities (creatures, players, permanents)
  → Can be repeated for multi-target spells

Step 5: Mana Payment (optional, for complex cases)
  → Which lands to tap
  → Usually auto-resolved, but matters for multi-color mana bases
```

Each step outputs a probability distribution, conditioned on all previous steps. Training uses teacher forcing (provide ground-truth previous steps during behavioral cloning). The pointer networks select from **actual entity embeddings** in the game state, not fixed slots, so this scales to any board size.

---

## Training Pipeline

### Phase 1: Behavioral Cloning (Imitation Learning)

Train the policy network to imitate the Forge AI's decisions. This bootstraps the agent with basic competence -- legal plays, reasonable card evaluation, basic combat math.

**Data collection:** Run Forge in observation mode (`-o`). The AI plays itself. For each decision, store the full game state (all zones, all cards with mechanics embeddings) and the AI's chosen action (as an index into the action list).

**What the agent learns:**
- When to play cards vs pass priority
- Basic card evaluation (don't waste removal on weak creatures)
- Combat fundamentals (attack/block decisions)
- Mana sequencing (tap lands correctly)

**What it does NOT learn:**
- Optimal play (Forge AI is not perfect)
- Long-term planning (BC is myopic -- predict next action, not game strategy)
- Meta knowledge (which decks beat which)

**Data requirements:** ~100K-500K decisions with full state encoding. At ~400 decisions/game, this is ~250-1250 games. Achievable in a few hours with Forge.

**Architecture note:** The data collection script (`collect_ai_training_data.py`) currently stores only 17 scalar features. It must be upgraded to store full card-level state using the ForgeGameStateEncoder, or store raw JSON for on-the-fly encoding during training.

### Phase 2: Self-Play Reinforcement Learning

Starting from the BC checkpoint, train via self-play with PPO. The agent plays against copies of itself (and past versions to prevent forgetting).

**Reward:**
- Terminal: +1 win, -1 loss
- Intermediate (annealed): life differential, board presence, card advantage

**MCTS integration:** The policy network guides Monte Carlo Tree Search. For each decision:
1. Run N simulations (default 800) using PUCT selection
2. Policy prior guides exploration; value network evaluates leaf nodes
3. Select action based on visit counts

**Self-play pool:** Maintain a pool of past checkpoints. Each game randomly selects an opponent from the pool. This prevents strategy collapse (rock-paper-scissors cycling).

### Phase 3: Curriculum and Format Scaling

```
Stage 1: Simplified Modern (50-card pool, no instants)
  → Learn: basic creatures, combat, land drops, sorcery-speed spells

Stage 2: Full Modern (5,000-card pool, all card types)
  → Learn: instant-speed interaction, stack usage, complex triggers

Stage 3: Commander 1v1 (27,000-card pool, 100-card singleton)
  → Learn: commander tax, larger boards, longer games, wider card diversity

Stage 4: Commander 4-player (future)
  → Learn: multi-opponent evaluation, threat assessment
  → Uses CTDE with 4-player value head (softmax over win probability per player)
```

---

## Parameter Budget and Practicality

### Current Architecture: 6.3M Parameters

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Zone Encoders (x4) | 3,773,440 | Self-attention over cards per zone |
| Stack Encoder | 682,340 | LIFO-ordered attention |
| Global Encoder | 29,472 | Life, mana, turn, phase |
| Cross-Zone Attention | 395,264 | Inter-zone relationships |
| Combine Network | 986,112 | Final aggregation to 512-dim |
| Policy Head | 237,465 | Action selection |
| Value Head | 198,401 | Position evaluation |
| **Total** | **6,302,494** | |

### Scaling for Commander

Commander needs larger zone capacities (100 battlefield permanents vs 50):

| Config | Encoder Params | Total | Inference Latency |
|--------|---------------|-------|-------------------|
| Modern 1v1 (current) | 5.9M | 6.3M | ~3ms |
| Commander 1v1 (scaled zones) | ~8.5M | ~9M | ~5ms |
| Commander 4p (4x player encoding) | ~15M | ~16M | ~8ms |
| With auto-regressive action head | +2-5M | ~20M | ~12ms |

All configurations are well within practical limits. AlphaStar uses 55M parameters at inference with no issues. AlphaZero chess uses ~25-45M. The sweet spot for card game AI appears to be 5-50M parameters.

### Inference During Gameplay

Measured on Apple Silicon MPS (CUDA will be faster):

| Batch Size | Latency | Throughput |
|------------|---------|------------|
| 1 (single decision) | 2.78ms | 360/sec |
| 32 (MCTS batch) | 0.30ms/sample | 107K/sec |

Even with 800 MCTS simulations per decision, each requiring a forward pass, total think time per move is ~240ms batched. This is fast enough for real-time play recommendations.

---

## Deck Recommendation System (Goal 2)

**Approach:** Run round-robin tournaments in Forge between candidate decks. Use win rates to evaluate deck quality. The same policy network that plays games also evaluates positions during games, providing insight into *why* certain decks win.

**Architecture:**
1. **Meta snapshot:** Scrape current meta decks from MTGGoldfish
2. **Tournament simulation:** Play all pairwise matchups (N decks = N*(N-1)/2 matchups, K games each)
3. **Win rate matrix:** Which deck beats which, and by how much
4. **Card impact analysis:** Which cards appear most in winning board states (via attention weights or ablation)
5. **Deck generation (future):** Use the card encoder to propose new deck compositions that fill gaps in the meta

**Reference:** Q-DeckRec (IEEE CIG 2018) uses Deep Q-Networks for deck construction as combinatorial optimization. Our approach is different -- we use gameplay simulation rather than abstract optimization, which captures emergent synergies that static analysis misses.

---

## Key References

### Directly Applicable
| Paper | Year | Relevance |
|-------|------|-----------|
| [PerfectDou](https://arxiv.org/abs/2203.16406) | 2022 | CTDE for card games -- train critic with perfect info, policy with partial |
| [Suphx](https://arxiv.org/abs/2003.13590) | 2020 | 4-player Mahjong with oracle guiding and feature dropout |
| [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) | 2019 | Entity encoder + pointer network + auto-regressive actions |
| [Mastering Hearthstone](https://arxiv.org/abs/2303.05197) | 2023 | Shared card embeddings + auto-regressive action decomposition |
| [Generalised Card Representations](https://arxiv.org/html/2407.05879v1) | 2024 | Property-based card encoding for MTG (transfers to unseen cards) |
| [Q-DeckRec](https://arxiv.org/abs/1806.09771) | 2018 | RL-based deck recommendation for CCGs |
| [Student of Games](https://www.science.org/doi/10.1126/sciadv.adg3256) | 2023 | Unified perfect/imperfect information game algorithm |

### Architecture Foundations
| Paper | Year | Relevance |
|-------|------|-----------|
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | 2019 | Attention-based encoding for variable-size sets |
| [OpenAI Five](https://cdn.openai.com/dota-2.pdf) | 2019 | Monolithic LSTM approach (159M params) -- proves flat architectures work |
| [ByteRL](https://arxiv.org/html/2404.16689v1) | 2024 | Conv1D + LSTM for card game state |

### Multi-Player / Imperfect Information
| Paper | Year | Relevance |
|-------|------|-----------|
| [Pluribus](https://www.science.org/doi/10.1126/science.aay2400) | 2019 | 6-player poker without alliance modeling |
| [ReBeL](https://arxiv.org/abs/2007.13544) | 2020 | Deep RL + search for imperfect info games |
| [AlphaHoldem](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf) | 2022 | End-to-end self-play for poker |

---

## Development Workstreams

The project splits naturally into three independent workstreams that can be developed in parallel by separate agents:

### Workstream A: Card Encoding & Vocabulary (src/mechanics/)

**Goal:** Ensure every MTG card can be accurately decomposed into mechanics primitives.

**Key tasks:**
- Validate vocabulary against full Scryfall card database
- Add missing primitives (type filters, modal choices, duration markers)
- Fix magnitude extraction (store power/toughness changes separately)
- Add conditional scoping (link conditions to the effects they gate)
- Write unit tests with tricky cards (Dragon's Rage Channeler, Cryptic Command, Omnath)
- Benchmark parser confidence across Standard/Modern/Commander card pools

**Files:** `src/mechanics/vocabulary.py`, `src/mechanics/card_parser.py`, `src/mechanics/precompute_embeddings.py`

### Workstream B: Data Pipeline & Training (scripts/, src/training/)

**Goal:** Collect rich training data from Forge and train the imitation learning model.

**Key tasks:**
- Upgrade `collect_ai_training_data.py` to store full card-level state (not 17 scalars)
- Implement auto-regressive action encoding (action type → source card → target)
- Train behavioral cloning model on Forge AI data
- Implement CTDE (critic sees all hands, policy sees legal observations)
- Set up self-play training loop with PPO
- Cloud deployment for training runs

**Files:** `scripts/collect_ai_training_data.py`, `scripts/train_imitation.py`, `src/training/self_play.py`, `src/forge/game_state_encoder.py`, `src/forge/policy_value_heads.py`

### Workstream C: Infrastructure & Forge Integration (infrastructure/, forge-repo/)

**Goal:** Reliable cloud training infrastructure and stable Forge communication.

**Key tasks:**
- Terraform modules for different training modes
- Docker builds for Forge daemon + training
- Forge daemon stability (handle long games, timeouts, reconnection)
- S3 data pipeline (collection → storage → training)
- Monitoring and cost controls

**Files:** `infrastructure/`, `scripts/deploy_*.sh`, `src/forge/forge_client.py`, `infrastructure/docker/Dockerfile.*`

---

## Current Status and Next Steps

### What Exists and Works
- [x] ForgeGameStateEncoder (5.9M params, hierarchical attention) -- `src/forge/game_state_encoder.py`
- [x] Policy/Value heads with action masking -- `src/forge/policy_value_heads.py`
- [x] MCTS integration -- `src/forge/mcts.py`
- [x] Forge TCP client -- `src/forge/forge_client.py`
- [x] Mechanics vocabulary (214 primitives) -- `src/mechanics/vocabulary.py`
- [x] Card text parser -- `src/mechanics/card_parser.py`
- [x] Pre-computed card embeddings -- `src/mechanics/precompute_embeddings.py`
- [x] Parallel self-play framework -- `src/training/parallel_selfplay.py`
- [x] AWS infrastructure (Terraform, ECR, S3) -- `infrastructure/`
- [x] Data collection from Forge (basic) -- `scripts/collect_ai_training_data.py`

### What Needs Work (Priority Order)
1. **Data collection upgrade** -- Store full card-level state, not 17 scalars
2. **Vocabulary validation** -- Test against full card pool, fill gaps
3. **Auto-regressive action head** -- Replace flat 153-action softmax with pointer networks
4. **CTDE implementation** -- Split critic/actor visibility for training
5. **Behavioral cloning training** -- End-to-end with rich state encoding
6. **Self-play fine-tuning** -- PPO on top of BC checkpoint
7. **Deck tournament system** -- Round-robin simulation for meta analysis
