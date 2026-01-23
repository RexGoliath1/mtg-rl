# Self-Play Training for MTG RL

## Overview

This document describes the self-play training infrastructure implemented for competitive MTG AI training, inspired by AlphaStar and OpenAI Five.

## Components Created

### 1. Self-Play Infrastructure (`self_play.py`)

**Key Components:**
- `EloTracker` - Tracks relative skill levels using Elo rating system
- `ModelPool` - Stores and manages past model checkpoints
- `SelfPlayGame` - Manages individual games between two models
- `SelfPlayTrainer` - Orchestrates the full training loop

**Usage:**
```bash
# Start self-play training
python self_play.py --games 100000 --checkpoint-interval 1000 --strategy recent

# With custom daemon
python self_play.py --host localhost --port 17171 --games 50000
```

### 2. Hierarchical Action Space (`hierarchical_actions.py`)

**Key Components:**
- `ActionParser` - Parses flat actions into hierarchical structure
- `PointerNetwork` - Selects from variable-size sets (cards, targets)
- `HierarchicalPolicyNetwork` - Multi-level action selection
- `ComplexMechanicsEncoder` - Handles 2024-2025 card mechanics

**Action Hierarchy:**
```
Level 1: Action Type  → {pass, play_spell, play_land, activate, attack, block}
Level 2: Card Selection → Which card from hand/battlefield
Level 3: Mode Selection → Kicker, X value, choose modes
Level 4: Target Selection → What to target
```

### 3. Regression Tests (`tests/test_self_play.py`)

**Test Coverage:**
- Elo rating system (win/loss updates, leaderboard)
- Model pool operations (add, sample, prune, persistence)
- Turn tracking across games
- Error persistence and recovery
- Multi-agent coordination
- Game state serialization

**Run tests:**
```bash
pytest tests/test_self_play.py -v
```

## Card Mechanics Support (2024-2025)

The system supports all major mechanics from recent sets:

| Set | Mechanics |
|-----|-----------|
| Aetherdrift (2025) | Start Your Engines, Exhaust |
| Duskmourn (2024) | Impending, Eerie, Survival |
| Bloomburrow (2024) | Offspring, Valiant, Expend, Forage, Gift |
| Thunder Junction (2024) | Plot, Spree, Saddle, Crime |
| Karlov Manor (2024) | Disguise, Cloak, Collect Evidence |
| Lost Caverns (2023) | Craft, Descend, Discover |
| Eldraine (2023) | Bargain, Celebration, Roles |

## Research Findings

### Training Data Sources

1. **17Lands** - Best source for draft/limited data
   - Aggregated game data with ML models trained
   - https://www.17lands.com/

2. **MTGATracker** - Arena game logs
   - Parse local log files
   - https://github.com/mtgatracker

3. **MTGO-Tracker** - MTGO replay data
   - Similar log parsing approach

### Recent RL Papers (2024-2025)

1. **Structured RL for Combinatorial Action Spaces** (NeurIPS 2025)
   - Embeds optimization layers in actor network
   - Better for routing/scheduling-like problems

2. **Transformer-Based MARL** (ICML 2025)
   - Graph Transformer for long-range dependencies
   - STACCA framework for shared representations

3. **Hierarchical RL with LLMs** (2025)
   - LLM-augmented action primitives
   - Useful for complex multi-step tasks

### Card Embeddings Research

1. **MTG Card Representations (2024)**
   - Siamese neural networks for card-deck relationships
   - 512-dimensional embedding space
   - Text + image + features fusion

2. **ByteRL Architecture**
   - 1D convolutions for card sequences
   - LSTM for battle stage
   - Separate draft/battle networks

## Linux/Cloud Deployment

### Docker Deployment
```bash
# Build
docker build -t mtg-training .

# Run training
docker run -d --name mtg-trainer \
  -v $(pwd)/model_pool:/forge/model_pool \
  mtg-training python self_play.py --games 1000000
```

### Cloud Scaling (AWS/GCP)

**Recommended Instance:**
- c5.4xlarge (16 vCPU, 32GB RAM)
- ~4 daemon instances per machine
- ~24,000 games/hour per instance

**Cost Estimate:**
| Games | Time | Cost |
|-------|------|------|
| 1M | 2 days | ~$100 |
| 10M | 18 days | ~$300 |
| 50M | 3 months | ~$1,500 |

### Multi-Machine Setup
```python
# Configure multiple daemons
config = SelfPlayConfig(
    daemon_host='10.0.0.1',  # First machine
    daemon_port=17171,
)

# Or use load balancer across machines
config = SelfPlayConfig(
    daemon_host='training-lb.internal',
    daemon_port=17171,
)
```

## Training Strategy

### Phase 1: Bootstrap (100K games)
- Random initialization
- Train against random baseline
- Goal: Learn basic land drops, spell casting

### Phase 2: Self-Play (1M games)
- 80% vs recent self, 20% vs pool
- Elo tracking enabled
- Goal: Develop basic strategies

### Phase 3: Competitive (10M+ games)
- Population-based training
- Multiple parallel trainers
- Goal: Beat strong AI baselines

## Monitoring

### Wandb Integration
```python
from training_dashboard import TrainingDashboard

dashboard = TrainingDashboard(config)
dashboard.run_training()
```

### Key Metrics
- Win rate (100-game rolling)
- Elo rating vs baseline
- Games per hour
- Policy/Value loss

## File Structure

```
mtg/
├── self_play.py           # Self-play trainer
├── hierarchical_actions.py # Hierarchical action space
├── policy_network.py      # Transformer policy
├── ppo_agent.py          # PPO algorithm
├── card_embeddings.py    # Card feature extraction
├── daemon_environment.py # Game environment
├── training_dashboard.py # Wandb dashboard
├── tests/
│   └── test_self_play.py # Regression tests
└── runs/
    └── selfplay_*/       # Training runs
        ├── model_pool/   # Checkpoints
        ├── config.json   # Run config
        └── training_metrics.json
```
