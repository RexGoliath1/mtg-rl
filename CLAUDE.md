# CLAUDE.md - MTG RL Project Context

## Project Overview

**ForgeRL** - A reinforcement learning system for Magic: The Gathering draft and gameplay, built on top of Forge (Java MTG engine).

**Goal**: Train an AI that can draft competitively and play games using human gameplay data (17lands.com) and self-play reinforcement learning.

---

## Development Rules

### Cloud-First for Long-Running Jobs

**CRITICAL**: Do NOT run jobs with >50 iterations locally. Deploy to AWS instead.

- Local: Quick tests (<50 games, <5 epochs), debugging, development
- Cloud: Training runs, data collection (>50 games), any job >10 minutes

### Package Management

Project uses **UV** (by Astral, makers of ruff) with `pyproject.toml`. All imports resolve via the installed `src` package — no `sys.path` hacks needed.

```bash
# Install (creates .venv automatically)
uv sync --extra dev          # dev (ruff, pytest)
uv sync --all-extras         # everything (training, data, embeddings, dev)

# Run commands through UV
uv run python3 scripts/card_recommender.py
uv run python3 -m pytest tests/ -v
```

---

## Quick Start for New Sessions

```bash
# Check project status
git status
git log --oneline -5

# Install dependencies
uv sync --extra dev

# Run tests (464 parser tests)
uv run python3 -m pytest tests/test_parser.py -v

# Lint
uv run python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb

# Train draft model (BC on 17lands data)
uv run python3 scripts/training_pipeline.py --mode bc --sets FDN --epochs 5
```

---

## Architecture Summary

### Neural Network Components

| File | Purpose | Params |
|------|---------|--------|
| `src/mechanics/vocabulary.py` | Mechanics primitives (VOCAB_SIZE=1387) | - |
| `src/mechanics/card_parser.py` | Oracle text -> mechanics | - |
| `src/mechanics/precompute_embeddings.py` | Pre-compute HDF5 embeddings | - |
| `src/forge/game_state_encoder.py` | Forge JSON -> tensor | 5.8M |
| `src/forge/policy_value_heads.py` | AlphaZero policy/value | 0.4M |
| `src/forge/mcts.py` | Monte Carlo Tree Search | - |
| `src/training/self_play.py` | Self-play training loop | - |
| `src/models/shared_card_encoder.py` | Simple card encoder (for draft) | 1.2M |
| `src/models/entity_encoder.py` | Full game state encoder (legacy) | 9.2M |
| `src/models/draft_policy.py` | Draft-specific policy network | 2.8M |
| `scripts/training_pipeline.py` | Unified BC + RL training | - |

### Training Pipeline

```
Phase 1: Behavioral Cloning (17lands data)
├── Download: uv run python3 scripts/download_17lands.py --sets FDN DSK BLB
├── Train: uv run python3 scripts/training_pipeline.py --mode bc --epochs 10
└── Output: checkpoints/bc_best.pt

Phase 2: RL Fine-tuning (Forge self-play)
├── Start Forge daemon: java -jar forge.jar daemon
├── Train: uv run python3 scripts/training_pipeline.py --mode rl --episodes 10000
└── Output: checkpoints/rl_best.pt

Phase 3: Evaluation
├── Eval: uv run python3 scripts/training_pipeline.py --mode eval
└── Compare against baselines
```

### Forge Integration

The Forge MTG engine is in `forge-repo/` (gitignored, separate repo).

**Repository**: `git@github.com:RexGoliath1/forge.git` (fork of Card-Forge/forge)

**Current Branch**: `feature/rl-daemon-mode`

```bash
# Clone our Forge fork (if not present)
git clone git@github.com:RexGoliath1/forge.git forge-repo
cd forge-repo && git checkout feature/rl-daemon-mode

# Build Forge
mvn package -DskipTests

# Run draft daemon
java -jar forge-gui-desktop/target/forge.jar daemon -p 17220
```

**Key Forge Files**:
- `forge-game/src/main/java/forge/game/card/Card.java` - Card representation
- `forge-game/src/main/java/forge/game/Game.java` - Game state
- `forge-game/src/main/java/forge/trackable/TrackableProperty.java` - State serialization
- `forge-gui-desktop/src/main/java/forge/view/ForgeDraftDaemon.java` - Our daemon (custom)

---

## Key Design Decisions

1. **Mechanics-Based Card Encoding**: Cards are multi-hot vectors over ~1387 primitives, not opaque embeddings
2. **AlphaZero Architecture**: Policy + Value networks with MCTS, trained via self-play
3. **Forge as Simulator**: Official MTG rules engine for game tree exploration
4. **No Meta Features**: Self-play learns optimal play, not human imitation
5. **HDF5 Storage**: Pre-computed card encodings (~1.3MB for all Commander cards)

### MTG Rules Reference

Full comprehensive rules: `MagicCompRules 20260116.txt` (932KB, project root)
Distilled reference for parser work: `docs/MTG_RULES_REFERENCE.md`

### Mechanics Vocabulary Approach

Instead of text embeddings or one-hot card IDs, we decompose cards into **mechanics primitives**:

```python
# Saw in Half encoded as mechanics sequence
[INSTANT_SPEED, TARGET_CREATURE, DESTROY, IF_TARGET_DIES, CREATE_TOKEN_COPY, HALF_STATS]

# Network learns: DESTROY + CREATE_TOKEN_COPY on own creature = double ETB triggers
# Discovered through MCTS self-play, not pre-coded
```

**Key files:**
- `src/mechanics/vocabulary.py` - 1387 mechanics primitives (VOCAB_SIZE)
- `src/mechanics/card_parser.py` - Oracle text -> mechanics sequence
- `research/alphazero_mtg_architecture.md` - Full architecture design

**Why this approach:**
- New mechanics = new combinations of existing primitives
- Transfers to any format (Draft, Standard, Commander)
- MCTS discovers card interactions, no manual coding
- ~1.3MB storage for all 30K Commander-legal cards (HDF5 format)

---

## Common Tasks

### Run Tests and Lint
```bash
uv run python3 -m pytest tests/test_parser.py -v    # 464 parser tests
uv run python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb
```

### Card Recommender
```bash
uv run python3 scripts/card_recommender.py    # http://localhost:8000
```

### Embedding Quiz
```bash
uv run python3 scripts/embedding_quiz.py      # http://localhost:8787
```

### Parser Coverage Report
```bash
uv run python3 scripts/parser_coverage_report.py --format standard
```

### Regenerate HDF5 Embeddings
```bash
uv run python3 -m src.mechanics.precompute_embeddings --format commander --bulk-json data/scryfall_bulk_cards.json
```

### Deploy to AWS
```bash
cd infrastructure && terraform apply -var="enable_training_instance=true"

# SSH into running instance
./scripts/ssh-instance.sh           # Interactive SSH
./scripts/ssh-instance.sh logs      # Tail collection logs

# Check S3 for results
aws s3 ls s3://mtg-rl-checkpoints-*/imitation_data/ --recursive
```

---

## Testing & Linting

### Before Committing

```bash
uv run python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb
uv run python3 -m pytest tests/test_parser.py -v
```

### CI

GitHub Actions runs on every push/PR via `.github/workflows/test.yml`:
- Uses `astral-sh/setup-uv@v5` + `uv sync --extra dev`
- Runs ruff lint + parser tests
- Free tier (GitHub-hosted runners)

---

## File Structure

```
mtg/
├── pyproject.toml            # UV/hatchling project config
├── uv.lock                   # Lockfile for reproducible installs
├── CLAUDE.md                 # This file
├── ARCHITECTURE.md           # Detailed architecture docs
├── DEPLOYMENT.md             # Consolidated deployment guide
├── WHITEPAPER.md             # Technical paper
│
├── src/                      # Installable Python package
│   ├── __init__.py
│   ├── agents/               # PPO agent, agent wrapper, self-play (Elo/ModelPool)
│   ├── data/                 # 17lands data loader, MTGGoldfish scraper
│   ├── environments/         # Draft env, daemon env, RL env
│   ├── forge/                # Forge client, state encoder, MCTS, policy/value heads
│   ├── mechanics/            # Vocabulary (1387 enums), card parser, HDF5 precompute
│   ├── models/               # Card embeddings, policy network, shared encoder, text embeddings
│   ├── training/             # Self-play (AlphaZero), forge imitation, parallel selfplay, profiler
│   └── utils/                # Checkpoint manager, evaluate, replay recorder, wandb integration
│
├── scripts/                  # Entry-point scripts (39 files)
│   ├── card_recommender.py   # EDHREC-inspired web app (http://localhost:8000)
│   ├── embedding_quiz.py     # Card encoding review tool (http://localhost:8787)
│   ├── training_pipeline.py  # Unified BC + RL training
│   ├── parser_coverage_report.py
│   ├── collect_ai_training_data.py
│   ├── deploy_data_collection.sh
│   └── ...
│
├── tests/                    # Test suite (464 parser tests + encoder/pipeline/self-play tests)
├── research/                 # Experimental files
├── infrastructure/           # Terraform (S3, ECR, IAM, EC2)
├── decks/                    # Deck files for training
├── docs/                     # MTG rules reference
├── data/                     # Training data, HDF5 embeddings, sidecar metadata (gitignored)
├── checkpoints/              # Model checkpoints (gitignored)
├── logs/                     # TensorBoard logs (gitignored)
└── forge-repo/               # Forge MTG engine (gitignored, separate repo)
```

---

## AWS Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Budget Limit** | $100/month | Hard-coded |
| **Region** | us-east-1 | Default for compute |
| **S3 Bucket** | `mtg-rl-checkpoints-20260124190118616600000001` | All artifacts |
| **Infrastructure** | Terraform | `infrastructure/main.tf` |

### S3 Lifecycle Rules (7 rules configured)
- Experiment checkpoints: IA at 30d, delete at 90d
- Promoted models: IA at 30d, never expires
- Imitation data: One Zone-IA at 30d, Glacier IR at 90d
- TensorBoard logs: delete at 60d
- Noncurrent versions: delete at 7d
- Versioning: suspended

### Cost Controls
- Budget alerts at 50%, 80%, 100% of $100
- Spot instances for training (70% cheaper)
- ECR lifecycle: keep only last 5 images

---

## Development Priorities

**IMPORTANT**: Prefer training-side edits over Forge edits when possible.
- Training code (Python) is faster to iterate and test
- Forge changes require rebuilding Docker image (~2 min)
- Only modify Forge for data collection improvements

---

## Known Issues / TODOs

- [ ] Checkpoint pruning in CheckpointManager (keep last 3 + best)
- [ ] EntityEncoder dimension mismatch with training pipeline (use SharedCardEncoder for now)
- [ ] Forge daemon integration (actual game simulation)
- [x] UV + pyproject.toml packaging (all sys.path hacks removed)
- [x] Codebase reorganized: 0 root Python files (was 42)
- [x] Mechanics vocabulary: VOCAB_SIZE=1387
- [x] Card text parser: 464 tests passing
- [x] HDF5 precomputed: 30,462 cards, 1.34 MB
- [x] Card recommender: EDHREC-inspired web app with curve-aware scoring
- [x] GitHub Actions CI: lint + tests via UV
- [x] S3 lifecycle rules: 7 rules, versioning suspended
- [x] AlphaZero policy/value network + MCTS
- [x] Self-play training loop
- [x] Parallel self-play training
- [x] Forge client TCP integration
- [x] Imitation learning data collection (1000 games, 400K decisions)

---

## Resuming Work

When starting a new Claude session:

1. **Read this file** and `ClaudeConversation.txt`
2. **Check git status**: `git status && git log --oneline -5`
3. **Run tests**: `uv run python3 -m pytest tests/test_parser.py -v`
