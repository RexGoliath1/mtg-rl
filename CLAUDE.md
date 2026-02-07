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

```bash
# LOCAL - Quick test (OK)
python scripts/collect_ai_training_data.py --games 20 --workers 4

# CLOUD - Full collection (deploy to AWS)
# 1. Push image to ECR
# 2. terraform apply -var="enable_training_instance=true"
# 3. SSH and run collection
```

---

## Quick Start for New Sessions

```bash
# Check project status
git status
git log --oneline -5

# Run tests
python -m pytest tests/ -v

# Train draft model (BC on 17lands data)
python training_pipeline.py --mode bc --sets FDN --epochs 5

# Play simulated drafts
python training_pipeline.py --mode play --num-games 3
```

---

## Architecture Summary

### Neural Network Components

| File | Purpose | Params |
|------|---------|--------|
| `src/mechanics/vocabulary.py` | Mechanics primitives (VOCAB_SIZE=1380) | - |
| `src/mechanics/card_parser.py` | Oracle text → mechanics | - |
| `src/mechanics/precompute_embeddings.py` | Pre-compute HDF5 embeddings | - |
| `src/forge/game_state_encoder.py` | Forge JSON → tensor | 5.8M |
| `src/forge/policy_value_heads.py` | AlphaZero policy/value | 0.4M |
| `src/forge/mcts.py` | Monte Carlo Tree Search | - |
| `src/training/self_play.py` | Self-play training loop | - |
| `shared_card_encoder.py` | Simple card encoder (for draft) | 1.2M |
| `entity_encoder.py` | Full game state encoder (legacy) | 9.2M |
| `draft_policy.py` | Draft-specific policy network | 2.8M |
| `training_pipeline.py` | Unified BC + RL training | - |

### Training Pipeline

```
Phase 1: Behavioral Cloning (17lands data)
├── Download: python scripts/download_17lands.py --sets FDN DSK BLB
├── Train: python training_pipeline.py --mode bc --epochs 10
└── Output: checkpoints/bc_best.pt

Phase 2: RL Fine-tuning (Forge self-play)
├── Start Forge daemon: java -jar forge.jar daemon
├── Train: python training_pipeline.py --mode rl --episodes 10000
└── Output: checkpoints/rl_best.pt

Phase 3: Evaluation
├── Eval: python training_pipeline.py --mode eval
└── Compare against baselines
```

### Forge Integration

The Forge MTG engine is in `forge-repo/` (gitignored, separate repo).

**Repository**: `git@github.com:RexGoliath1/forge.git` (fork of Card-Forge/forge)

**Current Branch**: `feature/rl-daemon-mode`

```bash
# Clone our Forge fork (if not present)
cd /Users/stevengonciar/git/mtg
git clone git@github.com:RexGoliath1/forge.git forge-repo
cd forge-repo
git checkout feature/rl-daemon-mode

# Add upstream for merging new mechanics
git remote add upstream https://github.com/Card-Forge/forge.git

# Build Forge
mvn package -DskipTests

# Run draft daemon
java -jar forge-gui-desktop/target/forge.jar daemon -p 17220
```

**Merging New Forge Updates (for new MTG mechanics)**:
```bash
cd forge-repo

# Fetch latest from upstream
git fetch upstream

# Merge into our branch
git checkout feature/rl-daemon-mode
git merge upstream/master

# Resolve any conflicts, then push
git push origin feature/rl-daemon-mode
```

**Key Forge Files**:
- `forge-game/src/main/java/forge/game/card/Card.java` - Card representation
- `forge-game/src/main/java/forge/game/Game.java` - Game state
- `forge-game/src/main/java/forge/trackable/TrackableProperty.java` - State serialization
- `forge-gui-desktop/src/main/java/forge/view/ForgeDraftDaemon.java` - Our daemon (custom)

---

## Key Design Decisions

1. **Mechanics-Based Card Encoding**: Cards are multi-hot vectors over ~1373 primitives, not opaque embeddings
2. **AlphaZero Architecture**: Policy + Value networks with MCTS, trained via self-play
3. **Forge as Simulator**: Official MTG rules engine for game tree exploration
4. **No Meta Features**: Self-play learns optimal play, not human imitation
5. **HDF5 Storage**: Pre-computed card encodings (~5-10MB for all Commander cards)

### MTG Rules Reference

Full comprehensive rules: `MagicCompRules 20260116.txt` (932KB, project root)
Distilled reference for parser work: `docs/MTG_RULES_REFERENCE.md`

The reference doc covers:
- **Cost taxonomy**: additional costs vs alternative costs vs cost reduction (with all keywords categorized)
- **All 189 keyword abilities** organized by category
- **Key rules concepts**: triggers, replacement effects, static abilities, layers
- **Common text patterns** for "as an additional cost" / "rather than pay" / cost reduction

### Mechanics Vocabulary Approach (NEW)

Instead of text embeddings or one-hot card IDs, we decompose cards into **mechanics primitives**:

```python
# Saw in Half encoded as mechanics sequence
[INSTANT_SPEED, TARGET_CREATURE, DESTROY, IF_TARGET_DIES, CREATE_TOKEN_COPY, HALF_STATS]

# Network learns: DESTROY + CREATE_TOKEN_COPY on own creature = double ETB triggers
# Discovered through MCTS self-play, not pre-coded
```

**Key files:**
- `src/mechanics/vocabulary.py` - ~1373 mechanics primitives (VOCAB_SIZE)
- `src/mechanics/card_parser.py` - Oracle text → mechanics sequence
- `research/alphazero_mtg_architecture.md` - Full architecture design

**Why this approach:**
- New mechanics = new combinations of existing primitives (Warp = ALT_COST + EXILE_TEMP + CAST_FROM_EXILE)
- Transfers to any format (Draft, Standard, Commander)
- MCTS discovers card interactions, no manual coding
- ~10MB storage for all 30K Commander-legal cards (HDF5 format)

---

## Common Tasks

### Download 17lands Data
```bash
python scripts/download_17lands.py --sets FDN DSK BLB MKM LCI
python scripts/download_17lands.py --status
```

### Train Model
```bash
# Behavioral cloning
python training_pipeline.py --mode bc --sets FDN --epochs 10

# With specific checkpoint
python training_pipeline.py --mode bc --checkpoint checkpoints/bc_epoch5.pt
```

### Test Components
```bash
python -m pytest tests/ -v
python shared_card_encoder.py  # Self-test
python entity_encoder.py       # Self-test
python text_embeddings.py      # Self-test
```

### Deploy to AWS
```bash
# Deploy imitation learning (10K games)
cd infrastructure
terraform apply -var="training_mode=imitation" -var="imitation_games=10000"

# Deploy BC training
terraform apply -var="training_mode=bc"

# SSH into running instance (uses AWS credentials, no PEM needed)
./scripts/ssh-instance.sh           # Interactive SSH
./scripts/ssh-instance.sh logs      # Tail collection logs
./scripts/ssh-instance.sh status    # Check daemon + collection status

# Check S3 for results
aws s3 ls s3://mtg-rl-checkpoints-*/imitation_data/ --recursive
```

---

## Git Workflow

### Commit Early and Often

**IMPORTANT**: Make small, focused commits frequently. This helps:
- Resume work after context loss
- Track changes incrementally
- Revert specific changes if needed

```bash
# Good commit pattern
git add -A && git commit -m "Add feature X"
git add -A && git commit -m "Fix bug in Y"
git add -A && git commit -m "Update tests for Z"

# Bad pattern (avoid)
git add -A && git commit -m "Big update with lots of changes"
```

### Merge Forge Updates

When new MTG mechanics are released, merge from upstream:

```bash
cd forge-repo

# Add upstream if not present
git remote add upstream https://github.com/Card-Forge/forge.git

# Fetch and merge
git fetch upstream
git checkout main
git merge upstream/master

# Resolve conflicts if any, then push
git push origin main
```

---

## Testing & Linting Requirements

### Before Committing

1. **Run linter**: `python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb`
2. **Run tests**: `python3 -m pytest tests/ -v`
3. **Test modified modules**: Run self-tests for changed files
4. **Verify imports**: `python -c "from training_pipeline import *"`

### Linting

We use `ruff` for linting. Install with: `pip install ruff`

```bash
# Check for issues
python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb

# Auto-fix safe issues
python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb --fix
```

### Test Coverage Goals

- [x] Unit tests for encoder components (tests/test_encoder.py)
- [x] Integration tests for training pipeline (tests/test_pipeline.py)
- [ ] End-to-end draft simulation tests
- [ ] API endpoint tests (when deployed)

---

## File Structure (Target)

The project should follow this structure. Files at root are being consolidated.

```
mtg/
├── CLAUDE.md                 # This file - project context
├── ARCHITECTURE.md           # Detailed architecture docs
├── DEPLOYMENT.md             # Consolidated deployment guide
├── WHITEPAPER.md             # Technical paper
├── README.md                 # Project README
│
├── src/                      # Source code (future refactor target)
│   ├── models/               # Neural network models
│   │   ├── shared_card_encoder.py
│   │   ├── entity_encoder.py
│   │   ├── draft_policy.py
│   │   └── text_embeddings.py
│   ├── training/             # Training pipelines
│   │   ├── train_draft.py
│   │   ├── training_pipeline.py
│   │   └── draft_training.py
│   ├── data/                 # Data loading
│   │   └── data_loader_17lands.py
│   └── environments/         # Gym environments
│       ├── draft_environment.py
│       └── daemon_environment.py
│
├── scripts/                  # Utility scripts
│   ├── download_17lands.py   # Data downloader
│   └── deploy.sh             # AWS deployment
├── infrastructure/           # Terraform/IaC
│   └── main.tf
├── tests/                    # Test suite
├── decks/                    # Deck files for training
├── checkpoints/              # Model checkpoints (gitignored)
├── data/                     # Training data (gitignored)
├── logs/                     # TensorBoard logs (gitignored)
└── forge-repo/               # Forge MTG engine (gitignored)
```

### Current Structure (Pre-Refactor)

Currently, all Python files are at root level. This is acceptable for rapid iteration
but should be organized into packages before production deployment.

**Core Files (keep at root for now):**
- `train_draft.py` - Primary draft training script
- `data_loader_17lands.py` - 17lands data loader (native format)
- `shared_card_encoder.py` - Card encoder

**Deprecated/Redundant (to consolidate):**
- `data_17lands.py` - Superseded by data_loader_17lands.py
- Several training scripts overlap - see redundancy analysis

---

## Resuming Work

When starting a new Claude session:

1. **Read this file first**: `cat CLAUDE.md`
2. **Check git status**: `git status && git log --oneline -5`
3. **Review recent changes**: `git diff HEAD~5`
4. **Run tests**: `python -m pytest tests/ -v`

### Key Context to Provide

When resuming, mention:
- Current task/goal
- Any errors encountered
- Which files you were working on

---

## AWS Configuration

### Key Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Budget Limit** | $100/month | Hard-coded, cannot be overridden |
| **Region** | us-east-1 | Default for compute (changed from us-west-2) |
| **Billing Region** | us-east-1 | Required for billing metrics |
| **Infrastructure** | Terraform | Chosen for scalability |

### Deployment Commands

```bash
# Initialize (first time)
cd infrastructure && terraform init

# Deploy base infrastructure (S3, ECR, IAM - no compute)
terraform apply

# Enable training instance when ready
terraform apply -var="enable_training_instance=true"

# Scale up instance type
terraform apply -var="training_instance_type=g4dn.2xlarge"

# Destroy everything
terraform destroy
```

### Cost Controls

- **Budget alerts** at 50%, 80%, 100% of $100
- **Hard limit** in scripts: $100/month max (cannot be overridden)
- **S3 lifecycle**: Auto-delete old checkpoints after 90 days
- **ECR lifecycle**: Keep only last 5 images
- **Spot instances**: 70% cheaper than on-demand

### Check Current Spending

```bash
aws ce get-cost-and-usage \
    --time-period Start=$(date -v-30d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --query 'ResultsByTime[0].Total.UnblendedCost'
```

---

## Training Run Monitoring

**IMPORTANT**: When kicking off long training runs, always provide these monitoring links.

### TensorBoard Access

```bash
# Option 1: SSM port forwarding (keyless, recommended)
brew install --cask session-manager-plugin  # One-time install
./scripts/connect_training.sh tensorboard
# Then open: http://localhost:6006

# Option 2: SSH port forwarding
./scripts/connect_training.sh ssh
# In another terminal:
ssh -i ~/.ssh/mtg-rl-training.pem -L 6006:localhost:6006 ubuntu@<INSTANCE_IP>
# Then open: http://localhost:6006

# Option 3: Direct (if security group allows)
# http://<INSTANCE_IP>:6006
```

### Check Training Status

```bash
# Quick status check
./scripts/connect_training.sh status

# View live training log
./scripts/connect_training.sh ssm
# Then: tail -f /home/ubuntu/mtg-rl/training.log

# Check S3 for checkpoints
aws s3 ls s3://<BUCKET>/checkpoints/ | tail -5
aws s3 cp s3://<BUCKET>/checkpoints/final_results.json -
```

### Weights & Biases (Future)

When W&B is configured:
- Dashboard: https://wandb.ai/your-org/mtg-rl
- Add to terraform.tfvars: `wandb_api_key = "your-key"`

### Model Artifacts Location

| Artifact | Location |
|----------|----------|
| Best checkpoint | `s3://<BUCKET>/checkpoints/best.pt` |
| Latest checkpoint | `s3://<BUCKET>/checkpoints/latest.pt` |
| TensorBoard logs | `s3://<BUCKET>/tensorboard-logs/` |
| Training results | `s3://<BUCKET>/checkpoints/final_results.json` |
| Local best | `checkpoints/draft_best.pt` |

---

## Development Priorities

**IMPORTANT**: Prefer training-side edits over Forge edits when possible.
- Training code (Python) is faster to iterate and test
- Forge changes require rebuilding Docker image (~2 min)
- Only modify Forge for data collection improvements

---

## Data Storage Best Practices

**Use HDF5 over JSON/JSONL for training data at scale.**

| Scale | JSONL | HDF5 (gzip) | Recommendation |
|-------|-------|-------------|----------------|
| <1K decisions | ~200 KB | ~10 KB | Either works |
| 10K decisions | ~2 MB | ~100 KB | HDF5 preferred |
| 100K+ decisions | ~20 MB | ~1 MB | **HDF5 required** |

**Why HDF5:**
- 20x smaller with compression
- Memory-mapped loading for large datasets
- Random access during training
- Stores numerical arrays efficiently

**Current encoding:** 17-dimensional state vector per decision:
- Player 1: life, hand_size, library_size, creatures, lands, other, mana (7)
- Player 2: same (7)
- Game: turn, phase_idx, is_game_over (3)

---

## Imitation Learning Strategy

**Goal:** Bootstrap a policy that understands basic game flow before self-play.

**Target:** 50,000 games across diverse decks (~15M decisions)

**Focus areas:**
1. Card selection (which spell to play)
2. Turn flow (when to pass priority)
3. Combat decisions (attackers/blockers)
4. Mana usage (tap correct lands)

**NOT focused on:**
- Win rate optimization (self-play handles this)
- Perfect play (just "reasonable" moves)
- Specific matchup knowledge

**Deck diversity is critical** - cover different:
- Colors (mono, 2-color, 3-color)
- Archetypes (aggro, midrange, control)
- Mechanics (tokens, counters, removal)

---

## Known Issues / TODOs

- [ ] EntityEncoder dimension mismatch with training pipeline (use SharedCardEncoder for now)
- [ ] NaN losses with synthetic data (normal - use real 17lands data)
- [x] Forge daemon integrated with observation mode (-o flag)
- [x] AWS cost controls configured ($100/month limit)
- [x] v2 hybrid encoder architecture implemented (hybrid_card_encoder.py)
- [x] Mechanics vocabulary defined (VOCAB_SIZE=1380) - src/mechanics/vocabulary.py
- [x] Card text parser implemented - src/mechanics/card_parser.py
- [x] Pre-embed all MTG cards to HDF5 format (data/card_mechanics_commander.h5)
- [x] Forge game state encoder (src/forge/game_state_encoder.py, 5.8M params)
- [x] AlphaZero-style policy/value network (src/forge/policy_value_heads.py)
- [x] MCTS integration with Forge (src/forge/mcts.py)
- [x] Self-play training loop (src/training/self_play.py)
- [ ] Forge daemon integration (actual game simulation)
- [x] Parallel self-play training (src/training/parallel_selfplay.py)
- [x] Training profiler and benchmarks (src/training/profiler.py)
- [x] MTGGoldfish deck scraper (src/data/mtggoldfish_decks.py)

---

## Gameplay Training (Modern Format)

### Format Choice: Modern
Selected Modern for training because:
- Diverse archetypes (aggro, control, combo, midrange, tempo)
- Stable card pool (no rotation)
- Complex decision trees
- Rich metagame data from MTGGoldfish

### Training Time Estimates

**Cloud Test Results (2026-01-25)** - Tesla T4 (g4dn.xlarge), 8 actors:

| Metric | Simulated (no Forge) | With Forge (estimated) |
|--------|---------------------|------------------------|
| Samples/second | 1,153 | ~50-100 |
| Samples/hour | 4,150,000 | ~180,000-360,000 |
| Time to 1M samples | 14.5 min | 3-6 hours |
| Cost to 1M samples | $0.04 | $0.50-1.00 |

The simulated test (no Forge) shows pure Python/GPU overhead is minimal.
When Forge is integrated, communication latency (~50ms/action) will be the bottleneck.

**Original Estimates (before cloud test)**:

| Configuration | Games/hr | Samples/hr | Time to 1M | AWS Cost |
|---------------|----------|------------|------------|----------|
| 1 actor, 50 MCTS | 300 | 24,000 | 1.7 days | $6.67 |
| 4 actors, 50 MCTS | 792 | 63,336 | 0.7 days | $2.53 |
| **8 actors, 50 MCTS** | **1,286** | **102,890** | **0.4 days** | **$1.56** |
| 16 actors (g4dn.12xl) | 2,089 | 167,146 | 0.2 days | $7.18 |

**Recommended**: 8 actors on g4dn.xlarge spot ($0.16/hr) - best cost/performance.

### Network Performance

Measured on Apple Silicon MPS (CUDA will be faster):

| Batch Size | Latency | Throughput |
|------------|---------|------------|
| 1 | 2.78ms | 360/sec |
| 32 | 0.30ms | 107,608/sec |
| 128 | 0.29ms | 442,326/sec |
| 256 | 0.30ms | 841,974/sec |

**Key insight**: Batching provides massive speedup. Single inference is 2.8ms but batched is 0.01ms per state.

### Training Commands

```bash
# Local test (verify pipeline)
python scripts/test_parallel_local.py

# Benchmark network
python scripts/benchmark_network.py

# View training estimates
python -m src.training.profiler

# Fetch Modern meta decks
python -m src.data.mtggoldfish_decks --format modern --top 20

# Run parallel self-play (when Forge is integrated)
python -m src.training.parallel_selfplay --actors 8 --iterations 100
```

### Forge Integration (NEW)

The Forge daemon is now integrated via `src/forge/forge_client.py`.

**Key Components:**
| File | Purpose |
|------|---------|
| `src/forge/forge_client.py` | TCP client for Forge daemon (port 17171) |
| `src/forge/state_mapper.py` | Maps Forge JSON → neural network tensors |
| `scripts/profile_forge_games.py` | Profiles game latency and throughput |
| `scripts/deploy_forge_test.sh` | Cloud deployment for Forge testing |

**Protocol:**
- Forge daemon listens on TCP port 17171
- Sends `DECISION:` prefixed JSON with full game state
- Game state includes: hand, battlefield, graveyard, exile, library_size, mana_pool
- Client responds with action index (or -1 to pass)

**Start Forge Daemon:**
```bash
# Build (once)
cd forge-repo && mvn package -DskipTests -pl forge-gui-desktop -am

# Start daemon (port 17171)
java -jar forge-gui-desktop/target/forge-gui-desktop-*-jar-with-dependencies.jar daemon

# Test connection
python -c "from src.forge import ForgeClient; c=ForgeClient(); c.connect(); print(c.get_status())"
```

**Profile Forge Games:**
```bash
# Local test (requires daemon running)
python scripts/profile_forge_games.py --games 10 --deck1 decks/competitive/mono_red_aggro.dck --deck2 decks/competitive/boros_aggro.dck

# Cloud deployment (automatic setup)
./scripts/deploy_forge_test.sh --games 100 --duration 25
```

**Collect AI Training Data (Imitation Learning):**
```bash
# Start daemon in Docker
docker run -d --name forge-daemon -p 17171:17171 mtg-daemon:debug

# Collect training data (AI observation mode)
python scripts/collect_ai_training_data.py --games 100 --output training_data

# Output: JSONL with (state, action) pairs from Forge AI
# - decision_type: choose_action, declare_attackers, declare_blockers
# - game_state: life totals, hand/library sizes, battlefield
# - actions: available actions
# - ai_choice: expert decision label
```

### Meta Decks

Current Modern meta decks stored in `data/decks/modern_meta.json`:
- Boros Energy, Ruby Storm, Eldrazi Tron, Jeskai Blink, Affinity
- Use MTGGoldfish scraper to update: `python -m src.data.mtggoldfish_decks`

## Active Training Run (2026-01-24)

**Status**: RUNNING (v2 hybrid encoder)
**Instance**: `i-0fe997b5fa21e192f` / `54.190.192.61` (g4dn.xlarge spot, us-west-2)
**S3 Bucket**: `mtg-rl-checkpoints-20260124190118616600000001`
**Encoder**: v2 hybrid (text embeddings + structural features)

### Monitor Training
```bash
# Check status (finds instance automatically)
./scripts/connect_training.sh status

# TensorBoard (SSM - keyless)
./scripts/connect_training.sh tensorboard
# Then open http://localhost:6006

# View live log
aws ssm start-session --target i-0fe997b5fa21e192f --region us-west-2
# Then: tail -f /home/ubuntu/mtg-rl/training.log
```

### Previous Run (v1)
- Results: 68.02% test accuracy, 94.38% top-3 accuracy
- Model: `s3://mtg-rl-checkpoints-20260124190118616600000001/checkpoints/best.pt`

### Monitoring

```bash
# Check training progress (live logs)
aws s3 cp s3://mtg-rl-checkpoints-20260124190118616600000001/logs/training_live.log - | tail -50

# TensorBoard (forward port first)
ssh -L 6006:localhost:6006 ubuntu@54.244.139.57
# Then open http://localhost:6006

# Check if training complete
aws s3 ls s3://mtg-rl-checkpoints-20260124190118616600000001/training_complete.json

# Download best model
aws s3 cp s3://mtg-rl-checkpoints-20260124190118616600000001/checkpoints/best.pt checkpoints/
```

### EC2 Console Location
**IMPORTANT**: Instances are in **us-west-2** region. Switch regions in AWS Console:
1. Click region dropdown (top-right)
2. Select "US West (Oregon) us-west-2"
3. Go to EC2 > Instances

---

## Agent Spawning Notes

For parallel work, consider spawning agents for:
- **Data processing**: Download and preprocess 17lands data
- **Training**: Run BC training on cloud GPU
- **Testing**: Write and run test suites
- **Forge integration**: Implement daemon communication

Use `Task` tool with appropriate `subagent_type` for each.
