# CLAUDE.md - MTG RL Project Context

## Project Overview

**ForgeRL** - A reinforcement learning system for Magic: The Gathering draft and gameplay, built on top of Forge (Java MTG engine).

**Goal**: Train an AI that can draft competitively and play games using human gameplay data (17lands.com) and self-play reinforcement learning.

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
| `shared_card_encoder.py` | Simple card encoder (for draft) | 1.2M |
| `entity_encoder.py` | Full game state encoder (for gameplay) | 9.2M |
| `draft_policy.py` | Draft-specific policy network | 2.8M |
| `text_embeddings.py` | LLM-based card text embeddings | External |
| `training_pipeline.py` | Unified BC + RL training | - |

### Training Pipeline

```
Phase 1: Behavioral Cloning (17lands data)
├── Download: python scripts/download_17lands.py --sets FDN DSK BLB
├── Train: python training_pipeline.py --mode bc --epochs 10
└── Output: checkpoints/bc_best.pt

Phase 2: RL Fine-tuning (Forge self-play)
├── Start Forge daemon: java -jar forge.jar --daemon
├── Train: python training_pipeline.py --mode rl --episodes 10000
└── Output: checkpoints/rl_best.pt

Phase 3: Evaluation
├── Eval: python training_pipeline.py --mode eval
└── Compare against baselines
```

### Forge Integration

The Forge MTG engine is in `forge-repo/` (gitignored, separate repo).

```bash
# Clone Forge (if not present)
git clone https://github.com/Card-Forge/forge.git forge-repo

# Build Forge
cd forge-repo && mvn package -DskipTests

# Run draft daemon
java -jar forge-gui-desktop/target/forge.jar --daemon --port 17220
```

**Key Forge Files**:
- `forge-game/src/main/java/forge/game/card/Card.java` - Card representation
- `forge-game/src/main/java/forge/game/Game.java` - Game state
- `forge-game/src/main/java/forge/trackable/TrackableProperty.java` - State serialization

---

## Key Design Decisions

1. **Shared Card Encoder**: Pre-train on draft, transfer to gameplay
2. **Entity-Based Architecture**: Like AlphaStar, each card is a full entity
3. **Text Embeddings**: Use sentence-transformers for ability semantics
4. **Multi-Layer Rate Limiting**: WAF + API Gateway + Application
5. **Forge as Simulator**: Official MTG rules engine, handles all mechanics

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
cd infrastructure && terraform apply
./scripts/deploy.sh
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

## Testing Requirements

### Before Committing

1. **Run existing tests**: `python -m pytest tests/ -v`
2. **Test modified modules**: Run self-tests for changed files
3. **Verify imports**: `python -c "from training_pipeline import *"`

### Test Coverage Goals

- [ ] Unit tests for all encoder components
- [ ] Integration tests for training pipeline
- [ ] End-to-end draft simulation tests
- [ ] API endpoint tests (when deployed)

---

## File Structure

```
mtg/
├── CLAUDE.md                 # This file - project context
├── ARCHITECTURE.md           # Detailed architecture docs
├── AWS_DEPLOYMENT.md         # AWS deployment guide
├── CLOUD_TRAINING.md         # Cloud GPU recommendations
├── WHITEPAPER.md             # Technical paper
├── shared_card_encoder.py    # Simple card encoder
├── entity_encoder.py         # Full game state encoder
├── draft_policy.py           # Draft policy network
├── text_embeddings.py        # Text embeddings
├── training_pipeline.py      # Training orchestration
├── data_17lands.py           # 17lands data loading
├── scripts/
│   ├── download_17lands.py   # Data downloader
│   └── deploy.sh             # AWS deployment
├── infrastructure/
│   └── main.tf               # Terraform config
├── tests/                    # Test suite
├── checkpoints/              # Model checkpoints (gitignored)
├── data/                     # Training data (gitignored)
└── forge-repo/               # Forge MTG engine (gitignored)
```

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

## Known Issues / TODOs

- [ ] EntityEncoder dimension mismatch with training pipeline (use SharedCardEncoder for now)
- [ ] NaN losses with synthetic data (normal - use real 17lands data)
- [ ] Forge daemon not integrated yet (simulated drafts work)
- [ ] Need to download actual 17lands data before real training

---

## Agent Spawning Notes

For parallel work, consider spawning agents for:
- **Data processing**: Download and preprocess 17lands data
- **Training**: Run BC training on cloud GPU
- **Testing**: Write and run test suites
- **Forge integration**: Implement daemon communication

Use `Task` tool with appropriate `subagent_type` for each.
