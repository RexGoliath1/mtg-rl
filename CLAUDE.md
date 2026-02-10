# CLAUDE.md - MTG RL Project Context

## Project Overview

**ForgeRL** - A reinforcement learning system for Magic: The Gathering draft and gameplay, built on top of Forge (Java MTG engine).

**Goal**: Train an AI that can draft competitively and play games using human gameplay data (17lands.com) and self-play reinforcement learning.

---

## Development Rules

### Git Workflow (Non-Negotiable)

**CRITICAL**: NEVER commit directly to main. All changes go through PRs.

1. **Create a GitHub Issue first** — every PR references `Closes #N`
2. **Create a feature branch** — `feat/description`, `fix/description`, `refactor/description`
3. **Commit on the branch** — commit early and often
4. **Push + open PR** — `gh pr create` with summary and test plan
5. **CI must pass** — `lint-and-test`, `docker-build`, `docker-smoke-test`
6. **User reviews and merges** — Claude codes, user approves

```bash
# Correct workflow:
gh issue create --title "Add feature X" --body "Description"
git checkout -b feat/feature-x
# ... make changes, commit ...
git push -u origin feat/feature-x
gh pr create --title "feat: Add feature X" --body "Closes #N ..."

# WRONG - never do this:
git commit -m "feat: something" && git push origin main  # NO!
```

**Branch protection**: `enforce_admins=true` — even repo owner cannot bypass CI.
**Merge via CLI**: `gh pr merge N --repo RexGoliath1/mtg-rl --merge` when UI is stale.
**Self-review on PR creation**: Review the diff as part of PR creation (free, no API cost).

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
| `src/mechanics/vocabulary.py` | Mechanics primitives (VOCAB_SIZE=1403) | - |
| `src/mechanics/card_parser.py` | Oracle text -> mechanics | - |
| `src/mechanics/precompute_embeddings.py` | Pre-compute HDF5 embeddings | - |
| `src/forge/game_state_encoder.py` | Forge JSON -> 768-dim tensor | 33.1M |
| `src/forge/policy_value_heads.py` | AlphaZero flat policy (203 actions) + value | 0.7M |
| `src/forge/strategic_core.py` | GRU for turn-level game trajectory | ~500K |
| `src/forge/turn_planner.py` | MLP for phase-level tactical planning | ~400K |
| `src/forge/hierarchical_network.py` | HierarchicalAlphaZeroNetwork wrapper | ~900K |
| `src/forge/ctde.py` | CTDE dual value heads (oracle + observable) | ~1.1M |
| `src/forge/autoregressive_head.py` | AlphaStar-style structured action head | ~600K |
| `src/forge/action_mapper.py` | Flat 203 <-> structured action mapping | - |
| `src/forge/opponent_model.py` | Belief model for MCTS determinization | ~2.2M |
| `src/forge/binary_state.py` | Fixed-width 1060-byte binary format | - |
| `src/forge/mcts.py` | Monte Carlo Tree Search | - |
| `src/training/self_play.py` | Self-play training loop | - |
| `src/models/shared_card_encoder.py` | Simple card encoder (for draft) | 1.2M |
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

The Forge MTG engine is in `forge-repo/` as a **git submodule** tracking `feature/rl-daemon-mode`.

**Repository**: `https://github.com/RexGoliath1/forge.git` (fork of Card-Forge/forge)

**Base image**: `ghcr.io/rexgoliath1/forge-daemon-base:<sha>` — built by Forge CI, consumed by `Dockerfile.daemon`.

```bash
# Initialize submodule (first clone only)
git submodule update --init

# Update to latest Forge
cd forge-repo && git pull origin feature/rl-daemon-mode && cd ..
git add forge-repo
git commit -m "chore: bump forge submodule to <new-sha>"

# Build daemon image using submodule SHA
FORGE_SHA=$(git ls-tree HEAD forge-repo | awk '{print $3}')
docker build --build-arg FORGE_SHA=$FORGE_SHA -f infrastructure/docker/Dockerfile.daemon .
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
6. **Binary State Format (v3)**: Fixed-width 2278-byte records for data collection (84% smaller than JSON)

### Binary Data Format

The binary format (`src/forge/binary_state.py`) defines fixed-width numpy structured dtypes:

| Dtype | Size | Fields |
|-------|------|--------|
| CARD_DTYPE | 12 bytes | card_id(u16), zone(u8), type_flags(u8), power(i8), toughness(i8), cmc(u8), state_flags(u8), damage(u8), counters(u8), attach_to(u8), controller(u8) |
| PLAYER_DTYPE | 15 bytes | life(i16), poison(u8), mana(6×u8), library_size(u8), hand_size(u8), lands_played(u8), energy(u8), storm_count(u8), status_flags(u8) |
| DECISION_DTYPE | 2278 bytes | header(8) + 4 players(60) + 150 cards(1800) + actions header(4) + 203 actions(406) |

**Wire protocol**: `DECISION_BIN:<base64(2278 bytes)>` per decision over existing TCP text protocol.
**Card ID mapping**: `CardIdLookup` maps card names → uint16 IDs (sorted, stable, 0=unknown).
**Encoding**: `ForgeGameStateEncoder.encode_from_binary()` maps binary records to same 768-dim tensors as JSON path.

```bash
# Generate card ID lookup from Scryfall data
uv run python3 -c "from src.forge.binary_state import CardIdLookup; l = CardIdLookup.from_scryfall('data/scryfall_bulk_cards.json'); l.save('data/card_id_lookup.json')"
```

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
├── .gitmodules               # Submodule config (forge-repo)
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
├── tests/                    # Test suite (494 parser + 116 HRL/binary + 15 HRL integration + encoder/pipeline tests)
├── research/                 # Experimental files
├── infrastructure/           # Terraform, Docker, deployment configs
│   ├── docker/              # Dockerfiles (sim, daemon, collection, training)
│   ├── docker-compose.yml   # All services (daemon, training, collection, tensorboard)
│   └── *.tf                 # Terraform (S3, ECR, IAM, EC2)
├── decks/                    # Deck files for training
├── docs/                     # MTG rules reference
├── data/                     # Training data, HDF5 embeddings, sidecar metadata (gitignored)
├── checkpoints/              # Model checkpoints (gitignored)
├── logs/                     # TensorBoard logs (gitignored)
└── forge-repo/               # Forge MTG engine (git submodule)
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

## Autonomous Session Protocol

### Permissions (Non-Negotiable)
- **CAN**: Write code, create branches, push branches, create PRs, create issues, deploy to AWS
- **CANNOT**: Merge PRs (user must review and approve all merges)
- **CANNOT**: Delete branches that have open PRs
- **CANNOT**: Force push to any branch

### Working Around Unmerged PRs
When you need functionality from a PR that hasn't been merged yet:
1. Create your new feature branch from the unmerged PR's branch (not from main)
2. Or cherry-pick the needed commits into your working branch
3. Document the dependency chain in your PR description
4. The user will merge in order when they review

### CI Monitoring
- **Check PR CI status every hour** during autonomous sessions
- If a check fails: diagnose, fix on the branch, push the fix
- If the fix requires changes to a different PR's branch: create a new PR instead
- Use `gh pr checks N --repo RexGoliath1/mtg-rl` to check status

### Daily Plan File
- The day's task plan lives in `~/.claude/projects/.../memory/daily_plan.md`
- This file survives context compaction and is the source of truth for the session
- Mark tasks as completed as you go
- Add new tasks discovered during work

### Daily Plan Lifecycle (Critical — Check FIRST on Boot)
Before executing any plan, check the date at the top of `daily_plan.md`:
- **FUTURE** (plan date > today): Do NOT execute. Ask the user if they want to start early or just chat.
- **CURRENT** (plan date = today): Execute normally. Morning = planning window, then work.
- **STALE** (plan date < today): Do NOT execute old tasks. Instead:
  1. Run `git log --since="<plan_date>"` and `gh pr list` to see what was actually completed
  2. Cross-reference plan tasks against commits/PRs
  3. Report findings to the user: "N of M tasks appear complete. These remain: ..."
  4. Offer to delete the stale plan and replan for today

### Context Compaction Recovery
When context is compressed mid-session:
1. Read `CLAUDE.md` (always loaded)
2. Read `memory/MEMORY.md` (always loaded, first 200 lines)
3. Read `memory/daily_plan.md` for today's task list and progress
4. Check `git status` and `git log` for what's been done
5. Resume from where you left off

---

## Development Priorities

**IMPORTANT**: Prefer training-side edits over Forge edits when possible.
- Training code (Python) is faster to iterate and test
- Forge changes require rebuilding Docker image (~2 min)
- Only modify Forge for data collection improvements

---

## Known Issues / TODOs

- [x] Checkpoint pruning in CheckpointManager (keep last 3 + best)
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
- [x] Imitation learning data collection (1000 games, 410K decisions, 3.1 GiB v2 HDF5)
- [x] HRL Phase 1: Strategic Core GRU + Turn Planner
- [x] HRL Phase 2a: CTDE dual value heads with oracle dropout
- [x] HRL Phase 2b: Auto-regressive action head (AlphaStar-style)
- [x] HRL Phase 2c: Opponent belief model for MCTS
- [x] Binary state contract (Python side, 2278 bytes/decision with Commander expansion)
- [x] Binary pipeline: encode_from_binary, collector parsing, BinaryDataset v3
- [x] Wire HRL modules into training pipeline (behind config flags)
- [ ] Binary state writer (Java/Forge side) — needed for binary data pipeline
- [ ] First training run on collected data (verify flat model trains)

---

## Resuming Work

When starting a new Claude session:

1. **Read this file** (always loaded automatically)
2. **Read `memory/daily_plan.md`** if it exists — contains today's task list and progress
3. **Check git status**: `git status && git log --oneline -10`
4. **Check open PRs**: `gh pr list --repo RexGoliath1/mtg-rl`
5. **Check GitHub issues**: `gh issue list --repo RexGoliath1/mtg-rl`
6. **Run tests**: `uv run python3 -m pytest tests/ -v`
