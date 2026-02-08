# ForgeRL

Reinforcement learning system for Magic: The Gathering draft and gameplay, built on [Forge](https://github.com/Card-Forge/forge) (open-source Java MTG engine).

## What it does

- **Drafts cards** using behavioral cloning on human gameplay data from [17lands.com](https://www.17lands.com)
- **Plays games** via AlphaZero-style self-play with MCTS against the Forge engine
- **Encodes cards** as mechanics-based feature vectors (not text embeddings), so the model learns card interactions through play
- **Recommends cards** for Commander decks via an EDHREC-inspired web tool

## Quick start

```bash
# Install (requires Python 3.11+)
uv sync --extra dev

# Run tests
uv run python3 -m pytest tests/test_parser.py -v

# Launch card recommender
uv run python3 scripts/card_recommender.py    # http://localhost:8000

# Train draft model (behavioral cloning)
uv run python3 scripts/training_pipeline.py --mode bc --sets FDN --epochs 5
```

## How card encoding works

Instead of treating cards as opaque IDs or text embeddings, we decompose each card's oracle text into ~1,400 mechanics primitives:

```
Lightning Bolt  ->  [INSTANT_SPEED, DEAL_DAMAGE, ANY_TARGET, DAMAGE_3]
Saw in Half     ->  [INSTANT_SPEED, TARGET_CREATURE, DESTROY, CREATE_TOKEN_COPY, HALF_STATS]
```

The neural network then learns which mechanic *combinations* are powerful through self-play, not pre-coded heuristics. This transfers across formats (Draft, Standard, Commander) and handles new cards immediately.

## Project structure

```
src/              Python package (mechanics, models, training, forge integration)
scripts/          Entry-point scripts (training, tools, deployment)
tests/            Test suite (460+ parser tests)
infrastructure/   Terraform, Docker, deployment configs
decks/            Deck files for Forge simulation
data/             Training data, HDF5 embeddings (gitignored)
```

## Training pipeline

1. **Behavioral Cloning**: Learn from 17lands human draft data
2. **Self-Play RL**: Fine-tune via AlphaZero (policy + value networks with MCTS) against Forge
3. **Evaluation**: Compare against baselines

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical design and [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment.

## Tools

| Tool | URL | Description |
|------|-----|-------------|
| Card Recommender | `localhost:8000` | EDHREC-style deck recommendations using mechanics similarity |
| Embedding Quiz | `localhost:8787` | Review and grade card encoding quality |
| Parser Coverage | CLI | Measure how well the parser handles oracle text |

## License

MIT License. See [NOTICE](NOTICE) for third-party attributions.

This project uses [Forge](https://github.com/Card-Forge/forge) (GPL-3.0)
as a separate daemon process, card data from [Scryfall](https://scryfall.com),
and game data from [17lands](https://www.17lands.com) (CC BY 4.0).

ForgeRL is unofficial Fan Content permitted under the [Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy).
Not approved/endorsed by Wizards. Portions of the materials used are
property of Wizards of the Coast. &copy; Wizards of the Coast LLC.
