ForgeRL Documentation
=====================

**ForgeRL** is a reinforcement learning system for Magic: The Gathering draft and gameplay, built on top of the Forge MTG engine.

The system uses mechanics-based card encoding (1387 primitives) and AlphaZero-style self-play to train agents that can draft competitively and play games.

Quick Start
-----------

Install dependencies::

    uv sync --extra dev

Run tests::

    uv run python3 -m pytest tests/test_parser.py -v

Train a draft model::

    uv run python3 scripts/training_pipeline.py --mode bc --sets FDN --epochs 5

Architecture Overview
---------------------

Core Components:

* **Mechanics System**: Decomposes MTG cards into 1387 primitive mechanics (not text embeddings)
* **Card Parser**: Converts Oracle text to mechanics sequences
* **Game State Encoder**: Encodes Forge game state as tensors
* **AlphaZero Network**: Policy + value heads with MCTS for gameplay
* **Training Pipeline**: Behavioral cloning on 17lands data + RL self-play
* **Forge Integration**: TCP client for the Forge MTG engine daemon

API Documentation
-----------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/mechanics
   api/forge
   api/models
   api/training
   api/agents
   api/data
   api/environments
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
