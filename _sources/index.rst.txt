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

.. toctree::
   :maxdepth: 2
   :caption: Project

   README
   WHITEPAPER
   ARCHITECTURE
   tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

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
