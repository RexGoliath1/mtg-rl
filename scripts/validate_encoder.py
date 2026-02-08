#!/usr/bin/env python3
"""
Validate the ForgeGameStateEncoder (v2) end-to-end.

Creates a mock Forge game_state JSON with realistic board state, runs it
through ForgeGameStateEncoder.encode_json(), checks output shape, prints
parameter count, and verifies save/load round-trip integrity.

Usage:
    uv run python3 scripts/validate_encoder.py
"""

import sys
import tempfile
from pathlib import Path

import torch

from src.forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig


# ---------------------------------------------------------------------------
# Mock game state -- a mid-game board with two players
# ---------------------------------------------------------------------------
MOCK_GAME_STATE = {
    "turn": 5,
    "phase": "main1",
    "activePlayer": 0,
    "priorityPlayer": 0,
    "players": [
        {
            "id": 0,
            "name": "Self",
            "life": 20,
            "poison": 0,
            "library_size": 50,
            "hand_size": 3,
            "lands_played_this_turn": 1,
            "max_land_plays": 1,
            "has_lost": False,
            "mana": {"W": 2, "U": 0, "B": 0, "R": 1, "G": 1, "C": 3},
            "hand": [
                {"name": "Lightning Bolt", "id": 1, "is_instant": True, "cmc": 1},
                {"name": "Counterspell", "id": 2, "is_instant": True, "cmc": 2},
                {"name": "Sol Ring", "id": 3, "is_artifact": True, "cmc": 1},
            ],
            "battlefield": [
                {
                    "name": "Grizzly Bears", "id": 10, "tapped": False,
                    "is_creature": True, "power": 2, "toughness": 2, "cmc": 2,
                    "summoningSickness": False,
                    "counters": [{"type": "p1p1", "count": 1}],
                },
                {"name": "Forest", "id": 11, "tapped": True, "is_land": True},
                {"name": "Mountain", "id": 12, "tapped": False, "is_land": True},
            ],
            "graveyard": [
                {"name": "Giant Growth", "id": 20, "is_instant": True, "cmc": 1},
            ],
            "exile": [],
            "command": [],
        },
        {
            "id": 1,
            "name": "Opponent",
            "life": 18,
            "poison": 2,
            "library_size": 45,
            "hand_size": 2,
            "lands_played_this_turn": 0,
            "max_land_plays": 1,
            "has_lost": False,
            "mana": {"W": 0, "U": 2, "B": 0, "R": 0, "G": 0, "C": 2},
            "hand": [
                {"name": "Unknown Card", "id": 100},
                {"name": "Unknown Card", "id": 101},
            ],
            "battlefield": [
                {
                    "name": "Snapcaster Mage", "id": 110, "tapped": False,
                    "is_creature": True, "power": 2, "toughness": 1, "cmc": 2,
                },
                {"name": "Island", "id": 111, "tapped": True, "is_land": True},
                {"name": "Island", "id": 112, "tapped": True, "is_land": True},
            ],
            "graveyard": [],
            "exile": [],
            "command": [],
        },
    ],
    "stack": [],
    "legalActions": ["cast:1:Lightning Bolt", "pass"],
}


def validate():
    """Run all validation checks. Returns True on success."""
    ok = True
    config = GameStateConfig()
    encoder = ForgeGameStateEncoder(config)
    encoder.eval()

    # ---- 1. Parameter count ------------------------------------------------
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[params]  total={total_params:,}  trainable={trainable:,}")

    # ---- 2. Encode empty-stack state ----------------------------------------
    with torch.no_grad():
        emb = encoder.encode_json(MOCK_GAME_STATE)

    expected_shape = (1, config.output_dim)
    if emb.shape != expected_shape:
        print(f"[FAIL]  shape {emb.shape} != expected {expected_shape}")
        ok = False
    else:
        print(f"[shape]  {emb.shape}  (output_dim={config.output_dim})")

    if torch.isnan(emb).any():
        print("[FAIL]  NaN in output embedding")
        ok = False
    else:
        print(f"[stats]  mean={emb.mean().item():.4f}  std={emb.std().item():.4f}")

    # ---- 3. Encode with non-empty stack -------------------------------------
    state_with_stack = {**MOCK_GAME_STATE, "stack": [
        {"name": "Lightning Bolt", "id": 1, "targets": [110]},
    ]}
    with torch.no_grad():
        emb_stack = encoder.encode_json(state_with_stack)

    diff = (emb - emb_stack).abs().mean().item()
    print(f"[stack]  mean abs diff from empty stack: {diff:.4f}")
    if diff == 0.0:
        print("[WARN]  stack made no difference -- may indicate a bug")

    # ---- 4. Save / load round-trip ------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "encoder.pt")
        encoder.save(path)

        loaded = ForgeGameStateEncoder.load(path)
        loaded.eval()

        with torch.no_grad():
            emb_orig = encoder.encode_json(MOCK_GAME_STATE)
            emb_loaded = loaded.encode_json(MOCK_GAME_STATE)

        max_diff = (emb_orig - emb_loaded).abs().max().item()
        if max_diff > 1e-5:
            print(f"[FAIL]  save/load max diff = {max_diff:.6f}  (expected ~0)")
            ok = False
        else:
            print(f"[save/load]  max diff = {max_diff:.6f}  OK")

    # ---- Summary ------------------------------------------------------------
    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
