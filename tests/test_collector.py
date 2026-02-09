"""Tests for the data collection pipeline.

Focuses on:
1. IncrementalHDF5Writer: resizable datasets, flush, close
2. Decision filtering: only training-relevant types are kept
3. Round-trip: data written == data read back
"""

import json

import h5py
import numpy as np

from scripts.collect_ai_training_data import (
    IncrementalHDF5Writer,
    TRAINING_DECISION_TYPES,
    save_to_hdf5,
)


def _make_decision(decision_type="choose_action", turn=1, num_actions=3):
    """Create a minimal DECISION dict for testing."""
    return {
        "decision_type": decision_type,
        "turn": turn,
        "game_state": {
            "is_game_over": False,
            "players": [
                {"life": 20, "hand_size": 7, "library_size": 53,
                 "battlefield_creatures": 0, "battlefield_lands": 1,
                 "battlefield_other": 0, "mana_pool": {"total": 1}},
                {"life": 20, "hand_size": 7, "library_size": 53,
                 "battlefield_creatures": 0, "battlefield_lands": 1,
                 "battlefield_other": 0, "mana_pool": {"total": 0}},
            ],
        },
        "actions": [{"action": f"action_{i}", "card": f"Card {i}"} for i in range(num_actions)],
        "ai_choice": {"action": "action_0", "card": "Card 0"},
    }


class TestIncrementalHDF5Writer:
    """IncrementalHDF5Writer creates resizable datasets and appends."""

    def test_single_flush(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = IncrementalHDF5Writer(path)
        decisions = [_make_decision(turn=t) for t in range(1, 6)]
        n = writer.flush(decisions)
        writer.close()

        assert n == 5
        assert writer.total_rows == 5

        with h5py.File(path, "r") as f:
            assert f["states"].shape == (5, 17)
            assert f["turns"].shape == (5,)
            assert f["choices"].shape == (5,)
            assert f["num_actions"].shape == (5,)
            assert f["decision_types"].shape == (5,)
            assert f["game_state_json"].shape == (5,)
            assert f.attrs["encoding_version"] == 2

    def test_multiple_flushes_append(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = IncrementalHDF5Writer(path)

        batch1 = [_make_decision(turn=1) for _ in range(3)]
        batch2 = [_make_decision(turn=2) for _ in range(7)]

        writer.flush(batch1)
        assert writer.total_rows == 3

        writer.flush(batch2)
        assert writer.total_rows == 10

        writer.close()

        with h5py.File(path, "r") as f:
            assert f["states"].shape == (10, 17)
            assert f["turns"][0] == 1
            assert f["turns"][3] == 2

    def test_empty_flush_is_noop(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = IncrementalHDF5Writer(path)
        n = writer.flush([])
        assert n == 0
        assert writer.total_rows == 0
        writer.close()

    def test_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = IncrementalHDF5Writer(path)
        writer.flush([_make_decision()])
        writer.set_metadata({"timestamp": "20260208", "num_games": 100, "extra": {"nested": True}})
        writer.close()

        with h5py.File(path, "r") as f:
            assert f.attrs["timestamp"] == "20260208"
            assert f.attrs["num_games"] == 100
            assert json.loads(f.attrs["extra"]) == {"nested": True}

    def test_game_state_json_roundtrip(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = IncrementalHDF5Writer(path)
        d = _make_decision()
        writer.flush([d])
        writer.close()

        with h5py.File(path, "r") as f:
            stored = f["game_state_json"][0]
            parsed = json.loads(stored)
            assert parsed["players"][0]["life"] == 20


class TestDecisionFiltering:
    """Only TRAINING_DECISION_TYPES should be kept."""

    def test_training_types_defined(self):
        assert "choose_action" in TRAINING_DECISION_TYPES
        assert "declare_attackers" in TRAINING_DECISION_TYPES
        assert "declare_blockers" in TRAINING_DECISION_TYPES

    def test_noise_types_excluded(self):
        noise = ["reveal", "choose_entity", "choose_ability",
                 "confirm_action", "play_trigger", "play_from_effect",
                 "announce_value", "choose_cards"]
        for t in noise:
            assert t not in TRAINING_DECISION_TYPES, f"{t} should be filtered"


class TestLegacySaveToHDF5:
    """save_to_hdf5 (batch mode) still works for backward compat."""

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "legacy.h5"
        decisions = [_make_decision(turn=i) for i in range(1, 4)]
        save_to_hdf5(decisions, path, {"timestamp": "test"})

        with h5py.File(path, "r") as f:
            assert f["states"].shape == (3, 17)
            assert f.attrs["encoding_version"] == 2
            assert f.attrs["timestamp"] == "test"

    def test_empty_decisions_skipped(self, tmp_path):
        path = tmp_path / "empty.h5"
        save_to_hdf5([], path, {})
        assert not path.exists()


class TestIncrementalMatchesLegacy:
    """Incremental writer produces identical data to legacy save_to_hdf5."""

    def test_data_matches(self, tmp_path):
        decisions = [_make_decision(turn=t, num_actions=t + 1) for t in range(1, 11)]

        # Legacy
        legacy_path = tmp_path / "legacy.h5"
        save_to_hdf5(decisions, legacy_path, {"timestamp": "x"})

        # Incremental (two flushes)
        inc_path = tmp_path / "incremental.h5"
        writer = IncrementalHDF5Writer(inc_path)
        writer.flush(decisions[:5])
        writer.flush(decisions[5:])
        writer.set_metadata({"timestamp": "x"})
        writer.close()

        with h5py.File(legacy_path, "r") as fl, h5py.File(inc_path, "r") as fi:
            np.testing.assert_array_equal(fl["states"][:], fi["states"][:])
            np.testing.assert_array_equal(fl["turns"][:], fi["turns"][:])
            np.testing.assert_array_equal(fl["choices"][:], fi["choices"][:])
            np.testing.assert_array_equal(fl["num_actions"][:], fi["num_actions"][:])
            np.testing.assert_array_equal(fl["decision_types"][:], fi["decision_types"][:])
            for i in range(10):
                assert fl["game_state_json"][i] == fi["game_state_json"][i]
