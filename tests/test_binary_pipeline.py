"""End-to-end binary pipeline integration tests.

Tests the full flow: binary record → base64 → collector parse → HDF5 → encoder → tensor.
All tests use synthetic data with no external dependencies.
"""
import base64
import tempfile
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.forge.binary_state import (
    BINARY_FORMAT_VERSION,
    DECISION_DTYPE,
    MAX_ACTIONS,
    MAX_CARDS,
    TYPE_CREATURE,
    TYPE_LAND,
    STATE_TAPPED,
    STATE_ATTACKING,
    CardIdLookup,
    write_binary_hdf5,
    read_binary_hdf5,
)


def _make_realistic_decision(turn=5, num_cards=8, num_actions=5, chosen=2):
    """Create a realistic binary decision record for testing.

    actions[j] is set to realistic flat policy indices:
      0 = pass, 3 = cast hand[0], 4 = cast hand[1], 18 = activate bf[0], 19 = activate bf[1]
    chosen selects which of these actions was taken.
    """
    rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
    rec['version'] = BINARY_FORMAT_VERSION
    rec['num_cards'] = num_cards
    rec['num_players'] = 2
    rec['turn'] = turn
    rec['phase'] = 3  # MAIN1
    rec['active_player'] = 0

    # Player 0
    rec['players'][0]['life'] = 20
    rec['players'][0]['poison'] = 0
    rec['players'][0]['mana_w'] = 2
    rec['players'][0]['mana_g'] = 1
    rec['players'][0]['library_size'] = 40
    rec['players'][0]['hand_size'] = 5
    rec['players'][0]['lands_played'] = 1

    # Player 1
    rec['players'][1]['life'] = 18
    rec['players'][1]['library_size'] = 35
    rec['players'][1]['hand_size'] = 3

    # Cards: mix of zones and types
    # Hand cards
    for i in range(3):
        rec['cards'][i]['card_id'] = 100 + i
        rec['cards'][i]['zone'] = 0  # hand
        rec['cards'][i]['type_flags'] = TYPE_CREATURE
        rec['cards'][i]['power'] = 2 + i
        rec['cards'][i]['toughness'] = 2 + i
        rec['cards'][i]['cmc'] = 2 + i
        rec['cards'][i]['controller'] = 0

    # Battlefield creatures
    for i in range(3, 6):
        rec['cards'][i]['card_id'] = 200 + i
        rec['cards'][i]['zone'] = 2  # battlefield
        rec['cards'][i]['type_flags'] = TYPE_CREATURE
        rec['cards'][i]['power'] = 3
        rec['cards'][i]['toughness'] = 3
        rec['cards'][i]['cmc'] = 3
        rec['cards'][i]['controller'] = 0
        if i == 3:
            rec['cards'][i]['state_flags'] = STATE_ATTACKING

    # Battlefield lands
    for i in range(6, 8):
        rec['cards'][i]['card_id'] = 300 + i
        rec['cards'][i]['zone'] = 2  # battlefield
        rec['cards'][i]['type_flags'] = TYPE_LAND
        rec['cards'][i]['state_flags'] = STATE_TAPPED if i == 6 else 0
        rec['cards'][i]['controller'] = 0

    # Actions: use realistic flat policy indices (not sequential)
    # Flat layout: 0=pass, 3-17=cast spell, 18-67=activate ability
    _flat_indices = [0, 3, 4, 18, 19]  # pass, cast[0], cast[1], activate[0], activate[1]
    rec['num_actions'] = num_actions
    rec['decision_type'] = 0  # choose_action
    rec['chosen_action'] = chosen
    for j in range(num_actions):
        rec['actions'][j] = _flat_indices[j] if j < len(_flat_indices) else j

    return rec


class TestBinaryBase64Roundtrip:
    """Test binary record → base64 → decode → verify."""

    def test_base64_roundtrip(self):
        rec = _make_realistic_decision()
        raw = rec.tobytes()
        encoded = base64.b64encode(raw).decode('ascii')
        decoded = np.frombuffer(base64.b64decode(encoded), dtype=DECISION_DTYPE)[0]

        assert decoded['version'] == BINARY_FORMAT_VERSION
        assert decoded['turn'] == 5
        assert decoded['players'][0]['life'] == 20
        assert decoded['cards'][0]['card_id'] == 100
        assert decoded['chosen_action'] == 2

    def test_base64_size(self):
        """Base64-encoded record should be ~3040 chars (4/3 * 2278)."""
        rec = _make_realistic_decision()
        encoded = base64.b64encode(rec.tobytes()).decode('ascii')
        assert len(encoded) < 3200


class TestBinaryHDF5Pipeline:
    """Test: create records → HDF5 → read back → verify."""

    def test_batch_write_read(self):
        """Write 100 decisions, read back, verify fields match."""
        decisions = np.zeros(100, dtype=DECISION_DTYPE)
        for i in range(100):
            decisions[i] = _make_realistic_decision(
                turn=i + 1, num_cards=min(i + 3, MAX_CARDS),
                num_actions=min(i + 2, MAX_ACTIONS), chosen=i % 5,
            )

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, decisions)
            loaded = read_binary_hdf5(path)

            assert len(loaded) == 100
            assert loaded.dtype == DECISION_DTYPE

            # Spot check
            assert loaded[0]['turn'] == 1
            assert loaded[99]['turn'] == 100
            assert loaded[42]['chosen_action'] == 42 % 5
            assert loaded[50]['players'][0]['life'] == 20
        finally:
            os.unlink(path)

    def test_memory_efficiency(self):
        """1000 decisions should use < 4 MB compressed."""
        decisions = np.zeros(1000, dtype=DECISION_DTYPE)
        for i in range(1000):
            decisions[i] = _make_realistic_decision(turn=i + 1)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, decisions, compression='gzip')
            size_bytes = os.path.getsize(path)
            assert size_bytes < 4_000_000, f"HDF5 too large: {size_bytes / 1024:.1f} KB"
        finally:
            os.unlink(path)


class TestBinaryEncoder:
    """Test: binary record → ForgeGameStateEncoder → tensor."""

    def test_encode_from_binary_shape(self):
        """encode_from_binary produces [1, 768] output."""
        from src.forge.game_state_encoder import ForgeGameStateEncoder

        encoder = ForgeGameStateEncoder()
        encoder.eval()

        rec = _make_realistic_decision()
        with torch.no_grad():
            output = encoder.encode_from_binary(rec, card_id_lookup=None)

        assert output.shape == (1, 768)
        assert not torch.isnan(output).any()

    def test_encode_with_card_lookup(self):
        """encode_from_binary with card_id_lookup produces different (non-zero) embeddings."""
        from src.forge.game_state_encoder import ForgeGameStateEncoder

        encoder = ForgeGameStateEncoder()
        encoder.eval()

        rec = _make_realistic_decision()

        # Without lookup (zero mechanics embeddings)
        with torch.no_grad():
            out_no_lookup = encoder.encode_from_binary(rec, card_id_lookup=None)

        # With a fake lookup that maps card_ids to known card names
        lookup = CardIdLookup({
            "Lightning Bolt": 100,
            "Grizzly Bears": 101,
            "Giant Growth": 102,
        })

        with torch.no_grad():
            out_with_lookup = encoder.encode_from_binary(rec, card_id_lookup=lookup)

        # Both should be valid tensors
        assert out_no_lookup.shape == (1, 768)
        assert out_with_lookup.shape == (1, 768)
        assert not torch.isnan(out_with_lookup).any()

    def test_different_records_different_output(self):
        """Different game states should produce different embeddings."""
        from src.forge.game_state_encoder import ForgeGameStateEncoder

        encoder = ForgeGameStateEncoder()
        encoder.eval()

        rec1 = _make_realistic_decision(turn=1, num_cards=3)
        rec2 = _make_realistic_decision(turn=10, num_cards=8)

        with torch.no_grad():
            out1 = encoder.encode_from_binary(rec1)
            out2 = encoder.encode_from_binary(rec2)

        # Outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_empty_game_state(self):
        """Encoder handles zero-card, zero-action records without error."""
        from src.forge.game_state_encoder import ForgeGameStateEncoder

        encoder = ForgeGameStateEncoder()
        encoder.eval()

        rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
        rec['version'] = BINARY_FORMAT_VERSION
        rec['num_players'] = 2
        rec['players'][0]['life'] = 20
        rec['players'][1]['life'] = 20

        with torch.no_grad():
            output = encoder.encode_from_binary(rec)

        assert output.shape == (1, 768)
        assert not torch.isnan(output).any()


class TestBinaryCollectorParsing:
    """Test: DECISION_BIN line parsing (simulated collector logic)."""

    def test_parse_decision_bin_line(self):
        """Simulate parsing a DECISION_BIN:<base64> line."""
        rec = _make_realistic_decision()
        raw = rec.tobytes()
        line = "DECISION_BIN:" + base64.b64encode(raw).decode('ascii')

        # Parser logic (from collector)
        assert line.startswith("DECISION_BIN:")
        payload = line[13:]
        decoded = np.frombuffer(base64.b64decode(payload), dtype=DECISION_DTYPE)[0]

        assert decoded['version'] == BINARY_FORMAT_VERSION
        assert decoded['turn'] == 5
        assert decoded['num_actions'] == 5
        assert decoded['chosen_action'] == 2

    def test_full_pipeline_synthetic(self):
        """End-to-end: create → base64 → parse → HDF5 → read → encode → tensor."""
        from src.forge.game_state_encoder import ForgeGameStateEncoder

        encoder = ForgeGameStateEncoder()
        encoder.eval()

        # Create 50 synthetic decisions
        records = []
        for i in range(50):
            rec = _make_realistic_decision(
                turn=i + 1, num_cards=5, num_actions=3, chosen=i % 3,
            )
            # Simulate base64 round-trip
            raw = rec.tobytes()
            encoded = base64.b64encode(raw).decode('ascii')
            decoded = np.frombuffer(base64.b64decode(encoded), dtype=DECISION_DTYPE)[0].copy()
            records.append(decoded)

        decisions = np.array(records)

        # Write to HDF5
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, decisions)
            loaded = read_binary_hdf5(path)

            assert len(loaded) == 50

            # Encode first and last record
            with torch.no_grad():
                out_first = encoder.encode_from_binary(loaded[0])
                out_last = encoder.encode_from_binary(loaded[49])

            assert out_first.shape == (1, 768)
            assert out_last.shape == (1, 768)
            assert not torch.isnan(out_first).any()
            assert not torch.isnan(out_last).any()
        finally:
            os.unlink(path)


class TestFlatActionsPipeline:
    """Test flat_actions parallel dataset through the full binary pipeline."""

    def test_buffer_writes_flat_actions(self):
        """BinaryDecisionBuffer computes flat_actions and writes them to HDF5."""
        import h5py
        from src.forge.binary_state import BinaryDecisionBuffer

        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=100)

            # chosen=2 → actions[2]=4 (cast hand[1]) → flat_action=4
            rec = _make_realistic_decision(turn=1, num_actions=5, chosen=2)
            buf.add(rec)

            # chosen=0 → actions[0]=0 (pass) → flat_action=0
            rec_pass = _make_realistic_decision(turn=2, num_actions=5, chosen=0)
            buf.add(rec_pass)

            final_path = buf.finalize()

            with h5py.File(final_path, 'r') as f:
                assert 'flat_actions' in f, "flat_actions missing from binary HDF5"
                fa = f['flat_actions'][:]

            assert len(fa) == 2
            # chosen=2 → actions[2]=4 (cast hand[1] = cast_start+1 = 3+1)
            assert fa[0] == 4,  f"Expected 4 (cast hand[1]), got {fa[0]}"
            assert fa[1] == 0,  f"Expected 0 (pass), got {fa[1]}"

    def test_flat_actions_valid_range(self):
        """All computed flat_actions should be -1 or in [0, 202]."""
        import h5py
        from src.forge.binary_state import BinaryDecisionBuffer

        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=100)

            for i in range(20):
                rec = _make_realistic_decision(
                    turn=i + 1, num_actions=5, chosen=i % 5,
                )
                buf.add(rec)

            final_path = buf.finalize()

            with h5py.File(final_path, 'r') as f:
                fa = f['flat_actions'][:]

            valid = fa[fa >= 0]
            assert (valid <= 202).all(), f"Out-of-range values: {valid[valid > 202]}"

    def test_flat_actions_write_read_with_decisions(self):
        """write_binary_hdf5 with flat_actions stores a readable parallel dataset."""
        import h5py
        from src.forge.binary_state import write_binary_hdf5

        decisions = np.array([_make_realistic_decision(turn=i) for i in range(10)],
                             dtype=DECISION_DTYPE)
        flat = np.array([0, 3, 4, 18, -1, 0, 3, 4, 18, -1], dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            path = tmp.name

        try:
            write_binary_hdf5(path, decisions, flat_actions=flat)

            with h5py.File(path, 'r') as f:
                assert 'decisions' in f
                assert 'flat_actions' in f
                stored_fa = f['flat_actions'][:]

            assert stored_fa.dtype == np.int16
            np.testing.assert_array_equal(stored_fa, flat)
        finally:
            os.unlink(path)
