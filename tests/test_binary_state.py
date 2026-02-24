"""Tests for binary state contract (dtypes, card ID lookup, HDF5 I/O)."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.forge.binary_state import (
    BINARY_FORMAT_VERSION,
    CARD_DTYPE,
    DECISION_DTYPE,
    DECISION_PLAY_SPELL,
    MAX_ACTIONS,
    MAX_CARDS,
    MAX_PLAYERS,
    PLAYER_DTYPE,
    STATE_ATTACKING,
    STATE_TAPPED,
    TYPE_CREATURE,
    TYPE_LAND,
    BinaryDecisionBuffer,
    CardIdLookup,
    read_binary_hdf5,
    write_binary_hdf5,
)


# =============================================================================
# DTYPE SIZE TESTS
# =============================================================================


class TestDtypeSizes:
    """Verify that dtypes have the exact expected byte sizes."""

    def test_card_dtype_size(self):
        assert CARD_DTYPE.itemsize == 12

    def test_player_dtype_size(self):
        assert PLAYER_DTYPE.itemsize == 15

    def test_decision_dtype_size(self):
        assert DECISION_DTYPE.itemsize == 2278

    def test_decision_breakdown(self):
        """Verify the decision size adds up from components."""
        header = 2 + 1 + 1 + 2 + 1 + 1  # version, num_cards, num_players, turn, phase, active_player = 8
        players = MAX_PLAYERS * PLAYER_DTYPE.itemsize  # 4 * 15 = 60
        cards = MAX_CARDS * CARD_DTYPE.itemsize  # 150 * 12 = 1800
        actions_header = 1 + 1 + 2  # num_actions, decision_type, chosen_action = 4
        actions = MAX_ACTIONS * 2  # 203 * 2 = 406
        expected = header + players + cards + actions_header + actions
        assert expected == 2278
        assert DECISION_DTYPE.itemsize == expected


# =============================================================================
# DTYPE FIELD ACCESS TESTS
# =============================================================================


class TestDtypeFieldAccess:
    """Verify structured array field access works correctly."""

    def test_card_fields(self):
        card = np.zeros(1, dtype=CARD_DTYPE)[0]
        card['card_id'] = 1234
        card['zone'] = 2  # battlefield
        card['type_flags'] = TYPE_CREATURE | TYPE_LAND
        card['power'] = 5
        card['toughness'] = -1
        card['cmc'] = 3
        card['state_flags'] = STATE_TAPPED | STATE_ATTACKING
        card['damage'] = 2
        card['counters'] = 3
        card['attach_to'] = 0
        card['controller'] = 1

        assert card['card_id'] == 1234
        assert card['zone'] == 2
        assert card['type_flags'] & TYPE_CREATURE
        assert card['type_flags'] & TYPE_LAND
        assert card['power'] == 5
        assert card['toughness'] == -1
        assert card['state_flags'] & STATE_TAPPED
        assert card['state_flags'] & STATE_ATTACKING

    def test_player_fields(self):
        player = np.zeros(1, dtype=PLAYER_DTYPE)[0]
        player['life'] = 20
        player['poison'] = 0
        player['mana_w'] = 3
        player['library_size'] = 45
        player['hand_size'] = 7
        player['lands_played'] = 1
        player['energy'] = 4
        player['storm_count'] = 2
        player['status_flags'] = 0b101  # monarch + initiative

        assert player['life'] == 20
        assert player['mana_w'] == 3
        assert player['library_size'] == 45
        assert player['energy'] == 4
        assert player['storm_count'] == 2
        assert player['status_flags'] & 0b001  # monarch
        assert player['status_flags'] & 0b100  # initiative

    def test_player_negative_life(self):
        """Life can go negative (e.g., Phyrexian mana, damage)."""
        player = np.zeros(1, dtype=PLAYER_DTYPE)[0]
        player['life'] = -5
        assert player['life'] == -5

    def test_decision_fields(self):
        dec = np.zeros(1, dtype=DECISION_DTYPE)[0]
        dec['version'] = BINARY_FORMAT_VERSION
        dec['num_cards'] = 10
        dec['num_players'] = 2
        dec['turn'] = 5
        dec['phase'] = 3  # MAIN1
        dec['active_player'] = 0
        dec['num_actions'] = 5
        dec['decision_type'] = DECISION_PLAY_SPELL
        dec['chosen_action'] = 2

        # Set player data
        dec['players'][0]['life'] = 20
        dec['players'][1]['life'] = 15

        # Set card data
        dec['cards'][0]['card_id'] = 100
        dec['cards'][0]['type_flags'] = TYPE_CREATURE
        dec['cards'][0]['power'] = 3
        dec['cards'][0]['toughness'] = 3

        # Set action data
        dec['actions'][0] = 10
        dec['actions'][1] = 20
        dec['actions'][2] = 30

        assert dec['version'] == BINARY_FORMAT_VERSION
        assert dec['num_cards'] == 10
        assert dec['players'][0]['life'] == 20
        assert dec['players'][1]['life'] == 15
        assert dec['cards'][0]['card_id'] == 100
        assert dec['cards'][0]['power'] == 3
        assert dec['actions'][0] == 10
        assert dec['actions'][2] == 30

    def test_decision_array(self):
        """Test creating an array of decisions."""
        decisions = np.zeros(5, dtype=DECISION_DTYPE)
        for i in range(5):
            decisions[i]['turn'] = i + 1
            decisions[i]['num_cards'] = i * 2
            decisions[i]['version'] = BINARY_FORMAT_VERSION

        assert decisions[3]['turn'] == 4
        assert decisions[3]['num_cards'] == 6


# =============================================================================
# CARD ID LOOKUP TESTS
# =============================================================================


class TestCardIdLookup:
    """Test card name <-> uint16 ID mapping."""

    def test_basic_lookup(self):
        lookup = CardIdLookup({"Lightning Bolt": 1, "Counterspell": 2})
        assert lookup.get_id("Lightning Bolt") == 1
        assert lookup.get_id("Counterspell") == 2
        assert lookup.get_id("Unknown Card") == 0  # default for unknown

    def test_reverse_lookup(self):
        lookup = CardIdLookup({"Lightning Bolt": 1, "Counterspell": 2})
        assert lookup.get_name(1) == "Lightning Bolt"
        assert lookup.get_name(2) == "Counterspell"
        assert lookup.get_name(999) is None

    def test_contains(self):
        lookup = CardIdLookup({"Lightning Bolt": 1})
        assert "Lightning Bolt" in lookup
        assert "Counterspell" not in lookup

    def test_len(self):
        lookup = CardIdLookup({"A": 1, "B": 2, "C": 3})
        assert len(lookup) == 3

    def test_save_load_roundtrip(self):
        original = CardIdLookup({"Lightning Bolt": 1, "Counterspell": 2, "Sol Ring": 3})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name

        try:
            original.save(path)
            loaded = CardIdLookup.load(path)
            assert loaded.get_id("Lightning Bolt") == 1
            assert loaded.get_id("Counterspell") == 2
            assert loaded.get_id("Sol Ring") == 3
            assert len(loaded) == 3
        finally:
            os.unlink(path)

    def test_from_scryfall(self):
        """Test building lookup from minimal Scryfall-like data."""
        cards = [
            {"name": "Lightning Bolt", "layout": "normal"},
            {"name": "Counterspell", "layout": "normal"},
            {"name": "Sol Ring", "layout": "normal"},
            {"name": "Lightning Bolt", "layout": "normal"},  # duplicate
            {"name": "Token Card", "layout": "token"},  # skip
            {"name": "Digital Only", "layout": "normal", "digital": True},  # skip
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(cards, f)
            path = f.name

        try:
            lookup = CardIdLookup.from_scryfall(path)
            assert len(lookup) == 3
            # IDs should be sorted alphabetically: Counterspell=1, Lightning Bolt=2, Sol Ring=3
            assert lookup.get_id("Counterspell") == 1
            assert lookup.get_id("Lightning Bolt") == 2
            assert lookup.get_id("Sol Ring") == 3
        finally:
            os.unlink(path)

    def test_from_mechanics_h5(self):
        """Test building lookup from mechanics HDF5."""
        h5_path = "data/card_mechanics_commander.h5"
        if not os.path.exists(h5_path):
            pytest.skip("No mechanics HDF5 available")

        lookup = CardIdLookup.from_mechanics_h5(h5_path)
        assert len(lookup) > 0
        # All IDs should be positive (1-based)
        assert all(v > 0 for v in lookup._name_to_id.values())
        # Spot check a well-known card
        assert lookup.get_id("Lightning Bolt") > 0

    def test_java_binary_roundtrip(self):
        """Test saving and reading the Java binary lookup format."""
        import struct

        lookup = CardIdLookup({"Lightning Bolt": 42, "Sol Ring": 100})

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name

        try:
            lookup.save_java_lookup(path)

            # Read back and verify
            with open(path, 'rb') as f:
                num_entries = struct.unpack('<I', f.read(4))[0]
                assert num_entries == 2

                entries = {}
                for _ in range(num_entries):
                    card_id, name_len = struct.unpack('<HH', f.read(4))
                    name = f.read(name_len).decode('utf-8')
                    entries[name] = card_id

                assert entries["Lightning Bolt"] == 42
                assert entries["Sol Ring"] == 100
        finally:
            os.unlink(path)


# =============================================================================
# HDF5 I/O TESTS
# =============================================================================


class TestHDF5IO:
    """Test HDF5 read/write for binary decisions."""

    def _make_sample_decisions(self, n: int = 10) -> np.ndarray:
        """Create sample decision records for testing."""
        decisions = np.zeros(n, dtype=DECISION_DTYPE)
        for i in range(n):
            decisions[i]['version'] = BINARY_FORMAT_VERSION
            decisions[i]['num_cards'] = min(i + 1, MAX_CARDS)
            decisions[i]['num_players'] = 2
            decisions[i]['turn'] = i + 1
            decisions[i]['phase'] = i % 14
            decisions[i]['active_player'] = i % 2
            decisions[i]['num_actions'] = min(i + 3, MAX_ACTIONS)
            decisions[i]['decision_type'] = DECISION_PLAY_SPELL
            decisions[i]['chosen_action'] = i % 5

            # Players
            decisions[i]['players'][0]['life'] = 20 - i
            decisions[i]['players'][1]['life'] = 20

            # Cards
            for j in range(min(i + 1, MAX_CARDS)):
                decisions[i]['cards'][j]['card_id'] = (i * 10 + j) % 65535
                decisions[i]['cards'][j]['type_flags'] = TYPE_CREATURE
                decisions[i]['cards'][j]['power'] = 2
                decisions[i]['cards'][j]['toughness'] = 2

            # Actions
            for j in range(min(i + 3, MAX_ACTIONS)):
                decisions[i]['actions'][j] = j

        return decisions

    def test_write_read_roundtrip(self):
        """Write and read back, verify exact equality."""
        decisions = self._make_sample_decisions(100)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, decisions)
            loaded = read_binary_hdf5(path)

            assert loaded.dtype == DECISION_DTYPE
            assert len(loaded) == 100
            np.testing.assert_array_equal(loaded, decisions)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_all_fields(self):
        """Verify every field survives serialization."""
        dec = np.zeros(1, dtype=DECISION_DTYPE)
        dec[0]['version'] = BINARY_FORMAT_VERSION
        dec[0]['num_cards'] = 3
        dec[0]['num_players'] = 2
        dec[0]['turn'] = 42
        dec[0]['phase'] = 10
        dec[0]['active_player'] = 1
        dec[0]['players'][0]['life'] = -5  # negative life
        dec[0]['players'][0]['poison'] = 7
        dec[0]['players'][0]['mana_w'] = 3
        dec[0]['players'][0]['mana_u'] = 2
        dec[0]['players'][0]['mana_b'] = 1
        dec[0]['players'][0]['mana_r'] = 4
        dec[0]['players'][0]['mana_g'] = 0
        dec[0]['players'][0]['mana_c'] = 5
        dec[0]['players'][0]['library_size'] = 33
        dec[0]['players'][0]['hand_size'] = 7
        dec[0]['players'][0]['lands_played'] = 1
        dec[0]['players'][0]['energy'] = 6
        dec[0]['players'][0]['storm_count'] = 3
        dec[0]['players'][0]['status_flags'] = 0b011  # monarch + city's blessing
        dec[0]['cards'][0]['card_id'] = 12345
        dec[0]['cards'][0]['zone'] = 2
        dec[0]['cards'][0]['type_flags'] = TYPE_CREATURE | TYPE_LAND
        dec[0]['cards'][0]['power'] = -1
        dec[0]['cards'][0]['toughness'] = 127
        dec[0]['cards'][0]['cmc'] = 15
        dec[0]['cards'][0]['state_flags'] = STATE_TAPPED | STATE_ATTACKING
        dec[0]['cards'][0]['damage'] = 4
        dec[0]['cards'][0]['counters'] = 2
        dec[0]['cards'][0]['attach_to'] = 5
        dec[0]['cards'][0]['controller'] = 1
        dec[0]['num_actions'] = 3
        dec[0]['decision_type'] = 2
        dec[0]['chosen_action'] = 1
        dec[0]['actions'][0] = 100
        dec[0]['actions'][1] = 200
        dec[0]['actions'][2] = 300

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, dec)
            loaded = read_binary_hdf5(path)

            r = loaded[0]
            assert r['turn'] == 42
            assert r['phase'] == 10
            assert r['players'][0]['life'] == -5
            assert r['players'][0]['poison'] == 7
            assert r['players'][0]['mana_c'] == 5
            assert r['players'][0]['energy'] == 6
            assert r['players'][0]['storm_count'] == 3
            assert r['players'][0]['status_flags'] == 0b011
            assert r['cards'][0]['card_id'] == 12345
            assert r['cards'][0]['type_flags'] == (TYPE_CREATURE | TYPE_LAND)
            assert r['cards'][0]['power'] == -1
            assert r['cards'][0]['toughness'] == 127
            assert r['cards'][0]['state_flags'] == (STATE_TAPPED | STATE_ATTACKING)
            assert r['cards'][0]['damage'] == 4
            assert r['cards'][0]['attach_to'] == 5
            assert r['actions'][0] == 100
            assert r['actions'][2] == 300
        finally:
            os.unlink(path)

    def test_compression_reduces_size(self):
        """Gzip compression should significantly reduce file size for sparse data."""
        decisions = self._make_sample_decisions(1000)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            compressed_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            uncompressed_path = f.name

        try:
            write_binary_hdf5(compressed_path, decisions, compression='gzip')
            write_binary_hdf5(uncompressed_path, decisions, compression=None)

            compressed_size = os.path.getsize(compressed_path)
            uncompressed_size = os.path.getsize(uncompressed_path)

            # Compressed should be smaller (lots of zeros in sparse records)
            assert compressed_size < uncompressed_size
            # Raw: 1000 * 2278 = 2.28 MB, compressed should be much less
            assert compressed_size < 1_000_000  # less than 1 MB
        finally:
            os.unlink(compressed_path)
            os.unlink(uncompressed_path)

    def test_metadata_stored(self):
        """Verify metadata attributes are stored in HDF5."""
        decisions = self._make_sample_decisions(5)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, decisions, metadata={
                'game_count': 10,
                'collector_version': '1.0',
            })

            import h5py
            with h5py.File(path, 'r') as f:
                assert f.attrs['format_version'] == BINARY_FORMAT_VERSION
                assert f.attrs['encoding_version'] == 3
                assert f.attrs['num_decisions'] == 5
                assert f.attrs['game_count'] == 10
                assert f.attrs['collector_version'] == '1.0'
        finally:
            os.unlink(path)

    def test_empty_array(self):
        """Handle zero-length decision arrays."""
        decisions = np.zeros(0, dtype=DECISION_DTYPE)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            write_binary_hdf5(path, decisions)
            loaded = read_binary_hdf5(path)
            assert len(loaded) == 0
            assert loaded.dtype == DECISION_DTYPE
        finally:
            os.unlink(path)

    def test_wrong_dtype_rejected(self):
        """Writing non-DECISION_DTYPE arrays should raise."""
        bad_data = np.zeros(10, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Expected dtype"):
                write_binary_hdf5(path, bad_data)
        finally:
            os.unlink(path)

    def test_version_mismatch_rejected(self):
        """Reading non-v3 data should raise."""
        import h5py

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            with h5py.File(path, 'w') as f:
                f.create_dataset('decisions', data=np.zeros(5, dtype=DECISION_DTYPE))
                f.attrs['encoding_version'] = 2  # wrong version

            with pytest.raises(ValueError, match="encoding_version=3"):
                read_binary_hdf5(path)
        finally:
            os.unlink(path)


# =============================================================================
# BINARY DECISION BUFFER TESTS
# =============================================================================


class TestBinaryDecisionBuffer:
    """Test the streaming buffer for data collection."""

    def test_buffer_accumulates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=100)

            rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec['version'] = BINARY_FORMAT_VERSION
            rec['turn'] = 1

            for _ in range(50):
                buf.add(rec)

            assert buf.total_count == 50
            # No checkpoint yet (under save_interval)
            assert buf._checkpoint_count == 0

    def test_auto_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=10)

            rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec['version'] = BINARY_FORMAT_VERSION

            for i in range(25):
                rec['turn'] = i + 1
                buf.add(rec)

            # Should have created 2 checkpoints (at 10 and 20)
            assert buf._checkpoint_count == 2
            assert buf.total_count == 25

    def test_finalize_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=10)

            rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec['version'] = BINARY_FORMAT_VERSION

            for i in range(25):
                rec['turn'] = i + 1
                buf.add(rec)

            final_path = buf.finalize()
            assert os.path.exists(final_path)

            loaded = read_binary_hdf5(final_path)
            assert len(loaded) == 25

            # Verify turn numbers preserved in order
            for i in range(25):
                assert loaded[i]['turn'] == i + 1

    def test_empty_finalize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=10)
            final_path = buf.finalize()
            # No decisions -> no file content but path returned
            assert final_path.endswith("final.h5")

    def test_flat_action_computed_from_actions_array(self):
        """flat_action = actions[chosen_action] (flat 0-202 policy index)."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=100)

            # Record: 3 legal actions with flat indices [0, 3, 18], chose index 1 → flat=3
            rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec['version'] = BINARY_FORMAT_VERSION
            rec['num_actions'] = 3
            rec['chosen_action'] = 1
            rec['actions'][0] = 0    # pass
            rec['actions'][1] = 3    # cast spell (hand slot 0)
            rec['actions'][2] = 18   # activate (bf slot 0)
            buf.add(rec)

            # Record: pass only (actions[0]=0, chose 0 → flat=0)
            rec2 = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec2['version'] = BINARY_FORMAT_VERSION
            rec2['num_actions'] = 1
            rec2['chosen_action'] = 0
            rec2['actions'][0] = 0
            buf.add(rec2)

            # Record: chosen_action out of bounds → flat=-1
            rec3 = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec3['version'] = BINARY_FORMAT_VERSION
            rec3['num_actions'] = 2
            rec3['chosen_action'] = 5  # > num_actions
            rec3['actions'][0] = 0
            rec3['actions'][1] = 3
            buf.add(rec3)

            final_path = buf.finalize()
            assert os.path.exists(final_path)

            with h5py.File(final_path, 'r') as f:
                assert 'flat_actions' in f, "flat_actions dataset missing"
                fa = f['flat_actions'][:]

            assert len(fa) == 3
            assert fa[0] == 3,  f"Expected 3 (cast slot 0), got {fa[0]}"
            assert fa[1] == 0,  f"Expected 0 (pass), got {fa[1]}"
            assert fa[2] == -1, f"Expected -1 (out of bounds), got {fa[2]}"

    def test_flat_action_range_check(self):
        """Values > 202 in actions[] are stored as -1 (Java not yet updated)."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=100)

            rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
            rec['version'] = BINARY_FORMAT_VERSION
            rec['num_actions'] = 2
            rec['chosen_action'] = 0
            rec['actions'][0] = 999  # out-of-range flat index (Java hasn't set it yet)
            rec['actions'][1] = 3
            buf.add(rec)

            final_path = buf.finalize()
            with h5py.File(final_path, 'r') as f:
                fa = f['flat_actions'][:]

            assert fa[0] == -1, f"Expected -1 for out-of-range, got {fa[0]}"

    def test_flat_actions_in_checkpoint(self):
        """flat_actions dataset appears in checkpoint files too."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            buf = BinaryDecisionBuffer(tmpdir, save_interval=3)

            for j in range(6):
                rec = np.zeros(1, dtype=DECISION_DTYPE)[0]
                rec['version'] = BINARY_FORMAT_VERSION
                rec['num_actions'] = 2
                rec['chosen_action'] = j % 2
                rec['actions'][0] = 0
                rec['actions'][1] = 3
                buf.add(rec)

            # Two checkpoints should have been flushed automatically
            assert buf._checkpoint_count == 2

            cp1 = os.path.join(tmpdir, "checkpoint_0001.h5")
            assert os.path.exists(cp1)
            with h5py.File(cp1, 'r') as f:
                assert 'flat_actions' in f
                assert len(f['flat_actions']) == 3


# =============================================================================
# MEMORY EFFICIENCY TEST
# =============================================================================


class TestMemoryEfficiency:
    """Verify the binary format meets memory targets."""

    def test_1000_games_memory_estimate(self):
        """1000 games * ~400 decisions/game * 2278 bytes should be manageable."""
        decisions_per_game = 400
        num_games = 1000
        total_decisions = decisions_per_game * num_games  # 400,000

        raw_bytes = total_decisions * DECISION_DTYPE.itemsize
        raw_mb = raw_bytes / (1024 * 1024)

        # 400K * 2278 = ~868 MB raw, vs 3.54 GB with JSON
        assert raw_mb < 1000, f"Raw size {raw_mb:.1f} MB exceeds 1000 MB target"

        # With gzip compression (>50% for sparse data), should be well under 500 MB
        # This is just a sanity check on the math

    def test_per_decision_comparison(self):
        """Binary: 2278 bytes vs JSON: ~14,500 bytes = 84% reduction."""
        json_avg = 14500
        binary = DECISION_DTYPE.itemsize
        reduction = 1 - (binary / json_avg)
        assert reduction > 0.80, f"Reduction {reduction:.1%} < 80% target"
