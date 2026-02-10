"""
Binary State Contract for Forge <-> Python data exchange.

Defines fixed-width numpy structured dtypes that serve as:
1. The wire format (Forge Java -> Python collector)
2. The storage format (HDF5 compound datasets)
3. The training input format (zero-copy reads into DataLoader)

A single decision record is exactly 1060 bytes, compared to ~14.5 KB
for the variable-length JSON format (v2). This eliminates OOM issues
during data collection.

Usage:
    from src.forge.binary_state import (
        DECISION_DTYPE, CardIdLookup,
        write_binary_hdf5, read_binary_hdf5
    )

    # Build card ID lookup from Scryfall data
    lookup = CardIdLookup.from_scryfall("data/scryfall_bulk_cards.json")
    lookup.save("data/card_id_lookup.json")

    # Read/write HDF5
    decisions = np.zeros(1000, dtype=DECISION_DTYPE)
    write_binary_hdf5("output.h5", decisions)
    loaded = read_binary_hdf5("output.h5")
"""

import json
import os
from typing import Optional

import numpy as np

# Try to import h5py
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# =============================================================================
# BINARY DTYPE DEFINITIONS
# =============================================================================

# 12 bytes per card
CARD_DTYPE = np.dtype([
    ('card_id', '<u2'),       # uint16: Scryfall-derived ID (0=empty slot)
    ('zone', '<u1'),          # uint8: Zone enum (0=hand, 1=library, 2=battlefield, ...)
    ('type_flags', '<u1'),    # uint8: bit flags (creature|land|artifact|enchantment|pw|instant|sorcery|token)
    ('power', '<i1'),         # int8: current power (-128 to 127, -1 for non-creatures)
    ('toughness', '<i1'),     # int8: current toughness (-128 to 127, -1 for non-creatures)
    ('cmc', '<u1'),           # uint8: converted mana cost (0-255)
    ('state_flags', '<u1'),   # uint8: bit flags (tapped|summoning_sick|attacking|blocking|face_down|flipped|transformed|monstrous)
    ('damage', '<u1'),        # uint8: damage marked (0-255)
    ('counters', '<u1'),      # uint8: total counter count (0-255, most common case)
    ('attach_to', '<u1'),     # uint8: index of card this is attached to (0=none, 1-based)
    ('controller', '<u1'),    # uint8: player index who controls this (0-3)
])

# 15 bytes per player
PLAYER_DTYPE = np.dtype([
    ('life', '<i2'),          # int16: life total (-32768 to 32767)
    ('poison', '<u1'),        # uint8: poison counters (0-255)
    ('mana_w', '<u1'),        # uint8: white mana in pool
    ('mana_u', '<u1'),        # uint8: blue mana in pool
    ('mana_b', '<u1'),        # uint8: black mana in pool
    ('mana_r', '<u1'),        # uint8: red mana in pool
    ('mana_g', '<u1'),        # uint8: green mana in pool
    ('mana_c', '<u1'),        # uint8: colorless mana in pool
    ('library_size', '<u1'),  # uint8: cards remaining in library (0-255)
    ('hand_size', '<u1'),     # uint8: cards in hand (0-255)
    ('lands_played', '<u1'),  # uint8: lands played this turn
    ('energy', '<u1'),        # uint8: energy counters (Kaladesh block)
    ('storm_count', '<u1'),   # uint8: storm count this turn
    ('status_flags', '<u1'),  # uint8: bit 0=monarch, bit 1=city's blessing, bit 2=initiative
])

# Maximum counts
MAX_CARDS = 150    # Max cards tracked per decision (Commander: 100 card decks + tokens)
MAX_PLAYERS = 4    # Max players (Commander support)
MAX_ACTIONS = 203  # Max legal actions per decision (matches ActionConfig.total_actions)

# 2278 bytes per decision
DECISION_DTYPE = np.dtype([
    ('version', '<u2'),              # uint16: format version (3 = binary)
    ('num_cards', '<u1'),            # uint8: actual card count in this record
    ('num_players', '<u1'),          # uint8: actual player count
    ('turn', '<u2'),                 # uint16: turn number
    ('phase', '<u1'),                # uint8: Phase enum value
    ('active_player', '<u1'),        # uint8: whose turn it is (0-3)
    ('players', PLAYER_DTYPE, (MAX_PLAYERS,)),   # 4 * 12 = 48 bytes
    ('cards', CARD_DTYPE, (MAX_CARDS,)),          # 50 * 12 = 600 bytes
    ('num_actions', '<u1'),          # uint8: number of legal actions
    ('decision_type', '<u1'),        # uint8: decision type enum
    ('chosen_action', '<u2'),        # uint16: index of chosen action
    ('actions', '<u2', (MAX_ACTIONS,)),  # 200 * 2 = 400 bytes
])

# Verify sizes
assert CARD_DTYPE.itemsize == 12, f"CARD_DTYPE is {CARD_DTYPE.itemsize} bytes, expected 12"
assert PLAYER_DTYPE.itemsize == 15, f"PLAYER_DTYPE is {PLAYER_DTYPE.itemsize} bytes, expected 15"
assert DECISION_DTYPE.itemsize == 2278, f"DECISION_DTYPE is {DECISION_DTYPE.itemsize} bytes, expected 2278"

# Format version for binary records
BINARY_FORMAT_VERSION = 3


# =============================================================================
# TYPE FLAG BITS
# =============================================================================

TYPE_CREATURE     = 0b00000001
TYPE_LAND         = 0b00000010
TYPE_ARTIFACT     = 0b00000100
TYPE_ENCHANTMENT  = 0b00001000
TYPE_PLANESWALKER = 0b00010000
TYPE_INSTANT      = 0b00100000
TYPE_SORCERY      = 0b01000000
TYPE_TOKEN        = 0b10000000


# =============================================================================
# STATE FLAG BITS
# =============================================================================

STATE_TAPPED          = 0b00000001
STATE_SUMMONING_SICK  = 0b00000010
STATE_ATTACKING       = 0b00000100
STATE_BLOCKING        = 0b00001000
STATE_FACE_DOWN       = 0b00010000
STATE_FLIPPED         = 0b00100000
STATE_TRANSFORMED     = 0b01000000
STATE_MONSTROUS       = 0b10000000


# =============================================================================
# DECISION TYPE ENUM
# =============================================================================

DECISION_PLAY_SPELL     = 0
DECISION_ATTACKERS      = 1
DECISION_BLOCKERS       = 2
DECISION_TARGET         = 3
DECISION_MODE           = 4
DECISION_COST           = 5
DECISION_MULLIGAN       = 6
DECISION_OTHER          = 7


# =============================================================================
# CARD ID LOOKUP
# =============================================================================

class CardIdLookup:
    """
    Maps card names to uint16 IDs for the binary format.

    IDs are stable (sorted by name) so the same card always gets the same ID.
    ID 0 is reserved for empty/unknown cards.
    """

    def __init__(self, name_to_id: dict[str, int]):
        self._name_to_id = name_to_id
        self._id_to_name = {v: k for k, v in name_to_id.items()}

    def get_id(self, card_name: str) -> int:
        """Get card ID by name. Returns 0 for unknown cards."""
        return self._name_to_id.get(card_name, 0)

    def get_name(self, card_id: int) -> Optional[str]:
        """Get card name by ID. Returns None for unknown IDs."""
        return self._id_to_name.get(card_id)

    def __len__(self) -> int:
        return len(self._name_to_id)

    def __contains__(self, card_name: str) -> bool:
        return card_name in self._name_to_id

    @classmethod
    def from_scryfall(cls, bulk_json_path: str) -> 'CardIdLookup':
        """
        Build card ID lookup from Scryfall bulk data.

        Cards are sorted by name and assigned sequential uint16 IDs (1-based).
        Only unique card names are included (first occurrence wins).
        """
        with open(bulk_json_path, 'r') as f:
            cards = json.load(f)

        # Collect unique card names
        seen = set()
        names = []
        for card in cards:
            name = card.get('name', '')
            if not name or name in seen:
                continue
            # Skip tokens, art cards, and other non-game objects
            layout = card.get('layout', '')
            if layout in ('token', 'double_faced_token', 'art_series', 'emblem'):
                continue
            # Skip digital-only cards
            if card.get('digital', False):
                continue
            seen.add(name)
            names.append(name)

        # Sort for stable IDs
        names.sort()

        if len(names) > 65534:
            raise ValueError(
                f"Too many unique cards ({len(names)}) for uint16 IDs. "
                f"Max supported: 65534 (0 is reserved)."
            )

        # Assign IDs (1-based, 0 = empty/unknown)
        name_to_id = {name: i + 1 for i, name in enumerate(names)}

        return cls(name_to_id)

    @classmethod
    def from_mechanics_h5(cls, h5_path: str) -> 'CardIdLookup':
        """
        Build card ID lookup from existing mechanics HDF5 file.

        Uses the card_index attribute which maps card names to row indices.
        Assigns new uint16 IDs based on sorted card names.
        """
        if not HAS_H5PY:
            raise ImportError("h5py required for from_mechanics_h5")

        with h5py.File(h5_path, 'r') as f:
            card_index = json.loads(f.attrs['card_index'])

        names = sorted(card_index.keys())
        name_to_id = {name: i + 1 for i, name in enumerate(names)}
        return cls(name_to_id)

    def save(self, path: str) -> None:
        """Save lookup to JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._name_to_id, f, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> 'CardIdLookup':
        """Load lookup from JSON file."""
        with open(path, 'r') as f:
            name_to_id = json.load(f)
        # JSON keys are strings, values should be ints
        name_to_id = {k: int(v) for k, v in name_to_id.items()}
        return cls(name_to_id)

    def to_java_source(self) -> str:
        """
        Generate Java HashMap initialization code for CardIdLookup.java.

        Returns a string containing the Java source for the static initializer block.
        """
        lines = []
        lines.append("// Auto-generated by src/forge/binary_state.py")
        lines.append(f"// {len(self._name_to_id)} cards")
        lines.append("private static final HashMap<String, Integer> CARD_IDS = new HashMap<>();")
        lines.append("static {")
        for name, card_id in sorted(self._name_to_id.items()):
            escaped_name = name.replace('"', '\\"')
            lines.append(f'    CARD_IDS.put("{escaped_name}", {card_id});')
        lines.append("}")
        return '\n'.join(lines)

    def save_java_lookup(self, path: str) -> None:
        """Save as a binary packed file for Java consumption.

        Format: [uint32 num_entries] + [uint16 id, uint16 name_len, utf8 name]*
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        import struct
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', len(self._name_to_id)))
            for name, card_id in sorted(self._name_to_id.items()):
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<HH', card_id, len(name_bytes)))
                f.write(name_bytes)


# =============================================================================
# HDF5 I/O
# =============================================================================

def write_binary_hdf5(
    path: str,
    decisions: np.ndarray,
    metadata: Optional[dict] = None,
    chunk_size: int = 1000,
    compression: str = 'gzip',
    compression_opts: int = 4,
) -> None:
    """
    Write binary decision records to HDF5.

    Args:
        path: Output HDF5 file path
        decisions: Structured numpy array with DECISION_DTYPE
        metadata: Optional metadata dict to store as HDF5 attributes
        chunk_size: HDF5 chunk size for compression
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_opts: Compression level (1-9 for gzip)
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for write_binary_hdf5")

    if decisions.dtype != DECISION_DTYPE:
        raise ValueError(
            f"Expected dtype {DECISION_DTYPE}, got {decisions.dtype}"
        )

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    with h5py.File(path, 'w') as f:
        # Store as compound dataset
        if len(decisions) == 0:
            # h5py can't chunk empty datasets; store without chunking/compression
            f.create_dataset('decisions', data=decisions)
        else:
            chunks = min(chunk_size, len(decisions))
            f.create_dataset(
                'decisions',
                data=decisions,
                chunks=(chunks,),
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None,
            )

        # Metadata
        f.attrs['format_version'] = BINARY_FORMAT_VERSION
        f.attrs['encoding_version'] = 3  # v3 = binary
        f.attrs['num_decisions'] = len(decisions)
        f.attrs['record_size_bytes'] = DECISION_DTYPE.itemsize

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, dict):
                    f.attrs[key] = json.dumps(value)


def read_binary_hdf5(path: str) -> np.ndarray:
    """
    Read binary decision records from HDF5.

    Returns:
        Structured numpy array with DECISION_DTYPE (zero-copy when possible)
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for read_binary_hdf5")

    with h5py.File(path, 'r') as f:
        # Verify format version
        version = f.attrs.get('encoding_version', 0)
        if version != 3:
            raise ValueError(
                f"Expected encoding_version=3 (binary), got {version}. "
                f"Use the appropriate reader for v{version} data."
            )

        decisions = f['decisions'][:]

    return decisions


class BinaryDecisionBuffer:
    """
    Streaming buffer for collecting binary decisions during data collection.

    Accumulates decisions in memory and periodically flushes to HDF5.
    Used by the collector to avoid holding all decisions in memory.
    """

    def __init__(self, output_dir: str, save_interval: int = 500):
        self._output_dir = output_dir
        self._save_interval = save_interval
        self._buffer = []
        self._total_written = 0
        self._checkpoint_count = 0

    def add(self, record: np.void) -> None:
        """Add a single decision record to the buffer.

        Makes a deep copy since numpy void scalars share memory with their
        parent array.
        """
        self._buffer.append(np.array([record], dtype=DECISION_DTYPE)[0].copy())
        if len(self._buffer) >= self._save_interval:
            self.flush_checkpoint()

    def flush_checkpoint(self) -> Optional[str]:
        """Write buffered decisions to a checkpoint HDF5 file."""
        if not self._buffer:
            return None

        decisions = np.array(self._buffer, dtype=DECISION_DTYPE)
        self._checkpoint_count += 1
        path = os.path.join(
            self._output_dir,
            f"checkpoint_{self._checkpoint_count:04d}.h5"
        )
        write_binary_hdf5(path, decisions, metadata={
            'checkpoint_number': self._checkpoint_count,
            'decisions_in_file': len(decisions),
            'total_decisions_so_far': self._total_written + len(decisions),
        })
        self._total_written += len(decisions)
        self._buffer.clear()
        return path

    def finalize(self) -> str:
        """
        Flush remaining buffer and merge all checkpoints into final.h5.

        Returns path to the final HDF5 file.
        """
        # Flush any remaining
        self.flush_checkpoint()

        # Merge all checkpoint files
        final_path = os.path.join(self._output_dir, "final.h5")
        all_decisions = []

        for i in range(1, self._checkpoint_count + 1):
            cp_path = os.path.join(
                self._output_dir,
                f"checkpoint_{i:04d}.h5"
            )
            if os.path.exists(cp_path):
                all_decisions.append(read_binary_hdf5(cp_path))

        if all_decisions:
            merged = np.concatenate(all_decisions)
            write_binary_hdf5(final_path, merged, metadata={
                'total_decisions': len(merged),
                'num_checkpoints_merged': len(all_decisions),
            })

        return final_path

    @property
    def total_count(self) -> int:
        """Total decisions written + buffered."""
        return self._total_written + len(self._buffer)
