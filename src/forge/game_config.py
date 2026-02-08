"""
Format Configuration for Multi-Format MTG Support

Centralizes format-specific parameters (life totals, deck sizes, player counts)
so the training and evaluation pipeline can handle Commander, Modern, and Standard
without hardcoded assumptions scattered across the codebase.

Format-Dependent Limits and Compatibility Notes:
- ActionConfig.max_battlefield=50 is sufficient for all formats. Commander's 4-player
  games rarely exceed 40 total permanents across all players; the encoder aggregates
  all players' cards into a single zone tensor, so 50 is a comfortable ceiling.
- ActionConfig.max_hand_size=15 holds for all formats (max 7 starting + draw effects).
- GameStateConfig.life_bits=8 encodes up to 255 life, covering Commander's 40 and any
  reasonable lifegain scenarios.
- GameStateConfig.max_players=4 already accommodates Commander; Modern/Standard use
  only 2 of the 4 player slots (remaining slots stay zeroed).
- The Forge daemon handles format rules (starting life, deck legality) server-side.
  FormatConfig is used client-side for validation, default deck selection, and to
  configure the SimulatedForgeClient for offline testing.

Usage:
    from src.forge.game_config import FORMATS, FormatConfig

    fmt = FORMATS["commander"]
    print(fmt.starting_life)   # 40
    print(fmt.deck_size)       # 100
    print(fmt.max_players)     # 4
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FormatConfig:
    """Configuration for a specific MTG format.

    Attributes:
        name: Format identifier ("commander", "modern", "standard").
        starting_life: Starting life total for each player.
        deck_size: Minimum deck size (100 for Commander, 60 for others).
        max_hand_size: Maximum hand size before discard (7 for all formats).
        allow_commander: Whether the command zone is used.
        max_players: Maximum number of players in a game.
        default_deck_dir: Subdirectory under decks/ containing format-appropriate decks.
        singleton: Whether the format enforces singleton (1 copy per card, except basics).
    """
    name: str
    starting_life: int
    deck_size: int
    max_hand_size: int = 7
    allow_commander: bool = False
    max_players: int = 2
    default_deck_dir: str = ""
    singleton: bool = False


# ============================================================================
# FORMAT REGISTRY
# ============================================================================

FORMATS: dict[str, FormatConfig] = {
    "commander": FormatConfig(
        name="commander",
        starting_life=40,
        deck_size=100,
        max_hand_size=7,
        allow_commander=True,
        max_players=4,
        default_deck_dir="decks/",
        singleton=True,
    ),
    "modern": FormatConfig(
        name="modern",
        starting_life=20,
        deck_size=60,
        max_hand_size=7,
        allow_commander=False,
        max_players=2,
        default_deck_dir="decks/modern",
        singleton=False,
    ),
    "standard": FormatConfig(
        name="standard",
        starting_life=20,
        deck_size=60,
        max_hand_size=7,
        allow_commander=False,
        max_players=2,
        default_deck_dir="decks/competitive",
        singleton=False,
    ),
}

# Convenience aliases
COMMANDER = FORMATS["commander"]
MODERN = FORMATS["modern"]
STANDARD = FORMATS["standard"]

# All supported format names (for argparse choices, validation, etc.)
FORMAT_NAMES: list[str] = list(FORMATS.keys())


def get_format(name: str) -> FormatConfig:
    """Look up a format by name (case-insensitive).

    Args:
        name: Format name ("commander", "modern", "standard").

    Returns:
        FormatConfig for the requested format.

    Raises:
        ValueError: If the format name is not recognized.
    """
    key = name.lower().strip()
    if key not in FORMATS:
        raise ValueError(
            f"Unknown format '{name}'. Supported formats: {FORMAT_NAMES}"
        )
    return FORMATS[key]


def validate_deck_size(num_cards: int, fmt: FormatConfig) -> Optional[str]:
    """Check whether a deck meets the format's size requirement.

    Args:
        num_cards: Number of cards in the deck.
        fmt: Format configuration.

    Returns:
        None if valid, or an error message string.
    """
    if num_cards < fmt.deck_size:
        return (
            f"Deck has {num_cards} cards but {fmt.name} requires "
            f"at least {fmt.deck_size}"
        )
    return None
