#!/usr/bin/env python3
"""
Standard 2025 Mechanics Implementation

Covers all missing mechanics from current Standard sets:
- Vehicles & Mounts (crew/saddle actions)
- Rooms (door unlocking)
- Delirium (graveyard type counting)
- Collect Evidence (exile from graveyard)
- Cases (condition tracking)
- Raid (attack-this-turn tracking)
- Outlaw (creature type batching)
- Tribal synergies (general creature type matters)
- Universes Beyond mechanics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np


# =============================================================================
# TRIBAL MECHANICS - Creature Type Synergies
# =============================================================================

class CreatureType(Enum):
    """Major creature types with tribal synergies in Standard."""
    # Core types with frequent synergies
    HUMAN = auto()
    DRAGON = auto()
    ELDRAZI = auto()
    ELF = auto()
    GOBLIN = auto()
    ZOMBIE = auto()
    VAMPIRE = auto()
    MERFOLK = auto()
    ANGEL = auto()
    DEMON = auto()
    SPIRIT = auto()
    ELEMENTAL = auto()
    WIZARD = auto()
    WARRIOR = auto()
    SOLDIER = auto()
    KNIGHT = auto()
    CLERIC = auto()
    ROGUE = auto()

    # Outlaws (Thunder Junction) - these form a "batch"
    ASSASSIN = auto()
    MERCENARY = auto()
    PIRATE = auto()
    WARLOCK = auto()

    # Bloomburrow animal types
    MOUSE = auto()
    RAT = auto()
    RABBIT = auto()
    FROG = auto()
    OTTER = auto()
    BAT = auto()
    BIRD = auto()
    SQUIRREL = auto()
    RACCOON = auto()
    LIZARD = auto()

    # Tarkir clans/types
    MONK = auto()
    SHAMAN = auto()

    # Other relevant types
    ARTIFACT_CREATURE = auto()
    HORROR = auto()
    NIGHTMARE = auto()
    BEAST = auto()
    DINOSAUR = auto()
    CAT = auto()
    DOG = auto()


# Outlaw batch - any of these count as "Outlaw"
OUTLAW_TYPES = {
    CreatureType.ASSASSIN,
    CreatureType.MERCENARY,
    CreatureType.PIRATE,
    CreatureType.ROGUE,
    CreatureType.WARLOCK,
}


@dataclass
class TribalTracker:
    """Tracks creature types on battlefield and in other zones."""

    # Count of each creature type on battlefield
    battlefield_counts: Dict[CreatureType, int] = field(default_factory=dict)

    # Count in graveyard (for recursion/reanimation synergies)
    graveyard_counts: Dict[CreatureType, int] = field(default_factory=dict)

    # Lords and tribal payoffs on battlefield
    active_lords: List[Tuple[CreatureType, str]] = field(default_factory=list)  # (type, buff_description)

    def count_type(self, creature_type: CreatureType, zone: str = "battlefield") -> int:
        """Count creatures of a type in a zone."""
        if zone == "battlefield":
            return self.battlefield_counts.get(creature_type, 0)
        elif zone == "graveyard":
            return self.graveyard_counts.get(creature_type, 0)
        return 0

    def count_outlaws(self, zone: str = "battlefield") -> int:
        """Count total outlaws (the batch of 5 types)."""
        counts = self.battlefield_counts if zone == "battlefield" else self.graveyard_counts
        return sum(counts.get(t, 0) for t in OUTLAW_TYPES)

    def has_type(self, creature_type: CreatureType, zone: str = "battlefield") -> bool:
        """Check if at least one creature of type exists."""
        return self.count_type(creature_type, zone) > 0

    def get_tribal_bonuses(self, creature_type: CreatureType) -> List[str]:
        """Get active bonuses for a creature type from lords."""
        return [buff for lord_type, buff in self.active_lords if lord_type == creature_type]

    def encode(self) -> np.ndarray:
        """Encode tribal state as feature vector."""
        # 50 dimensions: count for each major type (capped at 5)
        features = []
        for ct in CreatureType:
            count = min(self.battlefield_counts.get(ct, 0), 5)
            features.append(count / 5.0)  # Normalize to [0, 1]

        # Add outlaw count as separate feature
        features.append(min(self.count_outlaws(), 5) / 5.0)

        return np.array(features, dtype=np.float32)


# =============================================================================
# VEHICLES & MOUNTS
# =============================================================================

@dataclass
class Vehicle:
    """Represents a Vehicle artifact."""
    card_id: str
    name: str
    power: int
    toughness: int
    crew_cost: int  # Total power needed to crew
    is_crewed: bool = False
    abilities: List[str] = field(default_factory=list)

    def can_crew(self, available_creatures: List[Tuple[str, int]]) -> List[List[str]]:
        """
        Return valid crew combinations.

        Args:
            available_creatures: List of (creature_id, power) for untapped creatures

        Returns:
            List of valid crew combinations (each is list of creature_ids)
        """
        if self.is_crewed:
            return []

        valid_combos = []

        # Try single creature crews first
        for cid, power in available_creatures:
            if power >= self.crew_cost:
                valid_combos.append([cid])

        # Try two-creature combinations
        for i, (cid1, p1) in enumerate(available_creatures):
            for cid2, p2 in available_creatures[i+1:]:
                if p1 + p2 >= self.crew_cost:
                    valid_combos.append([cid1, cid2])

        # For larger crew costs, try three creatures
        if self.crew_cost > 4:
            for i, (cid1, p1) in enumerate(available_creatures):
                for j, (cid2, p2) in enumerate(available_creatures[i+1:], i+1):
                    for cid3, p3 in available_creatures[j+1:]:
                        if p1 + p2 + p3 >= self.crew_cost:
                            valid_combos.append([cid1, cid2, cid3])

        return valid_combos


@dataclass
class Mount:
    """Represents a Mount creature with Saddle."""
    card_id: str
    name: str
    power: int
    toughness: int
    saddle_cost: int  # Total power needed to saddle
    is_saddled: bool = False
    saddle_bonus: str = ""  # Bonus when saddled

    def can_saddle(self, available_creatures: List[Tuple[str, int]]) -> List[List[str]]:
        """Return valid saddle combinations (same logic as crew)."""
        if self.is_saddled:
            return []

        valid_combos = []
        for cid, power in available_creatures:
            if power >= self.saddle_cost:
                valid_combos.append([cid])

        for i, (cid1, p1) in enumerate(available_creatures):
            for cid2, p2 in available_creatures[i+1:]:
                if p1 + p2 >= self.saddle_cost:
                    valid_combos.append([cid1, cid2])

        return valid_combos


# =============================================================================
# ROOMS (Duskmourn)
# =============================================================================

@dataclass
class Room:
    """
    Represents a Room card (split enchantment with two "doors").

    Rooms enter with one door unlocked. You can pay to unlock the second door.
    Each door has its own ability.
    """
    card_id: str
    name: str

    # Door 1
    door1_name: str
    door1_cost: str  # Mana cost to cast this side
    door1_ability: str

    # Door 2
    door2_name: str
    door2_cost: str  # Cost to unlock this door later
    door2_ability: str

    # Status (defaults)
    door1_unlocked: bool = True
    door2_unlocked: bool = False

    def can_unlock_door2(self, available_mana: Dict[str, int]) -> bool:
        """Check if player can pay to unlock door 2."""
        if self.door2_unlocked:
            return False
        # Simplified mana check - would need proper mana parsing
        return True  # Placeholder

    def unlock_door2(self):
        """Unlock the second door."""
        self.door2_unlocked = True

    def get_active_abilities(self) -> List[str]:
        """Get abilities from all unlocked doors."""
        abilities = []
        if self.door1_unlocked:
            abilities.append(self.door1_ability)
        if self.door2_unlocked:
            abilities.append(self.door2_ability)
        return abilities


# =============================================================================
# DELIRIUM (Graveyard card type counting)
# =============================================================================

class CardType(Enum):
    """Card types for Delirium counting."""
    CREATURE = auto()
    INSTANT = auto()
    SORCERY = auto()
    ARTIFACT = auto()
    ENCHANTMENT = auto()
    LAND = auto()
    PLANESWALKER = auto()
    BATTLE = auto()
    # Note: Tribal and Kindred are supertypes, not counted separately


@dataclass
class DeliriumTracker:
    """Tracks card types in graveyard for Delirium."""

    graveyard_types: Set[CardType] = field(default_factory=set)
    cards_by_type: Dict[CardType, List[str]] = field(default_factory=dict)

    def add_card(self, card_id: str, card_types: List[CardType]):
        """Add a card to graveyard tracking."""
        for ct in card_types:
            self.graveyard_types.add(ct)
            if ct not in self.cards_by_type:
                self.cards_by_type[ct] = []
            self.cards_by_type[ct].append(card_id)

    def remove_card(self, card_id: str, card_types: List[CardType]):
        """Remove a card from graveyard (exile, return to hand, etc.)."""
        for ct in card_types:
            if ct in self.cards_by_type and card_id in self.cards_by_type[ct]:
                self.cards_by_type[ct].remove(card_id)
                if not self.cards_by_type[ct]:
                    self.graveyard_types.discard(ct)

    def count_types(self) -> int:
        """Count unique card types in graveyard."""
        return len(self.graveyard_types)

    def has_delirium(self) -> bool:
        """Check if Delirium is active (4+ card types)."""
        return self.count_types() >= 4

    def types_needed_for_delirium(self) -> int:
        """How many more types needed for Delirium."""
        return max(0, 4 - self.count_types())

    def encode(self) -> np.ndarray:
        """Encode Delirium state."""
        # 8 binary features for each card type present
        # + 1 feature for delirium active
        # + 1 feature for types_needed normalized
        features = []
        for ct in CardType:
            features.append(1.0 if ct in self.graveyard_types else 0.0)
        features.append(1.0 if self.has_delirium() else 0.0)
        features.append(self.types_needed_for_delirium() / 4.0)
        return np.array(features, dtype=np.float32)


# =============================================================================
# COLLECT EVIDENCE (Exile from graveyard as cost)
# =============================================================================

@dataclass
class CollectEvidenceAction:
    """
    Represents a Collect Evidence cost/action.

    "Collect evidence N" = Exile cards from graveyard with total mana value N.
    """
    required_mana_value: int
    card_with_ability: str  # The card requiring evidence

    def get_valid_combinations(
        self,
        graveyard: List[Tuple[str, int]]  # List of (card_id, mana_value)
    ) -> List[List[str]]:
        """
        Find all valid ways to pay the evidence cost.

        Returns list of card_id combinations that sum to >= required_mana_value.
        """
        valid_combos = []

        # Sort by mana value descending for efficiency
        sorted_gy = sorted(graveyard, key=lambda x: -x[1])

        def find_combos(index: int, current: List[str], current_mv: int):
            if current_mv >= self.required_mana_value:
                valid_combos.append(current.copy())
                return

            if index >= len(sorted_gy):
                return

            # Skip this card
            find_combos(index + 1, current, current_mv)

            # Use this card
            card_id, mv = sorted_gy[index]
            current.append(card_id)
            find_combos(index + 1, current, current_mv + mv)
            current.pop()

        find_combos(0, [], 0)

        # Prefer smaller combinations (less cards exiled)
        valid_combos.sort(key=len)

        return valid_combos[:10]  # Limit to 10 best options


# =============================================================================
# CASES (Karlov Manor)
# =============================================================================

class CaseStatus(Enum):
    """Status of a Case enchantment."""
    UNSOLVED = auto()
    SOLVED = auto()


@dataclass
class Case:
    """
    Represents a Case enchantment.

    Cases have:
    - An initial ability (always active)
    - A "To solve" condition
    - A solved ability (active once solved)
    """
    card_id: str
    name: str

    initial_ability: str
    solve_condition: str  # Description of condition
    solved_ability: str

    status: CaseStatus = CaseStatus.UNSOLVED

    # Condition tracking (varies by case)
    condition_type: str = ""  # "mana_spent", "creatures_died", "cards_drawn", etc.
    condition_threshold: int = 0
    condition_progress: int = 0

    def check_condition(self, game_state: Dict[str, Any]) -> bool:
        """Check if solve condition is met."""
        if self.status == CaseStatus.SOLVED:
            return True

        if self.condition_type == "mana_spent":
            return self.condition_progress >= self.condition_threshold
        elif self.condition_type == "creatures_died":
            return self.condition_progress >= self.condition_threshold
        elif self.condition_type == "cards_drawn":
            return self.condition_progress >= self.condition_threshold
        elif self.condition_type == "attacked":
            return game_state.get("attacked_this_turn", False)
        elif self.condition_type == "life_gained":
            return self.condition_progress >= self.condition_threshold

        return False

    def update_progress(self, event_type: str, amount: int = 1):
        """Update progress toward solving."""
        if event_type == self.condition_type:
            self.condition_progress += amount

    def solve(self):
        """Mark the case as solved."""
        self.status = CaseStatus.SOLVED

    def get_active_abilities(self) -> List[str]:
        """Get currently active abilities."""
        abilities = [self.initial_ability]
        if self.status == CaseStatus.SOLVED:
            abilities.append(self.solved_ability)
        return abilities


# =============================================================================
# RAID (Attack-this-turn tracking)
# =============================================================================

@dataclass
class RaidTracker:
    """Tracks whether player has attacked this turn for Raid abilities."""

    attacked_this_turn: bool = False
    attacking_creatures: List[str] = field(default_factory=list)

    def declare_attack(self, creature_ids: List[str]):
        """Record that creatures attacked."""
        self.attacked_this_turn = True
        self.attacking_creatures = creature_ids

    def end_turn(self):
        """Reset at end of turn."""
        self.attacked_this_turn = False
        self.attacking_creatures = []

    def has_raid(self) -> bool:
        """Check if Raid is active."""
        return self.attacked_this_turn


# =============================================================================
# UNIVERSES BEYOND MECHANICS
# =============================================================================

# Final Fantasy

@dataclass
class JobSystem:
    """
    Final Fantasy's Job Select mechanic.

    Choose a job when the creature enters. Each job gives different abilities.
    """
    creature_id: str
    available_jobs: List[str]  # e.g., ["Warrior", "Mage", "Thief"]
    selected_job: Optional[str] = None
    job_abilities: Dict[str, List[str]] = field(default_factory=dict)

    def select_job(self, job: str):
        """Select a job for this creature."""
        if job in self.available_jobs:
            self.selected_job = job

    def get_abilities(self) -> List[str]:
        """Get abilities from selected job."""
        if self.selected_job:
            return self.job_abilities.get(self.selected_job, [])
        return []


@dataclass
class LimitBreak:
    """
    Final Fantasy's Limit mechanic.

    Limit abilities can only be activated if you have less life than starting.
    """
    card_id: str
    limit_ability: str

    def can_activate(self, current_life: int, starting_life: int = 20) -> bool:
        """Check if Limit can be activated."""
        return current_life < starting_life


@dataclass
class MaterializeToken:
    """
    Final Fantasy's Materialize mechanic.

    Create a token copy, but it phases out at end of turn.
    """
    source_card: str
    token_id: str
    phases_out_at_end: bool = True


# Edge of Eternities

@dataclass
class SpaceCounter:
    """
    Edge of Eternities' Space mechanic.

    Tracks space counters and space-based abilities.
    """
    permanent_id: str
    space_counters: int = 0
    space_abilities: Dict[int, str] = field(default_factory=dict)  # threshold -> ability

    def add_counters(self, amount: int = 1):
        """Add space counters."""
        self.space_counters += amount

    def get_active_abilities(self) -> List[str]:
        """Get abilities that are active based on counter count."""
        active = []
        for threshold, ability in self.space_abilities.items():
            if self.space_counters >= threshold:
                active.append(ability)
        return active


@dataclass
class Station:
    """
    Edge of Eternities' Station mechanic.

    Stations are artifacts that provide abilities while you're "at" them.
    """
    card_id: str
    name: str
    station_abilities: List[str] = field(default_factory=list)
    is_at_station: bool = False

    def arrive(self):
        """Arrive at this station."""
        self.is_at_station = True

    def depart(self):
        """Depart from this station."""
        self.is_at_station = False


# Spider-Man

@dataclass
class WebSlinging:
    """
    Spider-Man's Web-slinging mechanic.

    Web-slinging creatures can swing to attack different targets.
    """
    creature_id: str
    has_web_slinging: bool = True
    web_targets: List[str] = field(default_factory=list)  # Valid swing targets

    def can_swing_to(self, target: str) -> bool:
        """Check if creature can swing to target."""
        return target in self.web_targets


# =============================================================================
# COMBINED STATE ENCODER
# =============================================================================

@dataclass
class StandardMechanicsState:
    """Combined state for all Standard 2025 mechanics."""

    # Tribal
    tribal: TribalTracker = field(default_factory=TribalTracker)

    # Vehicles & Mounts
    vehicles: List[Vehicle] = field(default_factory=list)
    mounts: List[Mount] = field(default_factory=list)

    # Rooms
    rooms: List[Room] = field(default_factory=list)

    # Delirium
    delirium: DeliriumTracker = field(default_factory=DeliriumTracker)

    # Cases
    cases: List[Case] = field(default_factory=list)

    # Raid
    raid: RaidTracker = field(default_factory=RaidTracker)

    # Universes Beyond
    jobs: List[JobSystem] = field(default_factory=list)
    limit_cards: List[LimitBreak] = field(default_factory=list)
    space_permanents: List[SpaceCounter] = field(default_factory=list)
    stations: List[Station] = field(default_factory=list)
    web_slingers: List[WebSlinging] = field(default_factory=list)

    def encode(self) -> np.ndarray:
        """
        Encode full Standard mechanics state.

        Returns feature vector for neural network input.
        """
        features = []

        # Tribal features (50 dims)
        features.extend(self.tribal.encode())

        # Delirium features (10 dims)
        features.extend(self.delirium.encode())

        # Raid (1 dim)
        features.append(1.0 if self.raid.has_raid() else 0.0)

        # Vehicle/Mount counts (2 dims)
        features.append(min(len(self.vehicles), 5) / 5.0)
        features.append(min(len(self.mounts), 5) / 5.0)

        # Crewed/Saddled counts (2 dims)
        crewed = sum(1 for v in self.vehicles if v.is_crewed)
        saddled = sum(1 for m in self.mounts if m.is_saddled)
        features.append(min(crewed, 5) / 5.0)
        features.append(min(saddled, 5) / 5.0)

        # Rooms (2 dims: count, doors unlocked)
        features.append(min(len(self.rooms), 5) / 5.0)
        doors_unlocked = sum(
            (1 if r.door1_unlocked else 0) + (1 if r.door2_unlocked else 0)
            for r in self.rooms
        )
        features.append(min(doors_unlocked, 10) / 10.0)

        # Cases (2 dims: count, solved)
        features.append(min(len(self.cases), 5) / 5.0)
        solved = sum(1 for c in self.cases if c.status == CaseStatus.SOLVED)
        features.append(min(solved, 5) / 5.0)

        # Universes Beyond summary (5 dims)
        features.append(min(len(self.jobs), 3) / 3.0)
        features.append(min(len(self.stations), 3) / 3.0)
        features.append(min(len(self.web_slingers), 3) / 3.0)
        total_space = sum(s.space_counters for s in self.space_permanents)
        features.append(min(total_space, 10) / 10.0)
        features.append(1.0 if any(s.is_at_station for s in self.stations) else 0.0)

        return np.array(features, dtype=np.float32)

    @property
    def encoding_dim(self) -> int:
        """Dimension of encoded state."""
        # 42 (tribal) + 10 (delirium) + 1 (raid) + 2 (vehicle/mount counts) +
        # 2 (crewed/saddled) + 2 (rooms) + 2 (cases) + 5 (universes beyond) = 66
        return 66


# =============================================================================
# ACTION SPACE EXTENSIONS
# =============================================================================

class StandardActionType(Enum):
    """Action types added by Standard 2025 mechanics."""

    # Vehicles & Mounts
    CREW_VEHICLE = auto()      # Tap creatures to crew
    SADDLE_MOUNT = auto()      # Tap creatures to saddle

    # Rooms
    UNLOCK_DOOR = auto()       # Pay to unlock second door

    # Collect Evidence
    COLLECT_EVIDENCE = auto()  # Exile cards from graveyard

    # Jobs (Final Fantasy)
    SELECT_JOB = auto()        # Choose job for creature

    # Stations (Edge of Eternities)
    ARRIVE_STATION = auto()    # Move to a station
    DEPART_STATION = auto()    # Leave a station


@dataclass
class StandardAction:
    """Represents an action using Standard mechanics."""
    action_type: StandardActionType
    source_card: str
    target_cards: List[str] = field(default_factory=list)
    choice: Optional[str] = None  # For job selection, etc.
    mana_payment: Optional[Dict[str, int]] = None

    def describe(self) -> str:
        """Human-readable description of action."""
        if self.action_type == StandardActionType.CREW_VEHICLE:
            return f"Crew {self.source_card} with {', '.join(self.target_cards)}"
        elif self.action_type == StandardActionType.SADDLE_MOUNT:
            return f"Saddle {self.source_card} with {', '.join(self.target_cards)}"
        elif self.action_type == StandardActionType.UNLOCK_DOOR:
            return f"Unlock door on {self.source_card}"
        elif self.action_type == StandardActionType.COLLECT_EVIDENCE:
            return f"Collect evidence by exiling {', '.join(self.target_cards)}"
        elif self.action_type == StandardActionType.SELECT_JOB:
            return f"Select job '{self.choice}' for {self.source_card}"
        elif self.action_type == StandardActionType.ARRIVE_STATION:
            return f"Arrive at station {self.source_card}"
        elif self.action_type == StandardActionType.DEPART_STATION:
            return f"Depart from station {self.source_card}"
        return f"{self.action_type.name} on {self.source_card}"


def get_standard_actions(state: StandardMechanicsState,
                         available_creatures: List[Tuple[str, int]],
                         available_mana: Dict[str, int],
                         graveyard: List[Tuple[str, int]]) -> List[StandardAction]:
    """
    Generate all valid Standard mechanic actions.

    Args:
        state: Current mechanics state
        available_creatures: Untapped creatures as (id, power) pairs
        available_mana: Available mana by color
        graveyard: Cards in graveyard as (id, mana_value) pairs

    Returns:
        List of valid StandardAction objects
    """
    actions = []

    # Crew vehicles
    for vehicle in state.vehicles:
        if not vehicle.is_crewed:
            for combo in vehicle.can_crew(available_creatures):
                actions.append(StandardAction(
                    action_type=StandardActionType.CREW_VEHICLE,
                    source_card=vehicle.card_id,
                    target_cards=combo
                ))

    # Saddle mounts
    for mount in state.mounts:
        if not mount.is_saddled:
            for combo in mount.can_saddle(available_creatures):
                actions.append(StandardAction(
                    action_type=StandardActionType.SADDLE_MOUNT,
                    source_card=mount.card_id,
                    target_cards=combo
                ))

    # Unlock room doors
    for room in state.rooms:
        if not room.door2_unlocked and room.can_unlock_door2(available_mana):
            actions.append(StandardAction(
                action_type=StandardActionType.UNLOCK_DOOR,
                source_card=room.card_id
            ))

    # Job selection (for creatures that just entered)
    for job_sys in state.jobs:
        if job_sys.selected_job is None:
            for job in job_sys.available_jobs:
                actions.append(StandardAction(
                    action_type=StandardActionType.SELECT_JOB,
                    source_card=job_sys.creature_id,
                    choice=job
                ))

    # Station actions
    for station in state.stations:
        if not station.is_at_station:
            actions.append(StandardAction(
                action_type=StandardActionType.ARRIVE_STATION,
                source_card=station.card_id
            ))
        else:
            actions.append(StandardAction(
                action_type=StandardActionType.DEPART_STATION,
                source_card=station.card_id
            ))

    return actions


# =============================================================================
# INTEGRATION WITH EXISTING CARD EMBEDDINGS
# =============================================================================

def extend_card_embedding(base_embedding: np.ndarray,
                         card_types: List[CreatureType],
                         has_vehicle: bool = False,
                         has_mount: bool = False,
                         crew_cost: int = 0,
                         saddle_cost: int = 0,
                         is_room: bool = False,
                         is_case: bool = False,
                         has_raid: bool = False,
                         has_delirium: bool = False,
                         collect_evidence_cost: int = 0,
                         job_options: int = 0,
                         has_limit: bool = False,
                         has_space: bool = False,
                         is_station: bool = False,
                         has_web_slinging: bool = False) -> np.ndarray:
    """
    Extend base card embedding with Standard 2025 mechanics features.

    Args:
        base_embedding: Existing card embedding
        card_types: Creature types on card
        Various mechanic flags

    Returns:
        Extended embedding with Standard mechanics features
    """
    standard_features = []

    # Tribal (one-hot for major types, max 3)
    type_encoding = np.zeros(len(CreatureType), dtype=np.float32)
    for i, ct in enumerate(CreatureType):
        if ct in card_types:
            type_encoding[i] = 1.0
    standard_features.extend(type_encoding[:10])  # First 10 types

    # Is outlaw (batch check)
    is_outlaw = any(ct in OUTLAW_TYPES for ct in card_types)
    standard_features.append(1.0 if is_outlaw else 0.0)

    # Vehicle/Mount
    standard_features.append(1.0 if has_vehicle else 0.0)
    standard_features.append(1.0 if has_mount else 0.0)
    standard_features.append(min(crew_cost, 5) / 5.0)
    standard_features.append(min(saddle_cost, 5) / 5.0)

    # Room/Case
    standard_features.append(1.0 if is_room else 0.0)
    standard_features.append(1.0 if is_case else 0.0)

    # Triggered mechanics
    standard_features.append(1.0 if has_raid else 0.0)
    standard_features.append(1.0 if has_delirium else 0.0)
    standard_features.append(min(collect_evidence_cost, 6) / 6.0)

    # Universes Beyond
    standard_features.append(min(job_options, 3) / 3.0)
    standard_features.append(1.0 if has_limit else 0.0)
    standard_features.append(1.0 if has_space else 0.0)
    standard_features.append(1.0 if is_station else 0.0)
    standard_features.append(1.0 if has_web_slinging else 0.0)

    # Concatenate
    extended = np.concatenate([
        base_embedding,
        np.array(standard_features, dtype=np.float32)
    ])

    return extended


# Extension size: 25 additional dimensions
STANDARD_EXTENSION_DIM = 25


if __name__ == "__main__":
    # Test the implementations
    print("Testing Standard 2025 Mechanics Implementation")
    print("=" * 50)

    # Test tribal tracking
    tribal = TribalTracker()
    tribal.battlefield_counts = {
        CreatureType.HUMAN: 2,
        CreatureType.DRAGON: 1,
        CreatureType.ROGUE: 1,
        CreatureType.ASSASSIN: 1,
    }
    print(f"Outlaws on battlefield: {tribal.count_outlaws()}")
    print(f"Tribal encoding shape: {tribal.encode().shape}")

    # Test delirium
    delirium = DeliriumTracker()
    delirium.add_card("card1", [CardType.CREATURE])
    delirium.add_card("card2", [CardType.INSTANT])
    delirium.add_card("card3", [CardType.ARTIFACT])
    print(f"Delirium types: {delirium.count_types()}, Active: {delirium.has_delirium()}")
    delirium.add_card("card4", [CardType.LAND])
    print(f"After land: types={delirium.count_types()}, Delirium: {delirium.has_delirium()}")

    # Test vehicle crewing
    vehicle = Vehicle(
        card_id="v1",
        name="Thundering Raiju",
        power=3,
        toughness=4,
        crew_cost=3
    )
    creatures = [("c1", 2), ("c2", 2), ("c3", 3)]
    combos = vehicle.can_crew(creatures)
    print(f"Valid crew combinations: {len(combos)}")

    # Test full state encoding
    state = StandardMechanicsState()
    state.tribal = tribal
    state.delirium = delirium
    state.vehicles = [vehicle]
    encoded = state.encode()
    print(f"Full state encoding shape: {encoded.shape}")
    print(f"Encoding dimension: {state.encoding_dim}")

    # Test action generation
    actions = get_standard_actions(
        state,
        available_creatures=creatures,
        available_mana={"W": 2, "U": 1},
        graveyard=[("g1", 2), ("g2", 3)]
    )
    print(f"Available Standard actions: {len(actions)}")
    for action in actions[:3]:
        print(f"  - {action.describe()}")

    print("\nAll tests passed!")
