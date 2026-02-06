"""
Card Text Parser

Parses Scryfall oracle text into mechanics sequences.

Strategy:
1. Rule-based parsing for common patterns (handles ~80% of cards)
2. LLM fallback for complex/ambiguous text (handles remaining ~20%)

This is NOT meant to be exhaustive from day 1. We iterate:
- Start with common patterns
- Add rules as we encounter new patterns
- LLM handles edge cases during development
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import our vocabulary
try:
    from src.mechanics.vocabulary import Mechanic, CardEncoding, VOCAB_SIZE
except ImportError:
    from vocabulary import Mechanic, CardEncoding, VOCAB_SIZE


# =============================================================================
# KEYWORD PATTERNS
# =============================================================================

# Keywords that appear as single words
KEYWORD_ABILITIES = {
    # Evergreen
    "flying": Mechanic.FLYING,
    "trample": Mechanic.TRAMPLE,
    "first strike": Mechanic.FIRST_STRIKE,
    "double strike": Mechanic.DOUBLE_STRIKE,
    "deathtouch": Mechanic.DEATHTOUCH,
    "lifelink": Mechanic.LIFELINK,
    "vigilance": Mechanic.VIGILANCE,
    "reach": Mechanic.REACH,
    "haste": Mechanic.HASTE,
    "menace": Mechanic.MENACE,
    "defender": Mechanic.DEFENDER,
    "indestructible": Mechanic.INDESTRUCTIBLE,
    "hexproof": Mechanic.HEXPROOF,
    "flash": Mechanic.FLASH,

    # Combat keywords
    "skulk": Mechanic.SKULK,
    "fear": Mechanic.FEAR,
    "intimidate": Mechanic.INTIMIDATE,
    "shadow": Mechanic.SHADOW,
    "horsemanship": Mechanic.HORSEMANSHIP,
    "flanking": Mechanic.FLANKING,
    "bushido": Mechanic.BUSHIDO,
    "rampage": Mechanic.RAMPAGE,
    "provoke": Mechanic.PROVOKE,
    "afflict": Mechanic.AFFLICT,

    # Counter-related
    "undying": Mechanic.UNDYING,
    "persist": Mechanic.PERSIST,
    "modular": Mechanic.MODULAR,
    "evolve": Mechanic.EVOLVE,
    "graft": Mechanic.GRAFT,
    "devour": Mechanic.DEVOUR,
    "mentor": Mechanic.MENTOR,
    "training": Mechanic.TRAINING,
    "renown": Mechanic.RENOWN,
    "fabricate": Mechanic.FABRICATE,
    "backup": Mechanic.BACKUP,
    "riot": Mechanic.RIOT,
    "adapt": Mechanic.ADAPT,

    # Cost reduction
    "convoke": Mechanic.CONVOKE,
    "delve": Mechanic.DELVE,
    "affinity": Mechanic.AFFINITY,
    "improvise": Mechanic.IMPROVISE,
    "assist": Mechanic.ASSIST,
    "undaunted": Mechanic.UNDAUNTED,

    # Alternative casting
    "flashback": Mechanic.FLASHBACK,
    "retrace": Mechanic.RETRACE,
    "jump-start": Mechanic.JUMP_START,
    "escape": Mechanic.ESCAPE,
    "foretell": Mechanic.FORETELL,
    "disturb": Mechanic.DISTURB,
    "warp": Mechanic.WARP,
    "dash": Mechanic.DASH,
    "blitz": Mechanic.BLITZ,
    "evoke": Mechanic.EVOKE,
    "emerge": Mechanic.EMERGE,
    "mutate": Mechanic.MUTATE,
    "spectacle": Mechanic.SPECTACLE,
    "ninjutsu": Mechanic.NINJUTSU,
    "buyback": Mechanic.BUYBACK,
    "overload": Mechanic.OVERLOAD,
    "kicker": Mechanic.KICKER,
    "multikicker": Mechanic.MULTIKICKER,

    # Triggered keywords
    "landfall": Mechanic.LANDFALL,
    "constellation": Mechanic.CONSTELLATION,
    "heroic": Mechanic.HEROIC,
    "magecraft": Mechanic.MAGECRAFT,
    "prowess": Mechanic.PROWESS,
    "raid": Mechanic.RAID,
    "revolt": Mechanic.REVOLT,
    "morbid": Mechanic.MORBID,
    "exploit": Mechanic.EXPLOIT,
    "extort": Mechanic.EXTORT,
    "exalted": Mechanic.EXALTED,

    # Other
    "changeling": Mechanic.CHANGELING,
    "devoid": Mechanic.DEVOID,
    "storm": Mechanic.STORM,
    "cascade": Mechanic.CASCADE,
    "dredge": Mechanic.DREDGE,
    "suspend": Mechanic.SUSPEND,
    "miracle": Mechanic.MIRACLE,
    "rebound": Mechanic.REBOUND,
    "cipher": Mechanic.CIPHER,
    "hideaway": Mechanic.HIDEAWAY,
    "living weapon": Mechanic.LIVING_WEAPON,
    "reconfigure": Mechanic.RECONFIGURE,
    "toxic": Mechanic.TOXIC,
    "infect": Mechanic.INFECT,
    "wither": Mechanic.WITHER,
    "annihilator": Mechanic.ANNIHILATOR,
    "myriad": Mechanic.MYRIAD,
    "encore": Mechanic.ENCORE,
    "decayed": Mechanic.DECAYED,
    "prototype": Mechanic.PROTOTYPE,
    "transform": Mechanic.TRANSFORM,
    "daybound": Mechanic.DAYBOUND,
    "nightbound": Mechanic.NIGHTBOUND,

    # Multiplayer
    "goad": Mechanic.GOAD,
    "monarch": Mechanic.MONARCH,
    "initiative": Mechanic.INITIATIVE,

    # Dinosaur/Ixalan
    "enrage": Mechanic.DAMAGE_RECEIVED_TRIGGER,

    # Conditions
    "threshold": Mechanic.THRESHOLD,
    "delirium": Mechanic.DELIRIUM,
    "metalcraft": Mechanic.METALCRAFT,
    "ferocious": Mechanic.FEROCIOUS,
    "domain": Mechanic.DOMAIN,
    "descend": Mechanic.DESCEND,
    "corrupted": Mechanic.CORRUPTED,
    "coven": Mechanic.COVEN,
    "hellbent": Mechanic.HELLBENT,
    "party": Mechanic.PARTY,

    # Combat keywords (missing)
    "shroud": Mechanic.SHROUD,
    "protection": Mechanic.PROTECTION,
    "ward": Mechanic.WARD,
    "banding": Mechanic.BANDING,
    "absorb": Mechanic.ABSORB,
    "battle cry": Mechanic.BATTLE_CRY,
    "melee": Mechanic.MELEE,
    "split second": Mechanic.SPLIT_SECOND,

    # Temporal keywords (missing)
    "phasing": Mechanic.PHASING,
    "vanishing": Mechanic.VANISHING,
    "fading": Mechanic.FADING,
    "cumulative upkeep": Mechanic.CUMULATIVE_UPKEEP,
    "echo": Mechanic.ECHO,

    # Modal/split keywords (missing)
    "entwine": Mechanic.ENTWINE,
    "splice": Mechanic.SPLICE,
    "fuse": Mechanic.FUSE,
    "aftermath": Mechanic.AFTERMATH,
    "adventure": Mechanic.ADVENTURE,

    # Partner/companion (missing)
    "companion": Mechanic.COMPANION,
    "partner": Mechanic.PARTNER,

    # Aura/equipment keywords (missing)
    "equip": Mechanic.EQUIP,
    "bestow": Mechanic.BESTOW,
    "soulbond": Mechanic.SOULBOND,
    "regenerate": Mechanic.REGENERATE,

    # Tribute/choice (missing)
    "tribute": Mechanic.TRIBUTE,

    # Recent set mechanics (missing)
    "cleave": Mechanic.CLEAVE,
    "casualty": Mechanic.CASUALTY,
    # "connive" handled via PATTERNS to catch verb conjugations (connives, connived)
    "for mirrodin!": Mechanic.FOR_MIRRODIN,
    "incubate": Mechanic.INCUBATE,
    "learn": Mechanic.LEARN,
    "craft": Mechanic.CRAFT,
    "plot": Mechanic.PLOT,
    "morph": Mechanic.MORPH,
    "manifest": Mechanic.MANIFEST,
    "disguise": Mechanic.DISGUISE,
    "discover": Mechanic.DISCOVER,
    "monstrosity": Mechanic.MONSTROSITY,
    "support": Mechanic.SUPPORT,
    "bolster": Mechanic.BOLSTER,
    "transmute": Mechanic.TRANSMUTE,
    "forecast": Mechanic.FORECAST,
    "bloodthirst": Mechanic.BLOODTHIRST,
}


# =============================================================================
# TEXT PATTERN MATCHING
# =============================================================================

# Patterns for common text structures
PATTERNS = [
    # Targeting patterns
    (r"target creature", [Mechanic.TARGET_CREATURE]),
    (r"target player", [Mechanic.TARGET_PLAYER]),
    (r"target opponent", [Mechanic.TARGET_OPPONENT]),
    (r"target permanent", [Mechanic.TARGET_PERMANENT]),
    (r"target spell", [Mechanic.TARGET_SPELL]),
    (r"target artifact", [Mechanic.TARGET_ARTIFACT]),
    (r"target enchantment", [Mechanic.TARGET_ENCHANTMENT]),
    (r"target land", [Mechanic.TARGET_LAND]),
    (r"target (spell or ability|ability)", [Mechanic.TARGET_SPELL_OR_ABILITY]),
    (r"target card (in|from) (a |your )?graveyard", [Mechanic.TARGET_CARD_IN_GRAVEYARD]),
    (r"each creature", [Mechanic.TARGETS_EACH, Mechanic.TARGET_CREATURE]),
    (r"each opponent", [Mechanic.TARGETS_EACH, Mechanic.TARGET_OPPONENT]),
    (r"all creatures", [Mechanic.TARGETS_ALL, Mechanic.TARGET_CREATURE]),
    (r"each player", [Mechanic.TARGETS_EACH, Mechanic.TARGET_PLAYER]),

    # Removal patterns
    (r"destroy target", [Mechanic.DESTROY]),
    (r"destroys? (it|that|this|target)", [Mechanic.DESTROY]),
    (r"exile target", [Mechanic.EXILE]),
    (r"exiles? (it|that|this|target)", [Mechanic.EXILE]),
    (r"sacrifice (a|an|target)", [Mechanic.SACRIFICE]),
    (r"return (it|that|target).+to (its|their|your|owner's) (hand|owner's hand)", [Mechanic.BOUNCE_TO_HAND]),
    (r"put.+on (the bottom|top) of.+library", [Mechanic.BOUNCE_TO_LIBRARY]),
    (r"counter target spell", [Mechanic.COUNTER_SPELL]),
    (r"counter (it|that spell)", [Mechanic.COUNTER_SPELL]),
    (r"counter target.+ability", [Mechanic.COUNTER_ABILITY]),
    (r"deals? (\d+|x) damage", [Mechanic.DEAL_DAMAGE]),
    (r"loses? (\d+|x) life", [Mechanic.LOSE_LIFE]),
    (r"you lose life equal to", [Mechanic.LOSE_LIFE]),

    # Creation patterns
    (r"create(s)? (a|\d+|an?|two|three|four|five|x) .*(token|treasure|food|clue|blood)", [Mechanic.CREATE_TOKEN]),
    (r"create(s)?.+cop(y|ies) of", [Mechanic.CREATE_TOKEN_COPY]),
    (r"create(s)?.+treasure token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_TREASURE]),
    (r"create(s)?.+food token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_FOOD]),
    (r"create(s)?.+clue token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_CLUE]),
    (r"create(s)?.+blood token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_BLOOD]),
    (r"\binvestigate\b", [Mechanic.CREATE_CLUE]),

    # Card advantage
    (r"draw(s)? (a card|two cards|three cards|\d+ cards?)", [Mechanic.DRAW]),
    (r"may draw (a card|two cards|three cards|\d+ cards?)", [Mechanic.DRAW_OPTIONAL]),
    (r"scry (\d+|x)", [Mechanic.SCRY]),
    (r"surveil (\d+|x)", [Mechanic.SURVEIL]),
    (r"look at the top", [Mechanic.LOOK_AT_TOP]),
    (r"reveal", [Mechanic.REVEAL]),
    (r"search your library", [Mechanic.TUTOR_TO_HAND]),
    (r"search your library.+put.+onto the battlefield", [Mechanic.TUTOR_TO_BATTLEFIELD]),
    (r"return.+from.+graveyard to the battlefield", [Mechanic.REANIMATE]),
    (r"return.+from.+graveyard to your hand", [Mechanic.REGROWTH]),
    (r"discard(s)? (a card|two cards|three cards|\d+ cards?|your hand|that card|it|them)", [Mechanic.DISCARD]),
    (r"mill(s)? (\d+|x)", [Mechanic.MILL]),
    (r"you may cast.+from.+graveyard", [Mechanic.CAST_FROM_GRAVEYARD]),

    # Triggers
    (r"when(ever)? .+ enters( the battlefield)?", [Mechanic.ETB_TRIGGER]),
    (r"when(ever)? .+ leaves( the battlefield)?", [Mechanic.LTB_TRIGGER]),
    (r"when(ever)? .+ dies", [Mechanic.DEATH_TRIGGER]),
    (r"when(ever)? .+ attacks", [Mechanic.ATTACK_TRIGGER]),
    (r"when(ever)? .+ blocks", [Mechanic.BLOCK_TRIGGER]),
    (r"when(ever)? .+ becomes? blocked", [Mechanic.BLOCK_TRIGGER]),
    (r"when(ever)? .+ deals (combat )?damage", [Mechanic.DAMAGE_TRIGGER]),
    (r"when(ever)? .+ is dealt damage", [Mechanic.DAMAGE_RECEIVED_TRIGGER]),
    (r"when(ever)? you cast", [Mechanic.CAST_TRIGGER]),
    (r"when(ever)? an opponent casts", [Mechanic.OPPONENT_CASTS]),
    (r"at the beginning of your upkeep", [Mechanic.UPKEEP_TRIGGER]),
    (r"at the beginning of (each|your) end step", [Mechanic.END_STEP_TRIGGER]),
    (r"when(ever)? you draw", [Mechanic.DRAW_TRIGGER]),
    (r"when(ever)? you gain life", [Mechanic.GAIN_LIFE_TRIGGER]),
    (r"when(ever)? you lose life", [Mechanic.LOSE_LIFE_TRIGGER]),
    (r"when(ever)? a land enters", [Mechanic.LANDFALL]),
    (r"when(ever)? .+ deals combat damage to a player", [Mechanic.COMBAT_DAMAGE_TO_PLAYER]),

    # Conditions
    (r"if .+ dies this way", [Mechanic.IF_TARGET_DIES]),
    (r"if you control a commander", [Mechanic.IF_YOU_CONTROL_COMMANDER]),
    (r"if you control (a|an) creature", [Mechanic.IF_YOU_CONTROL_CREATURE]),
    (r"if you control (a|an) artifact", [Mechanic.IF_YOU_CONTROL_ARTIFACT]),
    (r"if you control (a|an) enchantment", [Mechanic.IF_YOU_CONTROL_ENCHANTMENT]),
    (r"if a creature (entered|died)", [Mechanic.IF_CREATURE_ENTERED]),
    (r"if you('ve)? gained.+life", [Mechanic.IF_LIFE_GAINED]),
    (r"unless (that player|they) pays?", [Mechanic.UNLESS_PAYS]),

    # Stats modification
    (r"gets? \+\d+/\+\d+", [Mechanic.PLUS_POWER, Mechanic.PLUS_TOUGHNESS]),
    (r"gets? -\d+/-\d+", [Mechanic.MINUS_POWER, Mechanic.MINUS_TOUGHNESS]),
    (r"gets? \+\d+/\+0", [Mechanic.PLUS_POWER]),
    (r"gets? \+0/\+\d+", [Mechanic.PLUS_TOUGHNESS]),
    (r"(other )?(creatures|permanents) you control get \+", [Mechanic.ANTHEM_EFFECT]),
    (r"\+1/\+1 counter", [Mechanic.PLUS_ONE_COUNTER]),
    (r"-1/-1 counter", [Mechanic.MINUS_ONE_COUNTER]),
    (r"half.+(power|toughness)", [Mechanic.HALF_STATS]),
    (r"double.+(power|toughness)", [Mechanic.DOUBLE_STATS]),

    # Mana
    (r"add \{", [Mechanic.ADD_MANA]),
    (r"add (one|two|three|\d+) mana", [Mechanic.ADD_MANA]),
    (r"mana of any color", [Mechanic.MANA_OF_ANY_COLOR]),
    (r"costs? \{?\d+\}? less to cast", [Mechanic.REDUCE_COST]),
    (r"costs? \{?\d+\}? more to cast", [Mechanic.INCREASE_COST]),
    (r"without paying (its|their) mana cost", [Mechanic.FREE_CAST_CONDITION]),
    (r"you may cast.+without paying", [Mechanic.FREE_CAST_CONDITION]),
    (r"you may choose new targets", [Mechanic.CHANGE_TARGETS]),

    # Special effects
    (r"gains? protection", [Mechanic.PROTECTION]),
    (r"can't be blocked\b", [Mechanic.UNBLOCKABLE]),
    (r"can't be countered", [Mechanic.CANT_BE_COUNTERED]),
    (r"tap target", [Mechanic.TAP]),
    (r"untap (target|it|them|\w+)", [Mechanic.UNTAP]),
    (r"fight(s)?", [Mechanic.FIGHT]),
    (r"proliferate", [Mechanic.PROLIFERATE]),
    (r"twice that many", [Mechanic.TOKEN_DOUBLER]),
    (r"trigger(s)? an additional time", [Mechanic.DOUBLE_TRIGGER]),
    (r"instead", [Mechanic.REPLACEMENT_EFFECT]),
    (r"prevent(s)? (all )?(the next )?\d* ?damage", [Mechanic.PREVENT_DAMAGE]),
    (r"prevent(s)? all (combat )?damage", [Mechanic.PREVENT_DAMAGE]),
    (r"damage.+(be )?prevented", [Mechanic.PREVENT_DAMAGE]),
    (r"gain control of", [Mechanic.GAIN_CONTROL]),
    (r"gains? control of", [Mechanic.GAIN_CONTROL]),
    (r"exchange control", [Mechanic.GAIN_CONTROL]),
    (r"crew\s+\d+", [Mechanic.CREW]),
    (r"exile the top.+you may (play|cast)", [Mechanic.IMPULSE_DRAW]),
    (r"exile.+from the top.+you may (play|cast)", [Mechanic.IMPULSE_DRAW]),
    (r"exile the top.+until end of turn", [Mechanic.IMPULSE_DRAW]),
    (r"phases? out", [Mechanic.PHASE_OUT]),
    (r"take an extra turn", [Mechanic.EXTRA_TURN]),
    (r"extra turn after this one", [Mechanic.EXTRA_TURN]),
    (r"you win the game", [Mechanic.WIN_GAME]),
    (r"(target player|that player|opponent) loses the game", [Mechanic.LOSE_GAME]),
    (r"you lose the game", [Mechanic.LOSE_GAME]),
    (r"as an additional cost.+sacrifice", [Mechanic.ADDITIONAL_COST, Mechanic.SACRIFICE]),

    # Gain control variants
    (r"(forestwalk|islandwalk|swampwalk|mountainwalk|plainswalk)", [Mechanic.LANDWALK]),
    (r"no more than (one|two|\d+) creatures? can (attack|block)", [Mechanic.COMBAT_RESTRICTION]),
    (r"only one creature can (attack|block)", [Mechanic.COMBAT_RESTRICTION]),
    (r"enters? (the battlefield )?with (\w+ ){0,3}\+1/\+1 counter", [Mechanic.ENTERS_WITH_COUNTERS, Mechanic.PLUS_ONE_COUNTER]),
    (r"enters? (the battlefield )?with (\w+ ){0,3}-1/-1 counter", [Mechanic.ENTERS_WITH_COUNTERS, Mechanic.MINUS_ONE_COUNTER]),
    (r"enters? (the battlefield )?with (\w+ ){0,3}counter", [Mechanic.ENTERS_WITH_COUNTERS]),
    (r"flip a coin", [Mechanic.COIN_FLIP]),
    (r"regenerate", [Mechanic.REGENERATE]),
    (r"enters? as a copy", [Mechanic.COPY_PERMANENT]),
    (r"becomes? a copy", [Mechanic.COPY_PERMANENT]),
    (r"stun counter", [Mechanic.STUN_COUNTER]),

    # Zones
    (r"from your hand", [Mechanic.FROM_HAND]),
    (r"from (your|a|the) graveyard", [Mechanic.FROM_GRAVEYARD]),
    (r"from exile", [Mechanic.FROM_EXILE]),
    (r"from (your|the top of your) library", [Mechanic.FROM_LIBRARY]),
    (r"put.+into (your|the) graveyard", [Mechanic.TO_GRAVEYARD]),
    (r"exile.+until", [Mechanic.EXILE_TEMPORARY]),
    (r"you may cast.+from exile", [Mechanic.CAST_FROM_EXILE]),
    (r"enters (the battlefield )?tapped", [Mechanic.TO_BATTLEFIELD_TAPPED]),

    # =========================================================================
    # MODAL / CHOICE
    # =========================================================================
    (r"choose one", [Mechanic.MODAL_CHOOSE_ONE]),
    (r"choose two", [Mechanic.MODAL_CHOOSE_TWO]),
    (r"choose three", [Mechanic.MODAL_CHOOSE_THREE]),
    (r"choose (up to )?(one or more|any number|x)", [Mechanic.MODAL_CHOOSE_X]),

    # =========================================================================
    # RECENT SET MECHANICS
    # =========================================================================
    (r"spree\b", [Mechanic.SPREE]),
    (r"offspring\b", [Mechanic.OFFSPRING]),
    (r"\beerie\b", [Mechanic.EERIE]),
    (r"\bsurvival\b", [Mechanic.SURVIVAL]),
    (r"impending\s+\d+", [Mechanic.IMPENDING]),
    (r"\bbargain\b", [Mechanic.BARGAIN]),
    (r"\bcelebrat(e|ion)\b", [Mechanic.CELEBRATION]),
    (r"\brole\b.+\btoken\b", [Mechanic.ROLE_TOKEN]),
    (r"\bcase\b", [Mechanic.CASE]),
    (r"\bsuspect\b", [Mechanic.SUSPECT]),
    (r"\bcloak\b", [Mechanic.CLOAK]),
    (r"\bconnive[sd]?\b", [Mechanic.CONNIVE]),
    (r"collect evidence\s+\d+", [Mechanic.COLLECT_EVIDENCE]),
    (r"commit(ted|s)? a crime", [Mechanic.COMMIT_A_CRIME]),
    (r"saddle\s+\d+", [Mechanic.SADDLE]),
    (r"gift a\b", [Mechanic.GIFT]),
    (r"\bdescend 4\b", [Mechanic.DESCEND_4]),
    (r"\bdescend 8\b", [Mechanic.DESCEND_8]),
    (r"fathomless descent", [Mechanic.FATHOMLESS_DESCENT]),
    (r"\bmap token", [Mechanic.MAP_TOKEN]),
    (r"\bvaliant\b", [Mechanic.VALIANT]),
    (r"\boutlaw\b", [Mechanic.OUTLAW]),

    # =========================================================================
    # TYPE FILTERS
    # =========================================================================
    (r"\bnonland\b", [Mechanic.FILTER_NONLAND]),
    (r"\bnoncreature\b", [Mechanic.FILTER_NONCREATURE]),
    (r"\bnontoken\b", [Mechanic.FILTER_NONTOKEN]),
    (r"\bnonartifact\b", [Mechanic.FILTER_NONARTIFACT]),

    # =========================================================================
    # DURATION MARKERS
    # =========================================================================
    (r"until end of turn", [Mechanic.UNTIL_END_OF_TURN]),
    (r"until your next turn", [Mechanic.UNTIL_YOUR_NEXT_TURN]),
    (r"as long as", [Mechanic.AS_LONG_AS]),

    # =========================================================================
    # LIFE GAIN
    # =========================================================================
    (r"gains?\s+\d+\s+life", [Mechanic.GAIN_LIFE]),
    (r"you gain\s+\d+\s+life", [Mechanic.GAIN_LIFE]),

    # =========================================================================
    # VARIABLE EFFECTS ("where X is" / "equal to" / "for each")
    # =========================================================================
    # Variable draw
    (r"draw(s)?\s+(a\s+)?cards?\s+for\s+each", [Mechanic.DRAW]),
    (r"draw(s)?\s+\w+\s+cards?,?\s+where", [Mechanic.DRAW]),
    (r"draw(s)?\s+cards?\s+equal\s+to", [Mechanic.DRAW]),

    # Variable damage
    (r"deals?\s+damage\s+(to\s+.+?\s+)?equal\s+to", [Mechanic.DEAL_DAMAGE]),
    (r"deals?\s+\w+\s+damage.+?where\s+\w+\s+is", [Mechanic.DEAL_DAMAGE]),
    (r"deals?\s+damage\s+to\s+.+?\s+for\s+each", [Mechanic.DEAL_DAMAGE]),

    # Variable life loss/gain
    (r"loses?\s+life\s+equal\s+to", [Mechanic.LOSE_LIFE]),
    (r"loses?\s+\w+\s+life,?\s+where", [Mechanic.LOSE_LIFE]),
    (r"gains?\s+life\s+equal\s+to", [Mechanic.GAIN_LIFE]),

    # Variable mill
    (r"mills?\s+\w+\s+cards?,?\s+where", [Mechanic.MILL]),
    (r"mills?\s+cards?\s+equal\s+to", [Mechanic.MILL]),

    # Variable scry/surveil
    (r"scry\s+\w+,?\s+where", [Mechanic.SCRY]),
    (r"surveil\s+\w+,?\s+where", [Mechanic.SURVEIL]),

    # Variable tokens
    (r"create\s+.*?\s+tokens?\s+for\s+each", [Mechanic.CREATE_TOKEN]),

    # =========================================================================
    # EQUIPMENT
    # =========================================================================
    (r"equipped creature", [Mechanic.EQUIP]),

    # =========================================================================
    # COUNTER TYPES
    # =========================================================================
    (r"experience counter", [Mechanic.EXPERIENCE_COUNTER]),
    (r"\{e\}|energy counter", [Mechanic.ENERGY_COUNTER]),
    (r"shield counter", [Mechanic.SHIELD_COUNTER]),
    (r"oil counter", [Mechanic.OIL_COUNTER]),
    (r"(flying|first strike|double strike|deathtouch|lifelink|vigilance|reach|trample|haste|menace|hexproof|indestructible) counter", [Mechanic.KEYWORD_COUNTER]),

    # =========================================================================
    # BECOMES CREATURE (manlands, Gideon, etc.)
    # =========================================================================
    (r"becomes?\s+a?\s*\d+/\d+.*creature", [Mechanic.BECOMES_CREATURE]),
    (r"is\s+(a|an)\s+.*creature.*as long as", [Mechanic.BECOMES_CREATURE, Mechanic.AS_LONG_AS]),

    # =========================================================================
    # AURA / ENCHANTMENT EFFECTS
    # =========================================================================
    (r"enchant creature", [Mechanic.TARGET_CREATURE]),
    (r"enchant (permanent|artifact|land|player)", []),
    (r"enchanted creature", []),
    (r"enchanted permanent", []),
    (r"can't attack or block", [Mechanic.CANT_ATTACK, Mechanic.CANT_BLOCK]),
    (r"can't attack\b", [Mechanic.CANT_ATTACK]),
    (r"can't block\b", [Mechanic.CANT_BLOCK]),
    (r"base power and toughness (\d+)/(\d+)", [Mechanic.SET_POWER, Mechanic.SET_TOUGHNESS]),
    (r"loses all (other )?abilities", [Mechanic.LOSES_ABILITIES]),

    # =========================================================================
    # STAX / HATE EFFECTS
    # =========================================================================
    (r"can't gain life", [Mechanic.CANT_GAIN_LIFE]),
    (r"can't be cast", [Mechanic.CANT_CAST]),
    (r"can't cast\b", [Mechanic.CANT_CAST]),
    (r"can't cast more than one", [Mechanic.CAST_RESTRICTION]),
    (r"can't cast additional", [Mechanic.CAST_RESTRICTION]),
    (r"can cast only", [Mechanic.CAST_RESTRICTION]),
    (r"would draw.+instead", [Mechanic.DRAW_REPLACEMENT, Mechanic.REPLACEMENT_EFFECT]),
    (r"can't enter the battlefield", [Mechanic.GRAVEYARD_HATE]),
    (r"if a card.+would be put into a graveyard.+exile", [Mechanic.GRAVEYARD_HATE, Mechanic.REPLACEMENT_EFFECT]),

    # =========================================================================
    # WORD-CONSUMING PATTERNS (no mechanics, just improve confidence)
    # =========================================================================
    # These patterns match common MTG phrases that follow variable effects.
    # They don't emit mechanics but consume words so confidence scoring
    # correctly reflects that this text has been understood.
    (r"where\s+\w+\s+is\s+the\s+number\s+of", []),
    (r"equal\s+to\s+the\s+number\s+of", []),
    (r"equal\s+to\s+(its|that\s+creature's|that\s+card's)\s+(power|toughness|mana\s+value|converted\s+mana\s+cost)", []),
    (r"for\s+each\s+(creature|land|artifact|enchantment|permanent|card|instant|sorcery|spell|player|opponent|color)", []),
    (r"(creatures?|lands?|artifacts?|enchantments?|permanents?|cards?)\s+(you\s+control|in\s+your\s+graveyard|in\s+your\s+hand)", []),
    (r"(creatures?|lands?|artifacts?|enchantments?|permanents?|cards?)\s+your\s+opponents?\s+controls?", []),
    (r"(you\s+control|your\s+opponents?\s+controls?)", []),
    (r"in\s+(your|the)\s+(graveyard|hand|library|exile)", []),
    (r"that\s+(died|entered|left)\s+this\s+turn", []),
    (r"among\s+(creatures|permanents|cards)", []),
    (r"(any|another|each|a|that)\s+target", []),
    (r"in\s+addition\s+to\s+its\s+other\s+types?", []),
    (r"put\s+a\s+\+1/\+1\s+counter\s+on", []),
    (r"activate\s+(this\s+ability\s+)?only", []),
    (r"this\s+(creature|spell|card|permanent)\s+(enters|costs?)", []),
    (r"\{t\}:\s+add\s+\{?[wubrgc]\}?", []),
    (r"would\s+gain\s+life.+loses\s+that\s+much\s+life", [Mechanic.CANT_GAIN_LIFE, Mechanic.REPLACEMENT_EFFECT]),
]


# =============================================================================
# PARSER
# =============================================================================

def strip_reminder_text(text: str) -> str:
    """Strip parenthetical reminder text from oracle text.

    Reminder text in MTG cards appears in parentheses and explains mechanics
    but doesn't represent parseable abilities. Stripping it before word
    counting prevents artificial confidence deflation.

    Example: "Flying (This creature can't be blocked except by creatures
    with flying or reach.)" -> "Flying "
    """
    return re.sub(r'\([^)]*\)', '', text)


@dataclass
class ParseResult:
    """Result of parsing card text."""
    mechanics: List[Mechanic]
    parameters: Dict[str, Any]
    confidence: float  # 0-1, how confident we are in the parse
    unparsed_text: str  # Text we couldn't parse (for LLM fallback)


def parse_oracle_text(oracle_text: str, card_type: str = "") -> ParseResult:
    """
    Parse oracle text into mechanics sequence.

    Args:
        oracle_text: The card's oracle text
        card_type: The card's type line (e.g., "Instant", "Creature — Human Wizard")

    Returns:
        ParseResult with mechanics, parameters, and confidence
    """
    text = oracle_text.lower()
    text = strip_reminder_text(text)
    mechanics = []
    parameters = {}
    matched_spans = []

    # Determine base timing from card type
    card_type_lower = card_type.lower()
    if "instant" in card_type_lower:
        mechanics.append(Mechanic.INSTANT_SPEED)
    elif "sorcery" in card_type_lower:
        mechanics.append(Mechanic.SORCERY_SPEED)
    elif any(t in card_type_lower for t in ["creature", "artifact", "enchantment", "planeswalker"]):
        # Permanents can have static/triggered/activated abilities
        if "when" in text or "whenever" in text or "at the beginning" in text:
            mechanics.append(Mechanic.TRIGGERED_ABILITY)
        if "{" in text and ":" in text:  # Activated ability pattern
            mechanics.append(Mechanic.ACTIVATED_ABILITY)

    # Check for keyword abilities
    for keyword, mechanic in KEYWORD_ABILITIES.items():
        # Match whole word
        pattern = r'(?<!with )(?<!without )\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text):
            mechanics.append(mechanic)
            matched_spans.append(keyword)

    # Extract numeric parameters from keyword abilities
    # Ward N
    ward_match = re.search(r'ward\s*\{(\d+)\}', text)
    if not ward_match:
        ward_match = re.search(r'ward\s+(\d+)', text)
    if ward_match:
        parameters["ward_cost"] = int(ward_match.group(1))
    # Toxic N
    toxic_match = re.search(r'toxic\s+(\d+)', text)
    if toxic_match:
        parameters["toxic_count"] = int(toxic_match.group(1))
    # Annihilator N
    annih_match = re.search(r'annihilator\s+(\d+)', text)
    if annih_match:
        parameters["annihilator_count"] = int(annih_match.group(1))
    # Kicker {N} or kicker cost
    kicker_match = re.search(r'kicker\s*\{(\d+)\}', text)
    if kicker_match:
        parameters["kicker_cost"] = int(kicker_match.group(1))

    # Apply text patterns
    for pattern, mechs in PATTERNS:
        match = re.search(pattern, text)
        if match:
            for m in mechs:
                if m not in mechanics:
                    mechanics.append(m)
            matched_spans.append(match.group())

            # Extract numeric parameters
            word_to_num = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
                           "four": 4, "five": 5, "six": 6, "seven": 7,
                           "x": "x"}
            numbers = re.findall(r'\d+', match.group())
            # Also check for word numbers in the match
            for word, val in word_to_num.items():
                if word in match.group().split() and not numbers:
                    numbers = [str(val)]
                    break
            if numbers:
                if Mechanic.DRAW in mechs or Mechanic.DRAW_OPTIONAL in mechs:
                    parameters["draw_count"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.DEAL_DAMAGE in mechs:
                    parameters["damage"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.CREATE_TOKEN in mechs:
                    parameters["token_count"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.SCRY in mechs:
                    parameters["scry_count"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.MILL in mechs:
                    parameters["mill_count"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.SURVEIL in mechs:
                    parameters["surveil_count"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.LOSE_LIFE in mechs:
                    parameters["life_loss"] = int(numbers[0]) if numbers[0] != 'x' else 'x'
                elif Mechanic.PLUS_ONE_COUNTER in mechs:
                    # Try to extract counter count from surrounding text
                    counter_match = re.search(r'(\d+)\s+\+1/\+1 counter', match.group())
                    if counter_match:
                        parameters["counter_count"] = int(counter_match.group(1))

            # Extract stat modifiers from +X/+Y or -X/-Y patterns
            stat_match = re.search(r'gets?\s+([+-])(\d+)/([+-])(\d+)', match.group())
            if stat_match:
                p_sign = 1 if stat_match.group(1) == '+' else -1
                t_sign = 1 if stat_match.group(3) == '+' else -1
                parameters["power_mod"] = p_sign * int(stat_match.group(2))
                parameters["toughness_mod"] = t_sign * int(stat_match.group(4))

    # Saga chapter parsing (use original text for roman numeral case matching)
    if "saga" in card_type_lower or re.search(r'^[IV]+\s*[—–\-]', oracle_text, re.MULTILINE):
        if Mechanic.SAGA not in mechanics:
            mechanics.append(Mechanic.SAGA)
        chapter_map = {"I": Mechanic.CHAPTER_I, "II": Mechanic.CHAPTER_II,
                       "III": Mechanic.CHAPTER_III, "IV": Mechanic.CHAPTER_IV}
        chapters = re.findall(r'^(I{1,3}|IV)\s*[—–\-]\s*(.+)$', oracle_text, re.MULTILINE)
        chapter_count = 0
        for chapter_num, chapter_text in chapters:
            if chapter_num in chapter_map:
                if chapter_map[chapter_num] not in mechanics:
                    mechanics.append(chapter_map[chapter_num])
                chapter_count += 1
                # Parse chapter text for sub-effects (non-recursive via patterns)
                chapter_text_lower = chapter_text.lower()
                for pattern, mechs in PATTERNS:
                    sub_match = re.search(pattern, chapter_text_lower)
                    if sub_match:
                        for m in mechs:
                            if m not in mechanics:
                                mechanics.append(m)
                        matched_spans.append(sub_match.group())
        if chapter_count > 0:
            parameters["chapter_count"] = chapter_count

    # Life gain parameter extraction
    life_gain_match = re.search(r'gains?\s+(\d+)\s+life', text)
    if not life_gain_match:
        life_gain_match = re.search(r'you gain\s+(\d+)\s+life', text)
    if life_gain_match:
        parameters["life_gain"] = int(life_gain_match.group(1))

    # Saddle parameter extraction
    saddle_match = re.search(r'saddle\s+(\d+)', text)
    if saddle_match:
        parameters["saddle_power"] = int(saddle_match.group(1))

    # Collect evidence parameter extraction
    evidence_match = re.search(r'collect evidence\s+(\d+)', text)
    if evidence_match:
        parameters["collect_evidence_mv"] = int(evidence_match.group(1))

    # Impending parameter extraction
    impending_match = re.search(r'impending\s+(\d+)', text)
    if impending_match:
        parameters["impending_cost"] = int(impending_match.group(1))

    # Equip cost parameter extraction
    equip_match = re.search(r'equip\s*\{(\d+)\}', text)
    if not equip_match:
        equip_complex = re.search(r'equip\s*(\{[^}]+\})+', text)
        if equip_complex:
            symbols = re.findall(r'\{([^}]+)\}', equip_complex.group())
            parameters["equip_cost"] = sum(int(s) if s.isdigit() else 1 for s in symbols)
    if equip_match:
        parameters["equip_cost"] = int(equip_match.group(1))

    # Becomes creature P/T parameter extraction
    becomes_match = re.search(r'becomes?\s+a?\s*(\d+)/(\d+)', text)
    if becomes_match:
        parameters["becomes_power"] = int(becomes_match.group(1))
        parameters["becomes_toughness"] = int(becomes_match.group(2))

    # Base power/toughness (auras like Darksteel Mutation, Unable to Scream)
    base_pt_match = re.search(r'base power and toughness (\d+)/(\d+)', text)
    if base_pt_match:
        parameters["set_power"] = int(base_pt_match.group(1))
        parameters["set_toughness"] = int(base_pt_match.group(2))

    # Crew power parameter extraction
    crew_match = re.search(r'crew\s+(\d+)', text)
    if crew_match:
        parameters["crew_power"] = int(crew_match.group(1))

    # Planeswalker loyalty ability parsing
    if "planeswalker" in card_type_lower:
        if Mechanic.LOYALTY_COUNTER not in mechanics:
            mechanics.append(Mechanic.LOYALTY_COUNTER)

        # Match: +N:, −N: (en-dash U+2212), -N: (hyphen), 0:
        loyalty_pattern = r'^([+\-\u2212])(\d+)\s*:\s*(.+)$'
        abilities = re.findall(loyalty_pattern, oracle_text, re.MULTILINE)

        loyalty_count = 0
        for sign, number, ability_text in abilities:
            loyalty_count += 1
            n = int(number)
            if sign == '+':
                if Mechanic.LOYALTY_PLUS not in mechanics:
                    mechanics.append(Mechanic.LOYALTY_PLUS)
            elif n == 0:
                if Mechanic.LOYALTY_ZERO not in mechanics:
                    mechanics.append(Mechanic.LOYALTY_ZERO)
            else:  # minus (hyphen or en-dash)
                if Mechanic.LOYALTY_MINUS not in mechanics:
                    mechanics.append(Mechanic.LOYALTY_MINUS)

            # Sub-parse ability text for effects (like saga chapters)
            ability_text_lower = ability_text.lower()
            for pattern, mechs in PATTERNS:
                sub_match = re.search(pattern, ability_text_lower)
                if sub_match:
                    for m in mechs:
                        if m not in mechanics:
                            mechanics.append(m)
                    matched_spans.append(sub_match.group())

        # Also handle bare "0:" (no sign prefix)
        zero_pattern = r'^0\s*:\s*(.+)$'
        zero_abilities = re.findall(zero_pattern, oracle_text, re.MULTILINE)
        for ability_text in zero_abilities:
            loyalty_count += 1
            if Mechanic.LOYALTY_ZERO not in mechanics:
                mechanics.append(Mechanic.LOYALTY_ZERO)
            ability_text_lower = ability_text.lower()
            for pattern, mechs in PATTERNS:
                sub_match = re.search(pattern, ability_text_lower)
                if sub_match:
                    for m in mechs:
                        if m not in mechanics:
                            mechanics.append(m)
                    matched_spans.append(sub_match.group())

        # Detect static abilities (lines with no +/-/0 prefix, not empty)
        for line in oracle_text.split('\n'):
            stripped_line = line.strip()
            if (stripped_line
                    and not re.match(r'^[+\-\u2212]\d+\s*:', stripped_line)
                    and not re.match(r'^0\s*:', stripped_line)):
                if Mechanic.LOYALTY_STATIC not in mechanics:
                    mechanics.append(Mechanic.LOYALTY_STATIC)
                break

        if loyalty_count > 0:
            parameters["loyalty_ability_count"] = loyalty_count

    # Calculate confidence based on how much text we parsed
    # Reminder text was already stripped at the top of this function
    stripped_text = text
    total_words = len(stripped_text.split())
    parsed_words = sum(len(span.split()) for span in matched_spans)
    if total_words == 0:
        confidence = 1.0  # No text to parse (e.g., vanilla creature)
    else:
        confidence = max(0.05, min(1.0, parsed_words / total_words))

    # Find unparsed text (for LLM fallback)
    # Use stripped text (no reminder text) as the baseline
    unparsed = stripped_text
    for span in matched_spans:
        unparsed = unparsed.replace(span, "")
    unparsed = " ".join(unparsed.split())  # Clean up whitespace

    return ParseResult(
        mechanics=mechanics,
        parameters=parameters,
        confidence=confidence,
        unparsed_text=unparsed if len(unparsed) > 10 else ""
    )


def parse_card(card_data: Dict[str, Any]) -> CardEncoding:
    """
    Parse a Scryfall card object into CardEncoding.

    Args:
        card_data: Scryfall API response for a single card

    Returns:
        CardEncoding object
    """
    # Handle double-faced/split cards
    if "card_faces" in card_data and card_data.get("card_faces"):
        # Use the front face for primary encoding
        front_face = card_data["card_faces"][0]
        name = card_data.get("name", front_face.get("name", "Unknown"))
        oracle_text = front_face.get("oracle_text", "")
        type_line = front_face.get("type_line", card_data.get("type_line", ""))
        mana_cost = front_face.get("mana_cost", "")

        # Also parse back face and combine mechanics
        back_face = card_data["card_faces"][1] if len(card_data["card_faces"]) > 1 else {}
        back_oracle = back_face.get("oracle_text", "")
        oracle_text = oracle_text + "\n" + back_oracle if back_oracle else oracle_text
    else:
        # Normal single-faced card
        name = card_data.get("name", "Unknown")
        oracle_text = card_data.get("oracle_text", "")
        type_line = card_data.get("type_line", "")
        mana_cost = card_data.get("mana_cost", "")

    cmc = int(card_data.get("cmc", 0))

    # Parse mana cost
    mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    # Single symbols: {W}, {U}, {B}, {R}, {G}, {C}, {1}, {2}, etc.
    for symbol in re.findall(r'\{(\w)\}', mana_cost):
        if symbol in mana_dict:
            mana_dict[symbol] += 1
        elif symbol.isdigit():
            mana_dict["C"] += int(symbol)

    # Hybrid mana: {W/U}, {B/G}, etc. — count each color
    for hybrid in re.findall(r'\{(\w)/(\w)\}', mana_cost):
        c1, c2 = hybrid
        if c1 in mana_dict and c1 != 'P':
            mana_dict[c1] += 1
        if c2 in mana_dict and c2 != 'P':
            mana_dict[c2] += 1

    # X costs: {X}
    x_count = mana_cost.count('{X}')
    parameters = {}
    if x_count:
        parameters["x_cost_count"] = x_count

    # Parse types (handle various dash characters and // for split cards)
    types = []
    subtypes = []
    # Remove any back-face type line (after //)
    if "//" in type_line:
        type_line = type_line.split("//")[0].strip()
    # Handle em-dash, en-dash, and regular dash
    for dash in ["—", "–", "-"]:
        if dash in type_line:
            parts = type_line.split(dash, 1)
            if len(parts) == 2:
                main_types, sub_types = parts
                types = [t.strip().lower() for t in main_types.split() if t.strip()]
                subtypes = [t.strip().lower() for t in sub_types.split() if t.strip()]
                break
    if not types:
        types = [t.strip().lower() for t in type_line.split() if t.strip()]

    # Parse oracle text
    result = parse_oracle_text(oracle_text, type_line)

    # Get power/toughness if creature
    power = None
    toughness = None
    if "creature" in types:
        power_str = card_data.get("power", "0")
        toughness_str = card_data.get("toughness", "0")
        try:
            power = int(power_str)
            toughness = int(toughness_str)
        except ValueError:
            # Handle * or X
            power = -1 if power_str in ["*", "X"] else 0
            toughness = -1 if toughness_str in ["*", "X"] else 0

    # Merge mana-parsed parameters with oracle text parameters
    merged_params = result.parameters.copy()
    merged_params.update(parameters)

    return CardEncoding(
        name=name,
        mana_cost=mana_dict,
        cmc=cmc,
        types=types,
        subtypes=subtypes,
        mechanics=result.mechanics,
        parameters=merged_params,
        power=power,
        toughness=toughness,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with some example cards
    test_cards = [
        {
            "name": "Saw in Half",
            "mana_cost": "{2}{B}",
            "cmc": 3,
            "type_line": "Instant",
            "oracle_text": "Destroy target creature. If that creature dies this way, its controller creates two tokens that are copies of that creature, except their power is half that creature's power and their toughness is half that creature's toughness. Round up each time."
        },
        {
            "name": "Rhystic Study",
            "mana_cost": "{2}{U}",
            "cmc": 3,
            "type_line": "Enchantment",
            "oracle_text": "Whenever an opponent casts a spell, you may draw a card unless that player pays {1}."
        },
        {
            "name": "Deflecting Swat",
            "mana_cost": "{2}{R}",
            "cmc": 3,
            "type_line": "Instant",
            "oracle_text": "If you control a commander, you may cast this spell without paying its mana cost.\nYou may choose new targets for target spell or ability."
        },
        {
            "name": "Mulldrifter",
            "mana_cost": "{4}{U}",
            "cmc": 5,
            "type_line": "Creature — Elemental",
            "oracle_text": "Flying\nWhen Mulldrifter enters the battlefield, draw two cards.\nEvoke {2}{U}",
            "power": "2",
            "toughness": "2"
        },
        {
            "name": "Starfield Vocalist",
            "mana_cost": "{3}{U}",
            "cmc": 4,
            "type_line": "Creature — Human Bard",
            "oracle_text": "Warp {1}{U}\nIf a permanent entering the battlefield causes a triggered ability of a permanent you control to trigger, that ability triggers an additional time.",
            "power": "2",
            "toughness": "4"
        },
    ]

    print("=" * 70)
    print("CARD PARSER TEST")
    print("=" * 70)

    for card_data in test_cards:
        encoding = parse_card(card_data)
        print(f"\n{encoding.name}:")
        print(f"  Type: {encoding.types}")
        print(f"  Cost: CMC {encoding.cmc}")
        print(f"  Mechanics ({len(encoding.mechanics)}):")
        for m in encoding.mechanics:
            print(f"    - {m.name}")
        if encoding.parameters:
            print(f"  Parameters: {encoding.parameters}")

    print("\n" + "=" * 70)
    print("Parser handles common patterns. Complex text falls back to LLM.")
    print("=" * 70)
