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
from typing import List, Dict, Any
from dataclasses import dataclass

# Import our vocabulary
try:
    from src.mechanics.vocabulary import Mechanic, CardEncoding
except ImportError:
    from vocabulary import Mechanic, CardEncoding


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
    "madness": Mechanic.MADNESS,
    "ninjutsu": Mechanic.NINJUTSU,
    "web-slinging": Mechanic.NINJUTSU,   # SPM — alt cost, return tapped creature to hand
    "sneak": Mechanic.NINJUTSU,           # TMT — alt cost, return unblocked attacker to hand
    "buyback": Mechanic.BUYBACK,
    "overload": Mechanic.OVERLOAD,
    "kicker": Mechanic.KICKER,
    "multikicker": Mechanic.MULTIKICKER,

    # Typecycling variants (land types → TUTOR_LAND, creature types → CREATURE_TYPE_MATTERS)
    "mountaincycling": Mechanic.CYCLING,
    "forestcycling": Mechanic.CYCLING,
    "islandcycling": Mechanic.CYCLING,
    "plainscycling": Mechanic.CYCLING,
    "swampcycling": Mechanic.CYCLING,
    "basic landcycling": Mechanic.CYCLING,
    "artifact landcycling": Mechanic.CYCLING,
    "wizardcycling": Mechanic.CYCLING,
    "slivercycling": Mechanic.CYCLING,

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
    "populate": Mechanic.POPULATE,
    "detain": Mechanic.DETAIN,
    "eminence": Mechanic.EMINENCE,

    # Dinosaur/Ixalan
    "enrage": Mechanic.DAMAGE_RECEIVED_TRIGGER,
    "explore": Mechanic.EXPLORE,

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
    "partner with": Mechanic.PARTNER_WITH,
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

    # Feature gap keywords
    "unearth": Mechanic.UNEARTH,
    "amass": Mechanic.AMASS,
    "channel": Mechanic.CHANNEL,
    "ascend": Mechanic.ASCEND,
    "level up": Mechanic.LEVEL_UP,
    "replicate": Mechanic.REPLICATE,
    "scavenge": Mechanic.SCAVENGE,
}


# =============================================================================
# KEYWORD COST TAXONOMY
# =============================================================================
# Maps keywords from KEYWORD_ABILITIES to their cost category.
# When a keyword is detected, we also fire the appropriate cost enum.
# See docs/MTG_RULES_REFERENCE.md for full rules classification.

KEYWORD_COST_CATEGORY = {
    # --- ADDITIONAL COSTS (Rule 118.8) ---
    # Paid ON TOP of the mana cost
    "kicker": Mechanic.ADDITIONAL_COST,
    "multikicker": Mechanic.ADDITIONAL_COST,
    "buyback": Mechanic.ADDITIONAL_COST,
    "entwine": Mechanic.ADDITIONAL_COST,
    "exploit": Mechanic.ADDITIONAL_COST,
    "casualty": Mechanic.ADDITIONAL_COST,
    "splice": Mechanic.ADDITIONAL_COST,
    "retrace": Mechanic.ADDITIONAL_COST,       # Cast from GY + discard land
    "jump-start": Mechanic.ADDITIONAL_COST,    # Cast from GY + discard card

    # --- ALTERNATIVE COSTS (Rule 118.9) ---
    # Replaces the mana cost entirely
    "flashback": Mechanic.ALTERNATIVE_COST,
    "overload": Mechanic.ALTERNATIVE_COST,
    "madness": Mechanic.ALTERNATIVE_COST,
    "evoke": Mechanic.ALTERNATIVE_COST,
    "dash": Mechanic.ALTERNATIVE_COST,
    "blitz": Mechanic.ALTERNATIVE_COST,
    "bestow": Mechanic.ALTERNATIVE_COST,
    "escape": Mechanic.ALTERNATIVE_COST,
    "disturb": Mechanic.ALTERNATIVE_COST,
    "foretell": Mechanic.ALTERNATIVE_COST,
    "miracle": Mechanic.ALTERNATIVE_COST,
    "morph": Mechanic.ALTERNATIVE_COST,
    "disguise": Mechanic.ALTERNATIVE_COST,
    "spectacle": Mechanic.ALTERNATIVE_COST,
    "ninjutsu": Mechanic.ALTERNATIVE_COST,
    "web-slinging": Mechanic.ALTERNATIVE_COST,
    "sneak": Mechanic.ALTERNATIVE_COST,
    "cleave": Mechanic.ALTERNATIVE_COST,
    "emerge": Mechanic.ALTERNATIVE_COST,
    "mutate": Mechanic.ALTERNATIVE_COST,
    "warp": Mechanic.ALTERNATIVE_COST,
    "plot": Mechanic.ALTERNATIVE_COST,

    # --- COST REDUCTION (not additional or alternative per rules) ---
    "convoke": Mechanic.REDUCE_COST,
    "delve": Mechanic.REDUCE_COST,
    "affinity": Mechanic.REDUCE_COST,
    "improvise": Mechanic.REDUCE_COST,
    "undaunted": Mechanic.REDUCE_COST,
    "assist": Mechanic.REDUCE_COST,

    # Feature gap cost categories
    "replicate": Mechanic.ADDITIONAL_COST,
}

# =============================================================================
# KEYWORD IMPLIED EFFECTS
# =============================================================================
# Maps keywords to mechanical effects hidden in their reminder text.
# Since strip_reminder_text() removes ALL parenthesized text before parsing,
# these effects would otherwise be invisible to the parser.
# Only includes high-value signals — effects the network can learn from.

KEYWORD_IMPLICATIONS = {
    # --- Additional cost implied effects ---
    "buyback": [Mechanic.TO_HAND],              # Return spell to hand instead of graveyard
    "exploit": [Mechanic.SACRIFICE],             # May sacrifice a creature on ETB
    "casualty": [Mechanic.SACRIFICE],            # Sacrifice creature with power N+
    "retrace": [Mechanic.DISCARD, Mechanic.CAST_FROM_GRAVEYARD],
    "jump-start": [Mechanic.DISCARD, Mechanic.CAST_FROM_GRAVEYARD],

    # --- Alternative cost implied effects ---
    "flashback": [Mechanic.CAST_FROM_GRAVEYARD],
    "escape": [Mechanic.CAST_FROM_GRAVEYARD],
    "disturb": [Mechanic.CAST_FROM_GRAVEYARD, Mechanic.TRANSFORM],
    "dash": [Mechanic.HASTE],                    # Gains haste, returns to hand at end step
    "blitz": [Mechanic.HASTE, Mechanic.DRAW],    # Haste + draw on death + sacrifice at end step
    "evoke": [Mechanic.SACRIFICE],               # Sacrifice on ETB when evoked
    "emerge": [Mechanic.SACRIFICE],              # Sacrifice creature as part of cost
    "ninjutsu": [Mechanic.BOUNCE_TO_HAND],       # Return unblocked attacker to hand
    "web-slinging": [Mechanic.BOUNCE_TO_HAND],   # Return tapped creature to hand
    "sneak": [Mechanic.BOUNCE_TO_HAND],           # Return unblocked attacker to hand
    "foretell": [Mechanic.CAST_FROM_EXILE],      # Exile face-down, cast later
    "plot": [Mechanic.CAST_FROM_EXILE],           # Exile, cast free on later turn
    "warp": [Mechanic.CAST_FROM_EXILE],           # Exile, recast later
    "madness": [Mechanic.DISCARD],               # Triggered by being discarded

    # --- Non-cost keywords with important implied effects ---
    "undying": [Mechanic.DEATH_TRIGGER, Mechanic.PLUS_ONE_COUNTER],
    "persist": [Mechanic.DEATH_TRIGGER, Mechanic.MINUS_ONE_COUNTER],
    "modular": [Mechanic.PLUS_ONE_COUNTER, Mechanic.DEATH_TRIGGER],
    "storm": [Mechanic.COPY_SPELL],
    "cascade": [Mechanic.FREE_CAST_CONDITION],
    "dredge": [Mechanic.MILL, Mechanic.REGROWTH],
    "encore": [Mechanic.CREATE_TOKEN, Mechanic.CAST_FROM_GRAVEYARD],
    "cycling": [Mechanic.DRAW, Mechanic.DISCARD],
    # Typecycling — all share DRAW+DISCARD from cycling, land variants also tutor
    "mountaincycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "forestcycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "islandcycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "plainscycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "swampcycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "basic landcycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "artifact landcycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.TUTOR_LAND],
    "wizardcycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.CREATURE_TYPE_MATTERS],
    "slivercycling": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.CREATURE_TYPE_MATTERS],
    "morph": [Mechanic.FACE_DOWN_MATTERS],
    "manifest": [Mechanic.FACE_DOWN_MATTERS],
    "disguise": [Mechanic.FACE_DOWN_MATTERS],
    "infect": [Mechanic.POISON_COUNTER, Mechanic.MINUS_ONE_COUNTER],
    "wither": [Mechanic.MINUS_ONE_COUNTER],
    "living weapon": [Mechanic.CREATE_TOKEN, Mechanic.EQUIP],
    "devour": [Mechanic.SACRIFICE, Mechanic.PLUS_ONE_COUNTER],
    "fabricate": [Mechanic.PLUS_ONE_COUNTER, Mechanic.CREATE_TOKEN],
    "extort": [Mechanic.GAIN_LIFE, Mechanic.LOSE_LIFE],
    "annihilator": [Mechanic.SACRIFICE],
    "mentor": [Mechanic.PLUS_ONE_COUNTER],
    "training": [Mechanic.PLUS_ONE_COUNTER],
    "evolve": [Mechanic.PLUS_ONE_COUNTER],
    "riot": [Mechanic.HASTE, Mechanic.PLUS_ONE_COUNTER],
    "adapt": [Mechanic.PLUS_ONE_COUNTER],
    "monstrosity": [Mechanic.PLUS_ONE_COUNTER],
    "renown": [Mechanic.PLUS_ONE_COUNTER],
    "support": [Mechanic.PLUS_ONE_COUNTER],
    "bolster": [Mechanic.PLUS_ONE_COUNTER],
    "backup": [Mechanic.PLUS_ONE_COUNTER],
    "decayed": [Mechanic.SACRIFICE],             # Sacrifice after attacking
    "afflict": [Mechanic.LOSE_LIFE],             # Opponent loses N when this is blocked
    "battle cry": [Mechanic.ANTHEM_EFFECT],       # Attacking creatures get +1/+0
    "exalted": [Mechanic.PLUS_POWER, Mechanic.PLUS_TOUGHNESS],
    "prowess": [Mechanic.PLUS_POWER, Mechanic.PLUS_TOUGHNESS],
    "flanking": [Mechanic.MINUS_POWER, Mechanic.MINUS_TOUGHNESS],
    "incubate": [Mechanic.CREATE_TOKEN, Mechanic.PLUS_ONE_COUNTER],
    "suspend": [Mechanic.CAST_FROM_EXILE, Mechanic.HASTE],
    "discover": [Mechanic.FREE_CAST_CONDITION],
    "learn": [Mechanic.DRAW, Mechanic.DISCARD],  # Draw a Lesson or loot
    "transmute": [Mechanic.TUTOR_TO_HAND, Mechanic.DISCARD],
    "forecast": [Mechanic.FROM_HAND],
    "converge": [Mechanic.MANA_FIXING],
    "goad": [Mechanic.ATTACKS_EACH_COMBAT],
    "provoke": [Mechanic.MUST_BE_BLOCKED],
    "populate": [Mechanic.CREATE_TOKEN],
    "detain": [Mechanic.CANT_ATTACK, Mechanic.CANT_BLOCK],
    "partner with": [Mechanic.TUTOR_TO_HAND],

    # Feature gap implications
    "unearth": [Mechanic.REANIMATE, Mechanic.SACRIFICE, Mechanic.HASTE],
    "amass": [Mechanic.CREATE_TOKEN, Mechanic.PLUS_ONE_COUNTER],
    "channel": [Mechanic.ACTIVATED_ABILITY, Mechanic.DISCARD, Mechanic.FROM_HAND],
    "ascend": [Mechanic.THRESHOLD_CONDITION],
    "level up": [Mechanic.ACTIVATED_ABILITY, Mechanic.PLUS_ONE_COUNTER],
    "replicate": [Mechanic.COPY_SPELL, Mechanic.ADDITIONAL_COST],
    "scavenge": [Mechanic.FROM_GRAVEYARD, Mechanic.PLUS_ONE_COUNTER, Mechanic.EXILE],
}


# =============================================================================
# TOKEN IMPLIED EFFECTS
# =============================================================================
# Maps token types to their inherent mechanical effects.
# When a card creates/mentions these tokens, the token's built-in abilities
# are added so the network understands what the token actually does.
# (Token abilities are in reminder text which gets stripped.)

TOKEN_IMPLICATIONS = {
    "food": [Mechanic.GAIN_LIFE, Mechanic.SACRIFICE],           # {2}, {T}, Sacrifice: Gain 3 life
    "clue": [Mechanic.DRAW, Mechanic.SACRIFICE],                # {2}, Sacrifice: Draw a card
    "treasure": [Mechanic.ADD_MANA, Mechanic.SACRIFICE],    # Sacrifice: Add one mana of any color
    "blood": [Mechanic.DRAW, Mechanic.DISCARD, Mechanic.SACRIFICE],  # {1}, {T}, Discard, Sacrifice: Draw
    "map": [Mechanic.EXPLORE, Mechanic.SACRIFICE],              # {1}, {T}, Sacrifice: Target creature explores
    "powerstone": [Mechanic.ADD_MANA],                      # {T}: Add {C} (can't cast nonartifact)
    "incubator": [Mechanic.CREATE_TOKEN, Mechanic.TRANSFORM],   # {2}: Transform into Phyrexian token
    "shard": [Mechanic.ADD_MANA, Mechanic.SACRIFICE],       # {2}, Sacrifice: Add 3 mana of one color
    "gold": [Mechanic.ADD_MANA, Mechanic.SACRIFICE],        # Sacrifice: Add one mana of any color
    "junk": [Mechanic.DRAW, Mechanic.SACRIFICE],                # {2}, {T}, Sacrifice: Draw (same as Clue)
}


# =============================================================================
# CREATURE TYPE DATABASE (from Scryfall catalog API)
# =============================================================================
# Complete list of all MTG creature types. Used for context-filtered detection
# of tribal/creature-type-matters cards. All map to CREATURE_TYPE_MATTERS.
# This is a reference database, NOT individual features in the vocabulary.
# Source: https://api.scryfall.com/catalog/creature-types (302 types)

ALL_CREATURE_TYPES = {
    "advisor", "aetherborn", "alien", "ally", "angel", "antelope", "ape",
    "archer", "archon", "armadillo", "army", "artificer", "assassin",
    "assembly-worker", "astartes", "atog", "aurochs", "automaton", "avatar",
    "azra", "badger", "balloon", "barbarian", "bard", "basilisk", "bat",
    "bear", "beast", "beaver", "beeble", "beholder", "berserker", "bird",
    "bison", "blinkmoth", "boar", "brainiac", "bringer", "brushwagg",
    "c'tan", "camarid", "camel", "capybara", "caribou", "carrier", "cat",
    "centaur", "chicken", "child", "chimera", "citizen", "cleric", "clown",
    "cockatrice", "construct", "coward", "coyote", "crab", "crocodile",
    "custodes", "cyberman", "cyclops", "dalek", "dauthi", "demigod",
    "demon", "deserter", "detective", "devil", "dinosaur", "djinn",
    "doctor", "dog", "dragon", "drake", "dreadnought", "drix", "drone",
    "druid", "dryad", "dwarf", "echidna", "efreet", "egg", "elder",
    "eldrazi", "elemental", "elephant", "elf", "elk", "employee", "eye",
    "faerie", "ferret", "fish", "flagbearer", "fox", "fractal", "frog",
    "fungus", "gamer", "gargoyle", "germ", "giant", "gith", "glimmer",
    "gnoll", "gnome", "goat", "goblin", "god", "golem", "gorgon",
    "graveborn", "gremlin", "griffin", "guest", "hag", "halfling",
    "hamster", "harpy", "head", "hedgehog", "hellion", "hero", "hippo",
    "hippogriff", "homarid", "homunculus", "horror", "horse", "human",
    "hydra", "hyena", "illusion", "imp", "incarnation", "inkling",
    "inquisitor", "insect", "jackal", "jellyfish", "juggernaut",
    "kangaroo", "kavu", "kirin", "kithkin", "knight", "kobold", "kor",
    "kraken", "lamia", "lammasu", "leech", "lemur", "leviathan",
    "lhurgoyf", "licid", "lizard", "llama", "lobster", "manticore",
    "masticore", "mercenary", "merfolk", "metathran", "minion", "minotaur",
    "mite", "mole", "monger", "mongoose", "monk", "monkey", "moogle",
    "moonfolk", "mount", "mouse", "mutant", "myr", "mystic", "naga",
    "nautilus", "necron", "nephilim", "nightmare", "nightstalker", "ninja",
    "noble", "noggle", "nomad", "nymph", "octopus", "ogre", "ooze", "orb",
    "orc", "orgg", "otter", "ouphe", "ox", "oyster", "pangolin", "peasant",
    "pegasus", "pentavite", "performer", "pest", "phelddagrif", "phoenix",
    "phyrexian", "pilot", "pincher", "pirate", "plant", "platypus",
    "porcupine", "possum", "praetor", "primarch", "prism", "processor",
    "qu", "rabbit", "raccoon", "ranger", "rat", "rebel", "reflection",
    "reveler", "rhino", "rigger", "robot", "rogue", "rukh", "sable",
    "salamander", "samurai", "sand", "saproling", "satyr", "scarecrow",
    "scientist", "scion", "scorpion", "scout", "sculpture", "seal", "serf",
    "serpent", "servo", "shade", "shaman", "shapeshifter", "shark", "sheep",
    "siren", "skeleton", "skrull", "skunk", "slith", "sliver", "sloth",
    "slug", "snail", "snake", "soldier", "soltari", "sorcerer", "spawn",
    "specter", "spellshaper", "sphinx", "spider", "spike", "spirit",
    "splinter", "sponge", "spy", "squid", "squirrel", "starfish",
    "surrakar", "survivor", "symbiote", "synth", "teddy", "tentacle",
    "tetravite", "thalakos", "thopter", "thrull", "tiefling", "time lord",
    "toy", "treefolk", "trilobite", "triskelavite", "troll", "turtle",
    "tyranid", "unicorn", "urzan", "utrom", "vampire", "varmint",
    "vedalken", "villain", "volver", "wall", "walrus", "warlock",
    "warrior", "weasel", "weird", "werewolf", "whale", "wizard", "wolf",
    "wolverine", "wombat", "worm", "wraith", "wurm", "yeti", "zombie",
    "zubera",
    # Plural forms for common types (oracle text uses both)
    "elves",  # "Elves you control" (elf → elves)
}

# Pre-compile tribal context regex for creature type detection.
# Only fires CREATURE_TYPE_MATTERS when a creature type appears in a
# genuinely tribal context, NOT just in token creation text.
# Examples that SHOULD match: "Dragon spells", "whenever a Goblin enters"
# Examples that should NOT match: "create a 1/1 green Elf Warrior creature token"
_CREATURE_TYPE_PATTERN_STR = "|".join(
    re.escape(t) for t in sorted(ALL_CREATURE_TYPES, key=len, reverse=True)
)
TRIBAL_CONTEXT_PATTERNS = [
    # "[Type] spell(s)" — "Dragon spells", "Goblin spell"
    re.compile(r"\b(" + _CREATURE_TYPE_PATTERN_STR + r")\s+spells?\b"),
    # "[Type] card(s)" — "search for a Dragon card"
    re.compile(r"\b(" + _CREATURE_TYPE_PATTERN_STR + r")\s+cards?\b"),
    # "cast a/an [Type]" — "whenever you cast a Dragon"
    re.compile(r"cast\s+(?:a|an)\s+(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
    # "whenever a/an [Type]" — tribal triggers
    re.compile(r"whenever\s+(?:a|an|another)\s+(?:\w+\s+)?(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
    # "each [Type]" — "each Goblin gets +1/+1"
    re.compile(r"each\s+(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
    # "number of [Type](s)" — "number of Zombies"
    re.compile(r"number\s+of\s+(" + _CREATURE_TYPE_PATTERN_STR + r")s?\b"),
    # "target [Type]" — type-specific targeting
    re.compile(r"target\s+(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
    # "sacrifice a/an [Type]" — tribal cost
    re.compile(r"sacrifice\s+(?:a|an)\s+(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
    # "[Type] creatures you control" — broader lord pattern
    re.compile(r"\b(" + _CREATURE_TYPE_PATTERN_STR + r")\s+creatures?\s+you\s+control"),
    # "[Type] you control" — already caught by existing pattern, but reinforces
    re.compile(r"\b(" + _CREATURE_TYPE_PATTERN_STR + r")s?\s+you\s+control"),
    # "is a [Type]" / "becomes a [Type]" — type assignment
    re.compile(r"(?:is|becomes?)\s+(?:a|an)\s+(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
    # "choose a [Type]" / "chosen [Type]" — type selection
    re.compile(r"(?:choose|chosen|named)\s+(?:a\s+)?(" + _CREATURE_TYPE_PATTERN_STR + r")\b"),
]


# =============================================================================
# NAMED MECHANIC UNDERLYING EFFECTS
# =============================================================================
# Maps ability words (named mechanics) to their actual underlying effects.
# These names appear in oracle text but their meaning isn't parseable from the name.
# "Commit a crime" = target something an opponent controls/owns → TARGET_OPPONENT
# "Celebration" = two+ nonland permanents entered this turn → ETB_TRIGGER
# We prefer mapping to underlying effects so set-unique names transfer knowledge.

NAMED_MECHANIC_EFFECTS = {
    "commit a crime": [Mechanic.TARGET_OPPONENT],               # OTJ — target opponent's stuff
    "celebration": [Mechanic.ETB_TRIGGER],                      # WOE — two+ nonland permanents ETB'd
    "raid": [Mechanic.ATTACK_TRIGGER],                          # KTK — attacked with a creature this turn
    "revolt": [Mechanic.LTB_TRIGGER],                           # AER — permanent left battlefield
    "morbid": [Mechanic.DEATH_TRIGGER],                         # ISD — a creature died this turn
    "eerie": [Mechanic.ETB_TRIGGER],                            # DSK — enchantment/room enters
    "survival": [Mechanic.DEATH_TRIGGER],                       # DSK — two+ creatures died this turn
    "valiant": [Mechanic.TARGET_CREATURE],                      # BLB — targeted 2+ times
    "landfall": [Mechanic.ETB_TRIGGER],                         # ZEN — land entered under your control
    "constellation": [Mechanic.ETB_TRIGGER],                    # JOU — enchantment enters
    "heroic": [Mechanic.TARGET_CREATURE],                       # THS — you cast a spell targeting this
    "magecraft": [Mechanic.COPY_SPELL],                         # STX — you cast/copy an instant/sorcery
    "corrupted": [Mechanic.POISON_COUNTER],                     # ONE — opponent has 3+ poison
    "domain": [Mechanic.COLOR_CONDITION],                       # INV — count basic land types
    "threshold": [Mechanic.FROM_GRAVEYARD],                     # ODY — 7+ cards in graveyard
    "delirium": [Mechanic.FROM_GRAVEYARD],                      # SOI — 4+ card types in graveyard
    "metalcraft": [Mechanic.TARGET_ARTIFACT],                   # SOM — control 3+ artifacts
    "ferocious": [Mechanic.POWER_TOUGHNESS_CONDITION],          # KTK — control creature with power 4+
    "hellbent": [Mechanic.HAND_SIZE_MATTERS],                   # DIS — no cards in hand
    "descend 4": [Mechanic.FROM_GRAVEYARD],                     # LCI — 4+ permanent cards in GY
    "descend 8": [Mechanic.FROM_GRAVEYARD],                     # LCI — 8+ permanent cards in GY
}


# =============================================================================
# TEXT PATTERN MATCHING
# =============================================================================

# Generic number word pattern — used in draw, token, discard, mill patterns
NUM = r"(?:a|an|\d+|x|one|two|three|four|five|six|seven|eight|nine|ten)"

# Patterns for common text structures
PATTERNS = [
    # Targeting patterns (compound BEFORE specific, opponent-specific BEFORE generic)
    (r"any target", [Mechanic.TARGET_ANY, Mechanic.TARGET_CREATURE, Mechanic.TARGET_PLAYER, Mechanic.TARGET_PLANESWALKER]),
    (r"target creature or planeswalker", [Mechanic.TARGET_CREATURE, Mechanic.TARGET_PLANESWALKER]),
    (r"target (artifact or enchantment|enchantment or artifact)", [Mechanic.TARGET_ARTIFACT, Mechanic.TARGET_ENCHANTMENT]),
    (r"target creature or enchantment", [Mechanic.TARGET_CREATURE, Mechanic.TARGET_ENCHANTMENT]),
    (r"target creature an opponent controls", [Mechanic.TARGET_CREATURE, Mechanic.TARGET_OPPONENT_CREATURE]),
    (r"creature an opponent controls", [Mechanic.TARGET_OPPONENT_CREATURE]),
    (r"target creature you don't control", [Mechanic.TARGET_CREATURE, Mechanic.TARGET_OPPONENT_CREATURE]),
    (r"target creature you control", [Mechanic.TARGET_CREATURE, Mechanic.TARGET_YOU_CONTROL]),
    (r"target (permanent|artifact|enchantment) you control", [Mechanic.TARGET_PERMANENT, Mechanic.TARGET_YOU_CONTROL]),
    (r"target creature", [Mechanic.TARGET_CREATURE]),
    (r"target planeswalker", [Mechanic.TARGET_PLANESWALKER]),
    (r"target player", [Mechanic.TARGET_PLAYER]),
    (r"target opponent", [Mechanic.TARGET_OPPONENT]),
    (r"target .{0,30}(artifact|creature|permanent|enchantment).+opponent controls", [Mechanic.TARGET_PERMANENT, Mechanic.TARGET_OPPONENT_PERMANENT]),
    (r"(artifact|creature|permanent|enchantment) an opponent controls", [Mechanic.TARGET_OPPONENT_PERMANENT]),
    (r"(creatures?|permanents?|artifacts?|enchantments?|lands?) your opponents? controls?", [Mechanic.TARGET_OPPONENT_CONTROLS]),
    (r"target permanent", [Mechanic.TARGET_PERMANENT]),
    (r"target spell", [Mechanic.TARGET_SPELL]),
    (r"target artifact", [Mechanic.TARGET_ARTIFACT]),
    (r"target enchantment", [Mechanic.TARGET_ENCHANTMENT]),
    (r"target land", [Mechanic.TARGET_LAND]),
    (r"target (spell or ability|ability)", [Mechanic.TARGET_SPELL_OR_ABILITY]),
    (r"target card (in|from) .{0,25}graveyard", [Mechanic.TARGET_CARD_IN_GRAVEYARD]),
    (r"each creature", [Mechanic.TARGETS_EACH, Mechanic.TARGET_CREATURE]),
    (r"each opponent", [Mechanic.TARGETS_EACH, Mechanic.TARGET_OPPONENT]),
    (r"all creatures", [Mechanic.TARGETS_ALL, Mechanic.TARGET_CREATURE]),
    (r"each player", [Mechanic.TARGETS_EACH, Mechanic.TARGET_PLAYER]),

    # Removal patterns
    (r"destroy (another )?(up to \w+ )?(target|all|each)", [Mechanic.DESTROY]),
    (r"destroys? (it|that|this|target)", [Mechanic.DESTROY]),
    (r"exile target", [Mechanic.EXILE]),
    (r"exiles? (it|that|this|target)", [Mechanic.EXILE]),
    (r"sacrifice (a|an|target)", [Mechanic.SACRIFICE]),
    (r"return (it|that|target)(?!.*from .*(graveyard|exile)).+to (its|their|your|owner's) (hand|owner's hand)", [Mechanic.BOUNCE_TO_HAND]),
    (r"put.+on (the bottom|top) of.+library", [Mechanic.BOUNCE_TO_LIBRARY]),
    (r"counter target spell", [Mechanic.COUNTER_SPELL]),
    (r"counter (it|that spell)", [Mechanic.COUNTER_SPELL]),
    (r"counter target.+ability", [Mechanic.COUNTER_ABILITY]),
    (r"deals? (\d+|x) damage", [Mechanic.DEAL_DAMAGE]),
    # Damage doublers/triplers (Gratuitous Violence, Fiery Emancipation)
    (r"would deal damage.+deals? double that damage", [Mechanic.DAMAGE_DOUBLER, Mechanic.REPLACEMENT_EFFECT]),
    (r"would deal damage.+deals? triple that damage", [Mechanic.DAMAGE_DOUBLER, Mechanic.REPLACEMENT_EFFECT]),
    (r"deals? double.{0,20}damage instead", [Mechanic.DAMAGE_DOUBLER, Mechanic.REPLACEMENT_EFFECT]),
    (r"deals? triple.{0,20}damage instead", [Mechanic.DAMAGE_DOUBLER, Mechanic.REPLACEMENT_EFFECT]),
    (r"loses? (\d+|x) life", [Mechanic.LOSE_LIFE]),
    (r"you lose life equal to", [Mechanic.LOSE_LIFE]),

    # Creation patterns
    (r"create(s)? " + NUM + r" .*?(token|treasure|food|clue|blood)", [Mechanic.CREATE_TOKEN]),
    (r"create(s)?.+cop(y|ies) of", [Mechanic.CREATE_TOKEN_COPY]),
    (r"create(s)?.+treasure token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_TREASURE]),
    (r"create(s)?.+food token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_FOOD]),
    (r"create(s)?.+clue token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_CLUE]),
    (r"create(s)?.+blood token", [Mechanic.CREATE_TOKEN, Mechanic.CREATE_BLOOD]),
    (r"\binvestigate\b", [Mechanic.CREATE_CLUE]),

    # Card advantage
    (r"draw(s)? (" + NUM + r" cards?|a card)", [Mechanic.DRAW]),
    (r"may draw (" + NUM + r" cards?|a card)", [Mechanic.DRAW_OPTIONAL]),
    (r"scry (\d+|x)", [Mechanic.SCRY]),
    (r"surveil (\d+|x)", [Mechanic.SURVEIL]),
    (r"look at the top", [Mechanic.LOOK_AT_TOP]),
    (r"reveal", [Mechanic.REVEAL]),
    (r"search your library", [Mechanic.TUTOR_TO_HAND]),
    (r"search your library.+put.+onto the battlefield", [Mechanic.TUTOR_TO_BATTLEFIELD]),
    (r"return.+from.+graveyard to the battlefield", [Mechanic.REANIMATE]),
    (r"return.+from.+graveyard to your hand", [Mechanic.REGROWTH]),
    (r"discard(s)? (" + NUM + r" cards?|a card|your hand|that card|it|them)", [Mechanic.DISCARD]),
    (r"mills?\s+" + NUM + r"\s+cards?", [Mechanic.MILL]),
    (r"you may cast.+from.+graveyard", [Mechanic.CAST_FROM_GRAVEYARD]),
    (r"cast .+from (your|a|the) graveyard", [Mechanic.CAST_FROM_GRAVEYARD]),
    (r"play .+from (your|a|the) graveyard", [Mechanic.CAST_FROM_GRAVEYARD]),

    # Loot / Rummage (draw+discard combo)
    (r"draw.{1,20}then.{1,10}discard", [Mechanic.LOOT]),
    (r"discard.{1,20}then.{1,10}draw", [Mechanic.LOOT]),

    # Triggers
    (r"when(ever)? .+ enters( the battlefield)?", [Mechanic.ETB_TRIGGER]),
    (r"when(ever)? .+ leaves( the battlefield)?", [Mechanic.LTB_TRIGGER]),
    (r"when(ever)? .+ dies", [Mechanic.DEATH_TRIGGER]),
    (r"is put into a graveyard from the battlefield", [Mechanic.DEATH_TRIGGER]),
    (r"when(ever)? .+ attacks?", [Mechanic.ATTACK_TRIGGER]),
    (r"when(ever)? .+ blocks", [Mechanic.BLOCK_TRIGGER]),
    (r"when(ever)? .+ becomes? blocked", [Mechanic.BLOCK_TRIGGER]),
    (r"when(ever)? .+ deals (combat )?damage", [Mechanic.DAMAGE_TRIGGER]),
    (r"when(ever)? .+ is dealt damage", [Mechanic.DAMAGE_RECEIVED_TRIGGER]),
    (r"when(ever)? you cast", [Mechanic.CAST_TRIGGER]),
    (r"when(ever)? an opponent casts", [Mechanic.OPPONENT_CASTS]),
    (r"at the beginning of your upkeep", [Mechanic.UPKEEP_TRIGGER]),
    (r"at the beginning of combat on your turn", [Mechanic.BEGINNING_OF_COMBAT_TRIGGER]),
    (r"at the beginning of (each|your) end step", [Mechanic.END_STEP_TRIGGER]),
    (r"when(ever)? you draw", [Mechanic.DRAW_TRIGGER]),
    (r"when(ever)? you gain or lose life", [Mechanic.GAIN_LIFE_TRIGGER, Mechanic.LOSE_LIFE_TRIGGER]),
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
    (r"was dealt damage this turn", [Mechanic.DEALT_DAMAGE_CONDITION]),
    (r"if you control (\w+) or more", [Mechanic.THRESHOLD_CONDITION]),
    (r"(for each|of that|of the chosen|from .{0,10}) color", [Mechanic.COLOR_CONDITION]),

    # Stats modification (digits or X)
    (r"gets? \+([1-9]\d*|x)/\+([1-9]\d*|x)", [Mechanic.PLUS_POWER, Mechanic.PLUS_TOUGHNESS]),
    (r"gets? -([1-9]\d*|x)/-([1-9]\d*|x)", [Mechanic.MINUS_POWER, Mechanic.MINUS_TOUGHNESS]),
    (r"gets? \+(\d+|x)/\+0", [Mechanic.PLUS_POWER]),
    (r"gets? \+0/\+(\d+|x)", [Mechanic.PLUS_TOUGHNESS]),
    (r"gets? -(\d+|x)/-0", [Mechanic.MINUS_POWER]),
    (r"gets? -0/-(\d+|x)", [Mechanic.MINUS_TOUGHNESS]),
    (r"(other )?(creatures|permanents) you control get \+", [Mechanic.ANTHEM_EFFECT]),
    (r"\+1/\+1 counter", [Mechanic.PLUS_ONE_COUNTER]),
    (r"-1/-1 counter", [Mechanic.MINUS_ONE_COUNTER]),
    (r"half.+(power|toughness)", [Mechanic.HALF_STATS]),
    (r"double.+(power|toughness)", [Mechanic.DOUBLE_STATS]),

    # Mana
    (r"add \{", [Mechanic.ADD_MANA]),
    (r"add (one|two|three|\d+) mana", [Mechanic.ADD_MANA]),
    (r"mana of any (color|type)", [Mechanic.MANA_OF_ANY_COLOR, Mechanic.MANA_FIXING]),
    # Fetch lands searching for two land types → MANA_FIXING
    (r"search.{0,40}(plains|island|swamp|mountain|forest).{0,15}or\s+(plains|island|swamp|mountain|forest)", [Mechanic.MANA_FIXING]),
    # "tap for any color" variant phrasing
    (r"tap.{0,20}for mana of any color", [Mechanic.MANA_FIXING, Mechanic.ADD_MANA]),
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
    (r"(?:return|put).{0,80}under your control", [Mechanic.GAIN_CONTROL]),  # theft, not "entered under your control"
    (r"crew\s+\d+", [Mechanic.CREW]),
    (r"exile the top.+(you|they|that player) may (play|cast)", [Mechanic.IMPULSE_DRAW]),
    (r"exile.+from the top.+(you|they|that player) may (play|cast)", [Mechanic.IMPULSE_DRAW]),
    (r"exile the top.+until end of turn", [Mechanic.IMPULSE_DRAW]),
    (r"exile.+cards?.+(you|they|that player) may (play|cast) (those|them|that)", [Mechanic.IMPULSE_DRAW]),
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
    (r"onto the battlefield tapped", [Mechanic.TO_BATTLEFIELD_TAPPED]),

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
    (r"spree\b", [Mechanic.SPREE, Mechanic.ADDITIONAL_COST]),
    (r"offspring\b", [Mechanic.OFFSPRING, Mechanic.ADDITIONAL_COST, Mechanic.CREATE_TOKEN]),
    (r"\beerie\b", [Mechanic.EERIE]),
    (r"\bsurvival\b", [Mechanic.SURVIVAL]),
    (r"impending\s+\d+", [Mechanic.IMPENDING, Mechanic.ALTERNATIVE_COST]),
    (r"\bbargain\b", [Mechanic.BARGAIN, Mechanic.ADDITIONAL_COST, Mechanic.SACRIFICE]),
    (r"\bcelebrat(e|ion)\b", [Mechanic.CELEBRATION]),
    (r"\brole\b.+\btoken\b", [Mechanic.ROLE_TOKEN]),
    (r"\bcase\b", [Mechanic.CASE]),
    (r"\bsuspect\b", [Mechanic.SUSPECT]),
    (r"\bcloak\b", [Mechanic.CLOAK, Mechanic.FACE_DOWN_MATTERS]),
    (r"\bconnive[sd]?\b", [Mechanic.CONNIVE, Mechanic.DRAW, Mechanic.DISCARD]),
    (r"collect evidence\s+\d+", [Mechanic.COLLECT_EVIDENCE, Mechanic.ADDITIONAL_COST]),
    (r"commit(ted|s)? a crime", [Mechanic.COMMIT_A_CRIME]),
    (r"saddle\s+\d+", [Mechanic.SADDLE]),
    (r"gift a\b", [Mechanic.GIFT, Mechanic.ADDITIONAL_COST]),
    (r"\bdescend 4\b", [Mechanic.DESCEND_4]),
    (r"\bdescend 8\b", [Mechanic.DESCEND_8]),
    (r"fathomless descent", [Mechanic.FATHOMLESS_DESCENT]),
    (r"\bmap token", [Mechanic.MAP_TOKEN]),
    (r"\bvaliant\b", [Mechanic.VALIANT]),
    (r"\boutlaw\b", [Mechanic.OUTLAW]),
    (r"\bescalate\b", [Mechanic.ESCALATE, Mechanic.ADDITIONAL_COST]),
    (r"\btiered\b", [Mechanic.ESCALATE, Mechanic.ADDITIONAL_COST]),
    (r"\bcycling\b", [Mechanic.CYCLING, Mechanic.DRAW, Mechanic.DISCARD]),
    (r"\bexert\b", [Mechanic.EXERT]),
    (r"play an additional land", [Mechanic.EXTRA_LAND_PLAY]),
    (r"\bvoid\b\s*—", [Mechanic.VOID]),
    (r"\broom\b", [Mechanic.ROOM]),
    (r"\bbehold\b", [Mechanic.REVEAL, Mechanic.CREATURE_TYPE_MATTERS]),

    # Face-down mechanics — morph/manifest/disguise/cloak text references
    (r"face[- ]down", [Mechanic.FACE_DOWN_MATTERS]),
    (r"face[- ]up", [Mechanic.FACE_DOWN_MATTERS]),
    (r"turned? face[- ]up", [Mechanic.FACE_DOWN_MATTERS]),
    (r"is turned face[- ]down", [Mechanic.FACE_DOWN_MATTERS]),

    # Land tutoring / ramp
    (r"search.{0,30}(for|your library).{0,30}basic land", [Mechanic.TUTOR_LAND]),
    (r"search.{0,30}(for|your library).{0,30}(land card|forest|plains|island|swamp|mountain)", [Mechanic.TUTOR_LAND]),

    # Your-turn condition
    (r"if it's your turn", [Mechanic.YOUR_TURN_CONDITION]),
    (r"during your turn", [Mechanic.YOUR_TURN_CONDITION]),
    (r"on your turn", [Mechanic.YOUR_TURN_CONDITION]),
    (r"only during your turn", [Mechanic.YOUR_TURN_CONDITION]),

    # Spacecraft → word-consuming (artifact creature subtype, no special mechanic)
    (r"\bspacecraft\b", []),

    # Unnamed cost patterns (not keyword-specific)
    (r"as an additional cost.+discard", [Mechanic.ADDITIONAL_COST, Mechanic.DISCARD]),
    (r"as an additional cost.+pay.+life", [Mechanic.ADDITIONAL_COST, Mechanic.PAY_LIFE]),
    (r"as an additional cost.+exile", [Mechanic.ADDITIONAL_COST, Mechanic.EXILE]),
    (r"as an additional cost.+tap", [Mechanic.ADDITIONAL_COST, Mechanic.TAP]),
    (r"rather than pay (this spell's|its) mana cost", [Mechanic.ALTERNATIVE_COST]),
    (r"you may pay .+ rather than", [Mechanic.ALTERNATIVE_COST]),
    (r"you may pay \{[^}]+\}\.\s*if you do", []),  # word-consuming: conditional pay

    # =========================================================================
    # QUIZ ROUND 2 DESIGN DECISIONS
    # =========================================================================

    # Creature type matters / tribal synergy — "[type]s you control" where [type] is not generic
    (r"\b(?:other )?(?!creatures?\b|permanents?\b|artifacts?\b|enchantments?\b|lands?\b|spells?\b|cards?\b|tokens?\b|players?\b|nonland|noncreature|nontoken|legendary|it\b|they\b|them\b|that\b|this\b|those\b|each\b|all\b|target\b)\w+s? you control", [Mechanic.CREATURE_TYPE_MATTERS]),
    (r"if you control (?:a|an) (?!creature|permanent|artifact|enchantment|land|spell|card|token|player|nonland|noncreature)\w+", [Mechanic.CREATURE_TYPE_MATTERS]),

    # Dies → return to battlefield (self-recurring, NOT persist/undying keyword)
    (r"when .{1,30} dies, return .{1,30} to the battlefield", [Mechanic.DIES_TO_BATTLEFIELD]),
    (r"when .{1,30} dies, put .{1,30} onto the battlefield", [Mechanic.DIES_TO_BATTLEFIELD]),

    # Finality counter
    (r"finality counter", [Mechanic.FINALITY_COUNTER]),

    # Once per turn restriction
    (r"this ability triggers only once each turn", [Mechanic.ONCE_PER_TURN]),
    (r"activate .{0,20}only once each turn", [Mechanic.ONCE_PER_TURN]),
    (r"activate .{0,30}only once\b", [Mechanic.ONCE_PER_TURN]),  # exhaust reminder text
    (r"\bexhaust\b.{0,5}—", [Mechanic.ONCE_PER_TURN, Mechanic.ACTIVATED_ABILITY]),  # exhaust keyword ability
    (r"activate.+only as a sorcery", [Mechanic.SORCERY_SPEED]),

    # Pay life as cost/condition
    (r"pay (\d+|x) life", [Mechanic.PAY_LIFE]),
    (r"pays? (\d+|x) life", [Mechanic.PAY_LIFE]),

    # Toughness-matters — Assault Formation, Doran, defenders
    (r"equal to .{0,30}toughness", [Mechanic.TOUGHNESS_MATTERS]),
    (r"total toughness", [Mechanic.TOUGHNESS_MATTERS]),
    (r"toughness among", [Mechanic.TOUGHNESS_MATTERS]),
    (r"with the greatest toughness", [Mechanic.TOUGHNESS_MATTERS]),
    (r"with the least toughness", [Mechanic.TOUGHNESS_MATTERS]),
    (r"assign .{0,20}toughness", [Mechanic.TOUGHNESS_MATTERS]),
    (r"toughness rather than.{0,10}power", [Mechanic.TOUGHNESS_MATTERS]),
    (r"deals? combat damage equal to its toughness", [Mechanic.TOUGHNESS_MATTERS]),

    # Power/toughness condition (stat gate)
    (r"(?:power|toughness) \d+ or (?:greater|more|less)", [Mechanic.POWER_TOUGHNESS_CONDITION]),
    (r"with (?:power|toughness) \d+ or (?:greater|more|less)", [Mechanic.POWER_TOUGHNESS_CONDITION]),
    (r"(?:power|toughness) (?:equal to|less than|greater than)", [Mechanic.POWER_TOUGHNESS_CONDITION]),

    # Effect multiplier — Doubling Season, Parallel Lives, Ojer Taq
    (r"(twice|double|two times) that many", [Mechanic.EFFECT_MULTIPLIER]),
    (r"(three|four|five|\d+) times that many", [Mechanic.EFFECT_MULTIPLIER]),
    (r"(twice|double) the number of", [Mechanic.EFFECT_MULTIPLIER]),
    (r"tokens? (are|is) created instead", [Mechanic.EFFECT_MULTIPLIER]),

    # Mana value condition — "mana value N or less/more"
    (r"mana value.{0,10}(\d+|x) or (less|fewer|more|greater)", [Mechanic.MANA_VALUE_CONDITION]),
    (r"mana value.{0,10}(less than|greater than|equal to)", [Mechanic.MANA_VALUE_CONDITION]),
    (r"(converted mana cost|mana value) (is |was )?\d+", [Mechanic.MANA_VALUE_CONDITION]),

    # Hand size matters — "for each card in your hand"
    (r"for each card in (your|their|a player's) hand", [Mechanic.HAND_SIZE_MATTERS]),
    (r"cards? in (your|their) hand", [Mechanic.HAND_SIZE_MATTERS]),
    (r"(equal to|less than|greater than) the number of cards in", [Mechanic.HAND_SIZE_MATTERS]),

    # Grants ability — lord/equipment granting keywords to others
    (r"\w+s? you control (have|gain) ", [Mechanic.GRANTS_ABILITY]),
    (r"equipped creature (has|gains) ", [Mechanic.GRANTS_ABILITY]),
    (r"as though (they|it) had ", [Mechanic.GRANTS_ABILITY]),

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
    (r"charge counter", [Mechanic.CHARGE_COUNTER]),
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
    # Forced combat patterns
    (r"attacks each (combat|turn) if able", [Mechanic.ATTACKS_EACH_COMBAT, Mechanic.MUST_ATTACK]),
    (r"must attack\b", [Mechanic.MUST_ATTACK, Mechanic.ATTACKS_EACH_COMBAT]),
    (r"must be blocked\b", [Mechanic.MUST_BE_BLOCKED]),
    (r"blocks?.{0,20}if able", [Mechanic.MUST_BE_BLOCKED]),
    (r"all creatures.{0,20}attack each", [Mechanic.ATTACKS_EACH_COMBAT]),
    (r"is goaded", [Mechanic.GOAD, Mechanic.ATTACKS_EACH_COMBAT]),
    (r"goaded", [Mechanic.GOAD, Mechanic.ATTACKS_EACH_COMBAT]),
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
    (r"shuffles?.{0,30}graveyard into.{0,15}library", [Mechanic.GRAVEYARD_SHUFFLE, Mechanic.GRAVEYARD_HATE]),

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
    (r"cycling\s+(\{[^}]+\})+", []),
    (r"whenever\s+you\s+cycle", [Mechanic.DISCARD_TRIGGER]),
    (r"destroy\s+all\s+(creatures|permanents|artifacts|enchantments)", [Mechanic.DESTROY, Mechanic.TARGETS_ALL]),
    (r"(tap|untap)\s+all\s+", []),
    (r"under\s+(your|its\s+owner's)\s+control", []),
    (r"(its|their)\s+owner's\s+hand", []),
    (r"any\s+(number|combination)\s+of", []),
    (r"whenever\s+you\s+tap\s+a\s+\w+\s+for\s+mana", [Mechanic.MANA_DOUBLER]),

    # Explore (LCI, BLB, etc.)
    (r"it\s+explores", [Mechanic.EXPLORE]),
    (r"explores?\b", [Mechanic.EXPLORE]),

    # Leylines
    (r"if this card is in your opening hand.+begin the game with it on the battlefield", [Mechanic.LEYLINE]),

    # DSK Rooms
    (r"(fully )?unlock(s|ed)?\s+(a|this)\s+room", [Mechanic.UNLOCK_ROOM]),

    # Shockland / painland entry
    (r"as\s+this\s+land\s+enters,?\s+you\s+may\s+pay\s+\d+\s+life", []),
    (r"if\s+you\s+don't,?\s+it\s+enters\s+tapped", [Mechanic.TO_BATTLEFIELD_TAPPED]),

    # Common "gains [keyword]" confidence booster
    (r"gains?\s+(flying|trample|first strike|double strike|deathtouch|lifelink|vigilance|reach|haste|menace|hexproof|indestructible)", []),
    (r"(it|that creature|this creature)\s+gains?\s+", []),

    # Fetch land patterns
    (r"sacrifice\s+this\s+(land|artifact|creature|enchantment|permanent)", [Mechanic.SACRIFICE]),
    (r"for\s+a\s+basic\s+land\s+card", []),
    (r"put\s+it\s+onto\s+the\s+battlefield\s+tapped", [Mechanic.TO_BATTLEFIELD_TAPPED]),
    (r"then\s+shuffle", []),

    # =========================================================================
    # QUICK-WIN PATTERN WIRING (zero-hit enums)
    # =========================================================================

    # Sacrifice trigger — "whenever you sacrifice" / "when you sacrifice" / "is sacrificed"
    (r"when(?:ever)? .{0,30}sacrifice", [Mechanic.SACRIFICE_TRIGGER]),
    (r"is sacrificed", [Mechanic.SACRIFICE_TRIGGER]),

    # The Ring — "the ring tempts you"
    (r"the ring tempts you", [Mechanic.THE_RING]),

    # Voting / council mechanics
    (r"\bvotes?\b", [Mechanic.VOTING]),
    (r"council's dilemma", [Mechanic.VOTING, Mechanic.COUNCIL_DILEMMA]),
    (r"will of the council", [Mechanic.VOTING, Mechanic.WILL_OF_COUNCIL]),

    # Time counters — suspend/vanishing use them
    (r"time counter", [Mechanic.TIME_COUNTER]),

    # Target up to X — "up to N target"
    (r"up to (\w+) target", [Mechanic.TARGET_UP_TO_X]),

    # Tutor to top of library — "search your library...put...on top"
    (r"search your library.{0,80}put.{0,40}on top", [Mechanic.TUTOR_TO_TOP]),
    (r"put .{0,20}on top of (your|their) library.{0,20}shuffle", [Mechanic.TUTOR_TO_TOP]),

    # Common text fragments
    (r"at\s+the\s+beginning\s+of\s+(your|the)\s+(first|second|next)\s+main\s+phase", []),
    (r"return\s+(?!.*from .*(graveyard|exile)).+?\s+to\s+(its|their)\s+owner's\s+hand", [Mechanic.BOUNCE_TO_HAND]),
    (r"you\s+may\s+(cast|play)\s+it", []),
    (r"if\s+you\s+do,?\s+", []),
    (r"attach\s+it\s+to\s+", []),

    # Feature gap patterns (top 15 implementation)
    (r"counter target .{0,30}spell", [Mechanic.COUNTER_SPELL]),
    (r"power and toughness are each equal to", [Mechanic.POWER_EQUAL_TO_X, Mechanic.SET_POWER, Mechanic.SET_TOUGHNESS]),
    (r"devotion to\s+(white|blue|black|red|green)", [Mechanic.DEVOTION, Mechanic.COLOR_CONDITION]),
    (r"additional combat phase", [Mechanic.EXTRA_COMBAT]),
    (r"choose a creature type", [Mechanic.CHOOSE_CREATURE_TYPE, Mechanic.CREATURE_TYPE_MATTERS]),
    (r"spells .{0,30}cost .{0,5}\{?\d+\}?.{0,5} more to cast", [Mechanic.INCREASE_COST]),
    (r"spells .{0,30}cost .{0,5}\{?\d+\}?.{0,5} less to cast", [Mechanic.REDUCE_COST]),
    (r"venture into the dungeon", [Mechanic.VENTURE]),
    (r"can't be blocked (except )?by\b", [Mechanic.CONDITIONAL_UNBLOCKABLE]),
    (r"each opponent sacrifices", [Mechanic.EDICT_EFFECT, Mechanic.SACRIFICE]),
    (r"each opponent discards", [Mechanic.MASS_DISCARD_OPPONENT, Mechanic.DISCARD]),
    (r"activated abilities .{0,30}can't be activated", [Mechanic.ABILITY_SHUTDOWN, Mechanic.STATIC_ABILITY]),

    # Confidence-booster patterns (consume text, no new enums)
    (r"whenever one or more", []),
    (r"protection from (white|blue|black|red|green|colorless|monocolored|multicolored|all|each|everything|instants?|sorceries|creatures?|enchantments?|artifacts?|planeswalkers?)", []),
    (r"if .{0,20}(?:was|is) kicked", []),
    (r"whenever? .{0,30}becomes? the target", []),
    (r"hexproof from\s+(?:white|blue|black|red|green|monocolored|multicolored)", []),
]

# Pre-compile all patterns at module load for ~19x speedup
# (avoids re.search recompiling 343+ patterns on every call)
PATTERNS = [(re.compile(p), m) for p, m in PATTERNS]

# Pre-compile keyword ability patterns (dynamic word-boundary patterns)
_COMPILED_KEYWORD_PATTERNS: list[tuple[re.Pattern, str, 'Mechanic']] = [
    (re.compile(r'(?<!without )(?<!lose )(?<!loses )\b' + re.escape(kw) + r'\b'), kw, mech)
    for kw, mech in KEYWORD_ABILITIES.items()
]

# Pre-compile token implication patterns
_COMPILED_TOKEN_PATTERNS: list[tuple[re.Pattern, str, list]] = [
    (re.compile(r'\b' + re.escape(token_type) + r'\b'), token_type, effects)
    for token_type, effects in TOKEN_IMPLICATIONS.items()
]

# Pre-compile named mechanic effect patterns
_COMPILED_NAMED_MECHANIC_PATTERNS: list[tuple[re.Pattern, list]] = [
    (re.compile(r'\b' + re.escape(name) + r'\b'), effects)
    for name, effects in NAMED_MECHANIC_EFFECTS.items()
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


def parse_oracle_text(oracle_text: str, card_type: str = "", card_name: str = "") -> ParseResult:
    """
    Parse oracle text into mechanics sequence.

    Args:
        oracle_text: The card's oracle text
        card_type: The card's type line (e.g., "Instant", "Creature — Human Wizard")
        card_name: The card's name (for self-reference stripping)

    Returns:
        ParseResult with mechanics, parameters, and confidence
    """
    text = oracle_text.lower()
    text = strip_reminder_text(text)

    # Strip card self-references to reduce unparsed noise
    # "Atraxa, Grand Unifier enters" → "this creature enters"
    if card_name:
        card_name_lower = card_name.lower()
        # Handle DFC names ("Front // Back")
        for name_part in card_name_lower.split(" // "):
            name_part = name_part.strip()
            if len(name_part) > 2:  # Avoid stripping tiny names like "It"
                # Determine replacement based on card type
                if "creature" in card_type.lower():
                    replacement = "this creature"
                elif "instant" in card_type.lower() or "sorcery" in card_type.lower():
                    replacement = "this spell"
                elif "planeswalker" in card_type.lower():
                    replacement = "this planeswalker"
                elif "land" in card_type.lower():
                    replacement = "this land"
                else:
                    replacement = "this permanent"
                text = text.replace(name_part, replacement)
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

    # Check for keyword abilities (pre-compiled patterns)
    for kw_pattern, keyword, mechanic in _COMPILED_KEYWORD_PATTERNS:
        match = kw_pattern.search(text)
        if match:
            # Check if keyword is in a "lose/loses" clause (e.g. "lose hexproof and indestructible")
            preceding = text[max(0, match.start() - 50):match.start()]
            if re.search(r'\bloses?\s+(?:\w+[\s,]+(?:and\s+)?)*$', preceding):
                continue
            mechanics.append(mechanic)
            matched_spans.append(keyword)

            # Add cost category (ADDITIONAL_COST / ALTERNATIVE_COST / REDUCE_COST)
            if keyword in KEYWORD_COST_CATEGORY:
                cost_cat = KEYWORD_COST_CATEGORY[keyword]
                if cost_cat not in mechanics:
                    mechanics.append(cost_cat)

            # Add implied effects (what the reminder text describes)
            if keyword in KEYWORD_IMPLICATIONS:
                for implied in KEYWORD_IMPLICATIONS[keyword]:
                    if implied not in mechanics:
                        mechanics.append(implied)

    # Fire token implied effects (Food → GAIN_LIFE+SACRIFICE, etc.)
    for token_pattern, token_type, implied_effects in _COMPILED_TOKEN_PATTERNS:
        if token_pattern.search(text):
            matched_spans.append(token_type)
            for implied in implied_effects:
                if implied not in mechanics:
                    mechanics.append(implied)

    # Fire named mechanic underlying effects (commit a crime → TARGET_OPPONENT, etc.)
    for named_pattern, underlying_effects in _COMPILED_NAMED_MECHANIC_PATTERNS:
        if named_pattern.search(text):
            for effect in underlying_effects:
                if effect not in mechanics:
                    mechanics.append(effect)

    # Creature type tribal detection (context-filtered)
    # Only fires when a creature type appears in a genuinely tribal context,
    # NOT in token creation text like "create a 1/1 green Elf Warrior creature token"
    if Mechanic.CREATURE_TYPE_MATTERS not in mechanics:
        for tribal_pattern in TRIBAL_CONTEXT_PATTERNS:
            if tribal_pattern.search(text):
                mechanics.append(Mechanic.CREATURE_TYPE_MATTERS)
                break

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

    # Apply text patterns (pre-compiled at module load)
    for pattern, mechs in PATTERNS:
        match = pattern.search(text)
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
                    # Strip P/T patterns (e.g., "3/1") to avoid confusing P/T with token count
                    match_text_no_pt = re.sub(r'\d+/\d+', '', match.group())
                    token_nums = re.findall(r'\d+', match_text_no_pt)
                    if token_nums:
                        parameters["token_count"] = int(token_nums[0]) if token_nums[0] != 'x' else 'x'
                    else:
                        # Word number like "a" or "two" — use word_to_num
                        for w, v in word_to_num.items():
                            if w in match_text_no_pt.split():
                                parameters["token_count"] = v if v != 'x' else 'x'
                                break
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
                # Check if this stat modification is variable ("for each" / "equal to")
                # Look at surrounding text since "for each" may be outside the match group
                stat_pos = text.find(stat_match.group())
                nearby_text = text[max(0, stat_pos):stat_pos + len(stat_match.group()) + 40] if stat_pos >= 0 else match.group()
                if re.search(r'(for each|equal to)', nearby_text):
                    parameters["power_mod"] = "x"
                    parameters["toughness_mod"] = "x"
                else:
                    p_val = p_sign * int(stat_match.group(2))
                    t_val = t_sign * int(stat_match.group(4))
                    if p_val != 0:
                        parameters["power_mod"] = p_val
                    if t_val != 0:
                        parameters["toughness_mod"] = t_val

    # Multi-color mana production → MANA_FIXING (dual lands, tri-lands, filter lands)
    # If card already has ADD_MANA but not MANA_FIXING, check for 2+ distinct color symbols
    if Mechanic.ADD_MANA in mechanics and Mechanic.MANA_FIXING not in mechanics:
        add_matches = re.findall(r'add\s+[^.]*', text)
        colors_in_add = set()
        for add_text in add_matches:
            colors_in_add.update(re.findall(r'\{([wubrg])\}', add_text))
        if len(colors_in_add) >= 2:
            mechanics.append(Mechanic.MANA_FIXING)

    # Saga chapter parsing (use original text for roman numeral case matching)
    if "saga" in card_type_lower or re.search(r'^[IV]+\s*[—–\-]', oracle_text, re.MULTILINE):
        if Mechanic.SAGA not in mechanics:
            mechanics.append(Mechanic.SAGA)
        chapters = re.findall(r'^(I{1,3}|IV)\s*[—–\-]\s*(.+)$', oracle_text, re.MULTILINE)
        chapter_count = 0
        for chapter_num, chapter_text in chapters:
            chapter_count += 1
            # Parse chapter text for sub-effects (pre-compiled patterns)
            chapter_text_lower = chapter_text.lower()
            for pattern, mechs in PATTERNS:
                sub_match = pattern.search(chapter_text_lower)
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

            # Sub-parse ability text for effects (pre-compiled patterns)
            ability_text_lower = ability_text.lower()
            for pattern, mechs in PATTERNS:
                sub_match = pattern.search(ability_text_lower)
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
                sub_match = pattern.search(ability_text_lower)
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

    # Emblem text sub-parsing
    # Pattern: 'you get an emblem with "..."' or 'each opponent gets an emblem with "..."'
    emblem_matches = re.finditer(
        r'(?:you|each opponent) gets? an emblem with "([^"]+)"',
        text
    )
    for emblem_match in emblem_matches:
        emblem_text = emblem_match.group(1)
        if Mechanic.CREATE_EMBLEM not in mechanics:
            mechanics.append(Mechanic.CREATE_EMBLEM)
        matched_spans.append(emblem_match.group())

        # Detect opponent-targeted emblems
        if "each opponent" in emblem_match.group():
            if Mechanic.EMBLEM_OPPONENT not in mechanics:
                mechanics.append(Mechanic.EMBLEM_OPPONENT)

        # Classify emblem ability type
        if re.search(r'^(whenever|when|at the beginning)', emblem_text):
            if Mechanic.EMBLEM_TRIGGERED not in mechanics:
                mechanics.append(Mechanic.EMBLEM_TRIGGERED)
        elif ':' in emblem_text and re.search(r'\w+.*:.*\w', emblem_text):
            if Mechanic.EMBLEM_ACTIVATED not in mechanics:
                mechanics.append(Mechanic.EMBLEM_ACTIVATED)
        else:
            if Mechanic.EMBLEM_STATIC not in mechanics:
                mechanics.append(Mechanic.EMBLEM_STATIC)

        # Sub-parse emblem text for effects (pre-compiled patterns)
        for pattern, mechs in PATTERNS:
            sub_match = pattern.search(emblem_text)
            if sub_match:
                for m in mechs:
                    if m not in mechanics:
                        mechanics.append(m)
                matched_spans.append(sub_match.group())

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
    has_x_cost = False
    if x_count:
        parameters["x_cost_count"] = x_count
        has_x_cost = True

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

    # Parse oracle text (pass name for self-reference stripping)
    result = parse_oracle_text(oracle_text, type_line, card_name=name)

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

    # Detect card layout (adventure, modal_dfc, etc.)
    layout = card_data.get("layout", "normal")
    layout_mechanics = list(result.mechanics)
    if layout == "adventure":
        if Mechanic.ADVENTURE_SPELL not in layout_mechanics:
            layout_mechanics.append(Mechanic.ADVENTURE_SPELL)
    elif layout == "modal_dfc":
        if Mechanic.MDFC not in layout_mechanics:
            layout_mechanics.append(Mechanic.MDFC)

    # Fire X_COST mechanic from mana cost {X}
    if has_x_cost and Mechanic.X_COST not in layout_mechanics:
        layout_mechanics.append(Mechanic.X_COST)

    # Merge mana-parsed parameters with oracle text parameters
    merged_params = result.parameters.copy()
    merged_params.update(parameters)

    return CardEncoding(
        name=name,
        mana_cost=mana_dict,
        cmc=cmc,
        types=types,
        subtypes=subtypes,
        mechanics=layout_mechanics,
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
