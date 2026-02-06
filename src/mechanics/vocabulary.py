"""
MTG Mechanics Vocabulary

A comprehensive vocabulary of ~150+ mechanics primitives that can represent
any Magic: The Gathering card. This is the foundation for the AlphaZero-style
card encoding system.

Design principles:
1. Primitives are ATOMIC - they represent single mechanical concepts
2. Cards are SEQUENCES of primitives
3. New mechanics = new combinations of existing primitives (usually)
4. When truly new, we add minimal new primitives

The network learns how primitives INTERACT through self-play, not pre-coding.
"""

from enum import IntEnum, auto
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class MechanicCategory(IntEnum):
    """Categories for organizing mechanics."""
    TIMING = 0
    TARGETING = 100
    REMOVAL = 200
    CREATION = 300
    CARD_ADVANTAGE = 400
    MANA_COST = 500
    TRIGGERS = 600
    CONDITIONS = 700
    COMBAT = 800
    STATS = 900
    ZONES = 1000
    COUNTERS = 1100
    KEYWORDS = 1200
    SPECIAL = 1300


class Mechanic(IntEnum):
    """
    Core mechanics vocabulary.

    Each mechanic is a primitive that can be combined to represent any card.
    The integer values are used as embedding indices.
    """

    # ==========================================================================
    # TIMING (0-99)
    # ==========================================================================
    INSTANT_SPEED = 1
    SORCERY_SPEED = 2
    FLASH = 3
    STATIC_ABILITY = 4          # Always-on effects
    ACTIVATED_ABILITY = 5       # {cost}: effect
    TRIGGERED_ABILITY = 6       # When/Whenever/At
    REPLACEMENT_EFFECT = 7      # "Instead" effects
    SPLIT_SECOND = 8

    # ==========================================================================
    # TARGETING (100-199)
    # ==========================================================================
    TARGET_CREATURE = 101
    TARGET_PLAYER = 102
    TARGET_OPPONENT = 103
    TARGET_PERMANENT = 104
    TARGET_SPELL = 105
    TARGET_ABILITY = 106
    TARGET_SPELL_OR_ABILITY = 107
    TARGET_ARTIFACT = 108
    TARGET_ENCHANTMENT = 109
    TARGET_PLANESWALKER = 110
    TARGET_LAND = 111
    TARGET_CARD_IN_GRAVEYARD = 112
    TARGET_CARD_IN_HAND = 113
    TARGET_SELF = 114           # "this creature"
    TARGET_ANY_CONTROLLER = 115  # yours or opponent's
    TARGET_YOU_CONTROL = 116
    TARGET_OPPONENT_CONTROLS = 117
    TARGETS_EACH = 118          # "each creature"
    TARGETS_ALL = 119           # "all creatures"
    TARGET_UP_TO_X = 120        # "up to X targets"
    NO_TARGET = 121             # affects without targeting

    # ==========================================================================
    # REMOVAL / INTERACTION (200-299)
    # ==========================================================================
    DESTROY = 201
    EXILE = 202
    SACRIFICE = 203
    BOUNCE_TO_HAND = 204
    BOUNCE_TO_LIBRARY = 205
    COUNTER_SPELL = 206
    COUNTER_ABILITY = 207
    PREVENT_DAMAGE = 208
    REDIRECT = 209
    CHANGE_TARGETS = 210
    PHASE_OUT = 211
    TAP = 212
    UNTAP = 213
    FIGHT = 214
    DEAL_DAMAGE = 215
    LOSE_LIFE = 216
    MILL = 217
    DISCARD = 218
    TUCK = 219                  # Put on bottom of library

    # ==========================================================================
    # CREATION (300-399)
    # ==========================================================================
    CREATE_TOKEN = 301
    CREATE_TOKEN_COPY = 302     # Token that's a copy
    COPY_SPELL = 303
    COPY_PERMANENT = 304
    CREATE_TREASURE = 305
    CREATE_FOOD = 306
    CREATE_CLUE = 307
    CREATE_BLOOD = 308
    POPULATE = 309              # Copy a token you control

    # ==========================================================================
    # CARD ADVANTAGE (400-499)
    # ==========================================================================
    DRAW = 401
    DRAW_OPTIONAL = 402         # "may draw"
    SCRY = 403
    SURVEIL = 404
    LOOK_AT_TOP = 405
    REVEAL = 406
    TUTOR_TO_HAND = 407
    TUTOR_TO_TOP = 408
    TUTOR_TO_BATTLEFIELD = 409
    IMPULSE_DRAW = 410          # Exile top, may play this turn
    REANIMATE = 411             # Return from graveyard to battlefield
    REGROWTH = 412              # Return from graveyard to hand
    CASCADE = 413
    DISCOVER = 414

    # ==========================================================================
    # MANA / COSTS (500-599)
    # ==========================================================================
    ADD_MANA = 501
    MANA_OF_ANY_COLOR = 502
    REDUCE_COST = 503
    INCREASE_COST = 504         # Tax effects
    ALTERNATIVE_COST = 505
    ADDITIONAL_COST = 506
    FREE_CAST_CONDITION = 507   # "If X, cast without paying"
    CONVOKE = 508
    DELVE = 509
    AFFINITY = 510
    IMPROVISE = 511
    TREASURE_SACRIFICE = 512
    X_COST = 513
    KICKER = 514
    MULTIKICKER = 515
    OVERLOAD = 516
    BUYBACK = 517
    FLASHBACK = 518
    RETRACE = 519
    JUMP_START = 520
    ESCAPE = 521
    FORETELL = 522
    WARP = 523                  # Edge of Eternities - alt cost, exile at end, recast later
    DASH = 524
    BLITZ = 525
    EVOKE = 526
    EMERGE = 527
    MUTATE = 528
    SPECTACLE = 529
    DISGUISE = 530
    MORPH = 531
    MANIFEST = 532
    PLOT = 533
    CRAFT = 534

    # ==========================================================================
    # TRIGGERS (600-699)
    # ==========================================================================
    ETB_TRIGGER = 601           # Enters the battlefield
    LTB_TRIGGER = 602           # Leaves the battlefield
    DEATH_TRIGGER = 603         # Dies (goes to graveyard from battlefield)
    ATTACK_TRIGGER = 604
    BLOCK_TRIGGER = 605
    DAMAGE_TRIGGER = 606        # Deals damage
    DAMAGE_RECEIVED_TRIGGER = 607
    CAST_TRIGGER = 608          # When you cast
    OPPONENT_CASTS = 609        # When opponent casts
    UPKEEP_TRIGGER = 610
    END_STEP_TRIGGER = 611
    DRAW_TRIGGER = 612          # When you draw
    DISCARD_TRIGGER = 613
    SACRIFICE_TRIGGER = 614
    GAIN_LIFE_TRIGGER = 615
    LOSE_LIFE_TRIGGER = 616
    LANDFALL = 617
    CONSTELLATION = 618
    HEROIC = 619
    MAGECRAFT = 620
    PROWESS = 621
    COMBAT_DAMAGE_TO_PLAYER = 622
    MORBID = 623                # If creature died this turn
    REVOLT = 624                # If permanent left battlefield
    VOID = 625                  # Edge of Eternities - if permanent left or warp used

    # ==========================================================================
    # CONDITIONS (700-799)
    # ==========================================================================
    IF_CONDITION = 701          # Generic conditional
    IF_TARGET_DIES = 702
    IF_YOU_CONTROL_CREATURE = 703
    IF_YOU_CONTROL_ARTIFACT = 704
    IF_YOU_CONTROL_ENCHANTMENT = 705
    IF_YOU_CONTROL_COMMANDER = 706
    IF_CREATURE_ENTERED = 707
    IF_SPELL_CAST = 708
    IF_MANA_SPENT = 709         # "If {R} was spent"
    IF_LIFE_GAINED = 710
    IF_CREATURE_DIED = 711
    IF_OPPONENT_ATTACKED = 712
    UNLESS_PAYS = 713           # "unless that player pays"
    THRESHOLD = 714             # 7+ cards in graveyard
    DELIRIUM = 715              # 4+ card types in graveyard
    METALCRAFT = 716            # 3+ artifacts
    FEROCIOUS = 717             # Control 4+ power creature
    DESCEND = 718               # Cards in graveyard count
    DOMAIN = 719                # Basic land types
    PARTY = 720                 # Cleric, Rogue, Warrior, Wizard

    # ==========================================================================
    # COMBAT (800-899)
    # ==========================================================================
    FLYING = 801
    TRAMPLE = 802
    FIRST_STRIKE = 803
    DOUBLE_STRIKE = 804
    DEATHTOUCH = 805
    LIFELINK = 806
    VIGILANCE = 807
    REACH = 808
    HASTE = 809
    MENACE = 810
    DEFENDER = 811
    INDESTRUCTIBLE = 812
    HEXPROOF = 813
    SHROUD = 814
    PROTECTION = 815
    WARD = 816
    SKULK = 817
    SHADOW = 818
    FEAR = 819
    INTIMIDATE = 820
    UNBLOCKABLE = 821
    MUST_ATTACK = 822
    CANT_ATTACK = 823
    CANT_BLOCK = 824
    ATTACKS_EACH_COMBAT = 825
    ANNIHILATOR = 826
    INFECT = 827
    WITHER = 828
    FLANKING = 829
    BANDING = 830
    HORSEMANSHIP = 831
    ABSORB = 832
    BATTLE_CRY = 833
    EXALTED = 834
    MELEE = 835
    MYRIAD = 836
    AFFLICT = 837
    GOAD = 838
    PROVOKE = 839
    RAMPAGE = 840
    BUSHIDO = 841
    NINJUTSU = 842

    # ==========================================================================
    # STATS / MODIFICATIONS (900-999)
    # ==========================================================================
    PLUS_POWER = 901
    PLUS_TOUGHNESS = 902
    MINUS_POWER = 903
    MINUS_TOUGHNESS = 904
    SET_POWER = 905             # "has base power X"
    SET_TOUGHNESS = 906
    HALF_STATS = 907            # Saw in Half
    DOUBLE_STATS = 908
    POWER_EQUALS_TOUGHNESS = 909
    POWER_EQUAL_TO_X = 910      # Power = cards in hand, etc.
    ANTHEM_EFFECT = 911         # +1/+1 to all creatures you control
    LORD_EFFECT = 912           # +1/+1 to creature type

    # ==========================================================================
    # ZONES (1000-1099)
    # ==========================================================================
    FROM_HAND = 1001
    FROM_GRAVEYARD = 1002
    FROM_EXILE = 1003
    FROM_LIBRARY = 1004
    FROM_BATTLEFIELD = 1005
    TO_HAND = 1006
    TO_GRAVEYARD = 1007
    TO_EXILE = 1008
    TO_LIBRARY_TOP = 1009
    TO_LIBRARY_BOTTOM = 1010
    TO_BATTLEFIELD = 1011
    TO_BATTLEFIELD_TAPPED = 1012
    EXILE_TEMPORARY = 1013      # Exile until end of turn / leaves
    EXILE_WITH_COUNTER = 1014   # Imprint, suspend, etc.
    CAST_FROM_EXILE = 1015      # Can cast the exiled card
    CAST_FROM_GRAVEYARD = 1016

    # ==========================================================================
    # COUNTERS (1100-1199)
    # ==========================================================================
    PLUS_ONE_COUNTER = 1101
    MINUS_ONE_COUNTER = 1102
    LOYALTY_COUNTER = 1103
    CHARGE_COUNTER = 1104
    TIME_COUNTER = 1105
    POISON_COUNTER = 1106
    EXPERIENCE_COUNTER = 1107
    ENERGY_COUNTER = 1108
    STUN_COUNTER = 1109
    SHIELD_COUNTER = 1110
    FINALITY_COUNTER = 1111
    PROLIFERATE = 1112
    MODULAR = 1113
    UNDYING = 1114
    PERSIST = 1115
    DEVOUR = 1116
    GRAFT = 1117
    EVOLVE = 1118
    ADAPT = 1119
    MONSTROSITY = 1120
    RENOWN = 1121
    SUPPORT = 1122
    BOLSTER = 1123
    FABRICATE = 1124
    TRAINING = 1125
    MENTOR = 1126
    RIOT = 1127
    BACKUP = 1128

    # ==========================================================================
    # KEYWORDS / SPECIAL (1200-1399)
    # ==========================================================================
    CHANGELING = 1201
    DEVOID = 1202
    PHASING = 1203
    SUSPEND = 1204
    VANISHING = 1205
    FADING = 1206
    CUMULATIVE_UPKEEP = 1207
    ECHO = 1208
    ENTWINE = 1209
    SPLICE = 1210
    FUSE = 1211
    AFTERMATH = 1212
    ADVENTURE = 1213
    COMPANION = 1214
    PARTNER = 1215
    PARTNER_WITH = 1216
    COMMANDER_TAX = 1217
    EMINENCE = 1218
    CIPHER = 1219
    HAUNT = 1220
    EXTORT = 1221
    DETAIN = 1222
    POPULATE_AURA = 1223
    BESTOW = 1224
    TRIBUTE = 1225
    RAID = 1226
    EXPLOIT = 1227
    DASH_HASTE = 1228
    REBOUND = 1229
    STORM = 1230
    DREDGE = 1231
    TRANSMUTE = 1232
    FORECAST = 1233
    HELLBENT = 1234
    BLOODTHIRST = 1235
    RADIANCE = 1236
    SPLICE_ARCANE = 1237
    SOULBOND = 1238
    MIRACLE = 1239
    UNDAUNTED = 1240
    ASSIST = 1241
    ENCORE = 1242
    DEMONSTATE = 1243
    CLEAVE = 1244
    CASUALTY = 1245
    BLITZ_DRAW = 1246
    CONNIVE = 1247
    HIDEAWAY = 1248
    PROTOTYPE = 1249
    LIVING_WEAPON = 1250
    RECONFIGURE = 1251
    FOR_MIRRODIN = 1252
    TOXIC = 1253
    CORRUPTED = 1254
    OIL_COUNTER = 1255
    INCUBATE = 1256
    TRANSFORM = 1257
    DAYBOUND = 1258
    NIGHTBOUND = 1259
    DISTURB = 1260
    DECAYED = 1261
    EXPLOIT_CREATURE = 1262
    COVEN = 1263
    LEARN = 1264
    WARD_COST = 1265
    MAGECRAFT_COPY = 1266
    DOUBLE_TRIGGER = 1267       # Panharmonicon effect
    TOKEN_DOUBLER = 1268        # Doubling Season effect
    DAMAGE_DOUBLER = 1269
    MANA_DOUBLER = 1270

    # Multiplayer-specific
    GOAD_CREATURE = 1301
    MONARCH = 1302
    INITIATIVE = 1303
    THE_RING = 1304
    VOTING = 1305
    COUNCIL_DILEMMA = 1306
    WILL_OF_COUNCIL = 1307
    TEMPTING_OFFER = 1308
    JOIN_FORCES = 1309
    MYRIAD_TOKENS = 1310

    # ==========================================================================
    # RECENT SET MECHANICS (1271-1295)
    # ==========================================================================
    SPREE = 1271                # OTJ - Choose modes by paying costs
    OFFSPRING = 1272            # BLB - Pay extra, create 1/1 token copy
    VALIANT = 1273              # BLB - When 2+ targets
    EERIE = 1274                # DSK - Enchantment/room enters
    SURVIVAL = 1275             # DSK - If 2+ creatures died
    IMPENDING = 1276            # DSK - Cheaper cost, enters without creature type
    ROOM = 1277                 # DSK - Door/room enchantments
    GIFT = 1278                 # FDN - May give gift to opponent
    BARGAIN = 1279              # WOE - Sacrifice artifact/enchantment/token as cost
    CELEBRATION = 1280          # WOE - Two+ nonland permanents entered
    ROLE_TOKEN = 1281           # WOE - Young Hero, Wicked, Royal, etc.
    CASE = 1282                 # MKM - Enchantments that "solve"
    SUSPECT = 1283              # MKM - Menace + can't block
    CLOAK = 1284                # MKM - Face-down 2/2, turn up for mana
    COLLECT_EVIDENCE = 1285     # MKM - Exile cards from GY with total MV
    COMMIT_A_CRIME = 1286       # OTJ - Target opponent or their stuff
    SADDLE = 1288               # OTJ - Tap creatures with total power
    OUTLAW = 1289               # OTJ - Assassin/Mercenary/Pirate/Rogue/Warlock
    DESCEND_4 = 1290            # LCI - 4+ permanent cards in GY
    DESCEND_8 = 1291            # LCI - 8+ permanent cards in GY
    FATHOMLESS_DESCENT = 1292   # LCI - Count permanents in GY
    MAP_TOKEN = 1294            # LCI - Explore token artifact
    DISCOVER_X = 1295           # LCI - Exile til cheaper nonland, cast or hand

    # ==========================================================================
    # MODAL / CHOICE (1296-1299)
    # ==========================================================================
    MODAL_CHOOSE_ONE = 1296
    MODAL_CHOOSE_TWO = 1297
    MODAL_CHOOSE_THREE = 1298
    MODAL_CHOOSE_X = 1299

    # ==========================================================================
    # TYPE FILTERS (1311-1314)
    # ==========================================================================
    FILTER_NONLAND = 1311
    FILTER_NONCREATURE = 1312
    FILTER_NONTOKEN = 1313
    FILTER_NONARTIFACT = 1314

    # ==========================================================================
    # DURATION MARKERS (1315-1318)
    # ==========================================================================
    UNTIL_END_OF_TURN = 1315
    UNTIL_YOUR_NEXT_TURN = 1316
    AS_LONG_AS = 1317
    PERPETUAL = 1318

    # ==========================================================================
    # SAGA / CHAPTER (1319-1323)
    # ==========================================================================
    SAGA = 1319
    CHAPTER_I = 1320
    CHAPTER_II = 1321
    CHAPTER_III = 1322
    CHAPTER_IV = 1323

    # ==========================================================================
    # LIFE EFFECTS (1324)
    # ==========================================================================
    GAIN_LIFE = 1324

    # ==========================================================================
    # EQUIPMENT / PLANESWALKER / TYPE CHANGE (1325-1331)
    # ==========================================================================
    EQUIP = 1325
    LOYALTY_PLUS = 1326        # Planeswalker "+N:" ability
    LOYALTY_MINUS = 1327       # Planeswalker "−N:" ability
    LOYALTY_ZERO = 1328        # Planeswalker "0:" ability
    LOYALTY_STATIC = 1329      # Planeswalker passive/static ability
    BECOMES_CREATURE = 1330    # "becomes a N/N creature" (manlands, Gideon)
    KEYWORD_COUNTER = 1331     # Flying counter, deathtouch counter, etc. (Ikoria)

    # ==========================================================================
    # STAX / HATE / RESTRICTION (1332-1337)
    # ==========================================================================
    LOSES_ABILITIES = 1332      # "loses all abilities" (Darksteel Mutation, Kenrith's Transformation)
    CANT_GAIN_LIFE = 1333       # "can't gain life" (Erebos, Sulfuric Vortex)
    CANT_CAST = 1334            # "can't be cast" / "can't cast" (Gaddock Teeg, Drannith Magistrate)
    CAST_RESTRICTION = 1335     # "can't cast more than one" (Rule of Law, Arcane Laboratory)
    DRAW_REPLACEMENT = 1336     # "If would draw...instead" (Notion Thief, Narset)
    GRAVEYARD_HATE = 1337       # "can't enter the battlefield from" / GY exile (Rest in Peace, Grafdigger's Cage)

    # ==========================================================================
    # CONTROL / VEHICLES / INTERACTION (1338-1349)
    # ==========================================================================
    GAIN_CONTROL = 1338         # "gain control of" (Control Magic, Agent of Treachery)
    CREW = 1339                 # "crew N" (vehicles: Smuggler's Copter, Esika's Chariot)
    CANT_BE_COUNTERED = 1340    # "can't be countered" (Cavern of Souls, Loxodon Smiter)
    EXTRA_TURN = 1341           # "take an extra turn" (Time Warp, Alrund's Epiphany)
    WIN_GAME = 1342             # "you win the game" (Thassa's Oracle, Lab Maniac)
    LOSE_GAME = 1343            # "loses the game" / "you lose the game"

    # ==========================================================================
    # SCRYFALL SAMPLE ROUND 1 FIXES (1344-1349)
    # ==========================================================================
    LANDWALK = 1344             # forestwalk, islandwalk, etc. (Lynx, Bog Wraith)
    COMBAT_RESTRICTION = 1345   # "no more than one creature can attack" (Silent Arbiter, Dueling Grounds)
    ENTERS_WITH_COUNTERS = 1346 # "enters with N +1/+1 counters" (Hagra Constrictor, Walking Ballista)
    COIN_FLIP = 1347            # "flip a coin" (Game of Chaos, Krark's Thumb)
    REGENERATE = 1348           # "regenerate" (Thrun, Ezuri)

    # ==========================================================================
    # STANDARD COVERAGE REPORT FIXES (1349-1351)
    # ==========================================================================
    EXPLORE = 1349              # "explores" (Jadelight Ranger, Merfolk Branchwalker)
    LEYLINE = 1350              # "in your opening hand...begin the game with it" (Leyline cycle)
    UNLOCK_ROOM = 1351          # "unlock a room" / "fully unlock" (DSK rooms)


# Total vocabulary size
VOCAB_SIZE = max(m.value for m in Mechanic) + 1


@dataclass
class CardEncoding:
    """Encoded representation of a card."""
    name: str
    mana_cost: Dict[str, int]  # {"W": 1, "U": 0, "B": 0, "R": 0, "G": 0, "C": 2}
    cmc: int
    types: List[str]  # ["creature", "artifact"]
    subtypes: List[str]  # ["human", "wizard"]
    mechanics: List[Mechanic]  # Sequence of mechanics
    parameters: Dict[str, Any]  # Numeric parameters (token count, damage amount, etc.)
    power: Optional[int] = None
    toughness: Optional[int] = None
    loyalty: Optional[int] = None


def encode_card_to_vector(card: CardEncoding, vocab_size: int = VOCAB_SIZE) -> Dict[str, Any]:
    """
    Convert a CardEncoding to a feature vector suitable for neural network input.

    Returns:
        Dict with:
        - mechanics_multihot: [vocab_size] binary vector
        - cmc: int
        - colors: [5] binary vector (WUBRG)
        - is_creature: bool
        - is_instant: bool
        - etc.
    """
    # Mechanics as multi-hot encoding
    mechanics_multihot = [0] * vocab_size
    for m in card.mechanics:
        mechanics_multihot[m.value] = 1

    # Colors
    colors = [
        1 if card.mana_cost.get("W", 0) > 0 else 0,
        1 if card.mana_cost.get("U", 0) > 0 else 0,
        1 if card.mana_cost.get("B", 0) > 0 else 0,
        1 if card.mana_cost.get("R", 0) > 0 else 0,
        1 if card.mana_cost.get("G", 0) > 0 else 0,
    ]

    # Types
    types_lower = [t.lower() for t in card.types]

    return {
        "mechanics_multihot": mechanics_multihot,
        "mechanics_sequence": [m.value for m in card.mechanics],
        "cmc": card.cmc,
        "colors": colors,
        "color_count": sum(colors),
        "is_creature": "creature" in types_lower,
        "is_instant": "instant" in types_lower,
        "is_sorcery": "sorcery" in types_lower,
        "is_artifact": "artifact" in types_lower,
        "is_enchantment": "enchantment" in types_lower,
        "is_planeswalker": "planeswalker" in types_lower,
        "is_land": "land" in types_lower,
        "power": card.power,
        "toughness": card.toughness,
        "parameters": card.parameters,
    }


# =============================================================================
# EXAMPLE ENCODINGS
# =============================================================================

# Saw in Half {2}{B}
SAW_IN_HALF = CardEncoding(
    name="Saw in Half",
    mana_cost={"B": 1, "C": 2},
    cmc=3,
    types=["instant"],
    subtypes=[],
    mechanics=[
        Mechanic.INSTANT_SPEED,
        Mechanic.TARGET_CREATURE,
        Mechanic.TARGET_ANY_CONTROLLER,
        Mechanic.DESTROY,
        Mechanic.IF_TARGET_DIES,
        Mechanic.CREATE_TOKEN_COPY,
        Mechanic.HALF_STATS,
    ],
    parameters={"token_count": 2, "stat_multiplier": 0.5, "rounding": "up"},
)

# Deflecting Swat {2}{R}
DEFLECTING_SWAT = CardEncoding(
    name="Deflecting Swat",
    mana_cost={"R": 1, "C": 2},
    cmc=3,
    types=["instant"],
    subtypes=[],
    mechanics=[
        Mechanic.INSTANT_SPEED,
        Mechanic.FREE_CAST_CONDITION,
        Mechanic.IF_YOU_CONTROL_COMMANDER,
        Mechanic.TARGET_SPELL_OR_ABILITY,
        Mechanic.CHANGE_TARGETS,
    ],
    parameters={"alternative_cost": 0},
)

# Rhystic Study {2}{U}
RHYSTIC_STUDY = CardEncoding(
    name="Rhystic Study",
    mana_cost={"U": 1, "C": 2},
    cmc=3,
    types=["enchantment"],
    subtypes=[],
    mechanics=[
        Mechanic.STATIC_ABILITY,
        Mechanic.OPPONENT_CASTS,
        Mechanic.DRAW_OPTIONAL,
        Mechanic.UNLESS_PAYS,
    ],
    parameters={"tax_amount": 1, "draw_count": 1},
)


if __name__ == "__main__":
    print(f"Mechanics Vocabulary Size: {VOCAB_SIZE}")
    print(f"Number of defined mechanics: {len(Mechanic)}")
    print()

    for card in [SAW_IN_HALF, DEFLECTING_SWAT, RHYSTIC_STUDY]:
        vec = encode_card_to_vector(card)
        print(f"{card.name}:")
        print(f"  Mechanics: {[m.name for m in card.mechanics]}")
        print(f"  Mechanic count: {sum(vec['mechanics_multihot'])}")
        print(f"  CMC: {vec['cmc']}, Colors: {vec['colors']}")
        print()
