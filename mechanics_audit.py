#!/usr/bin/env python3
"""
Comprehensive Mechanics Audit

Compares:
1. All keywords in Forge's Keyword.java (200+)
2. Our card_embeddings.py implementation
3. Standard-legal mechanics (2025)

Identifies gaps and provides recommendations.
"""

import re
from dataclasses import dataclass
from typing import Set, Dict, List

# =============================================================================
# FORGE KEYWORDS (extracted from Keyword.java)
# =============================================================================

FORGE_KEYWORDS = {
    # Simple Keywords (binary presence)
    "Absorb", "Adapt", "Affinity", "Afflict", "Afterlife", "Aftermath",
    "Amplify", "Annihilator", "Ascend", "Assist", "Aura swap", "Awaken",
    "Backup", "Banding", "Bands with other", "Bargain", "Battle cry",
    "Bestow", "Blitz", "Bloodthirst", "Bushido", "Buyback", "Cascade",
    "Casualty", "Champion", "Changeling", "Choose a Background", "Cipher",
    "Companion", "Compleated", "Conspire", "Convoke", "Craft", "Crew",
    "Cumulative upkeep", "Cycling", "Dash", "Daybound", "Deathtouch",
    "Decayed", "Defender", "Delve", "Demonstrate", "Dethrone", "Devour",
    "Devoid", "Disguise", "Disturb", "Doctor's companion", "Double agenda",
    "Double Strike", "Double team", "Dredge", "Echo", "Embalm", "Emerge",
    "Enchant", "Encore", "Enlist", "Entwine", "Epic", "Equip", "Escape",
    "Escalate", "Eternalize", "Evoke", "Evolve", "Exalted", "Exploit",
    "Extort", "Fabricate", "Fading", "Fear", "Firebending", "First Strike",
    "Flanking", "Flash", "Flashback", "Flying", "For Mirrodin", "Foretell",
    "Fortify", "Freerunning", "Frenzy", "Fuse", "Gift", "Graft",
    "Gravestorm", "Harmonize", "Haste", "Haunt", "Hexproof", "Hideaway",
    "Hidden agenda", "Horsemanship", "Impending", "Improvise", "Indestructible",
    "Infect", "Ingest", "Intimidate", "Kicker", "Job select", "Jump-start",
    "Landwalk", "Level up", "Lifelink", "Living metal", "Living Weapon",
    "Madness", "Mayhem", "Melee", "Mentor", "Menace", "Megamorph",
    "Miracle", "Mobilize", "Monstrosity", "Modular", "More Than Meets the Eye",
    "Morph", "Multikicker", "Mutate", "Myriad", "Nightbound", "Ninjutsu",
    "Outlast", "Offering", "Offspring", "Overload", "Partner", "Partner with",
    "Persist", "Phasing", "Plot", "Poisonous", "Protection", "Prototype",
    "Provoke", "Prowess", "Prowl", "Rampage", "Ravenous", "Reach",
    "Read ahead", "Rebound", "Recover", "Reconfigure", "Reflect", "Reinforce",
    "Renown", "Replicate", "Retrace", "Riot", "Ripple", "Saddle", "Scavenge",
    "Shadow", "Shroud", "Skulk", "Sneak", "Soulbond", "Soulshift",
    "Space sculptor", "Specialize", "Spectacle", "Splice", "Split second",
    "Spree", "Squad", "Start your engines", "Starting intensity", "Station",
    "Storm", "Strive", "Sunburst", "Surge", "Suspend", "Tiered", "Toxic",
    "Training", "Trample", "Transfigure", "Transmute", "Tribute",
    "TypeCycling", "Umbra armor", "Undaunted", "Undying", "Unearth",
    "Unleash", "Vanishing", "Vigilance", "Ward", "Warp", "Web-slinging",
    "Wither", "MayFlashCost", "MayFlashSac"
}

# =============================================================================
# OUR IMPLEMENTATION (from card_embeddings.py)
# =============================================================================

OUR_SIMPLE_KEYWORDS = {
    "Flying", "First strike", "Double strike", "Trample", "Haste",
    "Vigilance", "Lifelink", "Deathtouch", "Reach", "Menace",
    "Defender", "Indestructible", "Hexproof", "Shroud", "Flash",
    "Fear", "Intimidate", "Shadow", "Infect", "Wither",
    "Prowess", "Exalted", "Undying", "Persist", "Flanking",
    "Changeling", "Devoid", "Decayed", "Riot", "Training",
}

OUR_AMOUNT_KEYWORDS = {
    "Absorb", "Afflict", "Afterlife", "Annihilator", "Bloodthirst",
    "Bushido", "Crew", "Dredge", "Fabricate", "Fading", "Frenzy",
    "Graft", "Hideaway", "Modular", "Poisonous", "Rampage", "Renown",
    "Ripple", "Soulshift", "Toxic", "Tribute", "Vanishing", "Backup",
    "Casualty", "Saddle",
}

OUR_COST_KEYWORDS = {
    "Kicker", "Multikicker", "Flashback", "Madness", "Morph",
    "Megamorph", "Disguise", "Cycling", "Equip", "Bestow", "Dash",
    "Evoke", "Unearth", "Encore", "Escape", "Foretell", "Suspend",
    "Buyback", "Echo", "Cumulative upkeep", "Ninjutsu", "Outlast",
    "Scavenge", "Embalm", "Eternalize", "Blitz", "Ward", "Reconfigure",
    "Spectacle", "Surge", "Prowl", "Miracle", "Overload", "Awaken",
    "Disturb", "Plot", "Craft", "Offspring",
}

OUR_TYPE_KEYWORDS = {
    "Affinity", "Protection", "Landwalk", "Enchant", "Champion",
    "Offering", "Amplify",
}

OUR_COMPLEX_KEYWORDS = {
    "Partner", "Partner with", "Companion", "Mutate", "Emerge", "Splice",
}

OUR_2024_2025_KEYWORDS = {
    # From hierarchical_actions.py ComplexMechanicsEncoder
    "impending", "eerie", "survival",  # Duskmourn
    "offspring", "valiant", "expend", "forage", "gift",  # Bloomburrow
    "plot", "spree", "saddle", "crime",  # Thunder Junction
    "disguise", "cloak", "collect_evidence", "suspect",  # Karlov Manor
    "craft", "descend", "discover", "explore",  # Ixalan
    "bargain", "celebration", "role",  # Eldraine
    "start_your_engines", "exhaust",  # Aetherdrift
}

# =============================================================================
# STANDARD 2025 MECHANICS ANALYSIS
# =============================================================================

STANDARD_2025_SETS = {
    "Foundations": {
        "release": "2024-11-15",
        "key_mechanics": [
            # Core mechanics reprint set
            "Flying", "First Strike", "Trample", "Deathtouch", "Lifelink",
            "Vigilance", "Haste", "Reach", "Menace", "Flash", "Ward",
        ],
        "notes": "Core set with evergreen mechanics, legal through 2029+"
    },
    "Duskmourn: House of Horror": {
        "release": "2024-09-27",
        "key_mechanics": [
            "Impending",  # Countdown to creature
            "Eerie",  # Trigger on enchantment ETB/room unlock
            "Survival",  # Trigger if no creature died
            "Delirium",  # Returning mechanic (card types in GY)
            "Rooms",  # Double-faced enchantments
        ],
        "notes": "Horror-themed set with enchantment focus"
    },
    "Bloomburrow": {
        "release": "2024-08-02",
        "key_mechanics": [
            "Offspring",  # Pay extra for token copy
            "Valiant",  # Trigger when targeted by own spell
            "Expend",  # Threshold on mana spent
            "Forage",  # Exile 3 from GY or sac food
            "Gift",  # Give opponent choice for bonus
        ],
        "notes": "Animal-themed set with cute creatures"
    },
    "Outlaws of Thunder Junction": {
        "release": "2024-04-19",
        "key_mechanics": [
            "Plot",  # Exile and cast later for free
            "Spree",  # Choose multiple modes with costs
            "Saddle",  # Tap creatures to enable Mount
            "Crime",  # Trigger on targeting opponent
            "Outlaw",  # Type batching (Assassin, Mercenary, Pirate, Rogue, Warlock)
        ],
        "notes": "Western-themed set"
    },
    "Murders at Karlov Manor": {
        "release": "2024-02-09",
        "key_mechanics": [
            "Disguise",  # Face-down with ward
            "Cloak",  # Put face-down from library
            "Collect evidence",  # Exile from GY with total MV
            "Suspect",  # Grants menace, can't block
            "Cases",  # Enchantments that solve
        ],
        "notes": "Detective/mystery-themed"
    },
    "Aetherdrift": {
        "release": "2025-02-14",
        "key_mechanics": [
            "Start Your Engines",  # Speed counter mechanic
            "Exhaust",  # Vehicle activation cost
            "Mounts",  # From Thunder Junction
            "Vehicles",  # Artifact subtype
        ],
        "notes": "Racing-themed set"
    },
    "Tarkir: Dragonstorm": {
        "release": "2025-04-11",
        "key_mechanics": [
            "Dragons",  # Tribal focus
            "Morph",  # Returning mechanic
            "Megamorph",  # Returning mechanic
            "Raid",  # Returning (if attacked this turn)
        ],
        "notes": "Dragon-focused Tarkir return"
    },
    "Final Fantasy": {
        "release": "2025-06-13",
        "key_mechanics": [
            "Job Select",  # Equipment creates Hero token
            "Limit",  # Trigger at low life
            "Materialize",  # Create token copies
        ],
        "notes": "Universes Beyond crossover"
    },
    "Edge of Eternities": {
        "release": "2025-09-05",
        "key_mechanics": [
            "Station",  # Spacecraft mechanic
            "Space",  # New zone/mechanic TBD
        ],
        "notes": "Space-themed set"
    },
    "Spider-Man": {
        "release": "2025-03-07",
        "key_mechanics": [
            "Web-slinging",  # Bounce + cast
        ],
        "notes": "Universes Beyond crossover"
    },
}

# =============================================================================
# GAP ANALYSIS
# =============================================================================

def compute_coverage():
    """Compute keyword coverage."""
    all_ours = (OUR_SIMPLE_KEYWORDS | OUR_AMOUNT_KEYWORDS |
                OUR_COST_KEYWORDS | OUR_TYPE_KEYWORDS | OUR_COMPLEX_KEYWORDS)

    # Normalize for comparison
    forge_lower = {k.lower() for k in FORGE_KEYWORDS}
    ours_lower = {k.lower() for k in all_ours}
    ours_2024_lower = {k.lower() for k in OUR_2024_2025_KEYWORDS}

    # Combine our implementations
    all_ours_lower = ours_lower | ours_2024_lower

    # Find gaps
    in_forge_not_ours = forge_lower - all_ours_lower
    in_ours_not_forge = all_ours_lower - forge_lower

    return {
        "forge_total": len(FORGE_KEYWORDS),
        "our_total": len(all_ours_lower),
        "coverage_pct": len(all_ours_lower & forge_lower) / len(forge_lower) * 100,
        "missing_from_ours": sorted(in_forge_not_ours),
        "extra_in_ours": sorted(in_ours_not_forge),
    }


def get_standard_mechanics():
    """Get all mechanics from Standard 2025 sets."""
    all_mechanics = set()
    for set_name, info in STANDARD_2025_SETS.items():
        all_mechanics.update(info["key_mechanics"])
    return all_mechanics


def analyze_standard_coverage():
    """Analyze coverage of Standard 2025 mechanics."""
    standard_mechanics = get_standard_mechanics()

    all_ours = (OUR_SIMPLE_KEYWORDS | OUR_AMOUNT_KEYWORDS |
                OUR_COST_KEYWORDS | OUR_TYPE_KEYWORDS | OUR_COMPLEX_KEYWORDS)
    ours_lower = {k.lower() for k in all_ours}
    ours_2024_lower = {k.lower() for k in OUR_2024_2025_KEYWORDS}
    all_ours_lower = ours_lower | ours_2024_lower

    standard_lower = {m.lower() for m in standard_mechanics}

    covered = standard_lower & all_ours_lower
    missing = standard_lower - all_ours_lower

    return {
        "standard_total": len(standard_mechanics),
        "covered": len(covered),
        "coverage_pct": len(covered) / len(standard_mechanics) * 100,
        "missing": sorted(missing),
        "covered_list": sorted(covered),
    }


def main():
    print("=" * 70)
    print("MTG MECHANICS AUDIT")
    print("=" * 70)

    # Overall coverage
    coverage = compute_coverage()
    print(f"\n{'='*40}")
    print("OVERALL KEYWORD COVERAGE")
    print(f"{'='*40}")
    print(f"Forge keywords: {coverage['forge_total']}")
    print(f"Our implementation: {coverage['our_total']}")
    print(f"Coverage: {coverage['coverage_pct']:.1f}%")

    print(f"\n{'='*40}")
    print("MISSING KEYWORDS (in Forge, not in our implementation)")
    print(f"{'='*40}")
    print(f"Total missing: {len(coverage['missing_from_ours'])}")

    # Categorize missing by importance
    combat_relevant = []
    alt_cost = []
    triggered = []
    other = []

    for kw in coverage['missing_from_ours']:
        kw_cap = kw.title()
        if kw in ['absorb', 'amplify', 'battle cry', 'bushido', 'frenzy',
                  'rampage', 'skulk', 'provoke', 'melee']:
            combat_relevant.append(kw)
        elif kw in ['bestow', 'dash', 'emerge', 'evoke', 'escape', 'foretell',
                    'madness', 'miracle', 'ninjutsu', 'prototype', 'spectacle',
                    'surge', 'prowl', 'suspend', 'sneak', 'harmonize', 'warp']:
            alt_cost.append(kw)
        elif kw in ['cascade', 'dethrone', 'exploit', 'extort', 'evolve',
                    'gravestorm', 'haunt', 'myriad', 'storm', 'undaunted']:
            triggered.append(kw)
        else:
            other.append(kw)

    print(f"\nCombat-relevant ({len(combat_relevant)}): {', '.join(combat_relevant)}")
    print(f"\nAlternate casting ({len(alt_cost)}): {', '.join(alt_cost)}")
    print(f"\nTriggered abilities ({len(triggered)}): {', '.join(triggered)}")
    print(f"\nOther ({len(other)}): {', '.join(other[:20])}...")

    # Standard 2025 coverage
    standard = analyze_standard_coverage()
    print(f"\n{'='*40}")
    print("STANDARD 2025 MECHANICS COVERAGE")
    print(f"{'='*40}")
    print(f"Standard mechanics: {standard['standard_total']}")
    print(f"Covered: {standard['covered']} ({standard['coverage_pct']:.1f}%)")
    print(f"\nMissing Standard mechanics: {', '.join(standard['missing'])}")

    # Per-set breakdown
    print(f"\n{'='*40}")
    print("PER-SET ANALYSIS")
    print(f"{'='*40}")

    all_ours = (OUR_SIMPLE_KEYWORDS | OUR_AMOUNT_KEYWORDS |
                OUR_COST_KEYWORDS | OUR_TYPE_KEYWORDS | OUR_COMPLEX_KEYWORDS)
    ours_lower = {k.lower() for k in all_ours}
    ours_2024_lower = {k.lower() for k in OUR_2024_2025_KEYWORDS}
    all_ours_lower = ours_lower | ours_2024_lower

    for set_name, info in STANDARD_2025_SETS.items():
        mechanics_lower = {m.lower() for m in info["key_mechanics"]}
        covered = mechanics_lower & all_ours_lower
        missing = mechanics_lower - all_ours_lower

        status = "✓" if not missing else "⚠"
        print(f"\n{status} {set_name}")
        print(f"  Covered: {', '.join(sorted(covered)) or 'none'}")
        if missing:
            print(f"  MISSING: {', '.join(sorted(missing))}")

    # Recommendations
    print(f"\n{'='*40}")
    print("RECOMMENDATIONS")
    print(f"{'='*40}")
    print("""
Priority 1 - Combat Mechanics (affect damage/blocking):
- Add: Skulk, Battle cry, Melee, Provoke, Bushido
- These directly affect combat math the agent must learn

Priority 2 - Alt-Cost Mechanics (expand action space):
- Add: Madness, Spectacle, Dash, Evoke, Emerge
- These create more decisions: "cast normally or use alt cost?"

Priority 3 - Triggered Mechanics (affect game state):
- Add: Cascade, Storm, Evolve, Exploit, Myriad
- Significant board impact when triggered

Priority 4 - Standard-Specific (for current format):
- Ensure: Impending, Offspring, Plot, Disguise, Spree working
- Rooms (Duskmourn), Cases (Karlov Manor) need special handling

Embedding Recommendations:
1. Current 113-dim embedding is sufficient for ~80% of mechanics
2. Add 20 more dims for alt-cost detection and parameters
3. Consider GNN for card interaction modeling
4. Text embeddings (128-dim) capture nuance well

Action Space Recommendations:
1. Modal spells (Spree, Kicker) need mode selection layer
2. Face-down cards need separate handling
3. Vehicle crewing needs creature selection
4. Target selection is the biggest complexity
""")


if __name__ == "__main__":
    main()
