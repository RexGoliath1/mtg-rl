# MTG Rules Reference for Parser Development

Distilled from MagicCompRules 20260116.txt (full rules: project root).
Focused on concepts relevant to card text parsing and mechanic detection.

---

## Cost Taxonomy

MTG has three distinct cost categories. This matters for parsing because they appear differently in card text.

### Additional Costs (Rule 118.8)
**"As an additional cost to cast this spell, [action]"**

- Paid ON TOP of the mana cost (or alternative cost)
- Can be mandatory or optional
- Multiple additional costs can stack
- Don't change the spell's mana cost/mana value

**Keywords that are additional costs:**
| Keyword | Rule | Pattern | Effect |
|---------|------|---------|--------|
| Kicker | 702.33 | "You may pay an additional [cost]" | Spell is "kicked" — bonus effect |
| Multikicker | 702.33c | "Pay additional [cost] any number of times" | Repeatable kicker |
| Buyback | 702.27 | "Pay additional [cost]" | Return to hand instead of graveyard |
| Entwine | 702.42 | "Pay additional [cost] to choose all modes" | Choose all modes on modal spell |
| Escalate | 702.120 | "For each mode beyond the first, pay [cost]" | Extra cost per extra mode |
| Replicate | 702.56 | "Pay [cost] any number of times" | Copy for each payment |
| Conspire | 702.78 | "Tap two creatures sharing a color" | Copy the spell |
| Squad | 702.157 | "Pay [cost] any number of times" | Token copies on ETB |
| Bargain | 702.166 | "Sacrifice an artifact, enchantment, or token" | Spell is "bargained" |
| Casualty | 702.153 | "Sacrifice creature with power N+" | Copy the spell |
| Offspring | 702.175 | "Pay additional [cost]" | 1/1 token copy on ETB |
| Splice | 702.47 | "Pay [cost], reveal from hand" | Add text to another spell |
| Offering | 702.48 | "Sacrifice a [type] permanent" | Cost reduction + instant timing |
| Spree | 702.172 | "Pay costs for chosen modes" | Modal with per-mode costs |
| Tiered | 702.183 | "Pay cost for chosen tier" | Modal with tiered costs |
| Gift | 702.174 | "Choose an opponent" | Opponent gets something |
| Jump-start | 702.133 | "Discard a card" (from graveyard) | Cast from graveyard |
| Retrace | 702.81 | "Discard a land" (from graveyard) | Cast from graveyard |

**Unnamed additional costs** (appear as rules text, not keywords):
- "As an additional cost to cast this spell, sacrifice a creature"
- "As an additional cost to cast this spell, pay N life"
- "As an additional cost to cast this spell, discard a card"
- "As an additional cost to cast this spell, exile N cards from your graveyard"
- "As an additional cost to cast this spell, tap an untapped creature you control"

### Alternative Costs (Rule 118.9)
**"You may pay [cost] rather than pay this spell's mana cost"**

- Replaces the mana cost entirely
- Only ONE alternative cost per spell
- Additional costs still apply on top
- Don't change the spell's mana value

**Keywords that are alternative costs:**
| Keyword | Rule | Pattern | Effect |
|---------|------|---------|--------|
| Flashback | 702.34 | "Cast from graveyard for [cost]" | One-time graveyard cast, then exile |
| Overload | 702.96 | "Pay [cost] instead of mana cost" | Replace "target" with "each" |
| Awaken | 702.113 | "Pay [cost] instead of mana cost" | Animate a land with +1/+1 counters |
| Surge | 702.117 | "Pay [cost] if another spell was cast" | Conditional cheaper cost |
| Spectacle | 702.137 | "Pay [cost] if opponent lost life" | Conditional cheaper cost |
| Madness | 702.35 | "Cast from exile for [cost] when discarded" | Cast when discarded |
| Evoke | 702.74 | "Cast for [cost], sacrifice on ETB" | Cheaper but temporary |
| Dash | 702.109 | "Cast for [cost], return to hand at end step" | Cheaper + haste, temporary |
| Prowl | 702.76 | "Pay [cost] if combat damage dealt by matching type" | Conditional cost |
| Bestow | 702.103 | "Cast as Aura for [cost]" | Becomes Aura instead of creature |
| Blitz | 702.152 | "Cast for [cost], haste + sacrifice + draw" | Aggressive temporary version |
| Cleave | 702.148 | "Pay [cost], remove bracketed text" | Text modification |
| Emerge | 702.119 | "Pay [cost] + sacrifice creature" | Cost reduced by sacrificed creature's MV |
| Morph | 702.37 | "Cast face-down for {3}" | 2/2 face-down creature |
| Disguise | 702.168 | "Cast face-down for {3}" | 2/2 face-down with ward {2} |
| Disturb | 702.146 | "Cast transformed from graveyard for [cost]" | Back-face from graveyard |
| Escape | 702.138 | "Cast from graveyard for [cost]" | Graveyard cast + exile cards |
| Foretell | 702.143 | "Exile for {2}, cast later for [cost]" | Split payment across turns |
| Miracle | 702.94 | "Cast for [cost] when drawn as first card" | Cheap if drawn at right time |
| Plot | 702.170 | "Exile for [cost], cast free later" | Pre-pay, cast free next turn+ |
| Warp | 702.185 | "Cast for [cost]" | Temporary, then cast from exile later |
| Freerunning | 702.173 | "Pay [cost] if Assassin/commander dealt damage" | Conditional cheaper cost |
| Harmonize | 702.180 | "Cast from GY, tap creature for reduction" | Graveyard + creature tap |
| Web-slinging | 702.188 | "Pay [cost] + return tapped creature" | Bounce creature as payment |

### Cost Reduction (NOT additional or alternative)
**"This spell costs {N} less to cast for each [condition]"**

These are applied AFTER the base cost is determined. Explicitly not additional or alternative costs per the rules.

| Keyword | Rule | Pattern |
|---------|------|---------|
| Convoke | 702.51 | Tap creatures to pay mana |
| Delve | 702.66 | Exile graveyard cards for generic mana |
| Improvise | 702.126 | Tap artifacts to pay generic mana |
| Affinity | 702.41 | "{1} less for each [type] you control" |
| Undaunted | 702.125 | "{1} less for each opponent" |
| Assist | 702.132 | Another player helps pay generic mana |

---

## All 189 Keyword Abilities by Category

### Evergreen (always in Standard)
Deathtouch, Defender, Double Strike, Enchant, Equip, First Strike, Flash,
Flying, Haste, Hexproof, Indestructible, Lifelink, Menace, Protection,
Reach, Shroud, Trample, Vigilance, Ward

### Combat
Banding, Flanking, Bushido, Exalted, Battle Cry, Melee, Enlist,
Myriad, Provoke, Rampage, Skulk, Intimidate, Fear, Horsemanship,
Shadow, Annihilator, Dethrone, Mentor, Training, Mobilize, Saddle

### Counters / ETB modification
Modular, Sunburst, Graft, Amplify, Fabricate, Devour, Renown,
Unleash, Riot, Backup, Tribute, Ravenous, Bloodthirst

### Triggered abilities
Cascade, Evolve, Exploit, Extort, Persist, Undying, Afterlife,
Afflict, Toxic, Poisonous, Prowess, Storm, Gravestorm, Ripple,
Haunt, Living Weapon, For Mirrodin!, Decayed, Boast, Job Select

### Graveyard interactions
Flashback, Retrace, Jump-start, Escape, Disturb, Aftermath,
Unearth, Embalm, Eternalize, Dredge, Recover, Scavenge,
Encore, Harmonize, Mayhem

### Activated abilities
Cycling, Level Up, Outlast, Ninjutsu, Transfigure, Aura Swap,
Transmute, Forecast, Reinforce, Fortify, Reconfigure, Craft,
Crew, Station, Exhaust

### Static / characteristic
Changeling, Devoid, Companion, Living Metal, Infinity

### Set-specific mechanics (recent)
Plot, Warp, Freerunning, Offspring, Bargain, Spree, Tiered,
Gift, Impending, Disguise, Solved, Start Your Engines!, Max Speed,
Firebending, Web-slinging

### Alternative/additional casting
Kicker, Multikicker, Buyback, Flashback, Overload, Bestow,
Evoke, Emerge, Dash, Blitz, Morph, Disguise, Surge, Spectacle,
Madness, Awaken, Cleave, Foretell, Escape, Miracle, Plot

---

## Key Rules Concepts for Parsing

### Triggers (Rule 603)
- **"When/Whenever [event]"** = triggered ability
- **"At the beginning of [phase/step]"** = triggered ability
- ETB triggers: "When [this] enters" (replaces old "enters the battlefield")
- LTB triggers: "When [this] leaves the battlefield"
- Dies = "is put into a graveyard from the battlefield" (creatures only in casual speech, but rules apply to all permanents)

### Replacement Effects (Rule 614)
- Use "instead" or "skip" or "enters with"
- "If [event] would [happen], [different thing] instead"
- Self-replacement: "enters with N counters" replaces entering normally
- "Can't" effects override replacement effects

### Static Abilities (Rule 604)
- Continuous effects that are always active
- "Has [keyword]", "[type]s you control have [keyword]"
- Characteristic-defining: changeling, devoid

### Activated Abilities (Rule 602)
- Format: "[Cost]: [Effect]"
- Cost before the colon, effect after
- Tap symbol {T} is a cost
- "Activate only as a sorcery" = timing restriction

### Modal Spells
- "Choose one —" / "Choose two —" / "Choose one or more —"
- Escalate, Entwine, Spree, Tiered modify modal choices
- Modes are chosen on cast, before targets

### Layers (Rule 613) — Order Effects Apply
1. Copy effects
2. Control-changing effects
3. Text-changing effects
4. Type-changing effects
5. Color-changing effects
6. Ability-adding/removing effects
7. Power/toughness changes (7a: characteristic-defining, 7b: set P/T, 7c: modifications, 7d: counters, 7e: switching)

---

## Rules Text Patterns for Parser

### Common "as an additional cost" forms
```
"As an additional cost to cast this spell, sacrifice a creature"
"As an additional cost to cast this spell, pay {N} life"
"As an additional cost to cast this spell, discard a card"
"As an additional cost to cast this spell, exile N cards from your graveyard"
"As an additional cost to cast this spell, you may [action]" (optional)
```

### Common "alternative cost" forms
```
"You may cast this spell by paying [cost] rather than its mana cost"
"You may pay [cost] rather than pay this spell's mana cost"
"Cast this spell without paying its mana cost"
"You may cast this card from your graveyard by paying [cost]"
```

### Common "cost reduction" forms
```
"This spell costs {N} less to cast for each [condition]"
"[Keyword] (For each generic mana..., you may [tap/exile] rather than pay)"
```

---

## Source
Full rules: `/MagicCompRules 20260116.txt` (932KB, Jan 16, 2026 edition)
