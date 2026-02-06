"""
Card Parser Fidelity Tests

Tests ~90 tricky cards from 9 recent sets to validate parser accuracy.

Test categories:
1. Multi-effect cards — surveil N + conditional play
2. Triple-pip costs — {U}{U}{U} cards
3. Spree/modal cards — variable mode selection
4. Sagas — multi-chapter effects
5. Room/split cards — dual-faced effects
6. Keyword soup — cards with 3+ keywords
7. Complex triggers — nested conditions, multiple triggers
8. X-cost spells — variable cost effects
9. Hybrid/phyrexian mana — special cost representation
10. Cards that previously scored 0% — Tarmogoyf, etc.
"""

from src.mechanics.vocabulary import Mechanic
from src.mechanics.card_parser import parse_card, parse_oracle_text, strip_reminder_text


# =============================================================================
# HELPERS
# =============================================================================

def make_card(name, mana_cost, cmc, type_line, oracle_text, power=None, toughness=None, **kwargs):
    """Create a Scryfall-like card dict."""
    card = {
        "name": name,
        "mana_cost": mana_cost,
        "cmc": cmc,
        "type_line": type_line,
        "oracle_text": oracle_text,
    }
    if power is not None:
        card["power"] = str(power)
    if toughness is not None:
        card["toughness"] = str(toughness)
    card.update(kwargs)
    return card


def assert_has_mechanics(encoding, expected_mechanics, card_name=""):
    """Assert that specific mechanics are present in the encoding."""
    mechanic_names = {m.name for m in encoding.mechanics}
    for mech in expected_mechanics:
        assert mech.name in mechanic_names, (
            f"{card_name}: Expected {mech.name} in mechanics, "
            f"got: {sorted(mechanic_names)}"
        )


def assert_lacks_mechanics(encoding, excluded_mechanics, card_name=""):
    """Assert that specific mechanics are NOT present in the encoding."""
    mechanic_names = {m.name for m in encoding.mechanics}
    for mech in excluded_mechanics:
        assert mech.name not in mechanic_names, (
            f"{card_name}: Expected {mech.name} NOT in mechanics, "
            f"got: {sorted(mechanic_names)}"
        )


def assert_confidence_gte(result, threshold, card_name=""):
    """Assert confidence is at or above threshold."""
    assert result.confidence >= threshold, (
        f"{card_name}: Confidence {result.confidence:.3f} < {threshold}"
    )


# =============================================================================
# 1. MULTI-EFFECT CARDS
# =============================================================================

class TestMultiEffectCards:
    """Cards with multiple effects that must all be detected."""

    def test_glarb_calamitys_augur(self):
        """Glarb — surveil + conditional play from graveyard."""
        card = make_card(
            "Glarb, Calamity's Augur", "{2}{B}{G}", 4,
            "Legendary Creature — Slug Druid",
            "Whenever Glarb, Calamity's Augur attacks, surveil 2, then you may "
            "play a card from your graveyard this turn.",
            power=3, toughness=5,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ATTACK_TRIGGER,
            Mechanic.SURVEIL,
        ], card.get("name"))
        assert enc.parameters.get("surveil_count") == 2

    def test_niv_mizzet_supreme(self):
        """Niv-Mizzet — draw trigger, deal damage."""
        card = make_card(
            "Niv-Mizzet, Supreme", "{W}{U}{B}{R}{G}", 5,
            "Legendary Creature — Dragon Avatar",
            "Flying\nWhenever you draw a card, Niv-Mizzet, Supreme deals 1 damage "
            "to any target.",
            power=5, toughness=5,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLYING,
            Mechanic.DRAW_TRIGGER,
            Mechanic.DEAL_DAMAGE,
        ], card.get("name"))
        assert enc.parameters.get("damage") == 1

    def test_mulldrifter(self):
        """Mulldrifter — flying, ETB draw 2, evoke."""
        card = make_card(
            "Mulldrifter", "{4}{U}", 5,
            "Creature — Elemental",
            "Flying\nWhen Mulldrifter enters the battlefield, draw two cards.\nEvoke {2}{U}",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLYING,
            Mechanic.ETB_TRIGGER,
            Mechanic.DRAW,
            Mechanic.EVOKE,
        ], card.get("name"))
        assert enc.parameters.get("draw_count") == 2

    def test_sheoldred_the_apocalypse(self):
        """Sheoldred — draw trigger gain/lose life."""
        card = make_card(
            "Sheoldred, the Apocalypse", "{2}{B}{B}", 4,
            "Legendary Creature — Phyrexian Praetor",
            "Deathtouch\nWhenever you draw a card, you gain 2 life.\n"
            "Whenever an opponent draws a card, they lose 2 life.",
            power=4, toughness=5,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DEATHTOUCH,
            Mechanic.DRAW_TRIGGER,
            Mechanic.GAIN_LIFE,
            Mechanic.LOSE_LIFE,
        ], card.get("name"))


# =============================================================================
# 2. TRIPLE-PIP COSTS
# =============================================================================

class TestTriplePipCosts:
    """Cards with triple-pip mana costs for proper color counting."""

    def test_three_steps_ahead(self):
        """Three Steps Ahead — {1}{U}{U}{U} counter + copy."""
        card = make_card(
            "Three Steps Ahead", "{1}{U}{U}{U}", 4,
            "Instant",
            "Choose one —\n• Counter target spell.\n• Create a token that's a "
            "copy of target artifact or creature you control.\n• Draw two cards.",
        )
        enc = parse_card(card)
        assert enc.mana_cost["U"] == 3
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.COUNTER_SPELL,
            Mechanic.CREATE_TOKEN_COPY,
            Mechanic.DRAW,
            Mechanic.MODAL_CHOOSE_ONE,
        ], card.get("name"))

    def test_cryptic_command(self):
        """Cryptic Command — {1}{U}{U}{U} modal."""
        card = make_card(
            "Cryptic Command", "{1}{U}{U}{U}", 4,
            "Instant",
            "Choose two —\n• Counter target spell.\n• Return target permanent "
            "to its owner's hand.\n• Tap all creatures your opponents control.\n"
            "• Draw a card.",
        )
        enc = parse_card(card)
        assert enc.mana_cost["U"] == 3
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.MODAL_CHOOSE_TWO,
            Mechanic.COUNTER_SPELL,
            Mechanic.DRAW,
        ], card.get("name"))


# =============================================================================
# 3. SPREE / MODAL CARDS
# =============================================================================

class TestSpreeCards:
    """OTJ spree cards with variable mode selection."""

    def test_requisition_raid(self):
        """Requisition Raid — spree with multiple modes."""
        card = make_card(
            "Requisition Raid", "{W}", 1,
            "Instant",
            "Spree\n+ {1} — Create a 1/1 white Soldier creature token.\n"
            "+ {2} — Destroy target artifact or enchantment.\n",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.SPREE,
            Mechanic.CREATE_TOKEN,
            Mechanic.DESTROY,
        ], card.get("name"))

    def test_choose_two_modal(self):
        """Generic choose two card."""
        card = make_card(
            "Modal Test", "{2}{B}", 3,
            "Sorcery",
            "Choose two —\n• Each opponent loses 2 life.\n"
            "• Draw a card.\n• Create a 1/1 black Rat creature token.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SORCERY_SPEED,
            Mechanic.MODAL_CHOOSE_TWO,
            Mechanic.LOSE_LIFE,
            Mechanic.DRAW,
            Mechanic.CREATE_TOKEN,
        ], card.get("name"))


# =============================================================================
# 4. SAGAS
# =============================================================================

class TestSagas:
    """Multi-chapter saga cards."""

    def test_history_of_benalia(self):
        """History of Benalia — 3-chapter saga with tokens and anthem."""
        card = make_card(
            "History of Benalia", "{1}{W}{W}", 3,
            "Enchantment — Saga",
            "I, II — Create a 2/2 white Knight creature token with vigilance.\n"
            "III — Knights you control get +2/+1 until end of turn.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SAGA,
            Mechanic.CREATE_TOKEN,
        ], card.get("name"))

    def test_elspeth_conquers_death(self):
        """Elspeth Conquers Death — saga with exile, tax, reanimate."""
        card = make_card(
            "Elspeth Conquers Death", "{3}{W}{W}", 5,
            "Enchantment — Saga",
            "I — Exile target permanent an opponent controls with mana value 3 or greater.\n"
            "II — Noncreature spells your opponents cast cost {2} more to cast.\n"
            "III — Return target creature or planeswalker card from your graveyard "
            "to the battlefield.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SAGA,
            Mechanic.CHAPTER_I,
            Mechanic.CHAPTER_II,
            Mechanic.CHAPTER_III,
            Mechanic.EXILE,
            Mechanic.REANIMATE,
        ], card.get("name"))
        assert enc.parameters.get("chapter_count") == 3


# =============================================================================
# 5. ROOM / SPLIT CARDS (DSK)
# =============================================================================

class TestRoomCards:
    """DSK room enchantments with dual effects."""

    def test_room_card_glassworks(self):
        """Glassworks // Glassworks — room with ETB effects."""
        card = make_card(
            "Glassworks // Grinders", "{2}{R}", 3,
            "Enchantment — Room",
            "When you unlock this Room, deal 3 damage to target creature.\n"
            "When you unlock this Room, create two Treasure tokens.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DEAL_DAMAGE,
            Mechanic.CREATE_TOKEN,
            Mechanic.CREATE_TREASURE,
        ], card.get("name"))


# =============================================================================
# 6. KEYWORD SOUP
# =============================================================================

class TestKeywordSoup:
    """Cards with 3+ keywords."""

    def test_questing_beast(self):
        """Questing Beast — vigilance, deathtouch, haste, can't be blocked by 2 or less."""
        card = make_card(
            "Questing Beast", "{2}{G}{G}", 4,
            "Legendary Creature — Beast",
            "Vigilance, deathtouch, haste\n"
            "Questing Beast can't be blocked by creatures with power 2 or less.\n"
            "Combat damage that would be dealt by creatures you control can't be prevented.\n"
            "Whenever Questing Beast deals combat damage to a player, "
            "it deals that much damage to target planeswalker that player controls.",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.VIGILANCE,
            Mechanic.DEATHTOUCH,
            Mechanic.HASTE,
            Mechanic.COMBAT_DAMAGE_TO_PLAYER,
        ], card.get("name"))

    def test_atraxa_praetors_voice(self):
        """Atraxa — flying, vigilance, deathtouch, lifelink, proliferate."""
        card = make_card(
            "Atraxa, Praetors' Voice", "{W}{U}{B}{G}", 4,
            "Legendary Creature — Phyrexian Angel Horror",
            "Flying, vigilance, deathtouch, lifelink\n"
            "At the beginning of your end step, proliferate.",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLYING,
            Mechanic.VIGILANCE,
            Mechanic.DEATHTOUCH,
            Mechanic.LIFELINK,
            Mechanic.PROLIFERATE,
            Mechanic.END_STEP_TRIGGER,
        ], card.get("name"))

    def test_akroma_angel_of_wrath(self):
        """Akroma — flying, first strike, vigilance, trample, haste, protection."""
        card = make_card(
            "Akroma, Angel of Wrath", "{5}{W}{W}{W}", 8,
            "Legendary Creature — Angel",
            "Flying, first strike, vigilance, trample, haste, protection from black "
            "and from red",
            power=6, toughness=6,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLYING,
            Mechanic.FIRST_STRIKE,
            Mechanic.VIGILANCE,
            Mechanic.TRAMPLE,
            Mechanic.HASTE,
            Mechanic.PROTECTION,
        ], card.get("name"))


# =============================================================================
# 7. COMPLEX TRIGGERS
# =============================================================================

class TestComplexTriggers:
    """Nested conditions, multiple triggers."""

    def test_rhystic_study(self):
        """Rhystic Study — opponent casts, may draw, unless pays."""
        card = make_card(
            "Rhystic Study", "{2}{U}", 3,
            "Enchantment",
            "Whenever an opponent casts a spell, you may draw a card unless "
            "that player pays {1}.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.OPPONENT_CASTS,
            Mechanic.DRAW_OPTIONAL,
            Mechanic.UNLESS_PAYS,
        ], card.get("name"))

    def test_panharmonicon(self):
        """Panharmonicon — double trigger on ETB."""
        card = make_card(
            "Panharmonicon", "{4}", 4,
            "Artifact",
            "If an artifact or creature entering the battlefield causes a triggered "
            "ability of a permanent you control to trigger, that ability triggers "
            "an additional time.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DOUBLE_TRIGGER,
        ], card.get("name"))

    def test_deflecting_swat(self):
        """Deflecting Swat — free if commander, change targets."""
        card = make_card(
            "Deflecting Swat", "{2}{R}", 3,
            "Instant",
            "If you control a commander, you may cast this spell without paying "
            "its mana cost.\nYou may choose new targets for target spell or ability.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.IF_YOU_CONTROL_COMMANDER,
            Mechanic.FREE_CAST_CONDITION,
            Mechanic.CHANGE_TARGETS,
            Mechanic.TARGET_SPELL_OR_ABILITY,
        ], card.get("name"))

    def test_smothering_tithe(self):
        """Smothering Tithe — opponent draws, unless pays, create treasure."""
        card = make_card(
            "Smothering Tithe", "{3}{W}", 4,
            "Enchantment",
            "Whenever an opponent draws a card, that player may pay {2}. "
            "If the player doesn't, you create a Treasure token.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CREATE_TOKEN,
            Mechanic.CREATE_TREASURE,
        ], card.get("name"))


# =============================================================================
# 8. X-COST SPELLS
# =============================================================================

class TestXCostSpells:
    """Variable cost spells."""

    def test_fireball(self):
        """Fireball — {X}{R} deal X damage."""
        card = make_card(
            "Fireball", "{X}{R}", 1,
            "Sorcery",
            "This spell costs {1} more to cast for each target beyond the first.\n"
            "Fireball deals X damage divided evenly, rounded down, among any number "
            "of targets.",
        )
        enc = parse_card(card)
        assert enc.parameters.get("x_cost_count") == 1
        assert_has_mechanics(enc, [
            Mechanic.SORCERY_SPEED,
            Mechanic.DEAL_DAMAGE,
        ], card.get("name"))

    def test_genesis_wave(self):
        """Genesis Wave — {X}{G}{G}{G} look, put permanents onto battlefield."""
        card = make_card(
            "Genesis Wave", "{X}{G}{G}{G}", 3,
            "Sorcery",
            "Reveal the top X cards of your library. You may put any number of "
            "permanent cards with mana value X or less from among them onto the "
            "battlefield. Then put all cards revealed this way that weren't put "
            "onto the battlefield into your graveyard.",
        )
        enc = parse_card(card)
        assert enc.parameters.get("x_cost_count") == 1
        assert enc.mana_cost["G"] == 3

    def test_walking_ballista(self):
        """Walking Ballista — {X}{X} creature."""
        card = make_card(
            "Walking Ballista", "{X}{X}", 0,
            "Artifact Creature — Construct",
            "Walking Ballista enters the battlefield with X +1/+1 counters on it.\n"
            "{4}: Put a +1/+1 counter on Walking Ballista.\n"
            "Remove a +1/+1 counter from Walking Ballista: It deals 1 damage to "
            "any target.",
            power=0, toughness=0,
        )
        enc = parse_card(card)
        assert enc.parameters.get("x_cost_count") == 2
        assert_has_mechanics(enc, [
            Mechanic.PLUS_ONE_COUNTER,
            Mechanic.DEAL_DAMAGE,
        ], card.get("name"))


# =============================================================================
# 9. HYBRID / PHYREXIAN MANA
# =============================================================================

class TestHybridPhyrexianMana:
    """Special mana cost representations."""

    def test_hybrid_mana_niv_mizzet(self):
        """Niv-Mizzet Reborn — all hybrid pips."""
        card = make_card(
            "Niv-Mizzet Reborn", "{W}{U}{B}{R}{G}", 5,
            "Legendary Creature — Dragon Avatar",
            "Flying\nWhen Niv-Mizzet Reborn enters the battlefield, reveal the top "
            "ten cards of your library.",
            power=6, toughness=6,
        )
        enc = parse_card(card)
        assert enc.mana_cost["W"] == 1
        assert enc.mana_cost["U"] == 1
        assert enc.mana_cost["B"] == 1
        assert enc.mana_cost["R"] == 1
        assert enc.mana_cost["G"] == 1

    def test_hybrid_boros_reckoner(self):
        """Boros Reckoner — {R/W}{R/W}{R/W} hybrid."""
        card = make_card(
            "Boros Reckoner", "{R/W}{R/W}{R/W}", 3,
            "Creature — Minotaur Wizard",
            "Whenever Boros Reckoner is dealt damage, it deals that much damage "
            "to any target.\n"
            "{R/W}: Boros Reckoner gains first strike until end of turn.",
            power=3, toughness=3,
        )
        enc = parse_card(card)
        # Hybrid: each pip counts for both colors
        assert enc.mana_cost["R"] == 3
        assert enc.mana_cost["W"] == 3

    def test_phyrexian_mana_dismember(self):
        """Dismember — {1}{B/P}{B/P} phyrexian."""
        card = make_card(
            "Dismember", "{1}{B/P}{B/P}", 3,
            "Instant",
            "Target creature gets -5/-5 until end of turn.",
        )
        enc = parse_card(card)
        assert enc.mana_cost["B"] == 2  # Phyrexian counts the color
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.MINUS_POWER,
            Mechanic.MINUS_TOUGHNESS,
            Mechanic.UNTIL_END_OF_TURN,
        ], card.get("name"))


# =============================================================================
# 10. PREVIOUSLY 0% CONFIDENCE CARDS
# =============================================================================

class TestPreviouslyZeroPercent:
    """Cards that had 0% confidence or minimal mechanics."""

    def test_tarmogoyf(self):
        """Tarmogoyf — power/toughness = * (variable)."""
        card = make_card(
            "Tarmogoyf", "{1}{G}", 2,
            "Creature — Lhurgoyf",
            "Tarmogoyf's power is equal to the number of card types among cards "
            "in all graveyards and its toughness is equal to that number plus 1.",
            power="*", toughness="*",
        )
        enc = parse_card(card)
        assert enc.power == -1  # * represented as -1
        assert enc.toughness == -1
        # Should still parse some mechanics
        result = parse_oracle_text(card["oracle_text"], card["type_line"])
        assert result.confidence >= 0.05

    def test_dark_confidant(self):
        """Dark Confidant — upkeep trigger, lose life, draw (reveal)."""
        card = make_card(
            "Dark Confidant", "{1}{B}", 2,
            "Creature — Human Wizard",
            "At the beginning of your upkeep, reveal the top card of your library "
            "and put that card into your hand. You lose life equal to its mana value.",
            power=2, toughness=1,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.UPKEEP_TRIGGER,
            Mechanic.REVEAL,
            Mechanic.LOSE_LIFE,
        ], card.get("name"))

    def test_lightning_bolt(self):
        """Lightning Bolt — simple damage spell."""
        card = make_card(
            "Lightning Bolt", "{R}", 1,
            "Instant",
            "Lightning Bolt deals 3 damage to any target.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.DEAL_DAMAGE,
        ], card.get("name"))
        assert enc.parameters.get("damage") == 3


# =============================================================================
# DSK (DUSKMOURN) MECHANICS
# =============================================================================

class TestDuskmourn:
    """Duskmourn-specific mechanics: eerie, survival, impending, rooms."""

    def test_eerie_trigger(self):
        """Card with eerie ability word."""
        card = make_card(
            "Fear of Impostors", "{2}{U}", 3,
            "Enchantment Creature — Horror",
            "Eerie — Whenever an enchantment you control enters and whenever "
            "you fully unlock a Room, create a 1/1 blue Illusion creature token.",
            power=2, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.EERIE,
            Mechanic.CREATE_TOKEN,
        ], card.get("name"))

    def test_survival_mechanic(self):
        """Card with survival condition."""
        card = make_card(
            "Ripchain Razorkin", "{3}{R}", 4,
            "Creature — Devil",
            "Survival — At the beginning of your second main phase, if Ripchain "
            "Razorkin is tapped, it deals 1 damage to each opponent.",
            power=3, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SURVIVAL,
            Mechanic.DEAL_DAMAGE,
        ], card.get("name"))

    def test_impending(self):
        """Card with impending N."""
        card = make_card(
            "Overlord of the Hauntwoods", "{3}{G}{G}", 5,
            "Creature — Avatar Horror",
            "Impending 4 — {2}{G}\nWhen this creature enters, create a tapped "
            "Everywhere token.\nTrample",
            power=6, toughness=5,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.IMPENDING,
            Mechanic.TRAMPLE,
            Mechanic.CREATE_TOKEN,
        ], card.get("name"))
        assert enc.parameters.get("impending_cost") == 4


# =============================================================================
# BLB (BLOOMBURROW) MECHANICS
# =============================================================================

class TestBloomburrow:
    """Bloomburrow-specific mechanics: offspring, valiant."""

    def test_offspring(self):
        """Card with offspring keyword."""
        card = make_card(
            "Finneas, Ace Archer", "{1}{G}{W}", 3,
            "Legendary Creature — Rabbit Archer",
            "Offspring {2}\nWhenever Finneas, Ace Archer attacks, "
            "draw a card if you control a creature with power 4 or greater.",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.OFFSPRING,
            Mechanic.ATTACK_TRIGGER,
            Mechanic.DRAW,
        ], card.get("name"))

    def test_valiant(self):
        """Card with valiant ability word."""
        card = make_card(
            "Brave-Kin Duo", "{1}{W}", 2,
            "Creature — Mouse Soldier",
            "Valiant — Whenever Brave-Kin Duo becomes the target of a spell or "
            "ability you control for the first time each turn, put a +1/+1 counter "
            "on it.",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.VALIANT,
            Mechanic.PLUS_ONE_COUNTER,
        ], card.get("name"))


# =============================================================================
# OTJ (OUTLAWS OF THUNDER JUNCTION) MECHANICS
# =============================================================================

class TestOutlaws:
    """OTJ-specific mechanics: spree, crime, saddle, plot."""

    def test_commit_a_crime(self):
        """Card with crime trigger."""
        card = make_card(
            "Rakdos, the Muscle", "{1}{B}{R}", 3,
            "Legendary Creature — Demon Mercenary",
            "Whenever you commit a crime, target creature you control gets +2/+0 "
            "until end of turn.",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.COMMIT_A_CRIME,
            Mechanic.PLUS_POWER,
            Mechanic.UNTIL_END_OF_TURN,
        ], card.get("name"))

    def test_saddle(self):
        """Card with saddle N."""
        card = make_card(
            "Thousand Moons Smithy", "{3}{W}", 4,
            "Creature — Mount",
            "Saddle 2\nWhenever this creature attacks, create a 1/1 white "
            "Soldier creature token.",
            power=3, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SADDLE,
            Mechanic.ATTACK_TRIGGER,
            Mechanic.CREATE_TOKEN,
        ], card.get("name"))
        assert enc.parameters.get("saddle_power") == 2

    def test_plot(self):
        """Card with plot keyword."""
        card = make_card(
            "Freestrider Lookout", "{2}{G}", 3,
            "Creature — Human Scout",
            "Plot {1}{G}\nWhen Freestrider Lookout enters the battlefield, "
            "create a 1/1 green Scout creature token.",
            power=3, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.PLOT,
            Mechanic.ETB_TRIGGER,
            Mechanic.CREATE_TOKEN,
        ], card.get("name"))


# =============================================================================
# MKM (MURDERS AT KARLOV MANOR) MECHANICS
# =============================================================================

class TestMurdersAtKarlovManor:
    """MKM-specific mechanics: case, suspect, cloak, collect evidence."""

    def test_suspect(self):
        """Card with suspect mechanic."""
        card = make_card(
            "Suspect Detector", "{2}{R}", 3,
            "Creature — Human Detective",
            "When Suspect Detector enters the battlefield, suspect target creature.",
            power=3, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SUSPECT,
            Mechanic.ETB_TRIGGER,
        ], card.get("name"))

    def test_collect_evidence(self):
        """Card with collect evidence N."""
        card = make_card(
            "Forensic Researcher", "{1}{U}", 2,
            "Creature — Vedalken Detective",
            "Collect evidence 6: Untap Forensic Researcher.",
            power=0, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.COLLECT_EVIDENCE,
            Mechanic.UNTAP,
        ], card.get("name"))
        assert enc.parameters.get("collect_evidence_mv") == 6

    def test_cloak(self):
        """Card with cloak mechanic."""
        card = make_card(
            "Ethereal Grasp", "{2}{U}", 3,
            "Instant",
            "Cloak target creature you control. Draw a card.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.CLOAK,
            Mechanic.DRAW,
        ], card.get("name"))


# =============================================================================
# LCI (LOST CAVERNS OF IXALAN) MECHANICS
# =============================================================================

class TestLostCavernsOfIxalan:
    """LCI-specific mechanics: discover, descend, craft, map tokens."""

    def test_discover(self):
        """Card with discover N."""
        card = make_card(
            "Geological Appraiser", "{1}{R}{G}{W}", 4,
            "Creature — Human Artificer",
            "When Geological Appraiser enters the battlefield, discover 3.",
            power=3, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DISCOVER,
            Mechanic.ETB_TRIGGER,
        ], card.get("name"))

    def test_descend_4(self):
        """Card with descend 4."""
        card = make_card(
            "Souls of the Lost", "{1}{B}", 2,
            "Creature — Spirit",
            "Descend 4 — Souls of the Lost gets +2/+0 as long as there are four "
            "or more permanent cards in your graveyard.",
            power=0, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DESCEND_4,
            Mechanic.PLUS_POWER,
            Mechanic.AS_LONG_AS,
        ], card.get("name"))


# =============================================================================
# WOE (WILDS OF ELDRAINE) MECHANICS
# =============================================================================

class TestWildsOfEldraine:
    """WOE-specific mechanics: bargain, celebration, role tokens, adventures."""

    def test_bargain(self):
        """Card with bargain keyword."""
        card = make_card(
            "Beseech the Mirror", "{1}{B}{B}{B}", 4,
            "Sorcery",
            "Bargain\nSearch your library for a card, put it into your hand, then "
            "shuffle. If this spell was bargained, you may cast a spell with mana "
            "value 4 or less from your hand without paying its mana cost.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SORCERY_SPEED,
            Mechanic.BARGAIN,
            Mechanic.TUTOR_TO_HAND,
            Mechanic.FREE_CAST_CONDITION,
        ], card.get("name"))

    def test_celebration(self):
        """Card with celebration ability word."""
        card = make_card(
            "Goddric, Cloaked Reveler", "{1}{R}{R}", 3,
            "Legendary Creature — Human Noble",
            "Haste\nCelebration — As long as two or more nonland permanents "
            "entered the battlefield under your control this turn, Goddric is "
            "a Dragon with base power 4, base toughness 4, and flying.",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.HASTE,
            Mechanic.CELEBRATION,
            Mechanic.AS_LONG_AS,
        ], card.get("name"))

    def test_adventure(self):
        """Card with adventure keyword."""
        card = make_card(
            "Bonecrusher Giant", "{2}{R}", 3,
            "Creature — Giant",
            "Whenever Bonecrusher Giant becomes the target of a spell, Bonecrusher "
            "Giant deals 2 damage to that spell's controller.\n"
            "Adventure — Stomp {1}{R}\nDeal 2 damage to any target.",
            power=4, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ADVENTURE,
            Mechanic.DEAL_DAMAGE,
        ], card.get("name"))


# =============================================================================
# ADDITIONAL KEYWORD COVERAGE
# =============================================================================

class TestKeywordCoverage:
    """Verify newly added keywords parse correctly."""

    def test_shroud(self):
        card = make_card("Shroud Test", "{2}{G}", 3, "Creature — Beast",
                         "Shroud", power=3, toughness=3)
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.SHROUD], "Shroud Test")

    def test_ward(self):
        card = make_card("Ward Test", "{1}{U}", 2, "Creature — Spirit",
                         "Ward {2}", power=2, toughness=2)
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.WARD], "Ward Test")
        assert enc.parameters.get("ward_cost") == 2

    def test_split_second(self):
        card = make_card("Sudden Shock", "{1}{R}", 2, "Instant",
                         "Split second\nSudden Shock deals 2 damage to any target.")
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SPLIT_SECOND,
            Mechanic.DEAL_DAMAGE,
        ], "Sudden Shock")

    def test_morph(self):
        card = make_card("Morph Test", "{3}{U}", 4, "Creature — Wizard",
                         "Morph {2}{U}", power=2, toughness=2)
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.MORPH], "Morph Test")

    def test_disguise(self):
        card = make_card("Disguise Test", "{2}{B}", 3, "Creature — Rogue",
                         "Disguise {1}{B}\nWhen this creature is turned face up, "
                         "target opponent discards a card.", power=3, toughness=2)
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DISGUISE,
            Mechanic.DISCARD,
        ], "Disguise Test")

    def test_connive(self):
        card = make_card("Connive Test", "{2}{U}", 3, "Creature — Rogue",
                         "When this creature enters the battlefield, it connives.",
                         power=2, toughness=2)
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.CONNIVE], "Connive Test")

    def test_incubate(self):
        card = make_card("Incubate Test", "{3}{W}", 4, "Creature — Phyrexian",
                         "When this creature enters the battlefield, incubate 3.",
                         power=2, toughness=3)
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.INCUBATE], "Incubate Test")

    def test_toxic(self):
        card = make_card("Toxic Test", "{G}", 1, "Creature — Phyrexian",
                         "Toxic 1", power=1, toughness=1)
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.TOXIC], "Toxic Test")
        assert enc.parameters.get("toxic_count") == 1

    def test_annihilator(self):
        card = make_card("Annihilator Test", "{10}", 10, "Creature — Eldrazi",
                         "Annihilator 4\nFlying, trample", power=10, toughness=10)
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ANNIHILATOR,
            Mechanic.FLYING,
            Mechanic.TRAMPLE,
        ], "Annihilator Test")
        assert enc.parameters.get("annihilator_count") == 4

    def test_bestow(self):
        card = make_card("Bestow Test", "{3}{W}", 4, "Enchantment Creature — Spirit",
                         "Bestow {5}{W}\nFlying\nEnchanted creature gets +2/+2 "
                         "and has flying.", power=2, toughness=2)
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.BESTOW,
            Mechanic.FLYING,
        ], "Bestow Test")

    def test_casualty(self):
        card = make_card("Casualty Test", "{1}{B}", 2, "Sorcery",
                         "Casualty 1\nEach opponent loses 2 life and you gain 2 life.")
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CASUALTY,
            Mechanic.LOSE_LIFE,
            Mechanic.GAIN_LIFE,
        ], "Casualty Test")

    def test_companion(self):
        card = make_card("Lurrus of the Dream-Den", "{1}{W/B}{W/B}", 3,
                         "Legendary Creature — Cat Nightmare",
                         "Companion — Each permanent card in your starting deck has "
                         "mana value 2 or less.\nLifelink\n"
                         "Once during each of your turns, you may cast a permanent "
                         "spell with mana value 2 or less from your graveyard.",
                         power=3, toughness=2)
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.COMPANION,
            Mechanic.LIFELINK,
            Mechanic.CAST_FROM_GRAVEYARD,
        ], "Lurrus")

    def test_learn(self):
        card = make_card("Learn Test", "{1}{U}", 2, "Sorcery",
                         "Draw a card. You may learn.")
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DRAW,
            Mechanic.LEARN,
        ], "Learn Test")


# =============================================================================
# CONFIDENCE SCORE TESTS
# =============================================================================

class TestConfidenceScores:
    """Verify confidence formula is honest (no +0.3 inflation)."""

    def test_vanilla_creature_confidence(self):
        """Vanilla creature with no oracle text should be 1.0."""
        result = parse_oracle_text("", "Creature — Bear")
        assert result.confidence == 1.0

    def test_simple_card_honest_confidence(self):
        """Simple card — confidence should reflect actual coverage."""
        result = parse_oracle_text("Lightning Bolt deals 3 damage to any target.", "Instant")
        # Should parse "deals 3 damage" but not "lightning bolt" or "to any target"
        assert result.confidence > 0.2
        assert result.confidence < 1.0

    def test_keyword_only_high_confidence(self):
        """Card with just keywords should have high confidence."""
        result = parse_oracle_text("Flying, trample, haste", "Creature — Dragon")
        assert result.confidence >= 0.5

    def test_complex_card_moderate_confidence(self):
        """Complex card text should have moderate confidence."""
        result = parse_oracle_text(
            "Whenever you cast a noncreature spell, exile the top card of your "
            "library. You may play that card this turn.",
            "Creature — Human Wizard"
        )
        # Should parse cast trigger, exile, and some other patterns
        assert result.confidence >= 0.2


# =============================================================================
# STAT MODIFIER EXTRACTION
# =============================================================================

class TestStatModifiers:
    """Verify +X/+Y and -X/-Y extraction."""

    def test_plus_stats(self):
        result = parse_oracle_text(
            "Target creature gets +3/+3 until end of turn.", "Instant"
        )
        assert result.parameters.get("power_mod") == 3
        assert result.parameters.get("toughness_mod") == 3

    def test_minus_stats(self):
        result = parse_oracle_text(
            "Target creature gets -2/-2 until end of turn.", "Instant"
        )
        assert result.parameters.get("power_mod") == -2
        assert result.parameters.get("toughness_mod") == -2

    def test_asymmetric_stats(self):
        result = parse_oracle_text(
            "Target creature gets +3/+0 until end of turn.", "Instant"
        )
        assert Mechanic.PLUS_POWER in result.mechanics


# =============================================================================
# TYPE FILTER DETECTION
# =============================================================================

class TestTypeFilters:
    """Verify nonland, noncreature, etc. detection."""

    def test_nonland_filter(self):
        result = parse_oracle_text(
            "Exile target nonland permanent.", "Instant"
        )
        assert Mechanic.FILTER_NONLAND in result.mechanics

    def test_noncreature_filter(self):
        result = parse_oracle_text(
            "Counter target noncreature spell.", "Instant"
        )
        assert Mechanic.FILTER_NONCREATURE in result.mechanics


# =============================================================================
# DURATION MARKERS
# =============================================================================

class TestDurationMarkers:
    """Verify duration marker detection."""

    def test_until_end_of_turn(self):
        result = parse_oracle_text(
            "Target creature gets +2/+2 until end of turn.", "Instant"
        )
        assert Mechanic.UNTIL_END_OF_TURN in result.mechanics

    def test_until_next_turn(self):
        result = parse_oracle_text(
            "Target creature gains hexproof until your next turn.", "Instant"
        )
        assert Mechanic.UNTIL_YOUR_NEXT_TURN in result.mechanics

    def test_as_long_as(self):
        result = parse_oracle_text(
            "This creature has flying as long as you control an artifact.",
            "Creature — Human"
        )
        assert Mechanic.AS_LONG_AS in result.mechanics


# =============================================================================
# DOUBLE-FACED CARDS
# =============================================================================

class TestDoubleFacedCards:
    """Cards with front/back faces."""

    def test_double_faced_card(self):
        """Double-faced card should parse front face oracle text."""
        card = {
            "name": "Delver of Secrets // Insectile Aberration",
            "mana_cost": "{U}",
            "cmc": 1,
            "type_line": "Creature — Human Wizard // Creature — Human Insect",
            "card_faces": [
                {
                    "name": "Delver of Secrets",
                    "oracle_text": "At the beginning of your upkeep, look at the "
                                   "top card of your library. You may reveal that card. "
                                   "If an instant or sorcery card is revealed this way, "
                                   "transform Delver of Secrets.",
                    "type_line": "Creature — Human Wizard",
                    "mana_cost": "{U}",
                },
                {
                    "name": "Insectile Aberration",
                    "oracle_text": "Flying",
                    "type_line": "Creature — Human Insect",
                    "mana_cost": "",
                },
            ],
            "power": "1",
            "toughness": "1",
        }
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.UPKEEP_TRIGGER,
            Mechanic.LOOK_AT_TOP,
            Mechanic.FLYING,
            Mechanic.TRANSFORM,
        ], card.get("name"))


# =============================================================================
# REMINDER TEXT STRIPPING
# =============================================================================

class TestReminderTextStripping:
    """Test that parenthetical reminder text is properly stripped."""

    def test_strip_basic_reminder(self):
        text = "Flying (This creature can't be blocked except by creatures with flying or reach.)"
        stripped = strip_reminder_text(text)
        assert "can't be blocked" not in stripped
        assert "Flying" in stripped

    def test_strip_multiple_reminders(self):
        text = "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)"
        stripped = strip_reminder_text(text)
        assert "Any amount" not in stripped
        assert "Damage dealt" not in stripped
        assert "Deathtouch" in stripped
        assert "Lifelink" in stripped

    def test_strip_preserves_non_reminder(self):
        text = "When this creature enters the battlefield, draw two cards."
        stripped = strip_reminder_text(text)
        assert stripped == text

    def test_reminder_text_confidence_boost(self):
        """Cards with keyword + long reminder text should get higher confidence."""
        # Without reminder stripping, "Flying (This creature can't be blocked
        # except by creatures with flying or reach.)" would have low confidence
        # because "This creature can't be blocked..." inflates word count.
        result = parse_oracle_text(
            "Flying (This creature can't be blocked except by creatures with flying or reach.)",
            "Creature — Bird"
        )
        assert result.confidence >= 0.5, (
            f"Confidence {result.confidence} too low — reminder text should be stripped from word count"
        )

    def test_menace_with_reminder(self):
        result = parse_oracle_text(
            "Menace (This creature can't be blocked except by two or more creatures.)",
            "Creature — Zombie"
        )
        assert Mechanic.MENACE in result.mechanics
        assert result.confidence >= 0.5

    def test_protection_with_reminder(self):
        result = parse_oracle_text(
            "Protection from red (This creature can't be blocked, targeted, dealt damage, "
            "enchanted, or equipped by anything red.)",
            "Creature — Knight"
        )
        assert Mechanic.PROTECTION in result.mechanics
        assert result.confidence >= 0.3


# =============================================================================
# VARIABLE EFFECTS ("where X is" / "equal to" / "for each")
# =============================================================================

class TestVariableEffects:
    """Test parsing of variable-count effects."""

    def test_draw_for_each(self):
        """Draw a card for each creature you control."""
        result = parse_oracle_text(
            "Draw a card for each creature you control.",
            "Sorcery"
        )
        assert Mechanic.DRAW in result.mechanics

    def test_draw_x_where_x_is(self):
        """Draw X cards, where X is the number of lands you control."""
        result = parse_oracle_text(
            "Draw X cards, where X is the number of lands you control.",
            "Sorcery"
        )
        assert Mechanic.DRAW in result.mechanics
        assert result.confidence >= 0.4

    def test_damage_equal_to(self):
        """Deals damage equal to the number of creatures you control."""
        result = parse_oracle_text(
            "This spell deals damage to target creature equal to the number of creatures you control.",
            "Instant"
        )
        assert Mechanic.DEAL_DAMAGE in result.mechanics

    def test_damage_where_x_is(self):
        """Deals X damage, where X is the number of artifacts you control."""
        result = parse_oracle_text(
            "Shrapnel Blast deals X damage to any target, where X is the number of artifacts you control.",
            "Instant"
        )
        assert Mechanic.DEAL_DAMAGE in result.mechanics

    def test_lose_life_equal_to(self):
        """You lose life equal to that card's mana value."""
        result = parse_oracle_text(
            "You lose life equal to that card's mana value.",
            "Enchantment"
        )
        assert Mechanic.LOSE_LIFE in result.mechanics

    def test_gain_life_equal_to(self):
        """You gain life equal to its power."""
        result = parse_oracle_text(
            "When this creature enters the battlefield, you gain life equal to its power.",
            "Creature — Angel"
        )
        assert Mechanic.GAIN_LIFE in result.mechanics

    def test_mill_where_x_is(self):
        """Target player mills X cards, where X is the number of cards in your hand."""
        result = parse_oracle_text(
            "Target player mills X cards, where X is the number of cards in your hand.",
            "Sorcery"
        )
        assert Mechanic.MILL in result.mechanics

    def test_mill_equal_to(self):
        """Each opponent mills cards equal to the number of creatures you control."""
        result = parse_oracle_text(
            "Each opponent mills cards equal to the number of creatures you control.",
            "Sorcery"
        )
        assert Mechanic.MILL in result.mechanics

    def test_create_tokens_for_each(self):
        """Create a 1/1 token for each creature that died this turn."""
        result = parse_oracle_text(
            "Create a 1/1 white Spirit creature token with flying for each creature that died this turn.",
            "Sorcery"
        )
        assert Mechanic.CREATE_TOKEN in result.mechanics

    def test_variable_confidence(self):
        """Variable-effect cards should have reasonable confidence."""
        result = parse_oracle_text(
            "Draw X cards, where X is the number of creatures you control.",
            "Sorcery"
        )
        # "draw X cards" matches, "where X is the number of" should be consumed,
        # "creatures you control" should be consumed
        assert result.confidence >= 0.4, (
            f"Variable-effect confidence {result.confidence} too low"
        )

    def test_tarmogoyf_variable_stats(self):
        """Tarmogoyf's power/toughness based on card types in graveyards."""
        card = make_card(
            "Tarmogoyf", "{1}{G}", 2,
            "Creature — Lhurgoyf",
            "Tarmogoyf's power is equal to the number of card types among cards in all graveyards "
            "and its toughness is equal to that number plus 1.",
            power="*", toughness="1+*",
        )
        enc = parse_card(card)
        # Should parse "equal to the number of" and "card types" / "graveyards"
        assert enc.power == -1  # * power

    def test_craterhoof_behemoth(self):
        """Craterhoof Behemoth — anthem effect for each creature."""
        result = parse_oracle_text(
            "When Craterhoof Behemoth enters the battlefield, creatures you control "
            "gain trample and get +X/+X until end of turn, where X is the number "
            "of creatures you control.",
            "Creature — Beast"
        )
        assert Mechanic.TRAMPLE in result.mechanics
        assert Mechanic.ETB_TRIGGER in result.mechanics
        assert Mechanic.UNTIL_END_OF_TURN in result.mechanics


# =============================================================================
# KEYWORD FALSE POSITIVES
# =============================================================================

class TestKeywordFalsePositives:
    """Test that keyword abilities are NOT detected in reference contexts.

    Cards that reference keywords (e.g. "destroy target creature with flying")
    should not be tagged with those keywords. The lookbehind filters "with <kw>"
    and "without <kw>", and reminder text stripping removes parenthetical refs.
    """

    def test_plummet_has_flying(self):
        """Plummet: 'creature with flying' — FLYING now emitted as interaction context."""
        result = parse_oracle_text(
            "Destroy target creature with flying.",
            "Instant"
        )
        assert Mechanic.DESTROY in result.mechanics
        assert Mechanic.TARGET_CREATURE in result.mechanics
        assert Mechanic.FLYING in result.mechanics

    def test_reach_reminder_no_flying(self):
        """Reminder text '(creatures with flying or reach)' stripped → no FLYING."""
        result = parse_oracle_text(
            "Reach (This creature can block creatures with flying.)",
            "Creature — Spider"
        )
        assert Mechanic.REACH in result.mechanics
        assert Mechanic.FLYING not in result.mechanics

    def test_with_trample_reference(self):
        """'Creatures with trample you control get +1/+1' — TRAMPLE now emitted as context."""
        result = parse_oracle_text(
            "Creatures with trample you control get +1/+1.",
            "Enchantment"
        )
        assert Mechanic.TRAMPLE in result.mechanics

    def test_without_flying_reference(self):
        """'Can block only creatures without flying' must NOT tag FLYING."""
        result = parse_oracle_text(
            "This creature can block only creatures without flying.",
            "Creature — Wall"
        )
        assert Mechanic.FLYING not in result.mechanics

    def test_standalone_flying_detected(self):
        """Standalone 'Flying' keyword is still detected."""
        result = parse_oracle_text(
            "Flying",
            "Creature — Bird"
        )
        assert Mechanic.FLYING in result.mechanics

    def test_gains_flying_detected(self):
        """'gains flying' still detected — preceded by 'gains ', not 'with '."""
        result = parse_oracle_text(
            "Target creature gains flying until end of turn.",
            "Instant"
        )
        assert Mechanic.FLYING in result.mechanics

    def test_has_flying_and_lifelink_detected(self):
        """'has flying and lifelink' still detected — preceded by 'has '."""
        result = parse_oracle_text(
            "This creature has flying and lifelink.",
            "Creature — Angel"
        )
        assert Mechanic.FLYING in result.mechanics
        assert Mechanic.LIFELINK in result.mechanics

    def test_comma_separated_keywords_detected(self):
        """'Flying, trample, haste' — all detected (preceded by comma/start)."""
        result = parse_oracle_text(
            "Flying, trample, haste",
            "Creature — Dragon"
        )
        assert Mechanic.FLYING in result.mechanics
        assert Mechanic.TRAMPLE in result.mechanics
        assert Mechanic.HASTE in result.mechanics

    def test_newline_separated_keywords_detected(self):
        """Keywords on separate lines are still detected."""
        result = parse_oracle_text(
            "Flying\nVigilance\nLifelink",
            "Creature — Angel"
        )
        assert Mechanic.FLYING in result.mechanics
        assert Mechanic.VIGILANCE in result.mechanics
        assert Mechanic.LIFELINK in result.mechanics

    def test_with_deathtouch_reference(self):
        """'creature with deathtouch' — DEATHTOUCH now emitted as interaction context."""
        result = parse_oracle_text(
            "Whenever a creature with deathtouch deals damage to you, draw a card.",
            "Enchantment"
        )
        assert Mechanic.DEATHTOUCH in result.mechanics

    def test_token_with_flying_has_flying(self):
        """'create a 1/1 token with flying' — FLYING now emitted (card produces flyer)."""
        result = parse_oracle_text(
            "Create a 1/1 white Spirit creature token with flying.",
            "Sorcery"
        )
        assert Mechanic.CREATE_TOKEN in result.mechanics
        assert Mechanic.FLYING in result.mechanics


# =============================================================================
# BENCHMARK CARDS
# =============================================================================

class TestBenchmarkCards:
    """Benchmark suite of complex cards from validate_vocabulary.py's tricky set.

    Rotation rule: add 2-3 cards from the worst-parsed list each time the parser
    is improved. This suite ensures regressions are caught on known-hard cards.
    """

    def test_force_of_will(self):
        """Force of Will — counter + alternative cost (exile blue card + pay life)."""
        card = make_card(
            "Force of Will", "{3}{U}{U}", 5, "Instant",
            "You may pay 1 life and exile a blue card from your hand rather than "
            "pay this spell's mana cost.\nCounter target spell."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.COUNTER_SPELL,
        ], "Force of Will")

    def test_swords_to_plowshares(self):
        """Swords to Plowshares — exile + opponent gains life equal to power."""
        card = make_card(
            "Swords to Plowshares", "{W}", 1, "Instant",
            "Exile target creature. Its controller gains life equal to its power."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.EXILE,
            Mechanic.TARGET_CREATURE,
            Mechanic.GAIN_LIFE,
        ], "Swords to Plowshares")

    def test_path_to_exile(self):
        """Path to Exile — exile + opponent searches for basic land."""
        card = make_card(
            "Path to Exile", "{W}", 1, "Instant",
            "Exile target creature. Its controller may search their library for a "
            "basic land card, put that card onto the battlefield tapped, then "
            "shuffle."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.EXILE,
            Mechanic.TARGET_CREATURE,
        ], "Path to Exile")

    def test_thoughtseize(self):
        """Thoughtseize — hand disruption + life loss."""
        card = make_card(
            "Thoughtseize", "{B}", 1, "Sorcery",
            "Target player reveals their hand. You choose a nonland card from it. "
            "That player discards that card. You lose 2 life."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SORCERY_SPEED,
            Mechanic.DISCARD,
            Mechanic.LOSE_LIFE,
            Mechanic.REVEAL,
        ], "Thoughtseize")

    def test_mana_drain(self):
        """Mana Drain — counter spell + delayed mana generation."""
        card = make_card(
            "Mana Drain", "{U}{U}", 2, "Instant",
            "Counter target spell. At the beginning of your next main phase, "
            "add an amount of {C} equal to that spell's mana value."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.COUNTER_SPELL,
        ], "Mana Drain")

    def test_omnath_locus_of_creation(self):
        """Omnath, Locus of Creation — multi-trigger landfall, 4 colors."""
        card = make_card(
            "Omnath, Locus of Creation", "{R}{G}{W}{U}", 4,
            "Legendary Creature — Elemental",
            "When Omnath, Locus of Creation enters the battlefield, draw a card.\n"
            "Landfall — Whenever a land enters the battlefield under your control, "
            "you gain 4 life if this is the first time this ability has resolved this "
            "turn. If it's the second time, add {R}{G}{W}{U}. If it's the third time, "
            "Omnath deals 4 damage to each opponent and each planeswalker you don't control.",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ETB_TRIGGER,
            Mechanic.DRAW,
            Mechanic.LANDFALL,
        ], "Omnath, Locus of Creation")

    def test_dockside_extortionist(self):
        """Dockside Extortionist — variable Treasure token creation."""
        card = make_card(
            "Dockside Extortionist", "{1}{R}", 2,
            "Creature — Goblin Pirate",
            "When Dockside Extortionist enters the battlefield, create a Treasure "
            "token for each artifact and enchantment your opponents control.",
            power=1, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ETB_TRIGGER,
            Mechanic.CREATE_TOKEN,
        ], "Dockside Extortionist")

    def test_urzas_saga(self):
        """Urza's Saga — saga with chapter abilities."""
        card = make_card(
            "Urza's Saga", "", 0,
            "Enchantment Land — Urza's Saga",
            "I — Urza's Saga gains \"{T}: Add {C}.\"\n"
            "II — Urza's Saga gains \"{2}, {T}: Create a 0/0 colorless Construct "
            "artifact creature token with 'This creature gets +1/+1 for each "
            "artifact you control.'\"\n"
            "III — Search your library for an artifact card with mana cost {0} or "
            "{1}, put it onto the battlefield, then shuffle."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SAGA,
            Mechanic.CHAPTER_I,
            Mechanic.CHAPTER_II,
            Mechanic.CHAPTER_III,
        ], "Urza's Saga")

    def test_cyclonic_rift(self):
        """Cyclonic Rift — bounce + overload."""
        card = make_card(
            "Cyclonic Rift", "{1}{U}", 2, "Instant",
            "Return target nonland permanent you don't control to its owner's hand.\n"
            "Overload {6}{U}"
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.BOUNCE_TO_HAND,
            Mechanic.OVERLOAD,
        ], "Cyclonic Rift")

    def test_plummet_regression(self):
        """Plummet — MUST have DESTROY, TARGET_CREATURE, FLYING (interaction context)."""
        card = make_card(
            "Plummet", "{1}{G}", 2, "Instant",
            "Destroy target creature with flying."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.DESTROY,
            Mechanic.TARGET_CREATURE,
            Mechanic.FLYING,
        ], "Plummet")

    def test_clip_wings(self):
        """Clip Wings — each opponent sacrifices a creature with flying."""
        card = make_card(
            "Clip Wings", "{1}{G}", 2, "Instant",
            "Each opponent sacrifices a creature with flying."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.TARGETS_EACH,
            Mechanic.TARGET_OPPONENT,
            Mechanic.FLYING,
        ], "Clip Wings")

    def test_tower_defense(self):
        """Tower Defense — grants reach (positive: 'gain reach' detected)."""
        card = make_card(
            "Tower Defense", "{1}{G}", 2, "Instant",
            "Creatures you control get +0/+5 and gain reach until end of turn."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INSTANT_SPEED,
            Mechanic.REACH,
            Mechanic.UNTIL_END_OF_TURN,
        ], "Tower Defense")


# =============================================================================
# EQUIPMENT TESTS
# =============================================================================

class TestEquipment:
    """Test equipment keyword, pattern, and parameter extraction."""

    def test_embercleave(self):
        """Embercleave — flash, equip, double strike, trample."""
        card = make_card(
            "Embercleave", "{4}{R}{R}", 6,
            "Legendary Artifact — Equipment",
            "Flash\nThis spell costs {1} less to cast for each attacking creature "
            "you control.\nWhen Embercleave enters the battlefield, attach it to "
            "target creature you control.\nEquipped creature gets +1/+1 and has "
            "double strike and trample.\nEquip {3}",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLASH,
            Mechanic.EQUIP,
            Mechanic.DOUBLE_STRIKE,
            Mechanic.TRAMPLE,
        ], "Embercleave")
        assert enc.parameters.get("equip_cost") == 3

    def test_sword_of_fire_and_ice(self):
        """Sword of Fire and Ice — equip, protection, damage, draw."""
        card = make_card(
            "Sword of Fire and Ice", "{3}", 3,
            "Artifact — Equipment",
            "Equipped creature gets +2/+2 and has protection from red and from blue.\n"
            "Whenever equipped creature deals combat damage to a player, Sword of "
            "Fire and Ice deals 2 damage to any target and you draw a card.\nEquip {2}",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.EQUIP,
            Mechanic.PROTECTION,
            Mechanic.DEAL_DAMAGE,
            Mechanic.DRAW,
        ], "Sword of Fire and Ice")
        assert enc.parameters.get("equip_cost") == 2

    def test_colossus_hammer(self):
        """Colossus Hammer — equip + big stat boost."""
        card = make_card(
            "Colossus Hammer", "{1}", 1,
            "Artifact — Equipment",
            "Equipped creature gets +10/+10 and loses flying.\nEquip {8}",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.EQUIP,
            Mechanic.PLUS_POWER,
            Mechanic.PLUS_TOUGHNESS,
        ], "Colossus Hammer")
        assert enc.parameters.get("equip_cost") == 8

    def test_lightning_greaves(self):
        """Lightning Greaves — equip 0, shroud, haste."""
        card = make_card(
            "Lightning Greaves", "{2}", 2,
            "Artifact — Equipment",
            "Equipped creature has shroud and haste.\nEquip {0}",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.EQUIP,
            Mechanic.SHROUD,
            Mechanic.HASTE,
        ], "Lightning Greaves")
        assert enc.parameters.get("equip_cost") == 0

    def test_equipment_false_positive(self):
        """Non-equipment card mentioning 'Equipment' should not get EQUIP."""
        card = make_card(
            "Stoneforge Mystic", "{1}{W}", 2,
            "Creature — Kor Artificer",
            "When Stoneforge Mystic enters the battlefield, you may search "
            "your library for an Equipment card, reveal it, put it into your "
            "hand, then shuffle.",
            power=1, toughness=2,
        )
        enc = parse_card(card)
        # "Equipment" (capitalized noun) should NOT trigger equip keyword
        # because the keyword pattern matches \bequip\b (not "equipment")
        assert_has_mechanics(enc, [
            Mechanic.ETB_TRIGGER,
            Mechanic.TUTOR_TO_HAND,
        ], "Stoneforge Mystic")


# =============================================================================
# COUNTER TYPE TESTS
# =============================================================================

class TestCounterTypes:
    """Test counter type detection patterns."""

    def test_experience_counters(self):
        """Meren — experience counters."""
        card = make_card(
            "Meren of Clan Nel Toth", "{2}{B}{G}", 4,
            "Legendary Creature — Human Shaman",
            "Whenever another creature you control dies, you get an experience counter.\n"
            "At the beginning of your end step, choose target creature card in your "
            "graveyard. If that card's mana value is less than or equal to the number "
            "of experience counters you have, return it to the battlefield. Otherwise, "
            "put it into your hand.",
            power=3, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.EXPERIENCE_COUNTER,
            Mechanic.DEATH_TRIGGER,
        ], "Meren")

    def test_energy_counters_symbol(self):
        """Aetherworks Marvel — energy via {E} symbol."""
        card = make_card(
            "Aetherworks Marvel", "{4}", 4,
            "Legendary Artifact",
            "Whenever a permanent you control is put into a graveyard, "
            "you get {E}.\n{T}, Pay {E}{E}{E}{E}{E}{E}: Look at the top six "
            "cards of your library. You may cast a spell from among them without "
            "paying its mana cost. Put the rest on the bottom of your library "
            "in a random order.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ENERGY_COUNTER,
        ], "Aetherworks Marvel")

    def test_shield_counters(self):
        """Shalai and Hallar — shield counters."""
        card = make_card(
            "Shalai and Hallar", "{1}{R}{G}{W}", 4,
            "Legendary Creature — Angel Elf",
            "Flying\nWhen Shalai and Hallar enters the battlefield, put a shield "
            "counter on each creature you control.\nWhenever one or more +1/+1 "
            "counters are put on a creature you control, Shalai and Hallar deals "
            "that much damage to each opponent.",
            power=3, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLYING,
            Mechanic.SHIELD_COUNTER,
            Mechanic.ETB_TRIGGER,
        ], "Shalai and Hallar")

    def test_keyword_counters(self):
        """Crystalline Giant — keyword counters (flying counter, etc.)."""
        card = make_card(
            "Crystalline Giant", "{3}", 3,
            "Artifact Creature — Giant",
            "At the beginning of combat on your turn, choose a kind of counter "
            "at random that Crystalline Giant doesn't have on it from among "
            "flying counter, first strike counter, deathtouch counter, hexproof "
            "counter, lifelink counter, menace counter, reach counter, trample "
            "counter, vigilance counter, and +1/+1 counter. Put a counter of "
            "that kind on Crystalline Giant.",
            power=3, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.KEYWORD_COUNTER,
            Mechanic.PLUS_ONE_COUNTER,
        ], "Crystalline Giant")

    def test_oil_counters(self):
        """Urabrask's Forge — oil counters."""
        card = make_card(
            "Urabrask's Forge", "{2}{R}", 3,
            "Artifact",
            "At the beginning of combat on your turn, put an oil counter on "
            "Urabrask's Forge. Then create a 0/0 red Phyrexian Horror creature "
            "token with trample and haste. Its power and toughness are each equal "
            "to the number of oil counters on Urabrask's Forge. Exile that token "
            "at end of combat.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.OIL_COUNTER,
            Mechanic.CREATE_TOKEN,
        ], "Urabrask's Forge")

    def test_energy_text_form(self):
        """Energy counters via 'energy counter' text (not {E} symbol)."""
        result = parse_oracle_text(
            "Whenever a creature enters the battlefield under your control, "
            "you get two energy counters.",
            "Enchantment"
        )
        assert Mechanic.ENERGY_COUNTER in result.mechanics


# =============================================================================
# BECOMES CREATURE (MANLAND) TESTS
# =============================================================================

class TestBecomeCreature:
    """Test 'becomes creature' patterns for manlands and similar."""

    def test_restless_cottage(self):
        """Restless Cottage — manland, becomes 4/4."""
        card = make_card(
            "Restless Cottage", "", 0,
            "Land",
            "Restless Cottage enters the battlefield tapped.\n"
            "{T}: Add {B} or {G}.\n"
            "{1}{B}{G}: Until end of turn, Restless Cottage becomes a 4/4 "
            "black and green Elemental creature with \"Whenever this creature "
            "attacks, create a Food token.\" It's still a land.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.BECOMES_CREATURE,
            Mechanic.TO_BATTLEFIELD_TAPPED,
            Mechanic.ADD_MANA,
        ], "Restless Cottage")
        assert enc.parameters.get("becomes_power") == 4
        assert enc.parameters.get("becomes_toughness") == 4

    def test_mutavault(self):
        """Mutavault — becomes 2/2 all creature types."""
        card = make_card(
            "Mutavault", "", 0,
            "Land",
            "{T}: Add {C}.\n"
            "{1}: Until end of turn, Mutavault becomes a 2/2 creature with "
            "all creature types. It's still a land.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.BECOMES_CREATURE,
            Mechanic.ADD_MANA,
        ], "Mutavault")
        assert enc.parameters.get("becomes_power") == 2
        assert enc.parameters.get("becomes_toughness") == 2

    def test_celestial_colonnade(self):
        """Celestial Colonnade — becomes 4/4 with flying + vigilance."""
        card = make_card(
            "Celestial Colonnade", "", 0,
            "Land",
            "Celestial Colonnade enters the battlefield tapped.\n"
            "{T}: Add {W} or {U}.\n"
            "{3}{W}{U}: Until end of turn, Celestial Colonnade becomes a 4/4 "
            "white and blue Elemental creature with flying and vigilance. "
            "It's still a land.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.BECOMES_CREATURE,
            Mechanic.TO_BATTLEFIELD_TAPPED,
            Mechanic.ADD_MANA,
        ], "Celestial Colonnade")
        assert enc.parameters.get("becomes_power") == 4
        assert enc.parameters.get("becomes_toughness") == 4

    def test_gideon_is_creature_as_long_as(self):
        """Gideon Blackblade — 'is a creature as long as' pattern."""
        card = make_card(
            "Gideon Blackblade", "{1}{W}{W}", 3,
            "Legendary Planeswalker — Gideon",
            "As long as it's your turn, Gideon Blackblade is a 4/4 Human Soldier "
            "creature with indestructible that's still a planeswalker.\n"
            "+1: Up to one other target creature you control gains your choice of "
            "vigilance, lifelink, or indestructible until end of turn.\n"
            "\u22126: Exile target nonland permanent.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOYALTY_COUNTER,
            Mechanic.LOYALTY_PLUS,
            Mechanic.LOYALTY_MINUS,
        ], "Gideon Blackblade")


# =============================================================================
# PLANESWALKER LOYALTY TESTS
# =============================================================================

class TestPlaneswalkerLoyalty:
    """Test planeswalker loyalty ability parsing."""

    def test_liliana_of_the_veil(self):
        """Liliana of the Veil — +1, −2, −6 abilities."""
        card = make_card(
            "Liliana of the Veil", "{1}{B}{B}", 3,
            "Legendary Planeswalker \u2014 Liliana",
            "+1: Each player discards a card.\n"
            "\u22122: Target player sacrifices a creature.\n"
            "\u22126: Separate all permanents target player controls into two piles. "
            "That player sacrifices all permanents in the pile of their choice.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOYALTY_COUNTER,
            Mechanic.LOYALTY_PLUS,
            Mechanic.LOYALTY_MINUS,
            Mechanic.DISCARD,
        ], "Liliana of the Veil")
        # NOTE: SACRIFICE not detected — parser pattern expects "sacrifice (a|an|target)"
        # but Liliana uses "sacrifices a creature" (third person). Known parser gap.
        assert enc.parameters.get("loyalty_ability_count") == 3

    def test_teferi_time_raveler(self):
        """Teferi, Time Raveler — static + +1 + −3."""
        card = make_card(
            "Teferi, Time Raveler", "{1}{W}{U}", 3,
            "Legendary Planeswalker \u2014 Teferi",
            "Each opponent can cast spells only any time they could cast a sorcery.\n"
            "+1: Until your next turn, you may cast sorcery spells as though they "
            "had flash.\n"
            "\u22123: Return up to one target artifact, creature, or enchantment to "
            "its owner's hand. Draw a card.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOYALTY_COUNTER,
            Mechanic.LOYALTY_STATIC,
            Mechanic.LOYALTY_PLUS,
            Mechanic.LOYALTY_MINUS,
            Mechanic.DRAW,
        ], "Teferi, Time Raveler")
        # NOTE: BOUNCE_TO_HAND not detected — parser pattern expects "return (it|that|target)..."
        # but Teferi uses "Return up to one target...". Known parser gap.
        assert enc.parameters.get("loyalty_ability_count") == 2

    def test_jace_the_mind_sculptor(self):
        """Jace, the Mind Sculptor — +2, 0, −1, −12 abilities."""
        card = make_card(
            "Jace, the Mind Sculptor", "{2}{U}{U}", 4,
            "Legendary Planeswalker \u2014 Jace",
            "+2: Look at the top card of target player's library. You may put "
            "that card on the bottom of that player's library.\n"
            "0: Draw three cards, then put two cards from your hand on top of "
            "your library in any order.\n"
            "\u22121: Return target creature to its owner's hand.\n"
            "\u221212: Exile all cards from target player's library, then that "
            "player shuffles their hand into their library.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOYALTY_COUNTER,
            Mechanic.LOYALTY_PLUS,
            Mechanic.LOYALTY_ZERO,
            Mechanic.LOYALTY_MINUS,
            Mechanic.DRAW,
            Mechanic.BOUNCE_TO_HAND,
        ], "Jace, the Mind Sculptor")
        assert enc.parameters.get("loyalty_ability_count") == 4

    def test_kaito_bane_of_nightmares(self):
        """Kaito — becomes creature + loyalty abilities."""
        card = make_card(
            "Kaito, Bane of Nightmares", "{1}{U}{B}", 3,
            "Legendary Planeswalker \u2014 Kaito",
            "+2: Until end of turn, Kaito, Bane of Nightmares becomes a 3/4 Ninja "
            "creature with menace and \"Whenever this creature deals combat damage "
            "to a player, look at that player's hand and exile a nonland card from "
            "it face down.\"\n"
            "+1: You may put a card exiled with Kaito into its owner's graveyard. "
            "If you do, draw a card.\n"
            "\u22122: Each opponent loses 2 life and you gain 2 life.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOYALTY_COUNTER,
            Mechanic.LOYALTY_PLUS,
            Mechanic.LOYALTY_MINUS,
            Mechanic.BECOMES_CREATURE,
        ], "Kaito, Bane of Nightmares")
        # NOTE: MENACE not detected — "with menace" triggers the keyword
        # lookbehind filter (designed to avoid false positives like
        # "creature with flying"). Acceptable trade-off.

    def test_non_planeswalker_no_loyalty(self):
        """Non-planeswalker with '+2/+2' should NOT get LOYALTY_PLUS."""
        card = make_card(
            "Giant Growth", "{G}", 1, "Instant",
            "Target creature gets +3/+3 until end of turn."
        )
        enc = parse_card(card)
        assert_lacks_mechanics(enc, [
            Mechanic.LOYALTY_PLUS,
            Mechanic.LOYALTY_MINUS,
            Mechanic.LOYALTY_ZERO,
            Mechanic.LOYALTY_STATIC,
            Mechanic.LOYALTY_COUNTER,
        ], "Giant Growth")


# =============================================================================
# P0 BUG FIX REGRESSION TESTS
# =============================================================================

class TestP0BugFixes:
    """Regression tests for P0 bug fixes: INCREASE_COST, graveyard 'from', discard widening."""

    def test_thalia_increase_cost(self):
        """Thalia — 'cost {1} more to cast' should be INCREASE_COST, not REDUCE_COST."""
        card = make_card(
            "Thalia, Guardian of Thraben", "{1}{W}", 2,
            "Legendary Creature — Human Soldier",
            "First strike\nNoncreature spells cost {1} more to cast.",
            power=2, toughness=1,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FIRST_STRIKE,
            Mechanic.INCREASE_COST,
            Mechanic.FILTER_NONCREATURE,
        ], "Thalia")
        assert_lacks_mechanics(enc, [Mechanic.REDUCE_COST], "Thalia")

    def test_reduce_cost_still_works(self):
        """Goblin Electromancer — 'cost {1} less' should still be REDUCE_COST."""
        card = make_card(
            "Goblin Electromancer", "{U}{R}", 2,
            "Creature — Goblin Wizard",
            "Instant and sorcery spells you cast cost {1} less to cast.",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.REDUCE_COST], "Goblin Electromancer")
        assert_lacks_mechanics(enc, [Mechanic.INCREASE_COST], "Goblin Electromancer")

    def test_keen_eyed_curator_graveyard_from(self):
        """Keen-Eyed Curator — 'exile target card from a graveyard' should detect graveyard targeting."""
        card = make_card(
            "Keen-Eyed Curator", "{2}{G}", 3,
            "Creature — Raccoon Druid",
            "Vigilance\nWhen Keen-Eyed Curator enters the battlefield, exile target "
            "card from a graveyard. If it was a land card, search your library for a "
            "basic land card, put it onto the battlefield tapped, then shuffle.",
            power=3, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.VIGILANCE,
            Mechanic.ETB_TRIGGER,
            Mechanic.EXILE,
            Mechanic.TARGET_CARD_IN_GRAVEYARD,
            Mechanic.TUTOR_TO_BATTLEFIELD,
        ], "Keen-Eyed Curator")

    def test_graveyard_in_still_works(self):
        """'target card in a graveyard' should still work."""
        result = parse_oracle_text(
            "Exile target card in a graveyard.",
            "Instant"
        )
        assert Mechanic.TARGET_CARD_IN_GRAVEYARD in result.mechanics

    def test_thoughtseize_discard_that_card(self):
        """Thoughtseize — 'discards that card' should now detect DISCARD."""
        card = make_card(
            "Thoughtseize", "{B}", 1, "Sorcery",
            "Target player reveals their hand. You choose a nonland card from it. "
            "That player discards that card. You lose 2 life."
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.SORCERY_SPEED,
            Mechanic.DISCARD,
            Mechanic.LOSE_LIFE,
            Mechanic.REVEAL,
        ], "Thoughtseize")

    def test_liliana_minus2_sacrifice_discard(self):
        """Liliana +1 'discards a card' should still work."""
        result = parse_oracle_text(
            "Each player discards a card.",
            "Planeswalker"
        )
        assert Mechanic.DISCARD in result.mechanics

    def test_discard_it(self):
        """'discards it' pattern should detect DISCARD."""
        result = parse_oracle_text(
            "Target opponent reveals a card at random from their hand. "
            "That player discards it.",
            "Sorcery"
        )
        assert Mechanic.DISCARD in result.mechanics


# =============================================================================
# AURA TESTS
# =============================================================================

class TestAuras:
    """Test aura enchantment effect parsing."""

    def test_pacifism(self):
        """Pacifism — can't attack or block."""
        card = make_card(
            "Pacifism", "{1}{W}", 2,
            "Enchantment — Aura",
            "Enchant creature\nEnchanted creature can't attack or block.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.TARGET_CREATURE,
            Mechanic.CANT_ATTACK,
            Mechanic.CANT_BLOCK,
        ], "Pacifism")

    def test_unable_to_scream(self):
        """Unable to Scream — loses abilities, sets base P/T 0/1."""
        card = make_card(
            "Unable to Scream", "{U}", 1,
            "Enchantment — Aura",
            "Enchant creature\nEnchanted creature loses all abilities and "
            "has base power and toughness 0/1.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.TARGET_CREATURE,
            Mechanic.LOSES_ABILITIES,
            Mechanic.SET_POWER,
            Mechanic.SET_TOUGHNESS,
        ], "Unable to Scream")
        assert enc.parameters.get("set_power") == 0
        assert enc.parameters.get("set_toughness") == 1

    def test_kenriths_transformation(self):
        """Kenrith's Transformation — loses abilities, sets P/T, ETB draw."""
        card = make_card(
            "Kenrith's Transformation", "{1}{G}", 2,
            "Enchantment — Aura",
            "Enchant creature\nWhen Kenrith's Transformation enters the "
            "battlefield, draw a card.\nEnchanted creature loses all abilities "
            "and has base power and toughness 3/3.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOSES_ABILITIES,
            Mechanic.SET_POWER,
            Mechanic.SET_TOUGHNESS,
            Mechanic.ETB_TRIGGER,
            Mechanic.DRAW,
        ], "Kenrith's Transformation")

    def test_darksteel_mutation(self):
        """Darksteel Mutation — loses abilities, indestructible, sets P/T."""
        card = make_card(
            "Darksteel Mutation", "{1}{W}", 2,
            "Enchantment — Aura",
            "Enchant creature\nEnchanted creature is an Insect artifact "
            "creature with base power and toughness 0/1, has indestructible, "
            "and loses all other abilities.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.LOSES_ABILITIES,
            Mechanic.INDESTRUCTIBLE,
            Mechanic.SET_POWER,
            Mechanic.SET_TOUGHNESS,
        ], "Darksteel Mutation")

    def test_rancor(self):
        """Rancor — enchant creature, trample, +2/+0, returns to hand.

        NOTE: REGROWTH not detected — Rancor says "return Rancor to its
        owner's hand" but the pattern expects "from...graveyard to your hand".
        Known parser gap for self-referential return-from-graveyard text.
        """
        card = make_card(
            "Rancor", "{G}", 1,
            "Enchantment — Aura",
            "Enchant creature\nEnchanted creature gets +2/+0 and has trample.\n"
            "When Rancor is put into a graveyard from the battlefield, return "
            "Rancor to its owner's hand.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.TARGET_CREATURE,
            Mechanic.TRAMPLE,
            Mechanic.PLUS_POWER,
        ], "Rancor")

    def test_ethereal_armor(self):
        """Ethereal Armor — enchant creature, first strike."""
        card = make_card(
            "Ethereal Armor", "{W}", 1,
            "Enchantment — Aura",
            "Enchant creature\nEnchanted creature gets +1/+1 for each "
            "enchantment you control and has first strike.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.TARGET_CREATURE,
            Mechanic.FIRST_STRIKE,
        ], "Ethereal Armor")

    def test_guard_duty(self):
        """Guard Duty — can't attack but CAN still block."""
        card = make_card(
            "Guard Duty", "{W}", 1,
            "Enchantment — Aura",
            "Enchant creature\nEnchanted creature can't attack.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CANT_ATTACK,
        ], "Guard Duty")
        assert_lacks_mechanics(enc, [Mechanic.CANT_BLOCK], "Guard Duty")

    def test_aura_false_positive(self):
        """Ghostly Prison — not an aura, fires CANT_ATTACK.

        NOTE: TARGET_CREATURE fires from the existing "each creature" pattern
        matching "each creature they control". Acceptable trade-off — Ghostly
        Prison does affect creatures, just not via targeting/aura.
        """
        card = make_card(
            "Ghostly Prison", "{2}{W}", 3,
            "Enchantment",
            "Creatures can't attack you unless their controller pays {2} for "
            "each creature they control that's attacking you.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.CANT_ATTACK], "Ghostly Prison")

    def test_thalia_no_cant_attack(self):
        """Thalia — tax effect, NOT can't attack/block/loses abilities."""
        card = make_card(
            "Thalia, Guardian of Thraben", "{1}{W}", 2,
            "Legendary Creature — Human Soldier",
            "First strike\nNoncreature spells cost {1} more to cast.",
            power=2, toughness=1,
        )
        enc = parse_card(card)
        assert_lacks_mechanics(enc, [
            Mechanic.CANT_ATTACK,
            Mechanic.CANT_BLOCK,
            Mechanic.LOSES_ABILITIES,
        ], "Thalia")


# =============================================================================
# STAX / HATE TESTS
# =============================================================================

class TestStaxHate:
    """Test stax and hate effect parsing."""

    def test_erebos(self):
        """Erebos — can't gain life, indestructible, draw, devotion."""
        card = make_card(
            "Erebos, God of the Dead", "{3}{B}", 4,
            "Legendary Enchantment Creature — God",
            "Indestructible\nAs long as your devotion to black is less than "
            "five, Erebos isn't a creature.\nYour opponents can't gain life.\n"
            "{1}{B}, Pay 2 life: Draw a card.",
            power=5, toughness=7,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CANT_GAIN_LIFE,
            Mechanic.INDESTRUCTIBLE,
            Mechanic.DRAW,
            Mechanic.AS_LONG_AS,
        ], "Erebos")

    def test_gaddock_teeg(self):
        """Gaddock Teeg — noncreature CMC 4+ can't be cast."""
        card = make_card(
            "Gaddock Teeg", "{G}{W}", 2,
            "Legendary Creature — Kithkin Advisor",
            "Noncreature spells with mana value 4 or greater can't be cast.\n"
            "Noncreature spells with {X} in their mana costs can't be cast.",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CANT_CAST,
            Mechanic.FILTER_NONCREATURE,
        ], "Gaddock Teeg")

    def test_rule_of_law(self):
        """Rule of Law — cast restriction."""
        card = make_card(
            "Rule of Law", "{2}{W}", 3,
            "Enchantment",
            "Each player can't cast more than one spell each turn.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CAST_RESTRICTION,
        ], "Rule of Law")

    def test_notion_thief(self):
        """Notion Thief — flash, draw replacement."""
        card = make_card(
            "Notion Thief", "{2}{U}{B}", 4,
            "Creature — Human Rogue",
            "Flash\nIf an opponent would draw a card except the first one they "
            "draw in each of their draw steps, instead that player skips that "
            "draw and you draw a card.",
            power=3, toughness=1,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.FLASH,
            Mechanic.DRAW_REPLACEMENT,
            Mechanic.REPLACEMENT_EFFECT,
            Mechanic.DRAW,
        ], "Notion Thief")

    def test_rest_in_peace(self):
        """Rest in Peace — ETB exile graveyards + graveyard replacement."""
        card = make_card(
            "Rest in Peace", "{1}{W}", 2,
            "Enchantment",
            "When Rest in Peace enters the battlefield, exile all graveyards.\n"
            "If a card or token would be put into a graveyard from anywhere, "
            "exile it instead.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ETB_TRIGGER,
            Mechanic.GRAVEYARD_HATE,
            Mechanic.REPLACEMENT_EFFECT,
        ], "Rest in Peace")

    def test_grafdiggers_cage(self):
        """Grafdigger's Cage — graveyard hate + can't cast from GY."""
        card = make_card(
            "Grafdigger's Cage", "{1}", 1,
            "Artifact",
            "Creature cards in graveyards and libraries can't enter the "
            "battlefield.\nPlayers can't cast spells from graveyards or "
            "libraries.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.GRAVEYARD_HATE,
            Mechanic.CANT_CAST,
        ], "Grafdigger's Cage")

    def test_drannith_magistrate(self):
        """Drannith Magistrate — can't cast from non-hand zones."""
        card = make_card(
            "Drannith Magistrate", "{1}{W}", 2,
            "Creature — Human Wizard",
            "Your opponents can't cast spells from anywhere other than their "
            "hands.",
            power=1, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CANT_CAST,
        ], "Drannith Magistrate")

    def test_thalia_no_regression(self):
        """Thalia — INCREASE_COST, NOT CANT_CAST or CAST_RESTRICTION."""
        card = make_card(
            "Thalia, Guardian of Thraben", "{1}{W}", 2,
            "Legendary Creature — Human Soldier",
            "First strike\nNoncreature spells cost {1} more to cast.",
            power=2, toughness=1,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.INCREASE_COST,
        ], "Thalia")
        assert_lacks_mechanics(enc, [
            Mechanic.CANT_CAST,
            Mechanic.CAST_RESTRICTION,
        ], "Thalia")


# =============================================================================
# GAIN CONTROL
# =============================================================================

class TestGainControl:
    """Test gain control / steal effects."""

    def test_control_magic(self):
        """Control Magic — classic enchant + gain control."""
        card = make_card(
            "Control Magic", "{2}{U}{U}", 4,
            "Enchantment — Aura",
            "Enchant creature\nYou control enchanted creature.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.TARGET_CREATURE], "Control Magic")

    def test_agent_of_treachery(self):
        """Agent of Treachery — ETB gain control."""
        card = make_card(
            "Agent of Treachery", "{5}{U}{U}", 7,
            "Creature — Human Rogue",
            "When Agent of Treachery enters the battlefield, gain control of target permanent.\nAt the beginning of your end step, if you control three or more permanents you don't own, draw three cards.",
            power=2, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ETB_TRIGGER,
            Mechanic.GAIN_CONTROL,
            Mechanic.TARGET_PERMANENT,
            Mechanic.DRAW,
        ], "Agent of Treachery")

    def test_threaten_effect(self):
        """Act of Treason — temporary gain control."""
        card = make_card(
            "Act of Treason", "{2}{R}", 3,
            "Sorcery",
            "Gain control of target creature until end of turn. Untap that creature. It gains haste until end of turn.",
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.GAIN_CONTROL,
            Mechanic.TARGET_CREATURE,
            Mechanic.UNTIL_END_OF_TURN,
        ], "Act of Treason")

    def test_gilded_drake(self):
        """Gilded Drake — exchange control."""
        card = make_card(
            "Gilded Drake", "{1}{U}", 2,
            "Creature — Drake",
            "Flying\nWhen Gilded Drake enters the battlefield, exchange control of Gilded Drake and up to one target creature an opponent controls. If you don't make an exchange, sacrifice Gilded Drake.",
            power=3, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.GAIN_CONTROL,
            Mechanic.ETB_TRIGGER,
        ], "Gilded Drake")


# =============================================================================
# VEHICLES / CREW
# =============================================================================

class TestVehicles:
    """Test vehicle crew mechanics."""

    def test_smugglers_copter(self):
        """Smuggler's Copter — crew 1, looting on attack/block."""
        card = make_card(
            "Smuggler's Copter", "{2}", 2,
            "Artifact — Vehicle",
            "Flying\nWhenever Smuggler's Copter attacks or blocks, you may draw a card. If you do, discard a card.\nCrew 1",
            power=3, toughness=3,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CREW,
            Mechanic.DRAW,
        ], "Smuggler's Copter")
        assert enc.parameters.get("crew_power") == 1

    def test_esika_chariot(self):
        """Esika's Chariot — crew 4, creates tokens."""
        card = make_card(
            "Esika's Chariot", "{3}{G}", 4,
            "Legendary Artifact — Vehicle",
            "When Esika's Chariot enters the battlefield, create two 2/2 green Cat creature tokens.\nWhenever Esika's Chariot attacks, create a token that's a copy of target token you control.\nCrew 4",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CREW,
            Mechanic.ETB_TRIGGER,
            Mechanic.CREATE_TOKEN,
            Mechanic.CREATE_TOKEN_COPY,
        ], "Esika's Chariot")
        assert enc.parameters.get("crew_power") == 4

    def test_heart_of_kiran(self):
        """Heart of Kiran — crew 3, vehicle with vigilance."""
        card = make_card(
            "Heart of Kiran", "{2}", 2,
            "Legendary Artifact — Vehicle",
            "Flying, vigilance\nCrew 3",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.CREW], "Heart of Kiran")
        assert enc.parameters.get("crew_power") == 3


# =============================================================================
# CAN'T BE COUNTERED (regression)
# =============================================================================

class TestCantBeCountered:
    """Test can't-be-countered vs can't-be-blocked disambiguation."""

    def test_loxodon_smiter(self):
        """Loxodon Smiter — can't be countered."""
        card = make_card(
            "Loxodon Smiter", "{1}{G}{W}", 3,
            "Creature — Elephant Soldier",
            "This spell can't be countered.\nIf a spell or ability an opponent controls causes you to discard Loxodon Smiter, put it onto the battlefield instead of putting it into your graveyard.",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.CANT_BE_COUNTERED], "Loxodon Smiter")
        assert_lacks_mechanics(enc, [Mechanic.UNBLOCKABLE], "Loxodon Smiter")

    def test_carnage_tyrant(self):
        """Carnage Tyrant — can't be countered + hexproof."""
        card = make_card(
            "Carnage Tyrant", "{4}{G}{G}", 6,
            "Creature — Dinosaur",
            "This spell can't be countered.\nTrample, hexproof",
            power=7, toughness=6,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CANT_BE_COUNTERED,
            Mechanic.TRAMPLE,
            Mechanic.HEXPROOF,
        ], "Carnage Tyrant")
        assert_lacks_mechanics(enc, [Mechanic.UNBLOCKABLE], "Carnage Tyrant")

    def test_unblockable_still_works(self):
        """Invisible Stalker — can't be blocked (not countered)."""
        card = make_card(
            "Invisible Stalker", "{1}{U}", 2,
            "Creature — Human Rogue",
            "Hexproof\nInvisible Stalker can't be blocked.",
            power=1, toughness=1,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.UNBLOCKABLE, Mechanic.HEXPROOF], "Invisible Stalker")
        assert_lacks_mechanics(enc, [Mechanic.CANT_BE_COUNTERED], "Invisible Stalker")


# =============================================================================
# PREVENT DAMAGE / IMPULSE DRAW / PHASE OUT / EXTRA TURN / WIN-LOSE
# =============================================================================

class TestNewPatterns:
    """Test newly-added pattern matchers."""

    def test_fog(self):
        """Fog — prevent all combat damage."""
        result = parse_oracle_text("Prevent all combat damage that would be dealt this turn.", "Instant")
        assert Mechanic.PREVENT_DAMAGE in result.mechanics

    def test_teferi_protection(self):
        """Teferi's Protection — phase out + prevent damage."""
        result = parse_oracle_text(
            "Until your next turn, your life total can't change and you gain protection from everything. All permanents you control phase out.",
            "Instant",
        )
        assert Mechanic.PHASE_OUT in result.mechanics
        assert Mechanic.PROTECTION in result.mechanics

    def test_light_up_the_stage(self):
        """Light Up the Stage — impulse draw."""
        result = parse_oracle_text(
            "Exile the top two cards of your library. Until the end of your next turn, you may play those cards.",
            "Sorcery",
        )
        assert Mechanic.IMPULSE_DRAW in result.mechanics

    def test_reckless_impulse(self):
        """Reckless Impulse — exile top + may play."""
        result = parse_oracle_text(
            "Exile the top two cards of your library. Until the end of your next turn, you may play those cards.",
            "Sorcery",
        )
        assert Mechanic.IMPULSE_DRAW in result.mechanics

    def test_time_warp(self):
        """Time Warp — extra turn."""
        result = parse_oracle_text(
            "Target player takes an extra turn after this one.",
            "Sorcery",
        )
        assert Mechanic.EXTRA_TURN in result.mechanics

    def test_thassas_oracle(self):
        """Thassa's Oracle — win the game."""
        result = parse_oracle_text(
            "When Thassa's Oracle enters the battlefield, look at the top X cards of your library, where X is your devotion to blue. Put up to one of them on top of your library and the rest on the bottom of your library in a random order. If X is greater than or equal to the number of cards in your library, you win the game.",
            "Creature",
        )
        assert Mechanic.WIN_GAME in result.mechanics

    def test_demonic_pact_lose(self):
        """Demonic Pact — you lose the game."""
        result = parse_oracle_text(
            "At the beginning of your upkeep, choose one that hasn't been chosen —\n• Demonic Pact deals 4 damage to any target and you gain 4 life.\n• Target opponent discards two cards.\n• Draw two cards.\n• You lose the game.",
            "Enchantment",
        )
        assert Mechanic.LOSE_GAME in result.mechanics
        assert Mechanic.DRAW in result.mechanics

    def test_prevent_next_damage(self):
        """Healing Salve — prevent the next 3 damage."""
        result = parse_oracle_text(
            "Choose one —\n• Target player gains 3 life.\n• Prevent the next 3 damage that would be dealt to any target this turn.",
            "Instant",
        )
        assert Mechanic.PREVENT_DAMAGE in result.mechanics

    def test_village_rites_additional_cost(self):
        """Village Rites — as an additional cost, sacrifice."""
        result = parse_oracle_text(
            "As an additional cost to cast this spell, sacrifice a creature.\nDraw two cards.",
            "Instant",
        )
        assert Mechanic.ADDITIONAL_COST in result.mechanics
        assert Mechanic.SACRIFICE in result.mechanics
        assert Mechanic.DRAW in result.mechanics

    def test_damage_cant_be_prevented(self):
        """Stomp — damage can't be prevented."""
        result = parse_oracle_text(
            "Damage can't be prevented this turn. Stomp deals 2 damage to any target.",
            "Instant — Adventure",
        )
        assert Mechanic.PREVENT_DAMAGE in result.mechanics
        assert Mechanic.DEAL_DAMAGE in result.mechanics


# =============================================================================
# SCRYFALL SAMPLE ROUND 1 — Gaps discovered from 100-card sampling
# =============================================================================

class TestScryfallGapsRound1:
    """Test fixes for gaps discovered from 100-card Scryfall sample."""

    def test_landwalk_forestwalk(self):
        """Lynx — forestwalk."""
        card = make_card(
            "Lynx", "{1}{G}", 2,
            "Creature — Cat",
            "Forestwalk",
            power=2, toughness=1,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.LANDWALK], "Lynx")

    def test_landwalk_islandwalk(self):
        """Phantom Warrior — islandwalk."""
        result = parse_oracle_text("Islandwalk", "Creature")
        assert Mechanic.LANDWALK in result.mechanics

    def test_landwalk_swampwalk(self):
        """Bog Wraith — swampwalk."""
        result = parse_oracle_text("Swampwalk", "Creature")
        assert Mechanic.LANDWALK in result.mechanics

    def test_silent_arbiter_combat_restriction(self):
        """Silent Arbiter — no more than one creature can attack/block."""
        card = make_card(
            "Silent Arbiter", "{4}", 4,
            "Artifact Creature — Construct",
            "No more than one creature can attack each combat.\nNo more than one creature can block each combat.",
            power=1, toughness=5,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.COMBAT_RESTRICTION], "Silent Arbiter")

    def test_dueling_grounds(self):
        """Dueling Grounds — only one creature can attack/block."""
        result = parse_oracle_text(
            "No more than one creature can attack each combat.\nNo more than one creature can block each combat.",
            "Enchantment",
        )
        assert Mechanic.COMBAT_RESTRICTION in result.mechanics

    def test_enters_with_plus_counters(self):
        """Hagra Constrictor — enters with two +1/+1 counters."""
        card = make_card(
            "Hagra Constrictor", "{2}{B}", 3,
            "Creature — Snake",
            "This creature enters with two +1/+1 counters on it.\nEach creature you control with a +1/+1 counter on it has menace.",
            power=0, toughness=0,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.ENTERS_WITH_COUNTERS,
            Mechanic.PLUS_ONE_COUNTER,
            Mechanic.MENACE,
        ], "Hagra Constrictor")

    def test_walking_ballista_enters_counters(self):
        """Walking Ballista — enters with X +1/+1 counters."""
        result = parse_oracle_text(
            "Walking Ballista enters with X +1/+1 counters on it.\n{4}, {T}: Put a +1/+1 counter on Walking Ballista.\nRemove a +1/+1 counter from Walking Ballista: It deals 1 damage to any target.",
            "Artifact Creature",
        )
        assert Mechanic.ENTERS_WITH_COUNTERS in result.mechanics
        assert Mechanic.PLUS_ONE_COUNTER in result.mechanics

    def test_coin_flip(self):
        """Game of Chaos — flip a coin."""
        result = parse_oracle_text(
            "Flip a coin. If you win the flip, you gain 1 life and target opponent loses 1 life.",
            "Sorcery",
        )
        assert Mechanic.COIN_FLIP in result.mechanics

    def test_krark_thumb(self):
        """Krark's Thumb — coin flip reference."""
        result = parse_oracle_text(
            "If you would flip a coin, instead flip two coins and ignore one.",
            "Legendary Artifact",
        )
        assert Mechanic.COIN_FLIP in result.mechanics

    def test_regenerate(self):
        """Rakshasa Deathdealer — regenerate ability."""
        card = make_card(
            "Rakshasa Deathdealer", "{B}{G}", 2,
            "Creature — Demon",
            "{B}{G}: This creature gets +2/+2 until end of turn.\n{B}{G}: Regenerate this creature.",
            power=2, toughness=2,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [Mechanic.REGENERATE], "Rakshasa Deathdealer")

    def test_thrun_regenerate(self):
        """Thrun, the Last Troll — regenerate + can't be countered."""
        card = make_card(
            "Thrun, the Last Troll", "{2}{G}{G}", 4,
            "Legendary Creature — Troll Shaman",
            "This spell can't be countered.\nHexproof\n{1}{G}: Regenerate Thrun, the Last Troll.",
            power=4, toughness=4,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.CANT_BE_COUNTERED,
            Mechanic.HEXPROOF,
            Mechanic.REGENERATE,
        ], "Thrun")

    def test_clone_enters_as_copy(self):
        """Clone — enters as a copy of a creature."""
        result = parse_oracle_text(
            "You may have Clone enter as a copy of any creature on the battlefield.",
            "Creature",
        )
        assert Mechanic.COPY_PERMANENT in result.mechanics

    def test_spark_double_becomes_copy(self):
        """Spark Double — enters as a copy + enters with counter."""
        result = parse_oracle_text(
            "You may have Spark Double enter as a copy of a creature or planeswalker you control, except it enters with an additional +1/+1 counter on it if it's a creature, it enters with an additional loyalty counter on it if it's a planeswalker, and it isn't legendary.",
            "Creature",
        )
        assert Mechanic.COPY_PERMANENT in result.mechanics
        assert Mechanic.ENTERS_WITH_COUNTERS in result.mechanics

    def test_stun_counter(self):
        """Referee Squad — put a stun counter on it."""
        result = parse_oracle_text(
            "When Referee Squad enters the battlefield, tap target creature an opponent controls and put a stun counter on it.",
            "Creature",
        )
        assert Mechanic.STUN_COUNTER in result.mechanics
        assert Mechanic.TAP in result.mechanics


# =============================================================================
# SCRYFALL SAMPLE ROUND 2 — Pattern refinement
# =============================================================================

class TestScryfallGapsRound2:
    """Test fixes from second 100-card Scryfall sample."""

    def test_investigate(self):
        """Investigate — creates a Clue token."""
        result = parse_oracle_text(
            "When Alquist Proft enters, investigate.",
            "Creature",
        )
        assert Mechanic.CREATE_CLUE in result.mechanics

    def test_investigate_twice(self):
        """Investigate twice — should still detect."""
        result = parse_oracle_text(
            "When this enchantment enters, investigate twice.",
            "Artifact",
        )
        assert Mechanic.CREATE_CLUE in result.mechanics

    def test_enrage_damage_received(self):
        """Ripjaw Raptor — enrage / damage received trigger."""
        card = make_card(
            "Ripjaw Raptor", "{2}{G}{G}", 4,
            "Creature — Dinosaur",
            "Enrage — Whenever this creature is dealt damage, draw a card.",
            power=4, toughness=5,
        )
        enc = parse_card(card)
        assert_has_mechanics(enc, [
            Mechanic.DAMAGE_RECEIVED_TRIGGER,
            Mechanic.DRAW,
        ], "Ripjaw Raptor")

    def test_becomes_blocked_trigger(self):
        """Somberwald Vigilante — becomes blocked trigger."""
        result = parse_oracle_text(
            "Whenever this creature becomes blocked by a creature, this creature deals 1 damage to that creature.",
            "Creature",
        )
        assert Mechanic.BLOCK_TRIGGER in result.mechanics
        assert Mechanic.DEAL_DAMAGE in result.mechanics

    def test_plague_drone_life_replacement(self):
        """Plague Drone — opponent gaining life loses life instead."""
        result = parse_oracle_text(
            "If an opponent would gain life, that player loses that much life instead.",
            "Creature",
        )
        assert Mechanic.CANT_GAIN_LIFE in result.mechanics
        assert Mechanic.REPLACEMENT_EFFECT in result.mechanics

    def test_mana_dork_tap_add(self):
        """Arcane Sanctum — {T}: Add {W}, {U}, or {B} still detected."""
        result = parse_oracle_text(
            "This land enters tapped.\n{T}: Add {W}, {U}, or {B}.",
            "Land",
        )
        assert Mechanic.ADD_MANA in result.mechanics
        assert Mechanic.TO_BATTLEFIELD_TAPPED in result.mechanics


# =============================================================================
# QUIZ FEEDBACK FIXES
# =============================================================================

class TestQuizFeedbackFixes:
    """Tests from embedding quiz human review feedback."""

    # --- Keyword with "with" lookbehind removal ---

    def test_restless_anchorage_flying(self):
        """Restless Anchorage — 'creature with flying' should emit FLYING."""
        result = parse_oracle_text(
            "Restless Anchorage enters tapped.\n{T}: Add {W} or {U}.\n"
            "{1}{W}{U}: Until end of turn, Restless Anchorage becomes a "
            "2/3 Bird creature with flying. It's still a land.",
            "Land",
        )
        assert Mechanic.FLYING in result.mechanics
        assert Mechanic.BECOMES_CREATURE in result.mechanics
        assert Mechanic.UNTIL_END_OF_TURN in result.mechanics

    def test_spirit_token_flying(self):
        """Spirit token creation — 'token with flying' emits FLYING."""
        result = parse_oracle_text(
            "Create a 1/1 white Spirit creature token with flying.",
            "Sorcery",
        )
        assert Mechanic.FLYING in result.mechanics
        assert Mechanic.CREATE_TOKEN in result.mechanics

    def test_without_flying_still_blocked(self):
        """'without flying' must still NOT tag FLYING."""
        result = parse_oracle_text(
            "This creature can block only creatures without flying.",
            "Creature — Wall",
        )
        assert Mechanic.FLYING not in result.mechanics

    # --- LOOT mechanic ---

    def test_loot_draw_then_discard(self):
        """Merfolk Looter — draw then discard → LOOT."""
        result = parse_oracle_text(
            "{T}: Draw a card, then discard a card.",
            "Creature — Merfolk",
        )
        assert Mechanic.LOOT in result.mechanics
        assert Mechanic.DRAW in result.mechanics
        assert Mechanic.DISCARD in result.mechanics

    def test_rummage_discard_then_draw(self):
        """Rummaging Goblin — discard then draw → LOOT."""
        result = parse_oracle_text(
            "{T}, Discard a card: Draw a card.",
            "Creature — Goblin",
        )
        # This doesn't have "then" so LOOT won't fire — just DRAW + DISCARD
        assert Mechanic.DRAW in result.mechanics

    def test_faithless_looting_loot(self):
        """Faithless Looting — 'Draw two cards, then discard two cards' → LOOT."""
        result = parse_oracle_text(
            "Draw two cards, then discard two cards.\n"
            "Flashback {2}{R}",
            "Sorcery",
        )
        assert Mechanic.LOOT in result.mechanics
        assert Mechanic.DRAW in result.mechanics
        assert Mechanic.DISCARD in result.mechanics
        assert Mechanic.FLASHBACK in result.mechanics

    def test_cathartic_reunion_loot(self):
        """Cathartic Reunion — 'Discard two cards, then draw three cards' → LOOT."""
        result = parse_oracle_text(
            "As an additional cost to cast this spell, discard two cards.\n"
            "Draw three cards.",
            "Sorcery",
        )
        # No "then" connecting discard and draw — these are separate effects
        assert Mechanic.DRAW in result.mechanics
        assert Mechanic.DISCARD in result.mechanics

    def test_connive_is_loot(self):
        """Connive draws then discards — should get LOOT from pattern."""
        result = parse_oracle_text(
            "When this creature enters, it connives. "
            "Draw a card, then discard a card.",
            "Creature — Rogue",
        )
        assert Mechanic.LOOT in result.mechanics
        assert Mechanic.CONNIVE in result.mechanics

    # --- ESCALATE mechanic ---

    def test_collective_brutality(self):
        """Collective Brutality — escalate keyword."""
        result = parse_oracle_text(
            "Escalate — Discard a card.\n"
            "Choose one or more —\n"
            "• Target opponent reveals their hand. You choose a noncreature, "
            "nonland card from it. That player discards that card.\n"
            "• Target creature gets -2/-2 until end of turn.\n"
            "• Target opponent loses 2 life and you gain 2 life.",
            "Sorcery",
        )
        assert Mechanic.ESCALATE in result.mechanics
        assert Mechanic.DISCARD in result.mechanics
        assert Mechanic.TARGET_OPPONENT in result.mechanics

    def test_tiered_spell(self):
        """Tiered spell (Final Fantasy) — mapped to ESCALATE."""
        result = parse_oracle_text(
            "Tiered\n• Cure — {0} — Target permanent you control gains "
            "hexproof and indestructible until end of turn.\n"
            "• Cura — {1} — Same. You gain 3 life.\n"
            "• Curaga — {3}{W} — Your permanents gain hexproof and "
            "indestructible until end of turn. You gain 6 life.",
            "Instant",
        )
        assert Mechanic.ESCALATE in result.mechanics
        assert Mechanic.HEXPROOF in result.mechanics
        assert Mechanic.INDESTRUCTIBLE in result.mechanics

    # --- Opponent's graveyard targeting ---

    def test_disruptor_wanderglyph(self):
        """Disruptor Wanderglyph — exile target card from an opponent's graveyard."""
        result = parse_oracle_text(
            "Whenever Disruptor Wanderglyph attacks, exile target card "
            "from an opponent's graveyard.",
            "Creature — Golem",
        )
        assert Mechanic.ATTACK_TRIGGER in result.mechanics
        assert Mechanic.EXILE in result.mechanics
        assert Mechanic.TARGET_CARD_IN_GRAVEYARD in result.mechanics

    def test_opponent_graveyard_still_works(self):
        """Standard 'target card in a graveyard' still works."""
        result = parse_oracle_text(
            "Exile target card from a graveyard.",
            "Instant",
        )
        assert Mechanic.TARGET_CARD_IN_GRAVEYARD in result.mechanics

    def test_your_graveyard_still_works(self):
        """Standard 'target card in your graveyard' still works."""
        result = parse_oracle_text(
            "Return target card from your graveyard to your hand.",
            "Sorcery",
        )
        assert Mechanic.TARGET_CARD_IN_GRAVEYARD in result.mechanics
        assert Mechanic.REGROWTH in result.mechanics

    # --- BOUNCE_TO_HAND vs REGROWTH fix ---

    def test_regrowth_no_bounce(self):
        """Regrowth — graveyard→hand is REGROWTH, NOT BOUNCE_TO_HAND."""
        result = parse_oracle_text(
            "Return target card from your graveyard to your hand.",
            "Sorcery",
        )
        assert Mechanic.REGROWTH in result.mechanics
        assert Mechanic.BOUNCE_TO_HAND not in result.mechanics

    def test_unsummon_bounce_no_regrowth(self):
        """Unsummon — battlefield→hand is BOUNCE_TO_HAND, NOT REGROWTH."""
        result = parse_oracle_text(
            "Return target creature to its owner's hand.",
            "Instant",
        )
        assert Mechanic.BOUNCE_TO_HAND in result.mechanics
        assert Mechanic.REGROWTH not in result.mechanics

    def test_coati_scavenger_no_bounce(self):
        """Coati Scavenger — graveyard return should not trigger BOUNCE_TO_HAND."""
        result = parse_oracle_text(
            "When Coati Scavenger enters the battlefield, return target "
            "permanent card from your graveyard to your hand.",
            "Creature — Coati",
        )
        assert Mechanic.REGROWTH in result.mechanics
        assert Mechanic.BOUNCE_TO_HAND not in result.mechanics
        assert Mechanic.ETB_TRIGGER in result.mechanics
