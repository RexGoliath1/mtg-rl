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
