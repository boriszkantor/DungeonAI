"""D&D 5E Level Progression Data.

This module contains all the static data needed for character level progression:
- XP thresholds for each level
- Proficiency bonus by level
- Class features by class and level
- Spell slots by class and level
- Hit dice by class

NEURO-SYMBOLIC PRINCIPLE:
This is TRUTH. All progression data comes from official 5E rules.
LLMs reference this data but cannot invent their own values.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# XP Thresholds (PHB p.15)
# =============================================================================

XP_THRESHOLDS: dict[int, int] = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
    6: 14000,
    7: 23000,
    8: 34000,
    9: 48000,
    10: 64000,
    11: 85000,
    12: 100000,
    13: 120000,
    14: 140000,
    15: 165000,
    16: 195000,
    17: 225000,
    18: 265000,
    19: 305000,
    20: 355000,
}


def get_level_for_xp(xp: int) -> int:
    """Determine character level based on XP."""
    for level in range(20, 0, -1):
        if xp >= XP_THRESHOLDS[level]:
            return level
    return 1


def get_xp_for_next_level(current_level: int) -> int | None:
    """Get XP needed for the next level. Returns None at level 20."""
    if current_level >= 20:
        return None
    return XP_THRESHOLDS[current_level + 1]


def get_xp_progress(xp: int, current_level: int) -> tuple[int, int]:
    """Get (current_xp_in_level, xp_needed_for_level).
    
    Returns:
        Tuple of (progress, total) for progress bar display.
    """
    if current_level >= 20:
        return (0, 0)  # Max level
    
    current_threshold = XP_THRESHOLDS[current_level]
    next_threshold = XP_THRESHOLDS[current_level + 1]
    
    progress = xp - current_threshold
    total = next_threshold - current_threshold
    
    return (progress, total)


# =============================================================================
# Proficiency Bonus by Level (PHB p.15)
# =============================================================================

def get_proficiency_bonus(level: int) -> int:
    """Get proficiency bonus for a given level."""
    if level < 1:
        return 2
    if level <= 4:
        return 2
    if level <= 8:
        return 3
    if level <= 12:
        return 4
    if level <= 16:
        return 5
    return 6  # Levels 17-20


# =============================================================================
# Hit Dice by Class
# =============================================================================

CLASS_HIT_DIE: dict[str, int] = {
    "Barbarian": 12,
    "Fighter": 10,
    "Paladin": 10,
    "Ranger": 10,
    "Bard": 8,
    "Cleric": 8,
    "Druid": 8,
    "Monk": 8,
    "Rogue": 8,
    "Warlock": 8,
    "Sorcerer": 6,
    "Wizard": 6,
}


def get_hit_die(class_name: str) -> int:
    """Get hit die size for a class."""
    return CLASS_HIT_DIE.get(class_name, 8)


def calculate_hp_increase(class_name: str, con_mod: int, roll: int | None = None) -> int:
    """Calculate HP increase on level up.
    
    Args:
        class_name: The class gaining a level.
        con_mod: Constitution modifier.
        roll: If provided, use this roll. Otherwise use average.
        
    Returns:
        HP gained this level.
    """
    hit_die = get_hit_die(class_name)
    
    if roll is not None:
        # Use actual roll (clamped to minimum 1)
        hp_from_die = max(1, roll)
    else:
        # Use average (rounded up per PHB)
        hp_from_die = (hit_die // 2) + 1
    
    # Always gain at least 1 HP per level
    return max(1, hp_from_die + con_mod)


# =============================================================================
# Spell Slots by Level (Full Casters)
# =============================================================================

# Full casters: Bard, Cleric, Druid, Sorcerer, Wizard
FULL_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {1: 2},
    2:  {1: 3},
    3:  {1: 4, 2: 2},
    4:  {1: 4, 2: 3},
    5:  {1: 4, 2: 3, 3: 2},
    6:  {1: 4, 2: 3, 3: 3},
    7:  {1: 4, 2: 3, 3: 3, 4: 1},
    8:  {1: 4, 2: 3, 3: 3, 4: 2},
    9:  {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    10: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
    11: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1},
    12: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1},
    13: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1},
    14: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1},
    15: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1},
    16: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1},
    17: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1},
    18: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1, 7: 1, 8: 1, 9: 1},
    19: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 2, 7: 1, 8: 1, 9: 1},
    20: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 2, 7: 2, 8: 1, 9: 1},
}

# Half casters: Paladin, Ranger (start at level 2)
HALF_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {},
    2:  {1: 2},
    3:  {1: 3},
    4:  {1: 3},
    5:  {1: 4, 2: 2},
    6:  {1: 4, 2: 2},
    7:  {1: 4, 2: 3},
    8:  {1: 4, 2: 3},
    9:  {1: 4, 2: 3, 3: 2},
    10: {1: 4, 2: 3, 3: 2},
    11: {1: 4, 2: 3, 3: 3},
    12: {1: 4, 2: 3, 3: 3},
    13: {1: 4, 2: 3, 3: 3, 4: 1},
    14: {1: 4, 2: 3, 3: 3, 4: 1},
    15: {1: 4, 2: 3, 3: 3, 4: 2},
    16: {1: 4, 2: 3, 3: 3, 4: 2},
    17: {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    18: {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    19: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
    20: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
}

# Warlock pact magic
WARLOCK_PACT_SLOTS: dict[int, tuple[int, int]] = {
    # level: (num_slots, slot_level)
    1:  (1, 1),
    2:  (2, 1),
    3:  (2, 2),
    4:  (2, 2),
    5:  (2, 3),
    6:  (2, 3),
    7:  (2, 4),
    8:  (2, 4),
    9:  (2, 5),
    10: (2, 5),
    11: (3, 5),
    12: (3, 5),
    13: (3, 5),
    14: (3, 5),
    15: (3, 5),
    16: (3, 5),
    17: (4, 5),
    18: (4, 5),
    19: (4, 5),
    20: (4, 5),
}

# Third casters: Eldritch Knight, Arcane Trickster (start at level 3)
THIRD_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {},
    2:  {},
    3:  {1: 2},
    4:  {1: 3},
    5:  {1: 3},
    6:  {1: 3},
    7:  {1: 4, 2: 2},
    8:  {1: 4, 2: 2},
    9:  {1: 4, 2: 2},
    10: {1: 4, 2: 3},
    11: {1: 4, 2: 3},
    12: {1: 4, 2: 3},
    13: {1: 4, 2: 3, 3: 2},
    14: {1: 4, 2: 3, 3: 2},
    15: {1: 4, 2: 3, 3: 2},
    16: {1: 4, 2: 3, 3: 3},
    17: {1: 4, 2: 3, 3: 3},
    18: {1: 4, 2: 3, 3: 3},
    19: {1: 4, 2: 3, 3: 3, 4: 1},
    20: {1: 4, 2: 3, 3: 3, 4: 1},
}

# Classify classes by caster type
FULL_CASTERS = {"Bard", "Cleric", "Druid", "Sorcerer", "Wizard"}
HALF_CASTERS = {"Paladin", "Ranger"}
PACT_CASTERS = {"Warlock"}
NON_CASTERS = {"Barbarian", "Fighter", "Monk", "Rogue"}


def get_spell_slots(class_name: str, level: int, subclass: str | None = None) -> dict[int, int]:
    """Get spell slots for a class at a given level.
    
    Args:
        class_name: The character's class.
        level: The character's level in that class.
        subclass: Subclass name (for third casters like Eldritch Knight).
        
    Returns:
        Dict of {spell_level: num_slots}.
    """
    # Check for third caster subclasses
    third_caster_subclasses = {"Eldritch Knight", "Arcane Trickster"}
    if subclass in third_caster_subclasses:
        return THIRD_CASTER_SLOTS.get(level, {})
    
    if class_name in FULL_CASTERS:
        return FULL_CASTER_SLOTS.get(level, {})
    elif class_name in HALF_CASTERS:
        return HALF_CASTER_SLOTS.get(level, {})
    elif class_name in PACT_CASTERS:
        # Warlock uses pact magic, not standard slots
        # Return empty - pact slots handled separately
        return {}
    else:
        return {}


def get_pact_magic(level: int) -> tuple[int, int] | None:
    """Get Warlock pact magic slots.
    
    Returns:
        Tuple of (num_slots, slot_level) or None if not a warlock level.
    """
    return WARLOCK_PACT_SLOTS.get(level)


# =============================================================================
# ASI Levels (Ability Score Improvements)
# =============================================================================

# Standard ASI levels for most classes
STANDARD_ASI_LEVELS = {4, 8, 12, 16, 19}

# Fighter gets extra ASIs
FIGHTER_ASI_LEVELS = {4, 6, 8, 12, 14, 16, 19}

# Rogue gets extra ASI
ROGUE_ASI_LEVELS = {4, 8, 10, 12, 16, 19}


def is_asi_level(class_name: str, level: int) -> bool:
    """Check if this level grants an ASI or Feat."""
    if class_name == "Fighter":
        return level in FIGHTER_ASI_LEVELS
    elif class_name == "Rogue":
        return level in ROGUE_ASI_LEVELS
    else:
        return level in STANDARD_ASI_LEVELS


# =============================================================================
# Class Features by Level
# =============================================================================

CLASS_FEATURES: dict[str, dict[int, list[str]]] = {
    "Barbarian": {
        1: ["Rage (2/day)", "Unarmored Defense"],
        2: ["Reckless Attack", "Danger Sense"],
        3: ["Primal Path"],  # Subclass
        4: ["ASI"],
        5: ["Extra Attack", "Fast Movement"],
        6: ["Path Feature"],
        7: ["Feral Instinct"],
        8: ["ASI"],
        9: ["Brutal Critical (1 die)"],
        10: ["Path Feature"],
        11: ["Relentless Rage"],
        12: ["ASI"],
        13: ["Brutal Critical (2 dice)"],
        14: ["Path Feature"],
        15: ["Persistent Rage"],
        16: ["ASI"],
        17: ["Brutal Critical (3 dice)"],
        18: ["Indomitable Might"],
        19: ["ASI"],
        20: ["Primal Champion", "Rage (Unlimited)"],
    },
    "Bard": {
        1: ["Spellcasting", "Bardic Inspiration (d6)"],
        2: ["Jack of All Trades", "Song of Rest (d6)"],
        3: ["Bard College", "Expertise"],  # Subclass
        4: ["ASI"],
        5: ["Bardic Inspiration (d8)", "Font of Inspiration"],
        6: ["Countercharm", "College Feature"],
        7: [],
        8: ["ASI"],
        9: ["Song of Rest (d8)"],
        10: ["Bardic Inspiration (d10)", "Expertise", "Magical Secrets"],
        11: [],
        12: ["ASI"],
        13: ["Song of Rest (d10)"],
        14: ["Magical Secrets", "College Feature"],
        15: ["Bardic Inspiration (d12)"],
        16: ["ASI"],
        17: ["Song of Rest (d12)"],
        18: ["Magical Secrets"],
        19: ["ASI"],
        20: ["Superior Inspiration"],
    },
    "Cleric": {
        1: ["Spellcasting", "Divine Domain"],  # Subclass
        2: ["Channel Divinity (1/rest)", "Domain Feature"],
        3: [],
        4: ["ASI"],
        5: ["Destroy Undead (CR 1/2)"],
        6: ["Channel Divinity (2/rest)", "Domain Feature"],
        7: [],
        8: ["ASI", "Destroy Undead (CR 1)", "Domain Feature"],
        9: [],
        10: ["Divine Intervention"],
        11: ["Destroy Undead (CR 2)"],
        12: ["ASI"],
        13: [],
        14: ["Destroy Undead (CR 3)"],
        15: [],
        16: ["ASI"],
        17: ["Destroy Undead (CR 4)", "Domain Feature"],
        18: ["Channel Divinity (3/rest)"],
        19: ["ASI"],
        20: ["Divine Intervention Improvement"],
    },
    "Druid": {
        1: ["Druidic", "Spellcasting"],
        2: ["Wild Shape", "Druid Circle"],  # Subclass
        3: [],
        4: ["ASI", "Wild Shape Improvement"],
        5: [],
        6: ["Circle Feature"],
        7: [],
        8: ["ASI", "Wild Shape Improvement"],
        9: [],
        10: ["Circle Feature"],
        11: [],
        12: ["ASI"],
        13: [],
        14: ["Circle Feature"],
        15: [],
        16: ["ASI"],
        17: [],
        18: ["Timeless Body", "Beast Spells"],
        19: ["ASI"],
        20: ["Archdruid"],
    },
    "Fighter": {
        1: ["Fighting Style", "Second Wind"],
        2: ["Action Surge (1 use)"],
        3: ["Martial Archetype"],  # Subclass
        4: ["ASI"],
        5: ["Extra Attack"],
        6: ["ASI"],
        7: ["Archetype Feature"],
        8: ["ASI"],
        9: ["Indomitable (1 use)"],
        10: ["Archetype Feature"],
        11: ["Extra Attack (2)"],
        12: ["ASI"],
        13: ["Indomitable (2 uses)"],
        14: ["ASI"],
        15: ["Archetype Feature"],
        16: ["ASI"],
        17: ["Action Surge (2 uses)", "Indomitable (3 uses)"],
        18: ["Archetype Feature"],
        19: ["ASI"],
        20: ["Extra Attack (3)"],
    },
    "Monk": {
        1: ["Unarmored Defense", "Martial Arts (d4)"],
        2: ["Ki (2 points)", "Unarmored Movement (+10 ft)"],
        3: ["Monastic Tradition", "Deflect Missiles"],  # Subclass
        4: ["ASI", "Slow Fall"],
        5: ["Extra Attack", "Stunning Strike", "Martial Arts (d6)"],
        6: ["Ki-Empowered Strikes", "Tradition Feature", "Unarmored Movement (+15 ft)"],
        7: ["Evasion", "Stillness of Mind"],
        8: ["ASI"],
        9: ["Unarmored Movement Improvement (+20 ft)"],
        10: ["Purity of Body", "Unarmored Movement (+20 ft)"],
        11: ["Tradition Feature", "Martial Arts (d8)"],
        12: ["ASI"],
        13: ["Tongue of the Sun and Moon"],
        14: ["Diamond Soul", "Unarmored Movement (+25 ft)"],
        15: ["Timeless Body"],
        16: ["ASI"],
        17: ["Tradition Feature", "Martial Arts (d10)"],
        18: ["Empty Body", "Unarmored Movement (+30 ft)"],
        19: ["ASI"],
        20: ["Perfect Self"],
    },
    "Paladin": {
        1: ["Divine Sense", "Lay on Hands"],
        2: ["Fighting Style", "Spellcasting", "Divine Smite"],
        3: ["Divine Health", "Sacred Oath"],  # Subclass
        4: ["ASI"],
        5: ["Extra Attack"],
        6: ["Aura of Protection"],
        7: ["Oath Feature"],
        8: ["ASI"],
        9: [],
        10: ["Aura of Courage"],
        11: ["Improved Divine Smite"],
        12: ["ASI"],
        13: [],
        14: ["Cleansing Touch"],
        15: ["Oath Feature"],
        16: ["ASI"],
        17: [],
        18: ["Aura Improvements"],
        19: ["ASI"],
        20: ["Oath Feature"],
    },
    "Ranger": {
        1: ["Favored Enemy", "Natural Explorer"],
        2: ["Fighting Style", "Spellcasting"],
        3: ["Ranger Archetype", "Primeval Awareness"],  # Subclass
        4: ["ASI"],
        5: ["Extra Attack"],
        6: ["Favored Enemy Improvement", "Natural Explorer Improvement"],
        7: ["Archetype Feature"],
        8: ["ASI", "Land's Stride"],
        9: [],
        10: ["Natural Explorer Improvement", "Hide in Plain Sight"],
        11: ["Archetype Feature"],
        12: ["ASI"],
        13: [],
        14: ["Favored Enemy Improvement", "Vanish"],
        15: ["Archetype Feature"],
        16: ["ASI"],
        17: [],
        18: ["Feral Senses"],
        19: ["ASI"],
        20: ["Foe Slayer"],
    },
    "Rogue": {
        1: ["Expertise", "Sneak Attack (1d6)", "Thieves' Cant"],
        2: ["Cunning Action"],
        3: ["Roguish Archetype", "Sneak Attack (2d6)"],  # Subclass
        4: ["ASI"],
        5: ["Uncanny Dodge", "Sneak Attack (3d6)"],
        6: ["Expertise"],
        7: ["Evasion", "Sneak Attack (4d6)"],
        8: ["ASI"],
        9: ["Archetype Feature", "Sneak Attack (5d6)"],
        10: ["ASI"],
        11: ["Reliable Talent", "Sneak Attack (6d6)"],
        12: ["ASI"],
        13: ["Archetype Feature", "Sneak Attack (7d6)"],
        14: ["Blindsense"],
        15: ["Slippery Mind", "Sneak Attack (8d6)"],
        16: ["ASI"],
        17: ["Archetype Feature", "Sneak Attack (9d6)"],
        18: ["Elusive"],
        19: ["ASI", "Sneak Attack (10d6)"],
        20: ["Stroke of Luck"],
    },
    "Sorcerer": {
        1: ["Spellcasting", "Sorcerous Origin"],  # Subclass
        2: ["Font of Magic (2 points)"],
        3: ["Metamagic (2 options)"],
        4: ["ASI"],
        5: [],
        6: ["Origin Feature"],
        7: [],
        8: ["ASI"],
        9: [],
        10: ["Metamagic (3 options)"],
        11: [],
        12: ["ASI"],
        13: [],
        14: ["Origin Feature"],
        15: [],
        16: ["ASI"],
        17: ["Metamagic (4 options)"],
        18: ["Origin Feature"],
        19: ["ASI"],
        20: ["Sorcerous Restoration"],
    },
    "Warlock": {
        1: ["Otherworldly Patron", "Pact Magic"],  # Subclass
        2: ["Eldritch Invocations (2)"],
        3: ["Pact Boon"],
        4: ["ASI"],
        5: ["Eldritch Invocations (3)"],
        6: ["Patron Feature"],
        7: ["Eldritch Invocations (4)"],
        8: ["ASI"],
        9: ["Eldritch Invocations (5)"],
        10: ["Patron Feature"],
        11: ["Mystic Arcanum (6th level)"],
        12: ["ASI", "Eldritch Invocations (6)"],
        13: ["Mystic Arcanum (7th level)"],
        14: ["Patron Feature"],
        15: ["Mystic Arcanum (8th level)", "Eldritch Invocations (7)"],
        16: ["ASI"],
        17: ["Mystic Arcanum (9th level)"],
        18: ["Eldritch Invocations (8)"],
        19: ["ASI"],
        20: ["Eldritch Master"],
    },
    "Wizard": {
        1: ["Spellcasting", "Arcane Recovery"],
        2: ["Arcane Tradition"],  # Subclass
        3: [],
        4: ["ASI"],
        5: [],
        6: ["Tradition Feature"],
        7: [],
        8: ["ASI"],
        9: [],
        10: ["Tradition Feature"],
        11: [],
        12: ["ASI"],
        13: [],
        14: ["Tradition Feature"],
        15: [],
        16: ["ASI"],
        17: [],
        18: ["Spell Mastery"],
        19: ["ASI"],
        20: ["Signature Spells"],
    },
}


def get_features_at_level(class_name: str, level: int) -> list[str]:
    """Get class features gained at a specific level."""
    if class_name not in CLASS_FEATURES:
        return []
    return CLASS_FEATURES[class_name].get(level, [])


def get_all_features_up_to_level(class_name: str, level: int) -> list[str]:
    """Get all class features from level 1 up to the given level."""
    if class_name not in CLASS_FEATURES:
        return []
    
    features = []
    for lvl in range(1, level + 1):
        features.extend(CLASS_FEATURES[class_name].get(lvl, []))
    return features


# =============================================================================
# Feature Resources (Limited-Use Features)
# =============================================================================

# Maps feature name patterns to their resource tracking info
# Format: pattern -> {"uses": int or level-based dict, "recharge": "short_rest"|"long_rest"|"dawn"}
FEATURE_RESOURCES: dict[str, dict[str, Any]] = {
    # Barbarian
    "Rage": {"uses": {1: 2, 3: 3, 6: 4, 12: 5, 17: 6, 20: -1}, "recharge": "long_rest", "description": "Enter a rage for extra damage and resistance"},
    "Relentless Rage": {"uses": None, "recharge": "at_will", "description": "Keep fighting when dropped to 0 HP (DC increases each use)"},
    
    # Bard
    "Bardic Inspiration": {"uses": "charisma_mod", "recharge": "long_rest", "description": "Grant an ally an inspiration die"},
    
    # Cleric
    "Channel Divinity": {"uses": {1: 1, 6: 2, 18: 3}, "recharge": "short_rest", "description": "Channel divine energy for special effects"},
    
    # Druid
    "Wild Shape": {"uses": 2, "recharge": "short_rest", "description": "Transform into a beast"},
    
    # Fighter
    "Second Wind": {"uses": 1, "recharge": "short_rest", "description": "Recover 1d10 + level HP as a bonus action"},
    "Action Surge": {"uses": {2: 1, 17: 2}, "recharge": "short_rest", "description": "Take an additional action"},
    "Indomitable": {"uses": {9: 1, 13: 2, 17: 3}, "recharge": "long_rest", "description": "Reroll a failed saving throw"},
    
    # Monk
    "Ki Points": {"uses": "level", "recharge": "short_rest", "description": "Fuel special monk abilities"},
    "Stillness of Mind": {"uses": None, "recharge": "at_will", "description": "End charmed/frightened as an action"},
    
    # Paladin
    "Lay on Hands": {"uses": "level_x5", "recharge": "long_rest", "description": "Heal with a pool of HP equal to level × 5"},
    "Divine Sense": {"uses": "charisma_mod_plus_1", "recharge": "long_rest", "description": "Detect celestials, fiends, and undead"},
    "Cleansing Touch": {"uses": "charisma_mod", "recharge": "long_rest", "description": "End one spell on yourself or willing creature"},
    
    # Ranger
    "Favored Foe": {"uses": "proficiency_bonus", "recharge": "long_rest", "description": "Mark a creature for extra damage"},
    
    # Rogue
    "Sneak Attack": {"uses": None, "recharge": "at_will", "description": "Extra damage once per turn when you have advantage"},
    "Cunning Action": {"uses": None, "recharge": "at_will", "description": "Dash, Disengage, or Hide as a bonus action"},
    "Evasion": {"uses": None, "recharge": "at_will", "description": "Take no damage on successful DEX saves"},
    "Stroke of Luck": {"uses": 1, "recharge": "short_rest", "description": "Turn a miss into a hit, or treat a check as 20"},
    
    # Sorcerer
    "Sorcery Points": {"uses": "level", "recharge": "long_rest", "description": "Create spell slots or fuel metamagic"},
    
    # Warlock
    "Eldritch Invocations": {"uses": None, "recharge": "at_will", "description": "Permanent magical abilities"},
    
    # Wizard
    "Arcane Recovery": {"uses": 1, "recharge": "long_rest", "description": "Recover spell slots during short rest"},
    
    # Common racial features
    "Breath Weapon": {"uses": 1, "recharge": "short_rest", "description": "Exhale destructive energy"},
    "Relentless Endurance": {"uses": 1, "recharge": "long_rest", "description": "Drop to 1 HP instead of 0"},
    "Infernal Legacy": {"uses": 1, "recharge": "long_rest", "description": "Cast a racial spell"},
    "Hellish Rebuke": {"uses": 1, "recharge": "long_rest", "description": "React to damage with fire"},
    "Fey Ancestry": {"uses": None, "recharge": "at_will", "description": "Advantage vs charm, immune to magical sleep"},
    "Lucky": {"uses": 3, "recharge": "long_rest", "description": "Reroll a d20"},
}


def get_feature_resource_info(feature_name: str, level: int = 1, stats: dict[str, int] | None = None) -> dict[str, Any] | None:
    """Get resource info for a feature if it has limited uses.
    
    Args:
        feature_name: Name of the feature (case-insensitive, partial match supported)
        level: Character level (for level-based uses)
        stats: Character stats dict (for stat-based uses like charisma_mod)
    
    Returns:
        Dict with 'uses_max', 'recharge', 'description' or None if at-will
    """
    # Check for exact or partial match
    feature_lower = feature_name.lower()
    matched_key = None
    
    for key in FEATURE_RESOURCES:
        if key.lower() in feature_lower or feature_lower in key.lower():
            matched_key = key
            break
    
    if matched_key is None:
        return None
    
    info = FEATURE_RESOURCES[matched_key]
    uses_spec = info.get("uses")
    recharge = info.get("recharge", "at_will")
    description = info.get("description", "")
    
    # Calculate actual uses
    if uses_spec is None:
        return {"uses_max": None, "recharge": recharge, "description": description}
    
    if uses_spec == -1:
        return {"uses_max": None, "recharge": "at_will", "description": description}  # Unlimited
    
    if isinstance(uses_spec, int):
        return {"uses_max": uses_spec, "recharge": recharge, "description": description}
    
    if isinstance(uses_spec, dict):
        # Level-based uses
        uses = 0
        for lvl, num in sorted(uses_spec.items()):
            if level >= lvl:
                uses = num
        if uses == -1:
            return {"uses_max": None, "recharge": "at_will", "description": description}
        return {"uses_max": uses, "recharge": recharge, "description": description}
    
    if isinstance(uses_spec, str):
        # Stat-based uses
        stats = stats or {}
        if uses_spec == "level":
            return {"uses_max": level, "recharge": recharge, "description": description}
        elif uses_spec == "level_x5":
            return {"uses_max": level * 5, "recharge": recharge, "description": description}
        elif uses_spec == "proficiency_bonus":
            from dnd_manager.models.progression import get_proficiency_bonus
            return {"uses_max": get_proficiency_bonus(level), "recharge": recharge, "description": description}
        elif uses_spec == "charisma_mod":
            cha = stats.get("charisma", 10)
            mod = max(1, (cha - 10) // 2)
            return {"uses_max": mod, "recharge": recharge, "description": description}
        elif uses_spec == "charisma_mod_plus_1":
            cha = stats.get("charisma", 10)
            mod = max(1, ((cha - 10) // 2) + 1)
            return {"uses_max": mod, "recharge": recharge, "description": description}
    
    return None


# =============================================================================
# Cantrips Known
# =============================================================================

CANTRIPS_KNOWN: dict[str, dict[int, int]] = {
    "Bard": {1: 2, 4: 3, 10: 4},
    "Cleric": {1: 3, 4: 4, 10: 5},
    "Druid": {1: 2, 4: 3, 10: 4},
    "Sorcerer": {1: 4, 4: 5, 10: 6},
    "Warlock": {1: 2, 4: 3, 10: 4},
    "Wizard": {1: 3, 4: 4, 10: 5},
}


def get_cantrips_known(class_name: str, level: int) -> int:
    """Get number of cantrips known at a level."""
    if class_name not in CANTRIPS_KNOWN:
        return 0
    
    cantrip_progression = CANTRIPS_KNOWN[class_name]
    known = 0
    for lvl, num in sorted(cantrip_progression.items()):
        if level >= lvl:
            known = num
    return known


# =============================================================================
# Spells Known (for spontaneous casters)
# =============================================================================

SPELLS_KNOWN: dict[str, dict[int, int]] = {
    "Bard": {
        1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11,
        9: 12, 10: 14, 11: 15, 12: 15, 13: 16, 14: 18,
        15: 19, 16: 19, 17: 20, 18: 22, 19: 22, 20: 22,
    },
    "Ranger": {
        1: 0, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 5,
        9: 6, 10: 6, 11: 7, 12: 7, 13: 8, 14: 8,
        15: 9, 16: 9, 17: 10, 18: 10, 19: 11, 20: 11,
    },
    "Sorcerer": {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
        9: 10, 10: 11, 11: 12, 12: 12, 13: 13, 14: 13,
        15: 14, 16: 14, 17: 15, 18: 15, 19: 15, 20: 15,
    },
    "Warlock": {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
        9: 10, 10: 10, 11: 11, 12: 11, 13: 12, 14: 12,
        15: 13, 16: 13, 17: 14, 18: 14, 19: 15, 20: 15,
    },
}


def get_spells_known(class_name: str, level: int) -> int:
    """Get number of spells known for spontaneous casters."""
    if class_name not in SPELLS_KNOWN:
        return 0
    return SPELLS_KNOWN[class_name].get(level, 0)


# =============================================================================
# Level Up Helper
# =============================================================================


def get_level_up_info(class_name: str, new_level: int, subclass: str | None = None) -> dict[str, Any]:
    """Get all the information needed for a level up.
    
    Args:
        class_name: The class being leveled.
        new_level: The level being gained.
        subclass: The subclass (if any).
        
    Returns:
        Dict with all level-up information.
    """
    return {
        "level": new_level,
        "class": class_name,
        "hit_die": get_hit_die(class_name),
        "proficiency_bonus": get_proficiency_bonus(new_level),
        "features": get_features_at_level(class_name, new_level),
        "is_asi_level": is_asi_level(class_name, new_level),
        "spell_slots": get_spell_slots(class_name, new_level, subclass),
        "cantrips_known": get_cantrips_known(class_name, new_level),
        "spells_known": get_spells_known(class_name, new_level),
        "pact_magic": get_pact_magic(new_level) if class_name == "Warlock" else None,
        "is_subclass_level": new_level == 3 or (new_level == 1 and class_name in ("Cleric", "Sorcerer", "Warlock")),
    }


# =============================================================================
# Skills and Skill Proficiencies
# =============================================================================

ALL_SKILLS = [
    "acrobatics", "animal_handling", "arcana", "athletics", "deception",
    "history", "insight", "intimidation", "investigation", "medicine",
    "nature", "perception", "performance", "persuasion", "religion",
    "sleight_of_hand", "stealth", "survival",
]

SKILL_ABILITIES: dict[str, str] = {
    "acrobatics": "dexterity",
    "animal_handling": "wisdom",
    "arcana": "intelligence",
    "athletics": "strength",
    "deception": "charisma",
    "history": "intelligence",
    "insight": "wisdom",
    "intimidation": "charisma",
    "investigation": "intelligence",
    "medicine": "wisdom",
    "nature": "intelligence",
    "perception": "wisdom",
    "performance": "charisma",
    "persuasion": "charisma",
    "religion": "intelligence",
    "sleight_of_hand": "dexterity",
    "stealth": "dexterity",
    "survival": "wisdom",
}

# Skills available to choose from for each class
CLASS_SKILL_OPTIONS: dict[str, list[str]] = {
    "Barbarian": ["animal_handling", "athletics", "intimidation", "nature", "perception", "survival"],
    "Bard": ALL_SKILLS,  # Bards can choose any skill
    "Cleric": ["history", "insight", "medicine", "persuasion", "religion"],
    "Druid": ["arcana", "animal_handling", "insight", "medicine", "nature", "perception", "religion", "survival"],
    "Fighter": ["acrobatics", "animal_handling", "athletics", "history", "insight", "intimidation", "perception", "survival"],
    "Monk": ["acrobatics", "athletics", "history", "insight", "religion", "stealth"],
    "Paladin": ["athletics", "insight", "intimidation", "medicine", "persuasion", "religion"],
    "Ranger": ["animal_handling", "athletics", "insight", "investigation", "nature", "perception", "stealth", "survival"],
    "Rogue": ["acrobatics", "athletics", "deception", "insight", "intimidation", "investigation", "perception", "performance", "persuasion", "sleight_of_hand", "stealth"],
    "Sorcerer": ["arcana", "deception", "insight", "intimidation", "persuasion", "religion"],
    "Warlock": ["arcana", "deception", "history", "intimidation", "investigation", "nature", "religion"],
    "Wizard": ["arcana", "history", "insight", "investigation", "medicine", "religion"],
}

# Number of skill proficiencies each class gets
CLASS_SKILL_COUNT: dict[str, int] = {
    "Barbarian": 2,
    "Bard": 3,
    "Cleric": 2,
    "Druid": 2,
    "Fighter": 2,
    "Monk": 2,
    "Paladin": 2,
    "Ranger": 3,
    "Rogue": 4,
    "Sorcerer": 2,
    "Warlock": 2,
    "Wizard": 2,
}


def get_skill_options(class_name: str) -> list[str]:
    """Get available skill options for a class."""
    return CLASS_SKILL_OPTIONS.get(class_name, [])


def get_skill_count(class_name: str) -> int:
    """Get number of skill proficiencies for a class."""
    return CLASS_SKILL_COUNT.get(class_name, 2)


# =============================================================================
# Racial Traits
# =============================================================================

RACIAL_TRAITS: dict[str, dict[str, Any]] = {
    "Human": {
        "ability_bonuses": {"strength": 1, "dexterity": 1, "constitution": 1, "intelligence": 1, "wisdom": 1, "charisma": 1},
        "size": "Medium",
        "speed": 30,
        "traits": ["Extra Language"],
        "languages": ["Common", "One extra language"],
    },
    "Elf": {
        "ability_bonuses": {"dexterity": 2},
        "size": "Medium",
        "speed": 30,
        "traits": ["Darkvision (60 ft)", "Fey Ancestry", "Trance", "Keen Senses"],
        "languages": ["Common", "Elvish"],
        "skill_proficiencies": ["perception"],
    },
    "High Elf": {
        "ability_bonuses": {"dexterity": 2, "intelligence": 1},
        "size": "Medium",
        "speed": 30,
        "traits": ["Darkvision (60 ft)", "Fey Ancestry", "Trance", "Keen Senses", "Elf Weapon Training", "Cantrip"],
        "languages": ["Common", "Elvish"],
        "skill_proficiencies": ["perception"],
    },
    "Wood Elf": {
        "ability_bonuses": {"dexterity": 2, "wisdom": 1},
        "size": "Medium",
        "speed": 35,
        "traits": ["Darkvision (60 ft)", "Fey Ancestry", "Trance", "Keen Senses", "Elf Weapon Training", "Fleet of Foot", "Mask of the Wild"],
        "languages": ["Common", "Elvish"],
        "skill_proficiencies": ["perception"],
    },
    "Dwarf": {
        "ability_bonuses": {"constitution": 2},
        "size": "Medium",
        "speed": 25,
        "traits": ["Darkvision (60 ft)", "Dwarven Resilience", "Dwarven Combat Training", "Stonecunning"],
        "languages": ["Common", "Dwarvish"],
    },
    "Hill Dwarf": {
        "ability_bonuses": {"constitution": 2, "wisdom": 1},
        "size": "Medium",
        "speed": 25,
        "traits": ["Darkvision (60 ft)", "Dwarven Resilience", "Dwarven Combat Training", "Stonecunning", "Dwarven Toughness"],
        "languages": ["Common", "Dwarvish"],
        "hp_bonus_per_level": 1,
    },
    "Mountain Dwarf": {
        "ability_bonuses": {"constitution": 2, "strength": 2},
        "size": "Medium",
        "speed": 25,
        "traits": ["Darkvision (60 ft)", "Dwarven Resilience", "Dwarven Combat Training", "Stonecunning", "Dwarven Armor Training"],
        "languages": ["Common", "Dwarvish"],
    },
    "Halfling": {
        "ability_bonuses": {"dexterity": 2},
        "size": "Small",
        "speed": 25,
        "traits": ["Lucky", "Brave", "Halfling Nimbleness"],
        "languages": ["Common", "Halfling"],
    },
    "Lightfoot Halfling": {
        "ability_bonuses": {"dexterity": 2, "charisma": 1},
        "size": "Small",
        "speed": 25,
        "traits": ["Lucky", "Brave", "Halfling Nimbleness", "Naturally Stealthy"],
        "languages": ["Common", "Halfling"],
    },
    "Stout Halfling": {
        "ability_bonuses": {"dexterity": 2, "constitution": 1},
        "size": "Small",
        "speed": 25,
        "traits": ["Lucky", "Brave", "Halfling Nimbleness", "Stout Resilience"],
        "languages": ["Common", "Halfling"],
    },
    "Dragonborn": {
        "ability_bonuses": {"strength": 2, "charisma": 1},
        "size": "Medium",
        "speed": 30,
        "traits": ["Draconic Ancestry", "Breath Weapon", "Damage Resistance"],
        "languages": ["Common", "Draconic"],
    },
    "Gnome": {
        "ability_bonuses": {"intelligence": 2},
        "size": "Small",
        "speed": 25,
        "traits": ["Darkvision (60 ft)", "Gnome Cunning"],
        "languages": ["Common", "Gnomish"],
    },
    "Rock Gnome": {
        "ability_bonuses": {"intelligence": 2, "constitution": 1},
        "size": "Small",
        "speed": 25,
        "traits": ["Darkvision (60 ft)", "Gnome Cunning", "Artificer's Lore", "Tinker"],
        "languages": ["Common", "Gnomish"],
    },
    "Half-Elf": {
        "ability_bonuses": {"charisma": 2},  # Plus 2 others of choice
        "size": "Medium",
        "speed": 30,
        "traits": ["Darkvision (60 ft)", "Fey Ancestry", "Skill Versatility"],
        "languages": ["Common", "Elvish", "One extra language"],
        "extra_skill_count": 2,
    },
    "Half-Orc": {
        "ability_bonuses": {"strength": 2, "constitution": 1},
        "size": "Medium",
        "speed": 30,
        "traits": ["Darkvision (60 ft)", "Menacing", "Relentless Endurance", "Savage Attacks"],
        "languages": ["Common", "Orc"],
        "skill_proficiencies": ["intimidation"],
    },
    "Tiefling": {
        "ability_bonuses": {"charisma": 2, "intelligence": 1},
        "size": "Medium",
        "speed": 30,
        "traits": ["Darkvision (60 ft)", "Hellish Resistance", "Infernal Legacy"],
        "languages": ["Common", "Infernal"],
    },
}


def get_racial_traits(race: str) -> dict[str, Any]:
    """Get racial traits for a race."""
    # Check for exact match first, then base race
    if race in RACIAL_TRAITS:
        return RACIAL_TRAITS[race]
    
    # Check base race (e.g., "High Elf" -> "Elf")
    for base_race in RACIAL_TRAITS:
        if base_race in race or race in base_race:
            return RACIAL_TRAITS[base_race]
    
    # Default human-like stats
    return {
        "ability_bonuses": {},
        "size": "Medium",
        "speed": 30,
        "traits": [],
        "languages": ["Common"],
    }


def get_ability_bonuses(race: str) -> dict[str, int]:
    """Get ability score bonuses for a race."""
    traits = get_racial_traits(race)
    return traits.get("ability_bonuses", {})


# =============================================================================
# Spellcasting Ability by Class
# =============================================================================

SPELLCASTING_ABILITY: dict[str, str] = {
    "Bard": "charisma",
    "Cleric": "wisdom",
    "Druid": "wisdom",
    "Paladin": "charisma",
    "Ranger": "wisdom",
    "Sorcerer": "charisma",
    "Warlock": "charisma",
    "Wizard": "intelligence",
}


def get_spellcasting_ability(class_name: str) -> str | None:
    """Get the spellcasting ability for a class."""
    return SPELLCASTING_ABILITY.get(class_name)


def is_spellcaster(class_name: str) -> bool:
    """Check if a class is a spellcaster."""
    return class_name in SPELLCASTING_ABILITY


# =============================================================================
# Starting Cantrips by Class
# =============================================================================

# These are PHB cantrips that are good starting choices
SUGGESTED_STARTING_CANTRIPS: dict[str, list[str]] = {
    "Bard": ["Vicious Mockery", "Minor Illusion"],
    "Cleric": ["Sacred Flame", "Guidance", "Light"],
    "Druid": ["Druidcraft", "Produce Flame"],
    "Sorcerer": ["Fire Bolt", "Light", "Prestidigitation", "Mage Hand"],
    "Warlock": ["Eldritch Blast", "Minor Illusion"],
    "Wizard": ["Fire Bolt", "Light", "Mage Hand"],
}

# Starting spells for level 1 (good defaults)
SUGGESTED_STARTING_SPELLS: dict[str, list[str]] = {
    "Bard": ["Healing Word", "Thunderwave", "Faerie Fire", "Dissonant Whispers"],
    "Cleric": ["Cure Wounds", "Bless", "Guiding Bolt", "Shield of Faith"],
    "Druid": ["Cure Wounds", "Entangle", "Thunderwave", "Faerie Fire"],
    "Sorcerer": ["Magic Missile", "Shield"],
    "Warlock": ["Hex", "Eldritch Blast"],  # EB is actually a cantrip
    "Wizard": ["Magic Missile", "Shield", "Mage Armor", "Sleep", "Detect Magic", "Find Familiar"],
    "Paladin": [],  # Paladins prepare spells, don't have spells known at level 1
    "Ranger": [],  # Rangers don't get spells until level 2
}


def get_starting_cantrips(class_name: str) -> list[str]:
    """Get suggested starting cantrips for a class."""
    return SUGGESTED_STARTING_CANTRIPS.get(class_name, [])


def get_starting_spells(class_name: str) -> list[str]:
    """Get suggested starting spells for a class."""
    return SUGGESTED_STARTING_SPELLS.get(class_name, [])


# =============================================================================
# Creature Morale Data
# =============================================================================

# Base morale and flee thresholds by creature type/category
# morale_base: Starting morale (0-100)
# flee_threshold: HP percentage at which creature considers fleeing
# intimidation_dc_mod: Modifier to Intimidation DC (negative = easier to intimidate)
CREATURE_MORALE: dict[str, dict[str, Any]] = {
    # Cowardly creatures - flee easily
    "goblin": {"morale_base": 30, "flee_threshold": 0.40, "intimidation_dc_mod": -5},
    "kobold": {"morale_base": 25, "flee_threshold": 0.50, "intimidation_dc_mod": -5},
    "bandit": {"morale_base": 35, "flee_threshold": 0.35, "intimidation_dc_mod": -3},
    "thug": {"morale_base": 40, "flee_threshold": 0.30, "intimidation_dc_mod": -2},
    "cultist": {"morale_base": 50, "flee_threshold": 0.25, "intimidation_dc_mod": 0},  # Fanatical but still mortal
    
    # Average creatures - standard morale
    "orc": {"morale_base": 55, "flee_threshold": 0.25, "intimidation_dc_mod": 0},
    "hobgoblin": {"morale_base": 60, "flee_threshold": 0.20, "intimidation_dc_mod": 2},
    "bugbear": {"morale_base": 50, "flee_threshold": 0.30, "intimidation_dc_mod": 0},
    "gnoll": {"morale_base": 55, "flee_threshold": 0.25, "intimidation_dc_mod": 0},
    "skeleton": {"morale_base": 100, "flee_threshold": 0.0, "intimidation_dc_mod": 100},  # Mindless undead
    "zombie": {"morale_base": 100, "flee_threshold": 0.0, "intimidation_dc_mod": 100},  # Mindless undead
    
    # Brave/Elite creatures - hard to break
    "knight": {"morale_base": 75, "flee_threshold": 0.15, "intimidation_dc_mod": 5},
    "veteran": {"morale_base": 70, "flee_threshold": 0.20, "intimidation_dc_mod": 3},
    "guard": {"morale_base": 55, "flee_threshold": 0.25, "intimidation_dc_mod": 0},
    "ogre": {"morale_base": 60, "flee_threshold": 0.20, "intimidation_dc_mod": 2},
    "troll": {"morale_base": 65, "flee_threshold": 0.15, "intimidation_dc_mod": 3},
    
    # Fearless creatures - fight to the death
    "dragon": {"morale_base": 95, "flee_threshold": 0.05, "intimidation_dc_mod": 10},
    "demon": {"morale_base": 90, "flee_threshold": 0.0, "intimidation_dc_mod": 10},
    "devil": {"morale_base": 85, "flee_threshold": 0.10, "intimidation_dc_mod": 8},
    "golem": {"morale_base": 100, "flee_threshold": 0.0, "intimidation_dc_mod": 100},  # Constructs don't flee
    "elemental": {"morale_base": 100, "flee_threshold": 0.0, "intimidation_dc_mod": 100},  # Elementals don't flee
    
    # Wild creatures - animals have survival instincts
    "wolf": {"morale_base": 45, "flee_threshold": 0.35, "intimidation_dc_mod": -2},
    "bear": {"morale_base": 55, "flee_threshold": 0.25, "intimidation_dc_mod": 0},
    "giant_spider": {"morale_base": 40, "flee_threshold": 0.40, "intimidation_dc_mod": -3},
    "owlbear": {"morale_base": 60, "flee_threshold": 0.20, "intimidation_dc_mod": 2},
}

# Default morale for creatures not in the list
DEFAULT_MORALE = {"morale_base": 50, "flee_threshold": 0.25, "intimidation_dc_mod": 0}

# Morale modifiers for combat events
MORALE_EVENTS: dict[str, int] = {
    "ally_killed": -15,  # When an ally dies
    "ally_fled": -10,  # When an ally flees
    "leader_killed": -25,  # When a leader/boss dies
    "taken_crit": -10,  # When hit by a critical hit
    "half_hp": -10,  # When reduced to half HP
    "quarter_hp": -15,  # When reduced to quarter HP
    "outnumbered": -10,  # When enemies outnumber allies
    "ally_healed": 5,  # When an ally is healed
    "enemy_killed": 10,  # When an enemy is killed
    "successful_hit": 5,  # When landing a hit
    "rallied": 20,  # When rallied by a leader
}


def get_creature_morale(creature_name: str) -> dict[str, Any]:
    """Get morale data for a creature type.
    
    Args:
        creature_name: Name of the creature (e.g., "goblin", "dragon").
        
    Returns:
        Dict with morale_base, flee_threshold, and intimidation_dc_mod.
    """
    # Normalize name and check for partial matches
    name_lower = creature_name.lower().strip()
    
    # Direct match
    if name_lower in CREATURE_MORALE:
        return CREATURE_MORALE[name_lower]
    
    # Partial match (e.g., "Goblin 1" matches "goblin")
    for creature, data in CREATURE_MORALE.items():
        if creature in name_lower or name_lower.startswith(creature):
            return data
    
    # Check for category keywords
    if "undead" in name_lower or "skeleton" in name_lower or "zombie" in name_lower:
        return CREATURE_MORALE.get("skeleton", DEFAULT_MORALE)
    if "dragon" in name_lower:
        return CREATURE_MORALE.get("dragon", DEFAULT_MORALE)
    if "demon" in name_lower or "fiend" in name_lower:
        return CREATURE_MORALE.get("demon", DEFAULT_MORALE)
    if "construct" in name_lower or "golem" in name_lower:
        return CREATURE_MORALE.get("golem", DEFAULT_MORALE)
    
    return DEFAULT_MORALE


def get_morale_modifier(event: str) -> int:
    """Get the morale modifier for a combat event.
    
    Args:
        event: The event that occurred.
        
    Returns:
        Morale modifier (positive or negative).
    """
    return MORALE_EVENTS.get(event, 0)


__all__ = [
    # XP
    "XP_THRESHOLDS",
    "get_level_for_xp",
    "get_xp_for_next_level",
    "get_xp_progress",
    # Proficiency
    "get_proficiency_bonus",
    # Hit dice
    "CLASS_HIT_DIE",
    "get_hit_die",
    "calculate_hp_increase",
    # Spell slots
    "FULL_CASTER_SLOTS",
    "HALF_CASTER_SLOTS",
    "WARLOCK_PACT_SLOTS",
    "THIRD_CASTER_SLOTS",
    "get_spell_slots",
    "get_pact_magic",
    # ASI
    "is_asi_level",
    # Features
    "CLASS_FEATURES",
    "get_features_at_level",
    "get_all_features_up_to_level",
    # Feature resources (limited use tracking)
    "FEATURE_RESOURCES",
    "get_feature_resource_info",
    # Spells known
    "get_cantrips_known",
    "get_spells_known",
    # Spellcasting
    "SPELLCASTING_ABILITY",
    "get_spellcasting_ability",
    "is_spellcaster",
    "get_starting_cantrips",
    "get_starting_spells",
    # Skills
    "ALL_SKILLS",
    "SKILL_ABILITIES",
    "CLASS_SKILL_OPTIONS",
    "CLASS_SKILL_COUNT",
    "get_skill_options",
    "get_skill_count",
    # Racial traits
    "RACIAL_TRAITS",
    "get_racial_traits",
    "get_ability_bonuses",
    # Creature morale
    "CREATURE_MORALE",
    "DEFAULT_MORALE",
    "MORALE_EVENTS",
    "get_creature_morale",
    "get_morale_modifier",
    # Helper
    "get_level_up_info",
]
