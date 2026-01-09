"""Constants for character creation and management."""

from __future__ import annotations


# =============================================================================
# Character Creation Constants
# =============================================================================


RACES = [
    "Human", "Elf (High)", "Elf (Wood)", "Elf (Dark/Drow)", 
    "Dwarf (Hill)", "Dwarf (Mountain)", "Halfling (Lightfoot)", 
    "Halfling (Stout)", "Gnome (Forest)", "Gnome (Rock)",
    "Half-Elf", "Half-Orc", "Tiefling", "Dragonborn"
]

CLASSES = [
    "Barbarian", "Bard", "Cleric", "Druid", "Fighter",
    "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer", 
    "Warlock", "Wizard"
]

BACKGROUNDS = [
    "Acolyte", "Charlatan", "Criminal", "Entertainer", "Folk Hero",
    "Guild Artisan", "Hermit", "Noble", "Outlander", "Sage",
    "Sailor", "Soldier", "Urchin"
]

ALIGNMENTS = [
    "Lawful Good", "Neutral Good", "Chaotic Good",
    "Lawful Neutral", "True Neutral", "Chaotic Neutral",
    "Lawful Evil", "Neutral Evil", "Chaotic Evil"
]

# Stat generation methods
STAT_METHODS = {
    "standard_array": "Standard Array (15, 14, 13, 12, 10, 8)",
    "point_buy": "Point Buy (27 points)",
    "roll": "Roll (4d6 drop lowest)",
    "manual": "Manual Entry",
}

STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]

# Point buy costs
POINT_BUY_COSTS = {
    8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7, 15: 9
}
POINT_BUY_TOTAL = 27

# Starting HP by class
CLASS_HIT_DIE = {
    "Barbarian": 12, "Bard": 8, "Cleric": 8, "Druid": 8,
    "Fighter": 10, "Monk": 8, "Paladin": 10, "Ranger": 10,
    "Rogue": 8, "Sorcerer": 6, "Warlock": 8, "Wizard": 6
}

# Saving throw proficiencies by class
CLASS_SAVE_PROFS = {
    "Barbarian": ["strength", "constitution"],
    "Bard": ["dexterity", "charisma"],
    "Cleric": ["wisdom", "charisma"],
    "Druid": ["intelligence", "wisdom"],
    "Fighter": ["strength", "constitution"],
    "Monk": ["strength", "dexterity"],
    "Paladin": ["wisdom", "charisma"],
    "Ranger": ["strength", "dexterity"],
    "Rogue": ["dexterity", "intelligence"],
    "Sorcerer": ["constitution", "charisma"],
    "Warlock": ["wisdom", "charisma"],
    "Wizard": ["intelligence", "wisdom"],
}
