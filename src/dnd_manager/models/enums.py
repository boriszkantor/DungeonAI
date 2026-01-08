"""Enumeration types for the D&D 5E AI Campaign Manager.

This module defines all enumeration types used throughout the application,
including ability scores, skills, alignments, damage types, and AI autonomy levels.
These enums serve as the foundation for type-safe D&D 5E mechanics.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum, auto


class Ability(StrEnum):
    """D&D 5E ability scores.

    The six core abilities that define a character's physical
    and mental characteristics.
    """

    STR = "strength"
    DEX = "dexterity"
    CON = "constitution"
    INT = "intelligence"
    WIS = "wisdom"
    CHA = "charisma"

    @property
    def full_name(self) -> str:
        """Get the full name of the ability.

        Returns:
            Full ability name (e.g., 'Strength' for STR).
        """
        return self.value.capitalize()

    @property
    def abbreviation(self) -> str:
        """Get the three-letter abbreviation.

        Returns:
            Three-letter abbreviation (e.g., 'STR').
        """
        return self.name


class Skill(StrEnum):
    """D&D 5E skills and their associated abilities.

    Each skill is linked to a primary ability score used
    for skill checks.
    """

    # Strength skills
    ATHLETICS = "athletics"

    # Dexterity skills
    ACROBATICS = "acrobatics"
    SLEIGHT_OF_HAND = "sleight_of_hand"
    STEALTH = "stealth"

    # Intelligence skills
    ARCANA = "arcana"
    HISTORY = "history"
    INVESTIGATION = "investigation"
    NATURE = "nature"
    RELIGION = "religion"

    # Wisdom skills
    ANIMAL_HANDLING = "animal_handling"
    INSIGHT = "insight"
    MEDICINE = "medicine"
    PERCEPTION = "perception"
    SURVIVAL = "survival"

    # Charisma skills
    DECEPTION = "deception"
    INTIMIDATION = "intimidation"
    PERFORMANCE = "performance"
    PERSUASION = "persuasion"

    @property
    def ability(self) -> Ability:
        """Get the primary ability score for this skill.

        Returns:
            The Ability enum value associated with this skill.
        """
        skill_abilities: dict[Skill, Ability] = {
            # Strength
            Skill.ATHLETICS: Ability.STR,
            # Dexterity
            Skill.ACROBATICS: Ability.DEX,
            Skill.SLEIGHT_OF_HAND: Ability.DEX,
            Skill.STEALTH: Ability.DEX,
            # Intelligence
            Skill.ARCANA: Ability.INT,
            Skill.HISTORY: Ability.INT,
            Skill.INVESTIGATION: Ability.INT,
            Skill.NATURE: Ability.INT,
            Skill.RELIGION: Ability.INT,
            # Wisdom
            Skill.ANIMAL_HANDLING: Ability.WIS,
            Skill.INSIGHT: Ability.WIS,
            Skill.MEDICINE: Ability.WIS,
            Skill.PERCEPTION: Ability.WIS,
            Skill.SURVIVAL: Ability.WIS,
            # Charisma
            Skill.DECEPTION: Ability.CHA,
            Skill.INTIMIDATION: Ability.CHA,
            Skill.PERFORMANCE: Ability.CHA,
            Skill.PERSUASION: Ability.CHA,
        }
        return skill_abilities[self]


class Alignment(StrEnum):
    """D&D 5E character alignments.

    The nine alignments representing a character's moral
    and ethical outlook.
    """

    LAWFUL_GOOD = "lawful_good"
    NEUTRAL_GOOD = "neutral_good"
    CHAOTIC_GOOD = "chaotic_good"
    LAWFUL_NEUTRAL = "lawful_neutral"
    TRUE_NEUTRAL = "true_neutral"
    CHAOTIC_NEUTRAL = "chaotic_neutral"
    LAWFUL_EVIL = "lawful_evil"
    NEUTRAL_EVIL = "neutral_evil"
    CHAOTIC_EVIL = "chaotic_evil"
    UNALIGNED = "unaligned"

    @property
    def display_name(self) -> str:
        """Get human-readable alignment name.

        Returns:
            Formatted alignment name (e.g., 'Lawful Good').
        """
        return self.value.replace("_", " ").title()


class AutonomyLevel(IntEnum):
    """AI autonomy levels for character control.

    Defines how much control the AI has over a character's
    actions during gameplay.

    Levels:
        NONE: Player has full control, AI provides no suggestions.
        SUGGESTIVE: AI suggests actions but player decides.
        ASSISTED: AI handles routine actions, player decides important ones.
        FULL_AUTO: AI has complete control over character actions.
    """

    NONE = 0
    SUGGESTIVE = 1
    ASSISTED = 2
    FULL_AUTO = 3

    @property
    def description(self) -> str:
        """Get a description of this autonomy level.

        Returns:
            Human-readable description of the autonomy level.
        """
        descriptions = {
            AutonomyLevel.NONE: "Player has full control",
            AutonomyLevel.SUGGESTIVE: "AI suggests actions, player decides",
            AutonomyLevel.ASSISTED: "AI handles routine actions automatically",
            AutonomyLevel.FULL_AUTO: "AI controls all actions",
        }
        return descriptions[self]


class DamageType(StrEnum):
    """D&D 5E damage types."""

    ACID = "acid"
    BLUDGEONING = "bludgeoning"
    COLD = "cold"
    FIRE = "fire"
    FORCE = "force"
    LIGHTNING = "lightning"
    NECROTIC = "necrotic"
    PIERCING = "piercing"
    POISON = "poison"
    PSYCHIC = "psychic"
    RADIANT = "radiant"
    SLASHING = "slashing"
    THUNDER = "thunder"


class Condition(StrEnum):
    """D&D 5E conditions that can affect creatures."""

    BLINDED = "blinded"
    CHARMED = "charmed"
    DEAFENED = "deafened"
    EXHAUSTION = "exhaustion"
    FRIGHTENED = "frightened"
    GRAPPLED = "grappled"
    INCAPACITATED = "incapacitated"
    INVISIBLE = "invisible"
    PARALYZED = "paralyzed"
    PETRIFIED = "petrified"
    POISONED = "poisoned"
    PRONE = "prone"
    RESTRAINED = "restrained"
    STUNNED = "stunned"
    UNCONSCIOUS = "unconscious"


class Size(StrEnum):
    """D&D 5E creature sizes."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"
    GARGANTUAN = "gargantuan"

    @property
    def space_feet(self) -> int:
        """Get the space controlled by a creature of this size in feet.

        Returns:
            Space in feet (e.g., 5 for Medium).
        """
        sizes = {
            Size.TINY: 2,
            Size.SMALL: 5,
            Size.MEDIUM: 5,
            Size.LARGE: 10,
            Size.HUGE: 15,
            Size.GARGANTUAN: 20,
        }
        return sizes[self]

    @property
    def hit_die(self) -> int:
        """Get the hit die size for creatures of this size.

        Returns:
            Hit die size (e.g., d8 for Medium returns 8).
        """
        hit_dice = {
            Size.TINY: 4,
            Size.SMALL: 6,
            Size.MEDIUM: 8,
            Size.LARGE: 10,
            Size.HUGE: 12,
            Size.GARGANTUAN: 20,
        }
        return hit_dice[self]


class CreatureType(StrEnum):
    """D&D 5E creature types."""

    ABERRATION = "aberration"
    BEAST = "beast"
    CELESTIAL = "celestial"
    CONSTRUCT = "construct"
    DRAGON = "dragon"
    ELEMENTAL = "elemental"
    FEY = "fey"
    FIEND = "fiend"
    GIANT = "giant"
    HUMANOID = "humanoid"
    MONSTROSITY = "monstrosity"
    OOZE = "ooze"
    PLANT = "plant"
    UNDEAD = "undead"


class ActionType(StrEnum):
    """Types of actions in D&D 5E combat."""

    ACTION = "action"
    BONUS_ACTION = "bonus_action"
    REACTION = "reaction"
    MOVEMENT = "movement"
    FREE_ACTION = "free_action"
    LEGENDARY_ACTION = "legendary_action"
    LAIR_ACTION = "lair_action"


class CombatantType(StrEnum):
    """Discriminator for combatant polymorphism."""

    PLAYER_CHARACTER = "player_character"
    MONSTER = "monster"
    NPC = "npc"


class SpellSchool(StrEnum):
    """D&D 5E schools of magic."""

    ABJURATION = "abjuration"
    CONJURATION = "conjuration"
    DIVINATION = "divination"
    ENCHANTMENT = "enchantment"
    EVOCATION = "evocation"
    ILLUSION = "illusion"
    NECROMANCY = "necromancy"
    TRANSMUTATION = "transmutation"


class RestType(StrEnum):
    """Types of rest in D&D 5E."""

    SHORT_REST = "short_rest"
    LONG_REST = "long_rest"


__all__ = [
    "Ability",
    "Skill",
    "Alignment",
    "AutonomyLevel",
    "DamageType",
    "Condition",
    "Size",
    "CreatureType",
    "ActionType",
    "CombatantType",
    "SpellSchool",
    "RestType",
]
