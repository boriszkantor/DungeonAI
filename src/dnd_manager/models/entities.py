"""Entity models for the D&D 5E AI Campaign Manager.

This module defines the core entity models representing game participants
including player characters, monsters, and NPCs. Uses discriminated unions
for polymorphic combatant handling.

Entities:
    Entity: Abstract base class for all game entities.
    PlayerCharacter: Player-controlled character with full customization.
    Monster: AI-controlled creature with challenge rating.
    NPC: Non-player character (ally, neutral, or enemy).
    Combatant: Discriminated union of all combat-capable entities.

DEPRECATED: Most functionality has been consolidated into models/ecs.py
with the ActorEntity class. This module is maintained for backward compatibility.
New code should use ActorEntity from models/ecs.py.
"""

from __future__ import annotations

import warnings

from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from dnd_manager.models.components import (
    ArmorComponent,
    HealthComponent,
    PersonaComponent,
    SavingThrowsComponent,
    SkillsComponent,
    SpeedComponent,
    StatsComponent,
    calculate_modifier,
)
from dnd_manager.models.enums import (
    Ability,
    Alignment,
    AutonomyLevel,
    CombatantType,
    CreatureType,
    DamageType,
    Size,
)


# =============================================================================
# Type Definitions
# =============================================================================

# Validated level (1-20)
CharacterLevel = Annotated[int, Field(ge=1, le=20, description="Character level (1-20)")]

# Challenge rating as string to support fractional CRs (e.g., "1/4", "1/2")
ChallengeRating = Annotated[
    str,
    Field(
        pattern=r"^(0|1/8|1/4|1/2|[1-9]|1[0-9]|2[0-9]|30)$",
        description="Challenge rating (0, 1/8, 1/4, 1/2, or 1-30)",
    ),
]


def cr_to_float(cr: str) -> float:
    """Convert challenge rating string to float.

    Args:
        cr: Challenge rating string (e.g., "1/4", "5").

    Returns:
        Numeric challenge rating.
    """
    if "/" in cr:
        num, denom = cr.split("/")
        return int(num) / int(denom)
    return float(cr)


def cr_to_proficiency_bonus(cr: str) -> int:
    """Calculate proficiency bonus from challenge rating.

    Args:
        cr: Challenge rating string.

    Returns:
        Proficiency bonus (2-9 based on CR).
    """
    cr_float = cr_to_float(cr)
    if cr_float < 5:
        return 2
    elif cr_float < 9:
        return 3
    elif cr_float < 13:
        return 4
    elif cr_float < 17:
        return 5
    elif cr_float < 21:
        return 6
    elif cr_float < 25:
        return 7
    elif cr_float < 29:
        return 8
    else:
        return 9


# =============================================================================
# Base Entity
# =============================================================================


class Entity(BaseModel):
    """Abstract base class for all game entities.

    Provides common fields and functionality shared by all entities
    in the game world, including unique identification and naming.

    Attributes:
        uid: Unique identifier for this entity instance.
        name: Display name of the entity.
        description: Optional detailed description.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "description": "Base class for all game entities"
        },
    )

    uid: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this entity",
    )
    name: str = Field(
        min_length=1,
        max_length=100,
        description="Display name of the entity",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of the entity",
    )


# =============================================================================
# Player Character
# =============================================================================


class InventoryItem(BaseModel):
    """An item in a character's inventory.

    Attributes:
        name: Item name.
        quantity: Number of this item.
        weight: Weight per item in pounds.
        description: Item description.
        equipped: Whether the item is currently equipped.
        magical: Whether the item is magical.
        attunement_required: Whether attunement is required.
        attuned: Whether the character is attuned to this item.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100, description="Item name")
    quantity: Annotated[int, Field(ge=1)] = Field(default=1, description="Item quantity")
    weight: Annotated[float, Field(ge=0)] = Field(default=0, description="Weight in pounds")
    description: str = Field(default="", max_length=500, description="Item description")
    equipped: bool = Field(default=False, description="Whether item is equipped")
    magical: bool = Field(default=False, description="Whether item is magical")
    attunement_required: bool = Field(default=False, description="Requires attunement")
    attuned: bool = Field(default=False, description="Currently attuned")


class ClassLevel(BaseModel):
    """A character class and level.

    Supports multiclassing by allowing multiple ClassLevel instances.

    Attributes:
        class_name: Name of the class.
        subclass: Optional subclass name.
        level: Level in this class (1-20).
        hit_die: Hit die size for this class.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    class_name: str = Field(
        min_length=1,
        max_length=50,
        description="Class name (e.g., 'Fighter', 'Wizard')",
    )
    subclass: str | None = Field(
        default=None,
        max_length=100,
        description="Subclass name (e.g., 'Champion', 'School of Evocation')",
    )
    level: CharacterLevel = Field(description="Level in this class")
    hit_die: Annotated[int, Field(ge=6, le=12)] = Field(
        default=8,
        description="Hit die size (d6, d8, d10, or d12)",
    )


class SpellSlots(BaseModel):
    """Spell slot tracking for spellcasters.

    Attributes:
        level_1 through level_9: Current and max slots per level.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    level_1: tuple[int, int] = Field(default=(0, 0), description="(current, max) 1st level slots")
    level_2: tuple[int, int] = Field(default=(0, 0), description="(current, max) 2nd level slots")
    level_3: tuple[int, int] = Field(default=(0, 0), description="(current, max) 3rd level slots")
    level_4: tuple[int, int] = Field(default=(0, 0), description="(current, max) 4th level slots")
    level_5: tuple[int, int] = Field(default=(0, 0), description="(current, max) 5th level slots")
    level_6: tuple[int, int] = Field(default=(0, 0), description="(current, max) 6th level slots")
    level_7: tuple[int, int] = Field(default=(0, 0), description="(current, max) 7th level slots")
    level_8: tuple[int, int] = Field(default=(0, 0), description="(current, max) 8th level slots")
    level_9: tuple[int, int] = Field(default=(0, 0), description="(current, max) 9th level slots")


class PlayerCharacter(Entity):
    """A player-controlled character with full customization.

    Represents a PC with complete D&D 5E character sheet data including
    ability scores, class levels, equipment, and AI persona configuration.

    Attributes:
        type: Discriminator field for polymorphism.
        stats: Ability scores component.
        health: Hit point tracking component.
        persona: AI behavior configuration.
        speed: Movement speeds.
        armor: Armor class calculation.
        saving_throws: Saving throw proficiencies.
        skills: Skill proficiencies and expertise.
        classes: List of class levels (supports multiclassing).
        race: Character race.
        background: Character background.
        alignment: Character alignment.
        experience_points: Current XP.
        inventory: List of inventory items.
        spell_slots: Spell slot tracking.
        known_spells: List of known spell names.
        prepared_spells: List of currently prepared spells.
        features: Class and racial features.
        languages: Known languages.
        tool_proficiencies: Tool proficiencies.
        weapon_proficiencies: Weapon proficiencies.
        armor_proficiencies: Armor proficiencies.
        notes: Player notes about the character.

    Example:
        >>> pc = PlayerCharacter(
        ...     name="Gandalf the Grey",
        ...     stats=StatsComponent(intelligence=18, wisdom=16),
        ...     health=HealthComponent(current_hp=45, max_hp=45),
        ...     classes=[ClassLevel(class_name="Wizard", level=10)],
        ...     race="Human",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "description": "Player character with full D&D 5E character sheet"
        },
    )

    # Discriminator for polymorphism
    type: Literal[CombatantType.PLAYER_CHARACTER] = Field(
        default=CombatantType.PLAYER_CHARACTER,
        description="Entity type discriminator",
    )

    # Core components
    stats: StatsComponent = Field(
        default_factory=StatsComponent,
        description="Ability scores (STR, DEX, CON, INT, WIS, CHA)",
    )
    health: HealthComponent = Field(
        description="Hit points and death save tracking",
    )
    persona: PersonaComponent = Field(
        description="AI behavior and personality configuration",
    )
    speed: SpeedComponent = Field(
        default_factory=SpeedComponent,
        description="Movement speeds (walk, fly, swim, etc.)",
    )
    armor: ArmorComponent = Field(
        default_factory=ArmorComponent,
        description="Armor class calculation",
    )
    saving_throws: SavingThrowsComponent = Field(
        default_factory=SavingThrowsComponent,
        description="Saving throw proficiencies",
    )
    skills: SkillsComponent = Field(
        default_factory=SkillsComponent,
        description="Skill proficiencies and expertise",
    )

    # Character progression
    classes: list[ClassLevel] = Field(
        min_length=1,
        description="Class levels (supports multiclassing)",
    )
    race: str = Field(
        min_length=1,
        max_length=50,
        description="Character race (e.g., 'Human', 'Elf')",
    )
    background: str = Field(
        default="",
        max_length=50,
        description="Character background (e.g., 'Soldier', 'Sage')",
    )
    alignment: Alignment = Field(
        default=Alignment.TRUE_NEUTRAL,
        description="Character alignment",
    )
    experience_points: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Current experience points",
    )

    # Equipment and spells
    inventory: list[InventoryItem] = Field(
        default_factory=list,
        description="Character inventory",
    )
    spell_slots: SpellSlots = Field(
        default_factory=SpellSlots,
        description="Available spell slots by level",
    )
    known_spells: list[str] = Field(
        default_factory=list,
        description="List of known spell names",
    )
    prepared_spells: list[str] = Field(
        default_factory=list,
        description="Currently prepared spells",
    )

    # Features and proficiencies
    features: list[str] = Field(
        default_factory=list,
        description="Class features, racial traits, and feats",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Known languages",
    )
    tool_proficiencies: list[str] = Field(
        default_factory=list,
        description="Tool proficiencies",
    )
    weapon_proficiencies: list[str] = Field(
        default_factory=list,
        description="Weapon proficiencies",
    )
    armor_proficiencies: list[str] = Field(
        default_factory=list,
        description="Armor proficiencies",
    )

    # Notes
    notes: str = Field(
        default="",
        max_length=5000,
        description="Player notes about the character",
    )

    @model_validator(mode="after")
    def set_default_persona(self) -> "PlayerCharacter":
        """Ensure persona defaults to NONE autonomy for PCs."""
        if self.persona.autonomy != AutonomyLevel.NONE:
            # Allow override, but default should be NONE
            pass
        return self

    @computed_field(description="Total character level across all classes")
    @property
    def total_level(self) -> int:
        """Calculate total level from all classes."""
        return sum(c.level for c in self.classes)

    @computed_field(description="Proficiency bonus based on total level")
    @property
    def proficiency_bonus(self) -> int:
        """Calculate proficiency bonus from level."""
        return (self.total_level - 1) // 4 + 2

    @computed_field(description="Calculated armor class")
    @property
    def armor_class(self) -> int:
        """Calculate total AC."""
        return self.armor.calculate_ac(self.stats.dexterity_modifier)

    @computed_field(description="Initiative modifier (DEX mod)")
    @property
    def initiative_modifier(self) -> int:
        """Calculate initiative modifier."""
        return self.stats.dexterity_modifier

    @computed_field(description="Passive Perception score")
    @property
    def passive_perception(self) -> int:
        """Calculate passive Perception."""
        from dnd_manager.models.enums import Skill

        perception_bonus = self.skills.get_skill_bonus(
            Skill.PERCEPTION, self.stats, self.proficiency_bonus
        )
        return 10 + perception_bonus


# =============================================================================
# Monster
# =============================================================================


class LegendaryAction(BaseModel):
    """A legendary action available to powerful monsters.

    Attributes:
        name: Action name.
        description: What the action does.
        cost: Number of legendary action points required.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100, description="Action name")
    description: str = Field(max_length=1000, description="Action description")
    cost: Annotated[int, Field(ge=1, le=3)] = Field(
        default=1,
        description="Legendary action cost (1-3)",
    )


class MonsterAction(BaseModel):
    """An action available to a monster.

    Attributes:
        name: Action name.
        description: Full action description.
        attack_bonus: Attack roll bonus (if attack).
        damage: Damage expression (e.g., "2d6+4").
        damage_type: Type of damage dealt.
        reach_range: Reach or range in feet.
        recharge: Recharge condition (e.g., "5-6").
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100, description="Action name")
    description: str = Field(max_length=2000, description="Full action description")
    attack_bonus: int | None = Field(default=None, description="Attack roll bonus")
    damage: str | None = Field(default=None, description="Damage expression (e.g., '2d6+4')")
    damage_type: DamageType | None = Field(default=None, description="Damage type")
    reach_range: str | None = Field(default=None, description="Reach or range")
    recharge: str | None = Field(default=None, description="Recharge condition (e.g., '5-6')")


class Monster(Entity):
    """A monster or creature controlled by the AI/DM.

    Represents a complete monster stat block with challenge rating,
    actions, and legendary abilities.

    Attributes:
        type: Discriminator field for polymorphism.
        stats: Ability scores.
        health: Hit points.
        persona: AI behavior (defaults to FULL_AUTO).
        speed: Movement speeds.
        armor: Armor class.
        saving_throws: Saving throw proficiencies.
        size: Creature size.
        creature_type: Type of creature.
        alignment: Creature alignment.
        challenge_rating: CR for XP and encounter balancing.
        armor_class_override: Explicit AC (bypasses calculation).
        damage_vulnerabilities: Damage types the creature is vulnerable to.
        damage_resistances: Damage types the creature resists.
        damage_immunities: Damage types the creature is immune to.
        condition_immunities: Conditions the creature is immune to.
        senses: Special senses (e.g., darkvision 60 ft.).
        languages: Known languages.
        actions: Available actions.
        legendary_actions: Legendary actions (if any).
        legendary_action_count: Number of legendary actions per round.
        lair_actions: Lair actions (if in lair).
        traits: Special traits and abilities.
        reactions: Reaction abilities.

    Example:
        >>> dragon = Monster(
        ...     name="Adult Red Dragon",
        ...     stats=StatsComponent(strength=27, constitution=25),
        ...     health=HealthComponent(current_hp=256, max_hp=256),
        ...     challenge_rating="17",
        ...     creature_type=CreatureType.DRAGON,
        ...     size=Size.HUGE,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "description": "Monster stat block with challenge rating and abilities"
        },
    )

    # Discriminator
    type: Literal[CombatantType.MONSTER] = Field(
        default=CombatantType.MONSTER,
        description="Entity type discriminator",
    )

    # Core components
    stats: StatsComponent = Field(
        default_factory=StatsComponent,
        description="Ability scores",
    )
    health: HealthComponent = Field(
        description="Hit points",
    )
    persona: PersonaComponent = Field(
        description="AI behavior configuration",
    )
    speed: SpeedComponent = Field(
        default_factory=SpeedComponent,
        description="Movement speeds",
    )
    armor: ArmorComponent = Field(
        default_factory=ArmorComponent,
        description="Armor class components",
    )
    saving_throws: SavingThrowsComponent = Field(
        default_factory=SavingThrowsComponent,
        description="Saving throw proficiencies",
    )

    # Monster-specific
    size: Size = Field(
        default=Size.MEDIUM,
        description="Creature size category",
    )
    creature_type: CreatureType = Field(
        default=CreatureType.HUMANOID,
        description="Creature type (e.g., dragon, undead)",
    )
    alignment: Alignment = Field(
        default=Alignment.UNALIGNED,
        description="Creature alignment",
    )
    challenge_rating: ChallengeRating = Field(
        default="1",
        description="Challenge rating (0, 1/8, 1/4, 1/2, or 1-30)",
    )
    armor_class_override: Annotated[int, Field(ge=1, le=30)] | None = Field(
        default=None,
        description="Explicit AC value (bypasses calculation)",
    )

    # Defenses
    damage_vulnerabilities: list[DamageType] = Field(
        default_factory=list,
        description="Damage types with vulnerability (double damage)",
    )
    damage_resistances: list[DamageType] = Field(
        default_factory=list,
        description="Damage types with resistance (half damage)",
    )
    damage_immunities: list[DamageType] = Field(
        default_factory=list,
        description="Damage types with immunity (no damage)",
    )
    condition_immunities: list[str] = Field(
        default_factory=list,
        description="Conditions the creature is immune to",
    )

    # Senses and communication
    senses: list[str] = Field(
        default_factory=list,
        description="Special senses (e.g., 'darkvision 60 ft.')",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Known languages",
    )

    # Actions and abilities
    actions: list[MonsterAction] = Field(
        default_factory=list,
        description="Available actions",
    )
    legendary_actions: list[LegendaryAction] = Field(
        default_factory=list,
        description="Legendary actions",
    )
    legendary_action_count: Annotated[int, Field(ge=0, le=5)] = Field(
        default=0,
        description="Number of legendary actions per round",
    )
    lair_actions: list[str] = Field(
        default_factory=list,
        description="Lair actions (if in lair)",
    )
    traits: list[str] = Field(
        default_factory=list,
        description="Special traits and passive abilities",
    )
    reactions: list[str] = Field(
        default_factory=list,
        description="Reaction abilities",
    )

    @model_validator(mode="after")
    def set_monster_autonomy(self) -> "Monster":
        """Ensure monsters default to full AI control."""
        # Monsters are AI-controlled by default
        return self

    @computed_field(description="Proficiency bonus from challenge rating")
    @property
    def proficiency_bonus(self) -> int:
        """Calculate proficiency bonus from CR."""
        return cr_to_proficiency_bonus(self.challenge_rating)

    @computed_field(description="Calculated or overridden armor class")
    @property
    def armor_class(self) -> int:
        """Get effective AC."""
        if self.armor_class_override is not None:
            return self.armor_class_override
        return self.armor.calculate_ac(self.stats.dexterity_modifier)

    @computed_field(description="Experience points awarded for defeating")
    @property
    def experience_points(self) -> int:
        """Calculate XP from challenge rating."""
        xp_by_cr: dict[str, int] = {
            "0": 10, "1/8": 25, "1/4": 50, "1/2": 100,
            "1": 200, "2": 450, "3": 700, "4": 1100, "5": 1800,
            "6": 2300, "7": 2900, "8": 3900, "9": 5000, "10": 5900,
            "11": 7200, "12": 8400, "13": 10000, "14": 11500, "15": 13000,
            "16": 15000, "17": 18000, "18": 20000, "19": 22000, "20": 25000,
            "21": 33000, "22": 41000, "23": 50000, "24": 62000, "25": 75000,
            "26": 90000, "27": 105000, "28": 120000, "29": 135000, "30": 155000,
        }
        return xp_by_cr.get(self.challenge_rating, 0)

    @computed_field(description="Whether this is a legendary creature")
    @property
    def is_legendary(self) -> bool:
        """Check if creature has legendary actions."""
        return self.legendary_action_count > 0 or len(self.legendary_actions) > 0


# =============================================================================
# NPC (Non-Player Character)
# =============================================================================


class NPC(Entity):
    """A non-player character (ally, neutral, or enemy).

    NPCs are simpler than full monsters but have persona for roleplay.

    Attributes:
        type: Discriminator for polymorphism.
        stats: Ability scores.
        health: Hit points.
        persona: AI behavior and personality.
        speed: Movement speeds.
        armor_class: Simple AC value.
        occupation: NPC's role or job.
        faction: Group or organization affiliation.
        attitude: Current attitude toward party.
        secrets: Hidden information for DM.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "description": "Non-player character for roleplay and simple combat"
        },
    )

    type: Literal[CombatantType.NPC] = Field(
        default=CombatantType.NPC,
        description="Entity type discriminator",
    )

    # Core components
    stats: StatsComponent = Field(
        default_factory=StatsComponent,
        description="Ability scores",
    )
    health: HealthComponent = Field(
        description="Hit points",
    )
    persona: PersonaComponent = Field(
        description="AI behavior and personality",
    )
    speed: SpeedComponent = Field(
        default_factory=SpeedComponent,
        description="Movement speeds",
    )

    # Simple stats
    armor_class: Annotated[int, Field(ge=1, le=30)] = Field(
        default=10,
        description="Armor class",
    )

    # Roleplay information
    occupation: str = Field(
        default="",
        max_length=100,
        description="NPC's role or job",
    )
    faction: str = Field(
        default="",
        max_length=100,
        description="Group or organization affiliation",
    )
    attitude: Literal["hostile", "unfriendly", "indifferent", "friendly", "helpful"] = Field(
        default="indifferent",
        description="Current attitude toward the party",
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="Hidden information (DM only)",
    )

    @computed_field(description="Proficiency bonus (assumes CR 0-1)")
    @property
    def proficiency_bonus(self) -> int:
        """NPCs have a flat +2 proficiency bonus."""
        return 2

    @computed_field(description="Initiative modifier")
    @property
    def initiative_modifier(self) -> int:
        """Calculate initiative from DEX."""
        return self.stats.dexterity_modifier


# =============================================================================
# Discriminated Union: Combatant
# =============================================================================

# Combatant is a discriminated union that can be PlayerCharacter, Monster, or NPC
# The discriminator field is 'type' which is set on each subclass

Combatant = Annotated[
    PlayerCharacter | Monster | NPC,
    Field(
        discriminator="type",
        description="A combat participant (PlayerCharacter, Monster, or NPC)",
    ),
]
"""Discriminated union of all combat-capable entity types.

The 'type' field serves as the discriminator:
- CombatantType.PLAYER_CHARACTER -> PlayerCharacter
- CombatantType.MONSTER -> Monster
- CombatantType.NPC -> NPC

Example:
    >>> import json
    >>> data = {"type": "player_character", "name": "Hero", ...}
    >>> combatant = TypeAdapter(Combatant).validate_python(data)
    >>> isinstance(combatant, PlayerCharacter)
    True
"""


# =============================================================================
# Factory Functions
# =============================================================================


def create_player_character(
    name: str,
    *,
    class_name: str = "Fighter",
    level: int = 1,
    race: str = "Human",
    stats: StatsComponent | None = None,
    max_hp: int = 10,
) -> PlayerCharacter:
    """Factory function to create a basic PlayerCharacter.

    Args:
        name: Character name.
        class_name: Primary class name.
        level: Character level.
        race: Character race.
        stats: Ability scores (defaults to standard array).
        max_hp: Maximum hit points.

    Returns:
        Configured PlayerCharacter instance.
    """
    if stats is None:
        stats = StatsComponent(
            strength=15, dexterity=14, constitution=13,
            intelligence=12, wisdom=10, charisma=8,
        )

    return PlayerCharacter(
        name=name,
        stats=stats,
        health=HealthComponent(current_hp=max_hp, max_hp=max_hp),
        persona=PersonaComponent(name=name, autonomy=AutonomyLevel.NONE),
        classes=[ClassLevel(class_name=class_name, level=level)],
        race=race,
    )


def create_monster(
    name: str,
    *,
    challenge_rating: str = "1",
    creature_type: CreatureType = CreatureType.HUMANOID,
    size: Size = Size.MEDIUM,
    stats: StatsComponent | None = None,
    max_hp: int = 22,
    armor_class: int = 13,
    directives: list[str] | None = None,
) -> Monster:
    """Factory function to create a Monster.

    Args:
        name: Monster name.
        challenge_rating: Challenge rating.
        creature_type: Type of creature.
        size: Creature size.
        stats: Ability scores.
        max_hp: Maximum hit points.
        armor_class: Armor class.
        directives: AI directives for behavior.

    Returns:
        Configured Monster instance.
    """
    if stats is None:
        stats = StatsComponent()

    return Monster(
        name=name,
        stats=stats,
        health=HealthComponent(current_hp=max_hp, max_hp=max_hp),
        persona=PersonaComponent(
            name=name,
            autonomy=AutonomyLevel.FULL_AUTO,
            directives=directives or ["Attack the nearest enemy"],
        ),
        challenge_rating=challenge_rating,
        creature_type=creature_type,
        size=size,
        armor_class_override=armor_class,
    )


# =============================================================================
# Deprecation Warning
# =============================================================================

def _issue_deprecation_warning() -> None:
    """Issue deprecation warning when this module is imported."""
    warnings.warn(
        "models.entities is being consolidated into models.ecs.ActorEntity. "
        "This compatibility layer will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


# Issue warning on import
_issue_deprecation_warning()


__all__ = [
    # Type definitions
    "CharacterLevel",
    "ChallengeRating",
    "cr_to_float",
    "cr_to_proficiency_bonus",
    # Base
    "Entity",
    # Supporting models
    "InventoryItem",
    "ClassLevel",
    "SpellSlots",
    "LegendaryAction",
    "MonsterAction",
    # Entities
    "PlayerCharacter",
    "Monster",
    "NPC",
    # Union
    "Combatant",
    # Factories
    "create_player_character",
    "create_monster",
]
