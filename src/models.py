"""
D&D 5E Campaign Manager - Data Layer
=====================================
Entity-Component-System (ECS) architecture using Pydantic V2.

Components are small, focused data containers.
Entities (Character, GameState) compose multiple components.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AbilityName(str, Enum):
    """The six core ability scores in D&D 5E."""
    STRENGTH = "strength"
    DEXTERITY = "dexterity"
    CONSTITUTION = "constitution"
    INTELLIGENCE = "intelligence"
    WISDOM = "wisdom"
    CHARISMA = "charisma"


class CharacterType(str, Enum):
    """Distinguishes between player characters and NPCs."""
    PLAYER = "player"
    NPC = "npc"
    COMPANION = "companion"
    ENEMY = "enemy"


class MessageRole(str, Enum):
    """Role of a message in the chat history."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DM = "dm"
    CHARACTER = "character"


class DamageType(str, Enum):
    """D&D 5E damage types for resistances/vulnerabilities."""
    BLUDGEONING = "bludgeoning"
    PIERCING = "piercing"
    SLASHING = "slashing"
    FIRE = "fire"
    COLD = "cold"
    LIGHTNING = "lightning"
    THUNDER = "thunder"
    ACID = "acid"
    POISON = "poison"
    NECROTIC = "necrotic"
    RADIANT = "radiant"
    FORCE = "force"
    PSYCHIC = "psychic"


class ConditionType(str, Enum):
    """D&D 5E conditions that can affect characters."""
    BLINDED = "blinded"
    CHARMED = "charmed"
    DEAFENED = "deafened"
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
    EXHAUSTION = "exhaustion"


# =============================================================================
# COMPONENT MODELS (ECS Building Blocks)
# =============================================================================

class CoreStats(BaseModel):
    """
    Component: The six ability scores and their computed modifiers.
    
    In D&D 5E, modifiers are calculated as: (score - 10) // 2
    These stats form the foundation of all character capabilities.
    """
    
    strength: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Physical power. Affects melee attacks, carrying capacity, and Athletics checks."
    )
    dexterity: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Agility and reflexes. Affects AC, ranged attacks, initiative, and Acrobatics/Stealth checks."
    )
    constitution: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Health and stamina. Affects HP and concentration saves."
    )
    intelligence: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Mental acuity and memory. Affects Arcana, History, Investigation, Nature, Religion checks."
    )
    wisdom: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Perception and insight. Affects Perception, Insight, Medicine, Survival, Animal Handling checks."
    )
    charisma: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Force of personality. Affects Persuasion, Deception, Intimidation, Performance checks."
    )

    @staticmethod
    def calculate_modifier(score: int) -> int:
        """Calculate the ability modifier from a score using D&D 5E rules."""
        return (score - 10) // 2

    @computed_field
    @property
    def str_mod(self) -> int:
        """Computed strength modifier."""
        return self.calculate_modifier(self.strength)

    @computed_field
    @property
    def dex_mod(self) -> int:
        """Computed dexterity modifier."""
        return self.calculate_modifier(self.dexterity)

    @computed_field
    @property
    def con_mod(self) -> int:
        """Computed constitution modifier."""
        return self.calculate_modifier(self.constitution)

    @computed_field
    @property
    def int_mod(self) -> int:
        """Computed intelligence modifier."""
        return self.calculate_modifier(self.intelligence)

    @computed_field
    @property
    def wis_mod(self) -> int:
        """Computed wisdom modifier."""
        return self.calculate_modifier(self.wisdom)

    @computed_field
    @property
    def cha_mod(self) -> int:
        """Computed charisma modifier."""
        return self.calculate_modifier(self.charisma)

    def get_modifier(self, ability: AbilityName) -> int:
        """Get the modifier for a specific ability by name."""
        score = getattr(self, ability.value)
        return self.calculate_modifier(score)


class Vitals(BaseModel):
    """
    Component: Combat-relevant vital statistics.
    
    Tracks the character's current survivability and combat readiness.
    """
    
    current_hp: int = Field(
        default=10,
        ge=0,
        description="Current hit points. When this reaches 0, the character falls unconscious."
    )
    max_hp: int = Field(
        default=10,
        ge=1,
        description="Maximum hit points. Determines total health pool."
    )
    temp_hp: int = Field(
        default=0,
        ge=0,
        description="Temporary hit points. Absorbed first before regular HP. Do not stack."
    )
    armor_class: int = Field(
        default=10,
        ge=0,
        alias="ac",
        description="Armor Class. Attack rolls must meet or exceed this to hit."
    )
    speed: int = Field(
        default=30,
        ge=0,
        description="Movement speed in feet per round."
    )
    initiative_bonus: int = Field(
        default=0,
        description="Bonus added to initiative rolls (typically DEX modifier + any features)."
    )

    @computed_field
    @property
    def is_bloodied(self) -> bool:
        """True if HP is at or below half maximum - useful for roleplay and abilities."""
        return self.current_hp <= self.max_hp // 2

    @computed_field
    @property
    def is_unconscious(self) -> bool:
        """True if HP has reached zero."""
        return self.current_hp == 0

    @model_validator(mode="after")
    def validate_hp_bounds(self) -> "Vitals":
        """Ensure current HP never exceeds max HP."""
        if self.current_hp > self.max_hp:
            self.current_hp = self.max_hp
        return self


class InventoryItem(BaseModel):
    """
    Component: A single item in a character's inventory.
    
    Can represent weapons, armor, consumables, or miscellaneous gear.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this item instance."
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Display name of the item."
    )
    description: str = Field(
        default="",
        description="Detailed description of the item's appearance and properties."
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="Number of this item held. Stackable items share an entry."
    )
    weight: float = Field(
        default=0.0,
        ge=0,
        description="Weight in pounds per unit."
    )
    value_gp: float = Field(
        default=0.0,
        ge=0,
        description="Value in gold pieces per unit."
    )
    equipped: bool = Field(
        default=False,
        description="Whether this item is currently equipped/worn."
    )
    magical: bool = Field(
        default=False,
        description="Whether this item is magical in nature."
    )
    attunement_required: bool = Field(
        default=False,
        description="Whether this item requires attunement to use."
    )
    attuned: bool = Field(
        default=False,
        description="Whether the character is currently attuned to this item."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags: 'weapon', 'armor', 'consumable', 'tool', etc."
    )


class ClassFeature(BaseModel):
    """
    Component: A class or racial feature/ability.
    
    Represents special abilities gained from class levels, race, or backgrounds.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this feature."
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Name of the feature (e.g., 'Rage', 'Sneak Attack', 'Wild Shape')."
    )
    source: str = Field(
        default="",
        description="Where this feature comes from (e.g., 'Barbarian 1', 'Half-Orc', 'Outlander')."
    )
    description: str = Field(
        default="",
        description="Full text description of what this feature does."
    )
    uses_max: int | None = Field(
        default=None,
        ge=0,
        description="Maximum uses per rest. None means unlimited or passive."
    )
    uses_current: int | None = Field(
        default=None,
        ge=0,
        description="Current remaining uses. None if unlimited."
    )
    recharge: Literal["short_rest", "long_rest", "dawn", "never"] | None = Field(
        default=None,
        description="When uses are restored. None if passive or unlimited."
    )
    active: bool = Field(
        default=False,
        description="Whether this feature is currently active (for toggle abilities like Rage)."
    )

    @model_validator(mode="after")
    def validate_uses(self) -> "ClassFeature":
        """Ensure current uses don't exceed max."""
        if self.uses_max is not None and self.uses_current is not None:
            if self.uses_current > self.uses_max:
                self.uses_current = self.uses_max
        return self


class SpellSlots(BaseModel):
    """
    Component: Spell slot tracking for spellcasters.
    
    Tracks available and used spell slots by level (1-9).
    """
    
    level_1: tuple[int, int] = Field(
        default=(0, 0),
        description="1st level slots: (current, maximum)."
    )
    level_2: tuple[int, int] = Field(
        default=(0, 0),
        description="2nd level slots: (current, maximum)."
    )
    level_3: tuple[int, int] = Field(
        default=(0, 0),
        description="3rd level slots: (current, maximum)."
    )
    level_4: tuple[int, int] = Field(
        default=(0, 0),
        description="4th level slots: (current, maximum)."
    )
    level_5: tuple[int, int] = Field(
        default=(0, 0),
        description="5th level slots: (current, maximum)."
    )
    level_6: tuple[int, int] = Field(
        default=(0, 0),
        description="6th level slots: (current, maximum)."
    )
    level_7: tuple[int, int] = Field(
        default=(0, 0),
        description="7th level slots: (current, maximum)."
    )
    level_8: tuple[int, int] = Field(
        default=(0, 0),
        description="8th level slots: (current, maximum)."
    )
    level_9: tuple[int, int] = Field(
        default=(0, 0),
        description="9th level slots: (current, maximum)."
    )
    pact_slots: tuple[int, int] = Field(
        default=(0, 0),
        description="Warlock pact magic slots: (current, maximum)."
    )
    pact_slot_level: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Level of pact magic slots (1-5)."
    )

    def get_slots(self, level: int) -> tuple[int, int]:
        """Get current and max slots for a specific spell level."""
        if level < 1 or level > 9:
            raise ValueError(f"Spell level must be 1-9, got {level}")
        return getattr(self, f"level_{level}")

    def use_slot(self, level: int) -> bool:
        """
        Attempt to use a spell slot of the given level.
        Returns True if successful, False if no slots available.
        """
        current, maximum = self.get_slots(level)
        if current > 0:
            setattr(self, f"level_{level}", (current - 1, maximum))
            return True
        return False

    def restore_slot(self, level: int, count: int = 1) -> None:
        """Restore spell slots of the given level."""
        current, maximum = self.get_slots(level)
        new_current = min(current + count, maximum)
        setattr(self, f"level_{level}", (new_current, maximum))

    def restore_all(self) -> None:
        """Restore all spell slots to maximum (long rest)."""
        for level in range(1, 10):
            _, maximum = self.get_slots(level)
            setattr(self, f"level_{level}", (maximum, maximum))
        # Restore pact slots too
        _, pact_max = self.pact_slots
        self.pact_slots = (pact_max, pact_max)


class Spell(BaseModel):
    """
    Component: A known or prepared spell.
    """
    
    name: str = Field(
        ...,
        min_length=1,
        description="Name of the spell."
    )
    level: int = Field(
        ...,
        ge=0,
        le=9,
        description="Spell level (0 for cantrips)."
    )
    school: str = Field(
        default="",
        description="School of magic (Evocation, Necromancy, etc.)."
    )
    casting_time: str = Field(
        default="1 action",
        description="Time required to cast (action, bonus action, reaction, etc.)."
    )
    range: str = Field(
        default="Self",
        description="Range of the spell."
    )
    components: str = Field(
        default="",
        description="Required components (V, S, M and materials)."
    )
    duration: str = Field(
        default="Instantaneous",
        description="How long the spell lasts."
    )
    description: str = Field(
        default="",
        description="Full spell description and effects."
    )
    prepared: bool = Field(
        default=True,
        description="Whether the spell is currently prepared (for prepared casters)."
    )
    ritual: bool = Field(
        default=False,
        description="Whether the spell can be cast as a ritual."
    )
    concentration: bool = Field(
        default=False,
        description="Whether the spell requires concentration."
    )


class RoleplayData(BaseModel):
    """
    Component: Narrative and personality information for roleplay.
    
    Contains all the flavor and story elements that make a character unique.
    """
    
    name: str = Field(
        ...,
        min_length=1,
        description="Character's full name or the name they go by."
    )
    title: str = Field(
        default="",
        description="Title or epithet (e.g., 'the Bold', 'Archmage of Waterdeep')."
    )
    race: str = Field(
        default="Human",
        description="Character's race/species."
    )
    character_class: str = Field(
        default="",
        description="Character's class(es) and levels (e.g., 'Fighter 5 / Wizard 2')."
    )
    background: str = Field(
        default="",
        description="Character's background (e.g., 'Noble', 'Criminal', 'Sage')."
    )
    alignment: str = Field(
        default="True Neutral",
        description="Character's moral and ethical alignment."
    )
    description: str = Field(
        default="",
        description="Physical appearance and distinguishing features."
    )
    personality_traits: list[str] = Field(
        default_factory=list,
        description="Distinctive personality quirks and behaviors."
    )
    ideals: list[str] = Field(
        default_factory=list,
        description="Core beliefs and principles that drive the character."
    )
    bonds: list[str] = Field(
        default_factory=list,
        description="Connections to people, places, or things that matter to the character."
    )
    flaws: list[str] = Field(
        default_factory=list,
        description="Weaknesses, vices, or fears that can be exploited."
    )
    backstory: str = Field(
        default="",
        description="The character's history and how they came to be an adventurer."
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="Hidden information known only to the DM. May be revealed during play."
    )
    notes: str = Field(
        default="",
        description="Miscellaneous notes about the character."
    )
    voice: str = Field(
        default="",
        description="Description of how the character speaks (accent, mannerisms, vocabulary)."
    )


class StatusEffect(BaseModel):
    """
    Component: An active condition or effect on a character.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for tracking this effect instance."
    )
    name: str = Field(
        ...,
        description="Name of the effect or condition."
    )
    condition: ConditionType | None = Field(
        default=None,
        description="Standard D&D condition if applicable."
    )
    source: str = Field(
        default="",
        description="What caused this effect (spell name, creature, trap, etc.)."
    )
    duration_rounds: int | None = Field(
        default=None,
        ge=0,
        description="Remaining duration in combat rounds. None if indefinite."
    )
    duration_description: str = Field(
        default="",
        description="Text description of duration (e.g., '1 minute', 'until dispelled')."
    )
    save_dc: int | None = Field(
        default=None,
        description="DC to save against this effect, if applicable."
    )
    save_ability: AbilityName | None = Field(
        default=None,
        description="Ability used for saving throws against this effect."
    )
    description: str = Field(
        default="",
        description="Description of the effect's impact."
    )


# =============================================================================
# ENTITY MODELS (Composed from Components)
# =============================================================================

class Character(BaseModel):
    """
    Entity: A complete D&D character (player, NPC, or creature).
    
    Composed of multiple components following ECS principles.
    This is the primary entity for representing any character in the game.
    """
    
    # Identity
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this character instance."
    )
    character_type: CharacterType = Field(
        default=CharacterType.NPC,
        description="Whether this is a player character, NPC, companion, or enemy."
    )
    level: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Total character level (sum of all class levels)."
    )
    experience_points: int = Field(
        default=0,
        ge=0,
        description="Total experience points earned."
    )
    proficiency_bonus: int = Field(
        default=2,
        ge=2,
        le=6,
        description="Proficiency bonus based on total level."
    )
    
    # Core Components
    stats: CoreStats = Field(
        default_factory=CoreStats,
        description="The six ability scores and their modifiers."
    )
    vitals: Vitals = Field(
        default_factory=Vitals,
        description="Hit points, AC, speed, and other combat vitals."
    )
    roleplay: RoleplayData = Field(
        ...,
        description="Name, personality, backstory, and other narrative elements."
    )
    
    # Collections
    inventory: list[InventoryItem | str] = Field(
        default_factory=list,
        description="Items carried by the character. Can be detailed items or simple strings."
    )
    features: list[ClassFeature] = Field(
        default_factory=list,
        description="Class features, racial traits, and other special abilities."
    )
    spells_known: list[Spell] = Field(
        default_factory=list,
        description="Spells the character knows or has in their spellbook."
    )
    spell_slots: SpellSlots = Field(
        default_factory=SpellSlots,
        description="Available spell slots by level."
    )
    conditions: list[StatusEffect] = Field(
        default_factory=list,
        description="Active conditions and effects on this character."
    )
    
    # Combat state
    death_saves: tuple[int, int] = Field(
        default=(0, 0),
        description="Death saving throws: (successes, failures). 3 of either ends the state."
    )
    
    # Currency
    copper: int = Field(default=0, ge=0, description="Copper pieces held.")
    silver: int = Field(default=0, ge=0, description="Silver pieces held.")
    electrum: int = Field(default=0, ge=0, description="Electrum pieces held.")
    gold: int = Field(default=0, ge=0, description="Gold pieces held.")
    platinum: int = Field(default=0, ge=0, description="Platinum pieces held.")

    # Proficiencies and Languages
    saving_throw_proficiencies: list[AbilityName] = Field(
        default_factory=list,
        description="Abilities the character is proficient in for saving throws."
    )
    skill_proficiencies: list[str] = Field(
        default_factory=list,
        description="Skills the character is proficient in."
    )
    skill_expertise: list[str] = Field(
        default_factory=list,
        description="Skills the character has expertise in (double proficiency)."
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages the character can speak, read, and write."
    )
    tool_proficiencies: list[str] = Field(
        default_factory=list,
        description="Tools the character is proficient with."
    )
    weapon_proficiencies: list[str] = Field(
        default_factory=list,
        description="Weapon categories or specific weapons the character is proficient with."
    )
    armor_proficiencies: list[str] = Field(
        default_factory=list,
        description="Armor types the character is proficient with."
    )
    
    # Resistances and Immunities
    damage_resistances: list[DamageType] = Field(
        default_factory=list,
        description="Damage types the character takes half damage from."
    )
    damage_immunities: list[DamageType] = Field(
        default_factory=list,
        description="Damage types the character is immune to."
    )
    damage_vulnerabilities: list[DamageType] = Field(
        default_factory=list,
        description="Damage types the character takes double damage from."
    )
    condition_immunities: list[ConditionType] = Field(
        default_factory=list,
        description="Conditions the character cannot be affected by."
    )

    @computed_field
    @property
    def total_wealth_gp(self) -> float:
        """Calculate total wealth in gold pieces."""
        return (
            self.copper / 100 +
            self.silver / 10 +
            self.electrum / 2 +
            self.gold +
            self.platinum * 10
        )

    @computed_field
    @property
    def is_spellcaster(self) -> bool:
        """True if the character has any spell slots or known spells."""
        has_slots = any(
            self.spell_slots.get_slots(level)[1] > 0 
            for level in range(1, 10)
        )
        has_spells = len(self.spells_known) > 0
        return has_slots or has_spells

    def update_stat(
        self,
        component: Literal["stats", "vitals"],
        field: str,
        value: Any,
        *,
        relative: bool = False
    ) -> None:
        """
        Safely modify a character stat with validation.
        
        Args:
            component: Which component to update ('stats' or 'vitals')
            field: The field name within the component
            value: The new value (absolute) or delta (if relative=True)
            relative: If True, add value to current; if False, set directly
            
        Raises:
            ValueError: If the component or field is invalid
            
        Example:
            character.update_stat("stats", "strength", 18)  # Set STR to 18
            character.update_stat("vitals", "current_hp", -5, relative=True)  # Take 5 damage
        """
        if component == "stats":
            target = self.stats
        elif component == "vitals":
            target = self.vitals
        else:
            raise ValueError(f"Invalid component '{component}'. Use 'stats' or 'vitals'.")
        
        if not hasattr(target, field):
            raise ValueError(f"Component '{component}' has no field '{field}'.")
        
        current_value = getattr(target, field)
        
        if relative:
            if not isinstance(current_value, (int, float)):
                raise ValueError(f"Cannot use relative update on non-numeric field '{field}'.")
            new_value = current_value + value
        else:
            new_value = value
        
        setattr(target, field, new_value)
        
        # Re-validate the component
        if component == "stats":
            self.stats = CoreStats.model_validate(self.stats.model_dump())
        else:
            self.vitals = Vitals.model_validate(self.vitals.model_dump())

    def take_damage(self, amount: int, damage_type: DamageType | None = None) -> int:
        """
        Apply damage to the character with resistance/vulnerability calculations.
        
        Args:
            amount: Base damage amount
            damage_type: Type of damage for resistance/vulnerability checks
            
        Returns:
            Actual damage taken after modifications
        """
        actual_damage = amount
        
        if damage_type:
            if damage_type in self.damage_immunities:
                actual_damage = 0
            elif damage_type in self.damage_resistances:
                actual_damage = amount // 2
            elif damage_type in self.damage_vulnerabilities:
                actual_damage = amount * 2
        
        # Temp HP absorbs first
        if self.vitals.temp_hp > 0:
            absorbed = min(self.vitals.temp_hp, actual_damage)
            self.vitals.temp_hp -= absorbed
            actual_damage -= absorbed
        
        # Then regular HP
        self.vitals.current_hp = max(0, self.vitals.current_hp - actual_damage)
        
        return actual_damage

    def heal(self, amount: int) -> int:
        """
        Heal the character, respecting max HP.
        
        Args:
            amount: Amount to heal
            
        Returns:
            Actual amount healed
        """
        old_hp = self.vitals.current_hp
        self.vitals.current_hp = min(self.vitals.max_hp, self.vitals.current_hp + amount)
        return self.vitals.current_hp - old_hp

    def add_condition(self, condition: StatusEffect) -> None:
        """Add a status effect/condition to the character."""
        if condition.condition and condition.condition in self.condition_immunities:
            return  # Character is immune
        self.conditions.append(condition)

    def remove_condition(self, condition_id: UUID) -> bool:
        """Remove a condition by ID. Returns True if found and removed."""
        for i, cond in enumerate(self.conditions):
            if cond.id == condition_id:
                self.conditions.pop(i)
                return True
        return False

    def has_condition(self, condition_type: ConditionType) -> bool:
        """Check if character currently has a specific condition."""
        return any(c.condition == condition_type for c in self.conditions)

    def short_rest(self) -> None:
        """Apply short rest benefits: restore features that recharge on short rest."""
        for feature in self.features:
            if feature.recharge == "short_rest" and feature.uses_max is not None:
                feature.uses_current = feature.uses_max
        # Restore warlock pact slots
        _, pact_max = self.spell_slots.pact_slots
        self.spell_slots.pact_slots = (pact_max, pact_max)

    def long_rest(self) -> None:
        """Apply long rest benefits: restore HP, all features, all spell slots."""
        self.vitals.current_hp = self.vitals.max_hp
        self.vitals.temp_hp = 0
        self.death_saves = (0, 0)
        
        for feature in self.features:
            if feature.uses_max is not None:
                feature.uses_current = feature.uses_max
        
        self.spell_slots.restore_all()

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


# =============================================================================
# MESSAGE & CHAT MODELS
# =============================================================================

class Message(BaseModel):
    """
    A single message in the chat history.
    
    Supports various roles for complex multi-character interactions.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this message."
    )
    role: MessageRole = Field(
        ...,
        description="Who sent this message (system, user, assistant, dm, character)."
    )
    content: str = Field(
        ...,
        description="The text content of the message."
    )
    character_id: UUID | None = Field(
        default=None,
        description="If role is 'character', the ID of the speaking character."
    )
    character_name: str | None = Field(
        default=None,
        description="Display name if this message is from a character."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this message was sent."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (dice rolls, references, etc.)."
    )


# =============================================================================
# GAME STATE MODEL
# =============================================================================

class Location(BaseModel):
    """
    Represents a location in the game world.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this location."
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Name of the location."
    )
    description: str = Field(
        default="",
        description="Detailed description of the location for narrative purposes."
    )
    region: str = Field(
        default="",
        description="Larger region or area this location belongs to."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization (dungeon, city, wilderness, etc.)."
    )
    connected_locations: list[UUID] = Field(
        default_factory=list,
        description="IDs of locations that can be traveled to from here."
    )
    npcs_present: list[UUID] = Field(
        default_factory=list,
        description="IDs of NPCs currently at this location."
    )
    items_present: list[InventoryItem] = Field(
        default_factory=list,
        description="Items that can be found/interacted with at this location."
    )
    hidden_details: str = Field(
        default="",
        description="DM-only information about hidden aspects of this location."
    )


class DMNotes(BaseModel):
    """
    Hidden state for the AI Dungeon Master.
    
    Contains information players shouldn't see but the DM needs to track.
    """
    
    session_goals: list[str] = Field(
        default_factory=list,
        description="Objectives for the current session."
    )
    plot_hooks: list[str] = Field(
        default_factory=list,
        description="Story hooks that can be introduced."
    )
    active_threats: list[str] = Field(
        default_factory=list,
        description="Current dangers or antagonist activities."
    )
    upcoming_encounters: list[str] = Field(
        default_factory=list,
        description="Planned encounters and their triggers."
    )
    faction_states: dict[str, str] = Field(
        default_factory=dict,
        description="Current state of various factions and their attitudes."
    )
    world_events: list[str] = Field(
        default_factory=list,
        description="Background events happening in the world."
    )
    player_secrets: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Secrets known about each player character (keyed by character ID string)."
    )
    npc_motivations: dict[str, str] = Field(
        default_factory=dict,
        description="Hidden motivations of key NPCs (keyed by NPC ID string)."
    )
    loot_tables: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Potential loot for different encounter types."
    )
    narrative_threads: list[str] = Field(
        default_factory=list,
        description="Ongoing storylines and their current states."
    )
    notes: str = Field(
        default="",
        description="Free-form DM notes and reminders."
    )


class CombatState(BaseModel):
    """
    Tracks the state of an active combat encounter.
    """
    
    is_active: bool = Field(
        default=False,
        description="Whether combat is currently happening."
    )
    round_number: int = Field(
        default=0,
        ge=0,
        description="Current combat round."
    )
    turn_order: list[UUID] = Field(
        default_factory=list,
        description="Character IDs in initiative order."
    )
    current_turn_index: int = Field(
        default=0,
        ge=0,
        description="Index in turn_order for whose turn it is."
    )
    initiative_rolls: dict[str, int] = Field(
        default_factory=dict,
        description="Initiative results keyed by character ID string."
    )
    lair_actions_used: bool = Field(
        default=False,
        description="Whether lair action was used this round."
    )
    legendary_actions_remaining: dict[str, int] = Field(
        default_factory=dict,
        description="Remaining legendary actions by creature ID."
    )


class GameState(BaseModel):
    """
    Entity: The complete state of a D&D campaign session.
    
    This is the root model that contains all game data needed for an AI DM
    to understand and continue the game.
    """
    
    # Session Identity
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this game session."
    )
    campaign_name: str = Field(
        default="Untitled Campaign",
        description="Name of the campaign."
    )
    session_number: int = Field(
        default=1,
        ge=1,
        description="Current session number for tracking."
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this game state was created."
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When this game state was last modified."
    )
    
    # World Context
    world_context: str = Field(
        default="",
        description="Summary of the current situation, recent events, and immediate context. "
                    "The LLM should read this to understand what's happening in the story."
    )
    world_lore: str = Field(
        default="",
        description="Background information about the campaign setting, history, and cosmology."
    )
    house_rules: list[str] = Field(
        default_factory=list,
        description="Custom rules and modifications to standard 5E rules."
    )
    
    # Location
    active_location: Location | None = Field(
        default=None,
        description="The current location where the party is. Contains description and relevant details."
    )
    known_locations: list[Location] = Field(
        default_factory=list,
        description="All locations the party has discovered or knows about."
    )
    
    # Characters
    party: list[Character] = Field(
        default_factory=list,
        description="All party members including the player character and AI companions. "
                    "The LLM controls companions and NPCs during roleplay."
    )
    npcs: list[Character] = Field(
        default_factory=list,
        description="Non-party NPCs that have been introduced or are relevant to the story."
    )
    
    # Combat
    combat: CombatState = Field(
        default_factory=CombatState,
        description="Current combat encounter state, if any."
    )
    
    # Narrative
    chat_history: list[Message] = Field(
        default_factory=list,
        description="Complete history of the session's messages and narration. "
                    "Includes player actions, DM descriptions, and character dialogue."
    )
    
    # DM State
    dm_notes: DMNotes = Field(
        default_factory=DMNotes,
        description="Hidden information for the AI DM. Contains secrets, plot hooks, "
                    "and behind-the-scenes narrative tracking. Never reveal to players."
    )
    
    # Time Tracking
    in_game_date: str = Field(
        default="Day 1",
        description="Current in-game date or time period."
    )
    in_game_time: str = Field(
        default="Morning",
        description="Current time of day in-game."
    )
    
    # Quest Tracking
    active_quests: list[str] = Field(
        default_factory=list,
        description="Currently active quest objectives the party is pursuing."
    )
    completed_quests: list[str] = Field(
        default_factory=list,
        description="Quests that have been completed."
    )

    def get_character_by_id(self, character_id: UUID) -> Character | None:
        """Find a character (party member or NPC) by their ID."""
        for char in self.party + self.npcs:
            if char.id == character_id:
                return char
        return None

    def get_character_by_name(self, name: str) -> Character | None:
        """Find a character by their name (case-insensitive)."""
        name_lower = name.lower()
        for char in self.party + self.npcs:
            if char.roleplay.name.lower() == name_lower:
                return char
        return None

    def add_to_history(
        self,
        role: MessageRole,
        content: str,
        character_id: UUID | None = None,
        character_name: str | None = None,
        **metadata: Any
    ) -> Message:
        """Add a new message to the chat history."""
        message = Message(
            role=role,
            content=content,
            character_id=character_id,
            character_name=character_name,
            metadata=metadata
        )
        self.chat_history.append(message)
        self.updated_at = datetime.now()
        return message

    def get_party_status_summary(self) -> str:
        """Generate a quick status summary of all party members."""
        lines = []
        for char in self.party:
            status = f"{char.roleplay.name}: {char.vitals.current_hp}/{char.vitals.max_hp} HP"
            if char.vitals.is_bloodied:
                status += " [BLOODIED]"
            if char.vitals.is_unconscious:
                status += " [UNCONSCIOUS]"
            if char.conditions:
                conds = ", ".join(c.name for c in char.conditions)
                status += f" ({conds})"
            lines.append(status)
        return "\n".join(lines)

    def start_combat(self, combatants: list[UUID], initiative_rolls: dict[str, int]) -> None:
        """Initialize combat with the given combatants and their initiative rolls."""
        self.combat.is_active = True
        self.combat.round_number = 1
        self.combat.initiative_rolls = initiative_rolls
        # Sort by initiative (highest first)
        self.combat.turn_order = sorted(
            combatants,
            key=lambda cid: initiative_rolls.get(str(cid), 0),
            reverse=True
        )
        self.combat.current_turn_index = 0

    def end_combat(self) -> None:
        """End the current combat encounter."""
        self.combat = CombatState()

    def next_turn(self) -> UUID | None:
        """Advance to the next turn in combat. Returns the ID of the character whose turn it is."""
        if not self.combat.is_active or not self.combat.turn_order:
            return None
        
        self.combat.current_turn_index += 1
        if self.combat.current_turn_index >= len(self.combat.turn_order):
            self.combat.current_turn_index = 0
            self.combat.round_number += 1
            self.combat.lair_actions_used = False
        
        return self.combat.turn_order[self.combat.current_turn_index]

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_character(
    name: str,
    character_type: CharacterType = CharacterType.NPC,
    **kwargs: Any
) -> Character:
    """
    Factory function to create a character with minimal required fields.
    
    Args:
        name: Character name
        character_type: Type of character (player, npc, companion, enemy)
        **kwargs: Additional fields to set on the character
        
    Returns:
        A new Character instance
    """
    roleplay = RoleplayData(name=name)
    return Character(
        character_type=character_type,
        roleplay=roleplay,
        **kwargs
    )


def create_game_state(
    campaign_name: str = "New Campaign",
    player_name: str = "Hero",
    **kwargs: Any
) -> GameState:
    """
    Factory function to create a new game state with a player character.
    
    Args:
        campaign_name: Name of the campaign
        player_name: Name of the player character
        **kwargs: Additional fields to set on the game state
        
    Returns:
        A new GameState instance with one player character
    """
    player = create_character(player_name, CharacterType.PLAYER)
    
    return GameState(
        campaign_name=campaign_name,
        party=[player],
        world_context=f"The adventure begins for {player_name}...",
        **kwargs
    )
