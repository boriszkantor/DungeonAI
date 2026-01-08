"""Entity Component System (ECS) models for DungeonAI.

This module implements a Pydantic V2-based ECS architecture where:
- Entities are the base containers (ActorEntity, ItemEntity, etc.)
- Components are modular data containers attached to entities
- Systems operate on entities with specific component combinations

The ECS pattern enables:
- Flexible entity composition
- Clean separation of concerns
- Type-safe state management
- Easy serialization/deserialization

NEURO-SYMBOLIC PRINCIPLE:
This module represents TRUTH. The Python ECS owns all game state.
LLMs may only READ state and EMIT StateUpdateRequests.
LLMs NEVER directly mutate these objects.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# =============================================================================
# Type Definitions
# =============================================================================


AbilityScore = Annotated[int, Field(ge=1, le=30, description="D&D ability score (1-30)")]
Level = Annotated[int, Field(ge=1, le=20, description="Character level (1-20)")]
ChallengeRating = Annotated[str, Field(pattern=r"^(\d+(/\d+)?|\d+)$", description="CR like '1/4', '1', '5'")]


class ActorType(StrEnum):
    """Type of actor entity for polymorphic handling."""

    PLAYER = "player"
    """Player-controlled character."""

    NPC_ALLY = "npc_ally"
    """AI-controlled ally NPC."""

    NPC_NEUTRAL = "npc_neutral"
    """Neutral NPC (shopkeeper, questgiver)."""

    MONSTER = "monster"
    """DM-controlled hostile creature."""


class Ability(StrEnum):
    """The six D&D ability scores."""

    STR = "strength"
    DEX = "dexterity"
    CON = "constitution"
    INT = "intelligence"
    WIS = "wisdom"
    CHA = "charisma"


class DamageType(StrEnum):
    """D&D 5E damage types."""

    SLASHING = "slashing"
    PIERCING = "piercing"
    BLUDGEONING = "bludgeoning"
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


class Condition(StrEnum):
    """D&D 5E conditions."""

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
    EXHAUSTION_1 = "exhaustion_1"
    EXHAUSTION_2 = "exhaustion_2"
    EXHAUSTION_3 = "exhaustion_3"
    EXHAUSTION_4 = "exhaustion_4"
    EXHAUSTION_5 = "exhaustion_5"
    EXHAUSTION_6 = "exhaustion_6"


class SpellSchool(StrEnum):
    """D&D 5E spell schools."""

    ABJURATION = "abjuration"
    CONJURATION = "conjuration"
    DIVINATION = "divination"
    ENCHANTMENT = "enchantment"
    EVOCATION = "evocation"
    ILLUSION = "illusion"
    NECROMANCY = "necromancy"
    TRANSMUTATION = "transmutation"


class EffectType(StrEnum):
    """Types of active effects that can be applied to actors."""
    
    AC_SET = "ac_set"  # Set AC to a value (Mage Armor, Barkskin)
    AC_BONUS = "ac_bonus"  # Add to AC (Shield, Shield of Faith)
    TEMP_HP = "temp_hp"  # Temporary hit points
    DAMAGE_RESISTANCE = "damage_resistance"  # Resistance to damage type
    DAMAGE_IMMUNITY = "damage_immunity"  # Immunity to damage type
    CONDITION_IMMUNITY = "condition_immunity"  # Immunity to condition
    ADVANTAGE_ATTACKS = "advantage_attacks"  # Advantage on attacks
    DISADVANTAGE_ATTACKS = "disadvantage_attacks"  # Attackers have disadvantage
    SPEED_BONUS = "speed_bonus"  # Speed increase
    STAT_BONUS = "stat_bonus"  # Bonus to ability score/check
    SAVE_BONUS = "save_bonus"  # Bonus to saving throws
    EXTRA_ATTACK = "extra_attack"  # Additional attack (Haste)
    INVISIBILITY = "invisibility"  # Invisible condition
    FLYING = "flying"  # Flying speed
    CUSTOM = "custom"  # Custom effect with description


class EffectDuration(StrEnum):
    """How effect duration is tracked."""
    
    INSTANTANEOUS = "instantaneous"  # Immediate, no duration
    UNTIL_START_OF_NEXT_TURN = "until_start_of_next_turn"  # Shield
    UNTIL_END_OF_NEXT_TURN = "until_end_of_next_turn"
    ROUNDS = "rounds"  # X combat rounds
    MINUTES = "minutes"  # X minutes (10 rounds = 1 minute)
    HOURS = "hours"  # X hours
    UNTIL_SHORT_REST = "until_short_rest"
    UNTIL_LONG_REST = "until_long_rest"
    UNTIL_DISPELLED = "until_dispelled"
    CONCENTRATION = "concentration"  # Ends when concentration breaks


# =============================================================================
# Active Effect Model
# =============================================================================


class ActiveEffect(BaseModel):
    """Represents an active buff, debuff, or condition on an actor.
    
    Effects are applied by spells, abilities, items, or conditions.
    They modify actor stats/behavior and expire based on duration.
    """
    
    model_config = ConfigDict(extra="ignore", use_enum_values=True)
    
    # Identity
    uid: UUID = Field(default_factory=uuid4, description="Unique effect ID")
    name: str = Field(description="Effect name (e.g., 'Mage Armor', 'Blessed')")
    source: str = Field(default="", description="What applied this (spell, item, feature)")
    source_actor: str = Field(default="", description="Who applied this effect")
    
    # Effect mechanics
    effect_type: EffectType = Field(description="Type of mechanical effect")
    value: int | None = Field(default=None, description="Numeric value (AC bonus, damage, etc.)")
    damage_type: DamageType | None = Field(default=None, description="For resistance/immunity")
    description: str = Field(default="", description="Human-readable description")
    
    # Duration tracking
    duration_type: EffectDuration = Field(description="How duration is measured")
    duration_value: int = Field(default=0, description="Duration amount (rounds, minutes, hours)")
    concentration: bool = Field(default=False, description="Requires concentration")
    
    # Combat timing
    applied_round: int = Field(default=0, description="Combat round when applied")
    applied_turn: int = Field(default=0, description="Turn index when applied")
    expires_round: int | None = Field(default=None, description="Round when effect expires")
    expires_turn: int | None = Field(default=None, description="Turn when effect expires")
    
    # State
    is_active: bool = Field(default=True, description="Whether effect is currently active")
    
    def should_expire_on_turn_start(self, current_round: int, current_turn: int) -> bool:
        """Check if effect should expire at start of this turn."""
        dur_type = self.duration_type.value if hasattr(self.duration_type, 'value') else str(self.duration_type)
        if dur_type == "until_start_of_next_turn":
            # Expires at start of the turn after it was applied
            if current_round > self.applied_round:
                return True
            if current_round == self.applied_round and current_turn >= self.applied_turn:
                # Same round but we've cycled back
                return current_turn != self.applied_turn or current_round > self.applied_round
        return False
    
    def should_expire_on_turn_end(self, current_round: int, current_turn: int) -> bool:
        """Check if effect should expire at end of this turn."""
        if self.expires_round is not None and self.expires_turn is not None:
            if current_round > self.expires_round:
                return True
            if current_round == self.expires_round and current_turn >= self.expires_turn:
                return True
        return False
    
    def should_expire_on_rest(self, rest_type: str) -> bool:
        """Check if effect expires on short or long rest."""
        dur_type = self.duration_type.value if hasattr(self.duration_type, 'value') else str(self.duration_type)
        if rest_type == "short":
            return dur_type == "until_short_rest"
        if rest_type == "long":
            return dur_type in ("until_short_rest", "until_long_rest")
        return False
    
    def calculate_expiration(self, current_round: int, current_turn: int) -> None:
        """Calculate when this effect expires based on duration."""
        dur_type = self.duration_type.value if hasattr(self.duration_type, 'value') else str(self.duration_type)
        if dur_type == "rounds" and self.duration_value > 0:
            self.expires_round = current_round + self.duration_value
            self.expires_turn = current_turn
        elif dur_type == "minutes" and self.duration_value > 0:
            # 1 minute = 10 rounds
            rounds = self.duration_value * 10
            self.expires_round = current_round + rounds
            self.expires_turn = current_turn


# =============================================================================
# Base Component
# =============================================================================


class Component(BaseModel):
    """Base class for all ECS components.
    
    Components are pure data containers with no behavior.
    They attach to entities to provide specific capabilities.
    """

    model_config = ConfigDict(
        frozen=False,  # Components can be mutated by the engine
        validate_assignment=True,
        extra="ignore",  # Ignore computed fields when deserializing
        use_enum_values=True,
    )


# =============================================================================
# Core Components
# =============================================================================


class StatsComponent(Component):
    """Ability scores and derived stats.
    
    This is the foundational component for any entity that
    participates in ability checks, saves, or combat.
    """

    strength: AbilityScore = Field(default=10, description="Physical power")
    dexterity: AbilityScore = Field(default=10, description="Agility and reflexes")
    constitution: AbilityScore = Field(default=10, description="Health and stamina")
    intelligence: AbilityScore = Field(default=10, description="Reasoning and memory")
    wisdom: AbilityScore = Field(default=10, description="Perception and insight")
    charisma: AbilityScore = Field(default=10, description="Force of personality")

    # Proficiency
    proficiency_bonus: int = Field(default=2, ge=2, le=6, description="Proficiency bonus")

    # Saving throw proficiencies
    save_proficiencies: list[Ability] = Field(default_factory=list)

    # Skill proficiencies (skill name -> multiplier: 1=proficient, 2=expertise)
    skill_proficiencies: dict[str, int] = Field(default_factory=dict)

    @field_validator("skill_proficiencies", mode="before")
    @classmethod
    def filter_none_skills(cls, v: Any) -> dict[str, int]:
        """Filter out None values from skill proficiencies (Vision AI may return nulls)."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {k: int(val) for k, val in v.items() if val is not None}
        return v

    @staticmethod
    def calc_modifier(score: int) -> int:
        """Calculate ability modifier from score."""
        return (score - 10) // 2

    @computed_field(description="Strength modifier")
    @property
    def str_mod(self) -> int:
        return self.calc_modifier(self.strength)

    @computed_field(description="Dexterity modifier")
    @property
    def dex_mod(self) -> int:
        return self.calc_modifier(self.dexterity)

    @computed_field(description="Constitution modifier")
    @property
    def con_mod(self) -> int:
        return self.calc_modifier(self.constitution)

    @computed_field(description="Intelligence modifier")
    @property
    def int_mod(self) -> int:
        return self.calc_modifier(self.intelligence)

    @computed_field(description="Wisdom modifier")
    @property
    def wis_mod(self) -> int:
        return self.calc_modifier(self.wisdom)

    @computed_field(description="Charisma modifier")
    @property
    def cha_mod(self) -> int:
        return self.calc_modifier(self.charisma)

    def get_save_bonus(self, ability: Ability) -> int:
        """Get saving throw bonus for an ability."""
        mod = self.calc_modifier(getattr(self, ability.value[:3] + "_mod", 0))
        # Recalculate from the score directly
        score = getattr(self, ability.value, 10)
        mod = self.calc_modifier(score)
        if ability in self.save_proficiencies:
            return mod + self.proficiency_bonus
        return mod

    def get_skill_bonus(self, skill: str, ability: Ability) -> int:
        """Get skill check bonus."""
        score = getattr(self, ability.value, 10)
        mod = self.calc_modifier(score)
        multiplier = self.skill_proficiencies.get(skill.lower(), 0)
        return mod + (self.proficiency_bonus * multiplier)


class HealthComponent(Component):
    """Hit points and death saves.
    
    Tracks current, maximum, and temporary HP along with
    death saving throw progress and active conditions.
    """

    hp_current: int = Field(default=10, ge=0, description="Current hit points")
    hp_max: int = Field(default=10, ge=1, description="Maximum hit points")
    hp_temp: int = Field(default=0, ge=0, description="Temporary hit points")

    death_saves_success: int = Field(default=0, ge=0, le=3)
    death_saves_failure: int = Field(default=0, ge=0, le=3)

    conditions: list[Condition] = Field(default_factory=list, description="Active conditions")

    @computed_field(description="HP as percentage")
    @property
    def hp_percentage(self) -> float:
        if self.hp_max <= 0:
            return 0.0
        return (self.hp_current / self.hp_max) * 100

    @computed_field(description="Whether entity is conscious")
    @property
    def is_conscious(self) -> bool:
        return self.hp_current > 0 and Condition.UNCONSCIOUS not in self.conditions

    @computed_field(description="Whether entity is dead")
    @property
    def is_dead(self) -> bool:
        return self.death_saves_failure >= 3

    @computed_field(description="Whether entity is dying")
    @property
    def is_dying(self) -> bool:
        return self.hp_current == 0 and not self.is_dead

    def apply_damage(self, amount: int, damage_type: DamageType | None = None) -> int:
        """Apply damage and return actual damage dealt.
        
        Temp HP absorbs damage first.
        """
        if amount <= 0:
            return 0

        actual = 0

        # Temp HP absorbs first
        if self.hp_temp > 0:
            absorbed = min(self.hp_temp, amount)
            self.hp_temp -= absorbed
            amount -= absorbed

        # Remaining damage to current HP
        if amount > 0:
            actual = min(self.hp_current, amount)
            self.hp_current = max(0, self.hp_current - amount)

        # Check for instant death (massive damage)
        if self.hp_current == 0 and amount >= self.hp_max:
            self.death_saves_failure = 3

        return actual

    def apply_healing(self, amount: int) -> int:
        """Apply healing and return actual HP restored."""
        if amount <= 0 or self.is_dead:
            return 0

        before = self.hp_current
        self.hp_current = min(self.hp_max, self.hp_current + amount)

        # Reset death saves on healing from 0
        if before == 0 and self.hp_current > 0:
            self.death_saves_success = 0
            self.death_saves_failure = 0
            if Condition.UNCONSCIOUS in self.conditions:
                self.conditions.remove(Condition.UNCONSCIOUS)

        return self.hp_current - before


class DefenseComponent(Component):
    """Armor class and resistances.
    
    Handles AC calculation and damage modifiers.
    """

    ac_base: int = Field(default=10, ge=0, description="Base armor class")
    ac_armor: int = Field(default=0, ge=0, description="AC from armor")
    ac_shield: int = Field(default=0, ge=0, description="AC from shield")
    ac_other: int = Field(default=0, description="Other AC modifiers")

    # Use DEX for AC (false for heavy armor)
    uses_dex: bool = Field(default=True)
    max_dex_bonus: int | None = Field(default=None, description="Max DEX to AC (medium armor)")

    resistances: list[DamageType] = Field(default_factory=list)
    immunities: list[DamageType] = Field(default_factory=list)
    vulnerabilities: list[DamageType] = Field(default_factory=list)

    condition_immunities: list[Condition] = Field(default_factory=list)

    def calculate_ac(self, dex_mod: int) -> int:
        """Calculate total AC given DEX modifier."""
        ac = self.ac_base + self.ac_armor + self.ac_shield + self.ac_other

        if self.uses_dex:
            dex_bonus = dex_mod
            if self.max_dex_bonus is not None:
                dex_bonus = min(dex_bonus, self.max_dex_bonus)
            ac += dex_bonus

        return ac

    def modify_damage(self, amount: int, damage_type: DamageType) -> int:
        """Modify damage based on resistances/immunities/vulnerabilities."""
        if damage_type in self.immunities:
            return 0
        if damage_type in self.resistances:
            return amount // 2
        if damage_type in self.vulnerabilities:
            return amount * 2
        return amount


class InventoryComponent(Component):
    """Equipment and carried items.
    
    Tracks all items an entity possesses, including
    equipped items and currency.
    """

    items: list["ItemStack"] = Field(default_factory=list)
    equipped: dict[str, UUID | None] = Field(
        default_factory=lambda: {
            "main_hand": None,
            "off_hand": None,
            "armor": None,
            "head": None,
            "cloak": None,
            "neck": None,
            "ring_1": None,
            "ring_2": None,
            "boots": None,
            "gloves": None,
        }
    )

    # Currency in copper pieces (1 gp = 100 cp)
    currency_cp: int = Field(default=0, ge=0)

    @computed_field(description="Gold pieces equivalent")
    @property
    def currency_gp(self) -> float:
        return self.currency_cp / 100

    def add_item(self, item: "ItemStack") -> None:
        """Add an item to inventory."""
        # Check if stackable item already exists
        for existing in self.items:
            if existing.item_id == item.item_id and existing.is_stackable:
                existing.quantity += item.quantity
                return
        self.items.append(item)

    def remove_item(self, item_uid: UUID, quantity: int = 1) -> bool:
        """Remove item(s) from inventory. Returns success."""
        for i, item in enumerate(self.items):
            if item.uid == item_uid:
                if item.quantity <= quantity:
                    self.items.pop(i)
                else:
                    item.quantity -= quantity
                return True
        return False

    def get_equipped_item(self, slot: str) -> "ItemStack | None":
        """Get the item in an equipment slot."""
        item_uid = self.equipped.get(slot)
        if item_uid is None:
            return None
        for item in self.items:
            if item.uid == item_uid:
                return item
        return None


class ItemType(StrEnum):
    """Types of items."""
    WEAPON = "weapon"
    ARMOR = "armor"
    SHIELD = "shield"
    GEAR = "gear"
    CONSUMABLE = "consumable"
    TOOL = "tool"
    TREASURE = "treasure"


class ItemStack(BaseModel):
    """A stack of items in inventory."""

    model_config = ConfigDict(frozen=False)

    uid: UUID = Field(default_factory=uuid4)
    item_id: str = Field(description="Reference to item definition")
    name: str = Field(description="Display name")
    quantity: int = Field(default=1, ge=1)
    is_stackable: bool = Field(default=True)
    equipped: bool = Field(default=False, description="Whether item is currently equipped")
    item_type: ItemType = Field(default=ItemType.GEAR)

    # Item properties
    weight: float = Field(default=0.0, ge=0)
    value_cp: int = Field(default=0, ge=0, description="Value in copper")
    description: str = Field(default="")
    
    # Weapon properties (if item_type == WEAPON)
    damage_dice: str | None = Field(default=None, description="e.g., '1d8'")
    damage_type: str | None = Field(default=None, description="e.g., 'slashing'")
    weapon_properties: list[str] = Field(default_factory=list, description="e.g., ['versatile', 'finesse']")
    range_normal: int | None = Field(default=None, description="Normal range in feet")
    range_long: int | None = Field(default=None, description="Long range in feet")
    
    # Armor properties (if item_type == ARMOR or SHIELD)
    ac_base: int | None = Field(default=None, description="Base AC for armor")
    ac_bonus: int | None = Field(default=None, description="AC bonus for shields")
    max_dex_bonus: int | None = Field(default=None, description="Max DEX bonus for medium armor")
    stealth_disadvantage: bool = Field(default=False)
    strength_requirement: int | None = Field(default=None)
    
    # General properties dict for anything else
    properties: dict[str, Any] = Field(default_factory=dict)


class SpellbookComponent(Component):
    """Spellcasting capabilities.
    
    Tracks known/prepared spells, spell slots, and
    spellcasting ability.
    """

    spellcasting_ability: Ability | None = Field(default=None)
    spell_save_dc: int = Field(default=10, ge=1)
    spell_attack_bonus: int = Field(default=0)

    # Spell slots: level -> (current, max)
    spell_slots: dict[int, tuple[int, int]] = Field(default_factory=dict)

    # Known spells (for spontaneous casters)
    spells_known: list[str] = Field(default_factory=list)

    # Prepared spells (for prepared casters)
    spells_prepared: list[str] = Field(default_factory=list)

    # Cantrips (always available)
    cantrips: list[str] = Field(default_factory=list)

    # Pact magic (Warlock)
    pact_slots_current: int = Field(default=0, ge=0)
    pact_slots_max: int = Field(default=0, ge=0)
    pact_slot_level: int = Field(default=0, ge=0, le=5)

    def has_slot(self, level: int) -> bool:
        """Check if a spell slot of the given level is available."""
        if level == 0:
            return True  # Cantrips don't use slots
        current, _ = self.spell_slots.get(level, (0, 0))
        return current > 0

    def use_slot(self, level: int) -> bool:
        """Use a spell slot. Returns success."""
        if level == 0:
            return True
        if level not in self.spell_slots:
            return False
        current, max_slots = self.spell_slots[level]
        if current <= 0:
            return False
        self.spell_slots[level] = (current - 1, max_slots)
        return True

    def restore_slots(self, level: int | None = None) -> None:
        """Restore spell slots. If level is None, restore all."""
        if level is None:
            for lvl in self.spell_slots:
                _, max_slots = self.spell_slots[lvl]
                self.spell_slots[lvl] = (max_slots, max_slots)
        elif level in self.spell_slots:
            _, max_slots = self.spell_slots[level]
            self.spell_slots[level] = (max_slots, max_slots)


class JournalComponent(Component):
    """Memory and narrative history for AI actors.
    
    This component enables AI-controlled entities to maintain
    context about past events, relationships, and goals.
    Also stores character sheet "page 2" content.
    """

    # Core memory entries
    memories: list["MemoryEntry"] = Field(default_factory=list)

    # Relationship tracking
    relationships: dict[str, "Relationship"] = Field(default_factory=dict)

    # Active goals/motivations
    goals: list[str] = Field(default_factory=list)

    # Personality traits for AI roleplay (Character Sheet Page 2)
    personality_traits: list[str] = Field(default_factory=list)
    ideals: list[str] = Field(default_factory=list)
    bonds: list[str] = Field(default_factory=list)
    flaws: list[str] = Field(default_factory=list)
    
    # Backstory and narrative (Character Sheet Page 2)
    backstory: str = Field(default="", description="Character's history and background story")
    allies_and_organizations: str = Field(default="", description="Allies, organizations, and factions")
    
    # Appearance details (Character Sheet Page 2)
    appearance: str = Field(default="", description="Physical appearance description")
    age: str = Field(default="", description="Character's age")
    height: str = Field(default="", description="Character's height")
    weight: str = Field(default="", description="Character's weight")
    eyes: str = Field(default="", description="Eye color")
    hair: str = Field(default="", description="Hair color/style")
    skin: str = Field(default="", description="Skin tone")
    
    # Additional treasure/notes (Character Sheet Page 2)
    treasure: str = Field(default="", description="Additional treasure and valuables")
    additional_features_traits: str = Field(default="", description="Additional features and traits not captured elsewhere")

    # Voice/speech pattern hints
    voice_style: str = Field(default="neutral")

    def add_memory(self, content: str, importance: int = 5) -> None:
        """Add a memory entry."""
        self.memories.append(MemoryEntry(
            content=content,
            importance=importance,
            timestamp=datetime.now(),
        ))

        # Keep only recent/important memories (max 100)
        if len(self.memories) > 100:
            self.memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            self.memories = self.memories[:100]

    def get_context_summary(self, max_tokens: int = 500) -> str:
        """Get a summary of journal for AI context."""
        lines = []

        if self.personality_traits:
            lines.append(f"Personality: {', '.join(self.personality_traits)}")
        
        if self.ideals:
            lines.append(f"Ideals: {', '.join(self.ideals)}")
        
        if self.bonds:
            lines.append(f"Bonds: {', '.join(self.bonds)}")
        
        if self.flaws:
            lines.append(f"Flaws: {', '.join(self.flaws)}")
        
        if self.backstory:
            # Truncate backstory if too long
            backstory_preview = self.backstory[:200] + "..." if len(self.backstory) > 200 else self.backstory
            lines.append(f"Backstory: {backstory_preview}")

        if self.goals:
            lines.append(f"Goals: {', '.join(self.goals[:3])}")

        if self.memories:
            recent = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:5]
            lines.append("Recent events: " + "; ".join(m.content for m in recent))

        return "\n".join(lines)
    
    def get_appearance_summary(self) -> str:
        """Get a formatted appearance summary."""
        parts = []
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.height:
            parts.append(f"Height: {self.height}")
        if self.weight:
            parts.append(f"Weight: {self.weight}")
        if self.eyes:
            parts.append(f"Eyes: {self.eyes}")
        if self.hair:
            parts.append(f"Hair: {self.hair}")
        if self.skin:
            parts.append(f"Skin: {self.skin}")
        if self.appearance:
            parts.append(self.appearance)
        return "; ".join(parts) if parts else ""


class MemoryEntry(BaseModel):
    """A single memory in the journal."""

    content: str = Field(description="What happened")
    importance: int = Field(default=5, ge=1, le=10, description="1-10 importance")
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """Relationship with another entity."""

    target_name: str
    attitude: int = Field(default=0, ge=-100, le=100, description="-100 hostile to 100 friendly")
    notes: list[str] = Field(default_factory=list)


class MovementComponent(Component):
    """Movement speeds and special movement."""

    speed_walk: int = Field(default=30, ge=0, description="Walking speed in feet")
    speed_fly: int = Field(default=0, ge=0)
    speed_swim: int = Field(default=0, ge=0)
    speed_climb: int = Field(default=0, ge=0)
    speed_burrow: int = Field(default=0, ge=0)

    # Movement remaining this turn
    movement_remaining: int = Field(default=30, ge=0)

    # Special movement traits
    can_hover: bool = Field(default=False)
    ignores_difficult_terrain: bool = Field(default=False)


class RechargeType(StrEnum):
    """When a feature's uses are restored."""
    AT_WILL = "at_will"  # No limit
    SHORT_REST = "short_rest"
    LONG_REST = "long_rest"
    DAWN = "dawn"  # Recharges at dawn
    ENCOUNTER = "encounter"  # Recharges after combat


class Feature(BaseModel):
    """A class or racial feature with optional limited uses.
    
    Examples:
    - Rage: 2 uses, recharges on long rest
    - Second Wind: 1 use, recharges on short rest
    - Sneak Attack: at will (no tracking needed)
    """
    model_config = ConfigDict(extra="ignore")
    
    name: str = Field(description="Feature name")
    description: str = Field(default="", description="What the feature does")
    
    # Usage tracking (None means at-will/unlimited)
    uses_current: int | None = Field(default=None, description="Current uses remaining")
    uses_max: int | None = Field(default=None, description="Max uses")
    recharge: RechargeType = Field(default=RechargeType.AT_WILL, description="When uses restore")
    
    # Source info
    source: str = Field(default="class", description="Where feature comes from: class, race, feat, item")
    level_gained: int = Field(default=1, description="Level when feature was gained")
    
    def use(self) -> bool:
        """Use this feature. Returns True if successful."""
        if self.uses_max is None:
            return True  # At-will, always succeeds
        if self.uses_current is None or self.uses_current <= 0:
            return False
        self.uses_current -= 1
        return True
    
    def restore(self, amount: int | None = None) -> None:
        """Restore uses. None = restore to max."""
        if self.uses_max is None:
            return  # At-will, nothing to restore
        if amount is None:
            self.uses_current = self.uses_max
        else:
            self.uses_current = min((self.uses_current or 0) + amount, self.uses_max)
    
    @property
    def has_uses(self) -> bool:
        """Check if feature has uses remaining."""
        if self.uses_max is None:
            return True
        return (self.uses_current or 0) > 0
    
    @property
    def uses_display(self) -> str:
        """Get display string for uses."""
        if self.uses_max is None:
            return "At Will"
        return f"{self.uses_current or 0}/{self.uses_max}"


class ClassFeatureComponent(Component):
    """Class-specific features and resources.
    
    Tracks class levels, subclasses, and class resources
    like Ki, Rage, Channel Divinity, etc.
    """

    classes: list["ClassLevel"] = Field(default_factory=list)

    # Class resources: resource_name -> (current, max) - legacy, kept for compatibility
    resources: dict[str, tuple[int, int]] = Field(default_factory=dict)

    # Features with full tracking (new system)
    tracked_features: dict[str, Feature] = Field(default_factory=dict, description="Features with usage tracking")
    
    # Simple feature names (legacy, for features without usage limits)
    features: list[str] = Field(default_factory=list)

    @computed_field(description="Total character level")
    @property
    def total_level(self) -> int:
        return sum(c.level for c in self.classes)

    @computed_field(description="Primary class")
    @property
    def primary_class(self) -> str:
        if not self.classes:
            return "Commoner"
        return max(self.classes, key=lambda c: c.level).class_name

    def use_resource(self, name: str, amount: int = 1) -> bool:
        """Use a class resource. Returns success."""
        if name not in self.resources:
            return False
        current, max_val = self.resources[name]
        if current < amount:
            return False
        self.resources[name] = (current - amount, max_val)
        return True

    def restore_resource(self, name: str, amount: int | None = None) -> None:
        """Restore a class resource. If amount is None, restore to max."""
        if name not in self.resources:
            return
        current, max_val = self.resources[name]
        if amount is None:
            self.resources[name] = (max_val, max_val)
        else:
            self.resources[name] = (min(current + amount, max_val), max_val)

    def add_feature(self, feature: Feature) -> None:
        """Add a tracked feature."""
        self.tracked_features[feature.name.lower()] = feature
    
    def get_feature(self, name: str) -> Feature | None:
        """Get a feature by name (case-insensitive)."""
        return self.tracked_features.get(name.lower())
    
    def use_feature(self, name: str) -> tuple[bool, str]:
        """Use a feature. Returns (success, message)."""
        feature = self.get_feature(name)
        if feature is None:
            # Check simple features list
            if name.lower() in [f.lower() for f in self.features]:
                return True, f"Used {name} (at-will feature)"
            return False, f"Feature '{name}' not found"
        
        if feature.use():
            uses_left = f" ({feature.uses_display} remaining)" if feature.uses_max else ""
            return True, f"Used {feature.name}{uses_left}"
        else:
            recharge_val = feature.recharge.value if hasattr(feature.recharge, 'value') else str(feature.recharge)
            return False, f"No uses of {feature.name} remaining! Recharges on {recharge_val.replace('_', ' ')}"
    
    def short_rest(self) -> list[str]:
        """Restore features that recharge on short rest. Returns list of restored features."""
        restored = []
        for feature in self.tracked_features.values():
            recharge = feature.recharge.value if hasattr(feature.recharge, 'value') else str(feature.recharge)
            if recharge in ("short_rest", "encounter"):
                if feature.uses_max and (feature.uses_current or 0) < feature.uses_max:
                    feature.restore()
                    restored.append(feature.name)
        return restored
    
    def long_rest(self) -> list[str]:
        """Restore all features (long rest restores everything). Returns list of restored features."""
        restored = []
        for feature in self.tracked_features.values():
            if feature.uses_max and (feature.uses_current or 0) < feature.uses_max:
                feature.restore()
                restored.append(feature.name)
        # Also restore legacy resources
        for name in self.resources:
            self.restore_resource(name)
        return restored
    
    def get_all_features(self) -> list[Feature | str]:
        """Get all features (tracked and simple)."""
        result: list[Feature | str] = list(self.tracked_features.values())
        result.extend(self.features)
        return result


class ClassLevel(BaseModel):
    """A class and level pair."""

    class_name: str
    level: Level = Field(default=1)
    subclass: str | None = Field(default=None)


# =============================================================================
# Actor Entity (Polymorphic)
# =============================================================================


class ActorEntity(BaseModel):
    """Base entity for all actors (players, NPCs, monsters).
    
    Uses discriminated union pattern for polymorphism.
    The `type` field determines how the entity is controlled.
    
    NEURO-SYMBOLIC PRINCIPLE:
    This is the source of truth. LLMs read this but never
    directly mutate it. All changes go through StateUpdateRequest.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="ignore",  # Ignore computed fields when deserializing
        use_enum_values=True,
    )

    # Identity
    uid: UUID = Field(default_factory=uuid4, description="Unique identifier")
    name: str = Field(default="Unknown Character", description="Display name")
    type: ActorType = Field(default=ActorType.PLAYER, description="Actor type for control logic")

    @field_validator("name", "race", "alignment", mode="before")
    @classmethod
    def default_none_strings(cls, v: Any) -> str:
        """Replace None values with sensible defaults for string fields."""
        if v is None:
            return "Unknown"
        return str(v)

    # Core components (always present)
    stats: StatsComponent = Field(default_factory=StatsComponent)
    health: HealthComponent = Field(default_factory=HealthComponent)
    defense: DefenseComponent = Field(default_factory=DefenseComponent)

    # Optional components
    inventory: InventoryComponent | None = Field(default=None)
    spellbook: SpellbookComponent | None = Field(default=None)
    journal: JournalComponent | None = Field(default=None)
    movement: MovementComponent | None = Field(default=None)
    class_features: ClassFeatureComponent | None = Field(default=None)

    # Metadata
    race: str = Field(default="Unknown")
    size: str = Field(default="Medium")
    alignment: str = Field(default="Neutral")
    background: str = Field(default="", description="D&D background (e.g., Soldier, Noble, Criminal)")
    challenge_rating: ChallengeRating | None = Field(default=None)

    # Combat state
    initiative: int = Field(default=0)
    is_in_combat: bool = Field(default=False)
    
    # Morale system (for NPCs/monsters)
    morale: int = Field(default=100, ge=0, le=100, description="Current morale (0=broken, 100=fearless)")
    morale_base: int = Field(default=50, ge=0, le=100, description="Base morale for this creature type")
    flee_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="HP percentage to consider fleeing")
    is_fleeing: bool = Field(default=False, description="Currently trying to flee")
    has_surrendered: bool = Field(default=False, description="Has surrendered to players")
    
    # Active effects (buffs, debuffs, conditions from spells/abilities)
    active_effects: list[ActiveEffect] = Field(default_factory=list, description="Active buffs and debuffs")
    
    # Experience (for player characters)
    experience_points: int = Field(default=0, ge=0, description="Current XP")
    pending_level_up: bool = Field(default=False, description="Whether character needs to level up")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @computed_field(description="Armor Class")
    @property
    def ac(self) -> int:
        return self.defense.calculate_ac(self.stats.dex_mod)

    @computed_field(description="Is player-controlled")
    @property
    def is_player_controlled(self) -> bool:
        return self.type == ActorType.PLAYER

    @computed_field(description="Is AI-controlled")
    @property
    def is_ai_controlled(self) -> bool:
        return self.type in (ActorType.NPC_ALLY, ActorType.MONSTER)

    @computed_field(description="Level or CR display")
    @property
    def level_display(self) -> str:
        if self.class_features:
            return f"Level {self.class_features.total_level}"
        if self.challenge_rating:
            return f"CR {self.challenge_rating}"
        return "Level 0"

    def to_summary(self) -> str:
        """Get a brief summary for AI context."""
        status = "conscious" if self.health.is_conscious else "unconscious"
        if self.health.is_dead:
            status = "dead"

        hp_str = f"{self.health.hp_current}/{self.health.hp_max}"
        
        # Include equipped weapons/items
        equipment = []
        if self.inventory:
            for item in self.inventory.items:
                if item.equipped:
                    equipment.append(item.name)
        
        equip_str = f", Equipped: {', '.join(equipment)}" if equipment else ""

        return f"{self.name} ({self.race} {self.level_display}) - HP: {hp_str}, AC: {self.ac}, Status: {status}{equip_str}"

    def add_xp(self, amount: int) -> bool:
        """Add XP and check for level up.
        
        Args:
            amount: XP to add.
            
        Returns:
            True if character leveled up (needs to complete level-up process).
        """
        from dnd_manager.models.progression import get_level_for_xp
        
        old_level = self.current_level
        self.experience_points += amount
        new_level = get_level_for_xp(self.experience_points)
        
        if new_level > old_level:
            self.pending_level_up = True
            return True
        return False

    @computed_field(description="Current level based on XP")
    @property
    def current_level(self) -> int:
        """Get current level based on XP."""
        from dnd_manager.models.progression import get_level_for_xp
        
        # Use class_features if available
        if self.class_features:
            return self.class_features.total_level
        
        # Fall back to XP-based calculation
        return get_level_for_xp(self.experience_points)

    @computed_field(description="XP needed for next level")
    @property
    def xp_to_next_level(self) -> int | None:
        """Get XP needed for next level."""
        from dnd_manager.models.progression import get_xp_for_next_level
        return get_xp_for_next_level(self.current_level)

    def level_up(
        self,
        hp_roll: int | None = None,
        asi_choices: dict[str, int] | None = None,
        new_spells: list[str] | None = None,
        subclass: str | None = None,
    ) -> dict[str, Any]:
        """Process a level up.
        
        Args:
            hp_roll: Rolled HP (or None to use average).
            asi_choices: Ability score increases {"strength": 1, "dexterity": 1}.
            new_spells: New spells learned this level.
            subclass: Subclass selection (if applicable).
            
        Returns:
            Dict with level-up results.
        """
        from dnd_manager.models.progression import (
            get_level_up_info,
            get_proficiency_bonus,
            calculate_hp_increase,
        )
        
        if not self.class_features or not self.class_features.classes:
            return {"error": "No class features to level up"}
        
        # Get primary class
        primary_class = self.class_features.classes[0]
        new_level = primary_class.level + 1
        
        if new_level > 20:
            return {"error": "Already at max level"}
        
        # Get level up info
        level_info = get_level_up_info(
            primary_class.class_name,
            new_level,
            subclass or primary_class.subclass,
        )
        
        results: dict[str, Any] = {
            "old_level": primary_class.level,
            "new_level": new_level,
            "class": primary_class.class_name,
            "features_gained": level_info["features"],
        }
        
        # Update class level
        primary_class.level = new_level
        
        # Update subclass if selecting one
        if subclass and level_info["is_subclass_level"]:
            primary_class.subclass = subclass
            results["subclass"] = subclass
        
        # Update proficiency bonus
        new_prof = get_proficiency_bonus(new_level)
        if new_prof != self.stats.proficiency_bonus:
            self.stats.proficiency_bonus = new_prof
            results["proficiency_bonus"] = new_prof
        
        # Calculate and add HP
        hp_gained = calculate_hp_increase(
            primary_class.class_name,
            self.stats.con_mod,
            hp_roll,
        )
        self.health.hp_max += hp_gained
        self.health.hp_current += hp_gained  # Also heal the gained HP
        results["hp_gained"] = hp_gained
        results["new_hp_max"] = self.health.hp_max
        
        # Apply ASI if this is an ASI level
        if level_info["is_asi_level"] and asi_choices:
            for ability, increase in asi_choices.items():
                current = getattr(self.stats, ability, 10)
                new_val = min(20, current + increase)  # Cap at 20
                setattr(self.stats, ability, new_val)
            results["asi_applied"] = asi_choices
        
        # Update spell slots for casters
        if self.spellbook and level_info["spell_slots"]:
            for spell_level, num_slots in level_info["spell_slots"].items():
                self.spellbook.spell_slots[spell_level] = (num_slots, num_slots)
            results["spell_slots"] = level_info["spell_slots"]
        
        # Add new spells if provided
        if new_spells and self.spellbook:
            self.spellbook.spells_known.extend(new_spells)
            results["new_spells"] = new_spells
        
        # Add features to class features
        if level_info["features"]:
            self.class_features.features.extend(level_info["features"])
        
        # Clear pending level up flag
        self.pending_level_up = False
        self.updated_at = datetime.now()
        
        return results
    
    # =========================================================================
    # Active Effect Management
    # =========================================================================
    
    def add_effect(self, effect: ActiveEffect, current_round: int = 0, current_turn: int = 0) -> None:
        """Add an active effect to this actor.
        
        Args:
            effect: The effect to add.
            current_round: Current combat round for expiration calculation.
            current_turn: Current turn index for expiration calculation.
        """
        effect.applied_round = current_round
        effect.applied_turn = current_turn
        effect.calculate_expiration(current_round, current_turn)
        self.active_effects.append(effect)
        self.updated_at = datetime.now()
    
    def remove_effect(self, effect_uid: UUID) -> ActiveEffect | None:
        """Remove an effect by UID.
        
        Args:
            effect_uid: The effect's unique ID.
            
        Returns:
            The removed effect, or None if not found.
        """
        for i, effect in enumerate(self.active_effects):
            if effect.uid == effect_uid:
                removed = self.active_effects.pop(i)
                self.updated_at = datetime.now()
                return removed
        return None
    
    def remove_effect_by_name(self, effect_name: str) -> list[ActiveEffect]:
        """Remove all effects with a given name.
        
        Args:
            effect_name: Name to match (case-insensitive).
            
        Returns:
            List of removed effects.
        """
        removed = []
        name_lower = effect_name.lower()
        self.active_effects = [
            e for e in self.active_effects
            if e.name.lower() != name_lower or (removed.append(e) and False)
        ]
        if removed:
            self.updated_at = datetime.now()
        return removed
    
    def get_effect(self, effect_name: str) -> ActiveEffect | None:
        """Get an active effect by name.
        
        Args:
            effect_name: Name to find (case-insensitive).
            
        Returns:
            The effect if found, None otherwise.
        """
        name_lower = effect_name.lower()
        for effect in self.active_effects:
            if effect.name.lower() == name_lower:
                return effect
        return None
    
    def has_effect(self, effect_name: str) -> bool:
        """Check if actor has an active effect by name."""
        return self.get_effect(effect_name) is not None
    
    def expire_effects_on_turn_start(self, current_round: int, current_turn: int) -> list[ActiveEffect]:
        """Expire effects that end at the start of this turn.
        
        Args:
            current_round: Current combat round.
            current_turn: Current turn index.
            
        Returns:
            List of expired effects.
        """
        expired = []
        remaining = []
        
        for effect in self.active_effects:
            if effect.should_expire_on_turn_start(current_round, current_turn):
                expired.append(effect)
                # Revert AC changes if applicable
                eff_type = effect.effect_type.value if hasattr(effect.effect_type, 'value') else str(effect.effect_type)
                if eff_type == "ac_bonus" and effect.value:
                    self.defense.ac_bonus -= effect.value
            else:
                remaining.append(effect)
        
        self.active_effects = remaining
        if expired:
            self.updated_at = datetime.now()
        return expired
    
    def expire_effects_on_turn_end(self, current_round: int, current_turn: int) -> list[ActiveEffect]:
        """Expire effects that end at the end of this turn.
        
        Args:
            current_round: Current combat round.
            current_turn: Current turn index.
            
        Returns:
            List of expired effects.
        """
        expired = []
        remaining = []
        
        for effect in self.active_effects:
            if effect.should_expire_on_turn_end(current_round, current_turn):
                expired.append(effect)
                # Revert AC changes
                eff_type = effect.effect_type.value if hasattr(effect.effect_type, 'value') else str(effect.effect_type)
                if eff_type == "ac_bonus" and effect.value:
                    self.defense.ac_bonus -= effect.value
            else:
                remaining.append(effect)
        
        self.active_effects = remaining
        if expired:
            self.updated_at = datetime.now()
        return expired
    
    def expire_effects_on_rest(self, rest_type: str) -> list[ActiveEffect]:
        """Expire effects that end on rest.
        
        Args:
            rest_type: "short" or "long".
            
        Returns:
            List of expired effects.
        """
        expired = []
        remaining = []
        
        for effect in self.active_effects:
            if effect.should_expire_on_rest(rest_type):
                expired.append(effect)
                # Revert modifications
                eff_type = effect.effect_type.value if hasattr(effect.effect_type, 'value') else str(effect.effect_type)
                if eff_type == "ac_bonus" and effect.value:
                    self.defense.ac_bonus -= effect.value
                elif eff_type == "ac_set" and effect.value:
                    # Reset to default AC calculation (10 + DEX)
                    self.defense.ac_base = 10
            else:
                remaining.append(effect)
        
        self.active_effects = remaining
        if expired:
            self.updated_at = datetime.now()
        return expired
    
    def clear_concentration_effects(self, caster_name: str) -> list[ActiveEffect]:
        """Clear all concentration effects from a specific caster.
        
        Args:
            caster_name: The caster who lost concentration.
            
        Returns:
            List of expired effects.
        """
        expired = []
        remaining = []
        
        for effect in self.active_effects:
            if effect.concentration and effect.source_actor.lower() == caster_name.lower():
                expired.append(effect)
                # Revert modifications
                eff_type = effect.effect_type.value if hasattr(effect.effect_type, 'value') else str(effect.effect_type)
                if eff_type == "ac_bonus" and effect.value:
                    self.defense.ac_bonus -= effect.value
                elif eff_type == "ac_set" and effect.value:
                    self.defense.ac_base = 10
            else:
                remaining.append(effect)
        
        self.active_effects = remaining
        if expired:
            self.updated_at = datetime.now()
        return expired
    
    def get_ac_with_effects(self) -> int:
        """Calculate AC including all active effects."""
        base_ac = self.defense.calculate_ac(self.stats.dex_mod)
        
        # AC_SET effects override base (use highest)
        set_acs = []
        for e in self.active_effects:
            eff_type = e.effect_type.value if hasattr(e.effect_type, 'value') else str(e.effect_type)
            if eff_type == "ac_set" and e.value and e.is_active:
                set_acs.append(e.value)
        
        if set_acs:
            base_ac = max(base_ac, max(set_acs))
        
        # AC_BONUS effects stack (already applied to defense.ac_bonus)
        # No additional calculation needed as they're applied when effect is added
        
        return base_ac
    
    # =========================================================================
    # Morale System
    # =========================================================================
    
    def check_morale(self) -> tuple[bool, str]:
        """Check if this actor should consider fleeing based on current state.
        
        Returns:
            Tuple of (should_flee, reason)
        """
        # Players don't flee automatically
        actor_type = self.type.value if hasattr(self.type, 'value') else str(self.type)
        if actor_type == "player":
            return False, "Players control their own actions"
        
        # Already fled or surrendered
        if self.is_fleeing:
            return True, "Already fleeing"
        if self.has_surrendered:
            return False, "Already surrendered"
        
        # Check HP threshold
        if self.health.hp_max > 0:
            hp_pct = self.health.hp_current / self.health.hp_max
            if hp_pct <= self.flee_threshold:
                return True, f"HP critically low ({int(hp_pct * 100)}%)"
        
        # Check morale threshold (below 25 = broken)
        if self.morale <= 25:
            return True, f"Morale broken ({self.morale}%)"
        
        return False, "Morale holding"
    
    def reduce_morale(self, amount: int, reason: str = "") -> tuple[int, bool]:
        """Reduce morale by an amount.
        
        Args:
            amount: How much to reduce morale.
            reason: Why morale is being reduced.
            
        Returns:
            Tuple of (new_morale, is_broken)
        """
        self.morale = max(0, self.morale - amount)
        is_broken = self.morale <= 25
        return self.morale, is_broken
    
    def restore_morale(self, amount: int | None = None) -> int:
        """Restore morale.
        
        Args:
            amount: How much to restore, or None for full restore.
            
        Returns:
            New morale value.
        """
        if amount is None:
            self.morale = self.morale_base
        else:
            self.morale = min(100, self.morale + amount)
        return self.morale
    
    @computed_field(description="Morale state description")
    @property
    def morale_state(self) -> str:
        """Get a description of current morale state."""
        if self.has_surrendered:
            return "surrendered"
        if self.is_fleeing:
            return "fleeing"
        if self.morale >= 75:
            return "confident"
        if self.morale >= 50:
            return "steady"
        if self.morale >= 25:
            return "shaken"
        return "broken"


# =============================================================================
# State Update Request (LLM -> Engine)
# =============================================================================


class StateUpdateRequest(BaseModel):
    """Request from LLM to update game state.
    
    NEURO-SYMBOLIC PRINCIPLE:
    LLMs cannot directly mutate state. They emit these requests
    which the Python engine validates and applies.
    """

    model_config = ConfigDict(frozen=True)

    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Target entity
    target_uid: UUID = Field(description="Entity to modify")

    # Update type
    update_type: Literal[
        "damage",
        "healing",
        "condition_add",
        "condition_remove",
        "resource_use",
        "resource_restore",
        "inventory_add",
        "inventory_remove",
        "position_change",
        "spell_slot_use",
        "hp_set",
        "death_save",
        "custom",
    ]

    # Update payload (structure depends on update_type)
    payload: dict[str, Any] = Field(default_factory=dict)

    # Validation status (set by engine)
    is_validated: bool = Field(default=False)
    validation_error: str | None = Field(default=None)


class StateUpdateResult(BaseModel):
    """Result of applying a state update."""

    request_id: UUID
    success: bool
    message: str
    old_value: Any = None
    new_value: Any = None


# =============================================================================
# Game State Container
# =============================================================================


class GameState(BaseModel):
    """The complete game state.
    
    This is the single source of truth for the entire session.
    It owns all entities and handles state transitions.
    
    NEURO-SYMBOLIC PRINCIPLE:
    Only the Python engine mutates this object.
    LLMs receive read-only snapshots and emit StateUpdateRequests.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Session identity
    session_id: UUID = Field(default_factory=uuid4)
    campaign_name: str = Field(default="Unnamed Campaign")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # All actors in the world
    actors: dict[UUID, ActorEntity] = Field(default_factory=dict)

    # Party members (subset of actors)
    party_uids: list[UUID] = Field(default_factory=list)

    # Current encounter/combat
    combat_active: bool = Field(default=False)
    combat_order: list[UUID] = Field(default_factory=list, description="Initiative order")
    combat_turn_index: int = Field(default=0)
    combat_round: int = Field(default=0)

    # Scene/location
    current_scene: str = Field(default="Unknown Location")
    scene_description: str = Field(default="")

    # Experience and progression
    party_xp: int = Field(default=0, ge=0)

    # State hash for change detection
    _last_hash: str = ""

    def add_actor(self, actor: ActorEntity) -> None:
        """Add an actor to the game state."""
        self.actors[actor.uid] = actor
        self.updated_at = datetime.now()

    def remove_actor(self, uid: UUID) -> ActorEntity | None:
        """Remove an actor and return it."""
        actor = self.actors.pop(uid, None)
        if uid in self.party_uids:
            self.party_uids.remove(uid)
        if uid in self.combat_order:
            self.combat_order.remove(uid)
        self.updated_at = datetime.now()
        return actor

    def get_actor(self, uid: UUID) -> ActorEntity | None:
        """Get an actor by UID."""
        return self.actors.get(uid)

    def get_party(self) -> list[ActorEntity]:
        """Get all party members."""
        return [self.actors[uid] for uid in self.party_uids if uid in self.actors]

    def get_current_combatant(self) -> ActorEntity | None:
        """Get the actor whose turn it is."""
        if not self.combat_active or not self.combat_order:
            return None
        uid = self.combat_order[self.combat_turn_index % len(self.combat_order)]
        return self.actors.get(uid)

    def advance_turn(self) -> None:
        """Advance to the next combatant's turn."""
        if not self.combat_active:
            return

        self.combat_turn_index += 1

        # Check for new round
        if self.combat_turn_index >= len(self.combat_order):
            self.combat_turn_index = 0
            self.combat_round += 1

        self.updated_at = datetime.now()

    def apply_update(self, request: StateUpdateRequest) -> StateUpdateResult:
        """Apply a validated state update request.
        
        This is the ONLY way LLM-driven changes enter the state.
        """
        actor = self.actors.get(request.target_uid)
        if actor is None:
            return StateUpdateResult(
                request_id=request.request_id,
                success=False,
                message=f"Actor {request.target_uid} not found",
            )

        try:
            if request.update_type == "damage":
                old_hp = actor.health.hp_current
                amount = request.payload.get("amount", 0)
                damage_type = request.payload.get("damage_type")
                
                # Apply resistance/immunity
                if damage_type:
                    amount = actor.defense.modify_damage(amount, DamageType(damage_type))
                
                actual = actor.health.apply_damage(amount)
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=True,
                    message=f"{actor.name} takes {actual} damage",
                    old_value=old_hp,
                    new_value=actor.health.hp_current,
                )

            elif request.update_type == "healing":
                old_hp = actor.health.hp_current
                amount = request.payload.get("amount", 0)
                actual = actor.health.apply_healing(amount)
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=True,
                    message=f"{actor.name} heals {actual} HP",
                    old_value=old_hp,
                    new_value=actor.health.hp_current,
                )

            elif request.update_type == "condition_add":
                condition = Condition(request.payload.get("condition"))
                if condition not in actor.health.conditions:
                    actor.health.conditions.append(condition)
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=True,
                    message=f"{actor.name} gains {condition.value}",
                )

            elif request.update_type == "condition_remove":
                condition = Condition(request.payload.get("condition"))
                if condition in actor.health.conditions:
                    actor.health.conditions.remove(condition)
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=True,
                    message=f"{actor.name} loses {condition.value}",
                )

            elif request.update_type == "spell_slot_use":
                if actor.spellbook is None:
                    return StateUpdateResult(
                        request_id=request.request_id,
                        success=False,
                        message=f"{actor.name} cannot cast spells",
                    )
                level = request.payload.get("level", 1)
                success = actor.spellbook.use_slot(level)
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=success,
                    message=f"Used level {level} spell slot" if success else "No slots available",
                )

            elif request.update_type == "death_save":
                result = request.payload.get("success", False)
                if result:
                    actor.health.death_saves_success += 1
                else:
                    actor.health.death_saves_failure += 1
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=True,
                    message=f"Death save {'success' if result else 'failure'}",
                )

            else:
                return StateUpdateResult(
                    request_id=request.request_id,
                    success=False,
                    message=f"Unknown update type: {request.update_type}",
                )

        except Exception as e:
            return StateUpdateResult(
                request_id=request.request_id,
                success=False,
                message=f"Update failed: {e}",
            )
        finally:
            self.updated_at = datetime.now()

    def get_state_hash(self) -> str:
        """Get a hash of the current state for change detection."""
        data = self.model_dump_json()
        return hashlib.md5(data.encode()).hexdigest()

    def to_ai_context(self, max_tokens: int = 2000) -> str:
        """Generate a context summary for AI prompting.
        
        This is the read-only view that LLMs receive.
        """
        lines = [
            f"# Game State - {self.campaign_name}",
            f"Location: {self.current_scene}",
            "",
        ]

        if self.scene_description:
            lines.append(f"Scene: {self.scene_description[:200]}")
            lines.append("")

        # Party status
        lines.append("## Party")
        for actor in self.get_party():
            lines.append(f"- {actor.to_summary()}")
        lines.append("")

        # Always show non-party entities (enemies, NPCs) so DM knows they exist
        non_party = [a for uid, a in self.actors.items() if uid not in self.party_uids]
        if non_party:
            lines.append("## Entities in Scene")
            for actor in non_party:
                status = ""
                if actor.health.is_dead:
                    status = " [DEAD]"
                elif not actor.health.is_conscious:
                    status = " [UNCONSCIOUS]"
                lines.append(f"- {actor.name}: HP {actor.health.hp_current}/{actor.health.hp_max}, AC {actor.ac}{status}")
            lines.append("")

        # Combat status
        if self.combat_active:
            lines.append(f"## Combat (Round {self.combat_round})")
            current = self.get_current_combatant()
            if current:
                lines.append(f"**Current Turn: {current.name}**")

            lines.append("Initiative Order:")
            for i, uid in enumerate(self.combat_order):
                actor = self.actors.get(uid)
                if actor:
                    is_current = (i == self.combat_turn_index)
                    marker = "→ " if is_current else "  "
                    lines.append(f"{marker}{actor.name} (Init: {actor.initiative}, HP: {actor.health.hp_current}/{actor.health.hp_max})")

        return "\n".join(lines)


# =============================================================================
# Factory Functions
# =============================================================================


def create_player_character(
    name: str,
    race: str = "Human",
    class_name: str = "Fighter",
    level: int = 1,
    stats: dict[str, int] | None = None,
) -> ActorEntity:
    """Create a new player character entity."""
    default_stats = {
        "strength": 10,
        "dexterity": 10,
        "constitution": 10,
        "intelligence": 10,
        "wisdom": 10,
        "charisma": 10,
    }
    if stats:
        default_stats.update(stats)

    # Calculate proficiency bonus from level
    prof_bonus = 2 + ((level - 1) // 4)

    stats_component = StatsComponent(
        **default_stats,
        proficiency_bonus=prof_bonus,
    )

    # Calculate HP (using CON mod + class hit die average)
    con_mod = StatsComponent.calc_modifier(default_stats["constitution"])
    hp = 10 + con_mod + ((level - 1) * (6 + con_mod))  # Simplified

    return ActorEntity(
        name=name,
        type=ActorType.PLAYER,
        race=race,
        stats=stats_component,
        health=HealthComponent(hp_current=hp, hp_max=hp),
        defense=DefenseComponent(),
        inventory=InventoryComponent(),
        spellbook=SpellbookComponent() if class_name in ("Wizard", "Sorcerer", "Cleric", "Bard", "Druid", "Warlock") else None,
        journal=JournalComponent(),
        movement=MovementComponent(),
        class_features=ClassFeatureComponent(
            classes=[ClassLevel(class_name=class_name, level=level)],
        ),
    )


def create_monster(
    name: str,
    cr: str = "1",
    hp: int = 22,
    ac: int = 13,
    stats: dict[str, int] | None = None,
) -> ActorEntity:
    """Create a monster entity."""
    default_stats = {
        "strength": 12,
        "dexterity": 12,
        "constitution": 12,
        "intelligence": 8,
        "wisdom": 10,
        "charisma": 8,
    }
    if stats:
        default_stats.update(stats)

    return ActorEntity(
        name=name,
        type=ActorType.MONSTER,
        challenge_rating=cr,
        stats=StatsComponent(**default_stats),
        health=HealthComponent(hp_current=hp, hp_max=hp),
        defense=DefenseComponent(ac_base=ac, uses_dex=False),
        journal=JournalComponent(
            goals=["Survive", "Defeat enemies"],
            personality_traits=["Hostile"],
        ),
        movement=MovementComponent(),
    )


__all__ = [
    # Types
    "AbilityScore",
    "Level",
    "ChallengeRating",
    "ActorType",
    "Ability",
    "DamageType",
    "Condition",
    "SpellSchool",
    "ItemType",
    "EffectType",
    "EffectDuration",
    # Components
    "Component",
    "StatsComponent",
    "HealthComponent",
    "DefenseComponent",
    "InventoryComponent",
    "ItemStack",
    "SpellbookComponent",
    "JournalComponent",
    "MemoryEntry",
    "Relationship",
    "MovementComponent",
    "ClassFeatureComponent",
    "ClassLevel",
    "Feature",
    "RechargeType",
    # Active Effects
    "ActiveEffect",
    # Entities
    "ActorEntity",
    # State Management
    "StateUpdateRequest",
    "StateUpdateResult",
    "GameState",
    # Factories
    "create_player_character",
    "create_monster",
]
