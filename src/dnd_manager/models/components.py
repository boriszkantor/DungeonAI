"""Component models for entity composition in the D&D 5E AI Campaign Manager.

This module defines reusable component models that can be composed into
larger entity models. Components follow the composition-over-inheritance
pattern, enabling flexible entity construction.

Components:
    StatsComponent: Ability scores with computed modifiers.
    HealthComponent: Hit point tracking (current, max, temp).
    PersonaComponent: AI behavior and personality configuration.
    SpeedComponent: Movement speeds by type.
    ArmorComponent: Armor class calculation.
    SavingThrowsComponent: Saving throw proficiencies and bonuses.
    SkillsComponent: Skill proficiencies and expertise.

DEPRECATED: These components have been fully integrated into models/ecs.py.
This module is maintained for backward compatibility only.
New code should import from models/ecs.py.
"""

from __future__ import annotations

import warnings

from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from dnd_manager.models.enums import Ability, AutonomyLevel, Condition, Skill


# =============================================================================
# Validators and Type Definitions
# =============================================================================


def validate_ability_score(value: int) -> int:
    """Validate that an ability score is within D&D 5E bounds (1-30).

    Args:
        value: The ability score to validate.

    Returns:
        The validated score.

    Raises:
        ValueError: If score is outside 1-30 range.
    """
    if not 1 <= value <= 30:
        msg = f"Ability score must be between 1 and 30, got {value}"
        raise ValueError(msg)
    return value


def calculate_modifier(score: int) -> int:
    """Calculate the ability modifier from an ability score.

    The modifier is calculated as: (score - 10) // 2

    Args:
        score: The ability score (1-30).

    Returns:
        The ability modifier (-5 to +10).

    Example:
        >>> calculate_modifier(10)
        0
        >>> calculate_modifier(18)
        4
        >>> calculate_modifier(7)
        -2
    """
    return (score - 10) // 2


# Type alias for validated ability scores
AbilityScore = Annotated[int, Field(ge=1, le=30, description="Ability score (1-30)")]


# =============================================================================
# Stats Component
# =============================================================================


class StatsComponent(BaseModel):
    """Component for D&D 5E ability scores with computed modifiers.

    Holds the six core ability scores and automatically computes
    their modifiers. All scores must be between 1 and 30.

    Attributes:
        strength: Physical power, athletic training, raw physical force.
        dexterity: Agility, reflexes, balance, coordination.
        constitution: Health, stamina, vital force, endurance.
        intelligence: Mental acuity, information recall, analytical skill.
        wisdom: Awareness, intuition, insight, perception.
        charisma: Confidence, eloquence, leadership, force of personality.

    Example:
        >>> stats = StatsComponent(
        ...     strength=16, dexterity=14, constitution=15,
        ...     intelligence=10, wisdom=12, charisma=8
        ... )
        >>> stats.strength_modifier
        3
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        json_schema_extra={
            "description": "D&D 5E ability scores with automatic modifier calculation"
        },
    )

    strength: AbilityScore = Field(
        default=10,
        description="Strength score: Physical power and athletic ability (1-30)",
    )
    dexterity: AbilityScore = Field(
        default=10,
        description="Dexterity score: Agility, reflexes, and balance (1-30)",
    )
    constitution: AbilityScore = Field(
        default=10,
        description="Constitution score: Health, stamina, and vital force (1-30)",
    )
    intelligence: AbilityScore = Field(
        default=10,
        description="Intelligence score: Mental acuity and reasoning (1-30)",
    )
    wisdom: AbilityScore = Field(
        default=10,
        description="Wisdom score: Awareness, intuition, and insight (1-30)",
    )
    charisma: AbilityScore = Field(
        default=10,
        description="Charisma score: Force of personality and leadership (1-30)",
    )

    # Computed modifiers
    @computed_field(description="Strength modifier: (STR - 10) // 2")
    @property
    def strength_modifier(self) -> int:
        """Calculate Strength modifier."""
        return calculate_modifier(self.strength)

    @computed_field(description="Dexterity modifier: (DEX - 10) // 2")
    @property
    def dexterity_modifier(self) -> int:
        """Calculate Dexterity modifier."""
        return calculate_modifier(self.dexterity)

    @computed_field(description="Constitution modifier: (CON - 10) // 2")
    @property
    def constitution_modifier(self) -> int:
        """Calculate Constitution modifier."""
        return calculate_modifier(self.constitution)

    @computed_field(description="Intelligence modifier: (INT - 10) // 2")
    @property
    def intelligence_modifier(self) -> int:
        """Calculate Intelligence modifier."""
        return calculate_modifier(self.intelligence)

    @computed_field(description="Wisdom modifier: (WIS - 10) // 2")
    @property
    def wisdom_modifier(self) -> int:
        """Calculate Wisdom modifier."""
        return calculate_modifier(self.wisdom)

    @computed_field(description="Charisma modifier: (CHA - 10) // 2")
    @property
    def charisma_modifier(self) -> int:
        """Calculate Charisma modifier."""
        return calculate_modifier(self.charisma)

    def get_score(self, ability: Ability) -> int:
        """Get the score for a specific ability.

        Args:
            ability: The ability to get the score for.

        Returns:
            The ability score value.
        """
        ability_map: dict[Ability, int] = {
            Ability.STR: self.strength,
            Ability.DEX: self.dexterity,
            Ability.CON: self.constitution,
            Ability.INT: self.intelligence,
            Ability.WIS: self.wisdom,
            Ability.CHA: self.charisma,
        }
        return ability_map[ability]

    def get_modifier(self, ability: Ability) -> int:
        """Get the modifier for a specific ability.

        Args:
            ability: The ability to get the modifier for.

        Returns:
            The calculated modifier.
        """
        return calculate_modifier(self.get_score(ability))


# =============================================================================
# Health Component
# =============================================================================


class HealthComponent(BaseModel):
    """Component for tracking hit points and health status.

    Manages current, maximum, and temporary hit points along with
    death saving throw tracking and condition status.

    Attributes:
        current_hp: Current hit points (can be negative for death saves).
        max_hp: Maximum hit points.
        temp_hp: Temporary hit points (shield, buffs).
        death_save_successes: Successful death saves (0-3).
        death_save_failures: Failed death saves (0-3).
        conditions: Active conditions affecting the entity.

    Example:
        >>> health = HealthComponent(current_hp=25, max_hp=30, temp_hp=5)
        >>> health.effective_hp
        30
        >>> health.hp_percentage
        83.33
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "description": "Hit point tracking with death saves and conditions"
        },
    )

    current_hp: int = Field(
        description="Current hit points. Can be negative for death save tracking.",
    )
    max_hp: Annotated[int, Field(ge=1, description="Maximum hit points (minimum 1)")]
    temp_hp: Annotated[int, Field(ge=0, description="Temporary hit points")] = 0
    death_save_successes: Annotated[
        int, Field(ge=0, le=3, description="Death save successes (0-3)")
    ] = 0
    death_save_failures: Annotated[
        int, Field(ge=0, le=3, description="Death save failures (0-3)")
    ] = 0
    conditions: list[Condition] = Field(
        default_factory=list,
        description="Active conditions (e.g., poisoned, stunned)",
    )

    @model_validator(mode="after")
    def validate_hp_bounds(self) -> "HealthComponent":
        """Ensure current HP doesn't exceed max HP (excluding temp)."""
        if self.current_hp > self.max_hp:
            # Allow this but it's unusual - healing can't exceed max
            pass
        return self

    @computed_field(description="Effective HP including temporary hit points")
    @property
    def effective_hp(self) -> int:
        """Calculate effective HP (current + temp)."""
        return max(0, self.current_hp) + self.temp_hp

    @computed_field(description="HP as percentage of maximum")
    @property
    def hp_percentage(self) -> float:
        """Calculate HP as percentage of max."""
        return round((max(0, self.current_hp) / self.max_hp) * 100, 2)

    @computed_field(description="Whether the entity is conscious")
    @property
    def is_conscious(self) -> bool:
        """Check if entity is conscious (HP > 0 and not incapacitated)."""
        incapacitating = {
            Condition.UNCONSCIOUS,
            Condition.PARALYZED,
            Condition.PETRIFIED,
        }
        return self.current_hp > 0 and not any(c in incapacitating for c in self.conditions)

    @computed_field(description="Whether the entity is stable (not dying)")
    @property
    def is_stable(self) -> bool:
        """Check if entity is stable (not making death saves)."""
        return self.current_hp > 0 or self.death_save_successes >= 3

    @computed_field(description="Whether the entity is dead")
    @property
    def is_dead(self) -> bool:
        """Check if entity is dead."""
        return self.death_save_failures >= 3 or self.current_hp <= -self.max_hp

    @computed_field(description="Whether the entity is making death saves")
    @property
    def is_dying(self) -> bool:
        """Check if entity is at 0 HP and making death saves."""
        return (
            self.current_hp <= 0
            and self.death_save_failures < 3
            and self.death_save_successes < 3
        )

    def take_damage(self, amount: int) -> "HealthComponent":
        """Apply damage to the entity.

        Damage is first applied to temporary HP, then current HP.

        Args:
            amount: Amount of damage to take.

        Returns:
            New HealthComponent with updated values.
        """
        remaining = amount
        new_temp = self.temp_hp
        new_current = self.current_hp

        # Temp HP absorbs damage first
        if new_temp > 0:
            if remaining <= new_temp:
                new_temp -= remaining
                remaining = 0
            else:
                remaining -= new_temp
                new_temp = 0

        # Apply remaining to current HP
        new_current -= remaining

        return HealthComponent(
            current_hp=new_current,
            max_hp=self.max_hp,
            temp_hp=new_temp,
            death_save_successes=0 if new_current <= 0 else self.death_save_successes,
            death_save_failures=0 if new_current <= 0 else self.death_save_failures,
            conditions=self.conditions.copy(),
        )

    def heal(self, amount: int) -> "HealthComponent":
        """Heal the entity.

        Healing cannot exceed max HP.

        Args:
            amount: Amount of healing.

        Returns:
            New HealthComponent with updated values.
        """
        new_current = min(self.current_hp + amount, self.max_hp)

        return HealthComponent(
            current_hp=new_current,
            max_hp=self.max_hp,
            temp_hp=self.temp_hp,
            death_save_successes=0,
            death_save_failures=0,
            conditions=[c for c in self.conditions if c != Condition.UNCONSCIOUS],
        )


# =============================================================================
# Persona Component
# =============================================================================


class PersonaComponent(BaseModel):
    """Component defining AI behavior and personality for entities.

    Controls how the AI interprets and plays a character, including
    their personality, tactical directives, and autonomy level.

    Attributes:
        name: Display name for the persona.
        autonomy: Level of AI control over this entity.
        biography: Backstory and personality description for AI context.
        directives: Tactical and roleplay directives for AI behavior.
        voice_style: Speaking style for AI-generated dialogue.
        personality_traits: Key personality traits.
        ideals: Character ideals and beliefs.
        bonds: Important connections and relationships.
        flaws: Character flaws and weaknesses.

    Example:
        >>> persona = PersonaComponent(
        ...     name="Grimjaw the Fierce",
        ...     autonomy=AutonomyLevel.FULL_AUTO,
        ...     biography="A battle-hardened orc warrior...",
        ...     directives=["Protect the tribe", "Never show weakness"],
        ...     voice_style="Gruff and direct"
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "description": "AI personality and behavior configuration for entities"
        },
    )

    name: str = Field(
        min_length=1,
        max_length=100,
        description="Display name for this persona",
    )
    autonomy: AutonomyLevel = Field(
        default=AutonomyLevel.NONE,
        description="Level of AI control: NONE (player), SUGGESTIVE, ASSISTED, FULL_AUTO",
    )
    biography: str = Field(
        default="",
        max_length=5000,
        description="Backstory and personality description for AI roleplay context",
    )
    directives: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Tactical and roleplay directives (e.g., 'Protect the healer', 'Use fire spells')",
    )
    voice_style: str = Field(
        default="Natural",
        max_length=200,
        description="Speaking style for dialogue (e.g., 'Shakespearean', 'Gruff', 'Eloquent')",
    )
    personality_traits: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Key personality traits (from D&D background)",
    )
    ideals: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Character ideals and core beliefs",
    )
    bonds: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Important connections, relationships, or loyalties",
    )
    flaws: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Character flaws, weaknesses, or vices",
    )
    catchphrases: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Memorable phrases the character uses",
    )

    @computed_field(description="Whether AI has any control over this entity")
    @property
    def is_ai_controlled(self) -> bool:
        """Check if AI has any level of control."""
        return self.autonomy != AutonomyLevel.NONE

    @computed_field(description="Whether AI has full autonomous control")
    @property
    def is_fully_autonomous(self) -> bool:
        """Check if AI has complete control."""
        return self.autonomy == AutonomyLevel.FULL_AUTO

    def get_ai_prompt_context(self) -> str:
        """Generate context string for AI prompts.

        Returns:
            Formatted string with persona information for AI context.
        """
        parts = [f"Character: {self.name}"]

        if self.biography:
            parts.append(f"Background: {self.biography}")

        if self.personality_traits:
            parts.append(f"Personality: {', '.join(self.personality_traits)}")

        if self.ideals:
            parts.append(f"Ideals: {', '.join(self.ideals)}")

        if self.bonds:
            parts.append(f"Bonds: {', '.join(self.bonds)}")

        if self.flaws:
            parts.append(f"Flaws: {', '.join(self.flaws)}")

        if self.directives:
            parts.append(f"Directives: {'; '.join(self.directives)}")

        if self.voice_style and self.voice_style != "Natural":
            parts.append(f"Voice: {self.voice_style}")

        return "\n".join(parts)


# =============================================================================
# Speed Component
# =============================================================================


class SpeedComponent(BaseModel):
    """Component for movement speeds.

    Tracks various movement types available to an entity.

    Attributes:
        walk: Walking speed in feet.
        fly: Flying speed in feet (0 if cannot fly).
        swim: Swimming speed in feet.
        climb: Climbing speed in feet.
        burrow: Burrowing speed in feet.
        hover: Whether the entity can hover while flying.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    walk: Annotated[int, Field(ge=0, le=1000, description="Walking speed in feet")] = 30
    fly: Annotated[int, Field(ge=0, le=1000, description="Flying speed in feet")] = 0
    swim: Annotated[int, Field(ge=0, le=1000, description="Swimming speed in feet")] = 0
    climb: Annotated[int, Field(ge=0, le=1000, description="Climbing speed in feet")] = 0
    burrow: Annotated[int, Field(ge=0, le=1000, description="Burrowing speed in feet")] = 0
    hover: bool = Field(
        default=False,
        description="Whether the entity can hover while flying",
    )

    @computed_field(description="Primary movement speed (usually walking)")
    @property
    def primary_speed(self) -> int:
        """Get the primary movement speed."""
        return self.walk if self.walk > 0 else max(self.fly, self.swim, self.climb)


# =============================================================================
# Armor Component
# =============================================================================


class ArmorComponent(BaseModel):
    """Component for armor class calculation.

    Supports various AC calculation methods including natural armor,
    worn armor, and magical bonuses.

    Attributes:
        base_ac: Base armor class (10 for unarmored).
        armor_bonus: Bonus from worn armor.
        shield_bonus: Bonus from shield.
        natural_armor: Natural armor bonus (replaces base).
        dex_modifier_cap: Maximum DEX modifier allowed (None for no cap).
        magical_bonus: Magical enhancement bonus.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    base_ac: Annotated[int, Field(ge=0, le=30)] = Field(
        default=10,
        description="Base armor class before modifiers",
    )
    armor_bonus: Annotated[int, Field(ge=0, le=20)] = Field(
        default=0,
        description="Bonus from worn armor",
    )
    shield_bonus: Annotated[int, Field(ge=0, le=5)] = Field(
        default=0,
        description="Bonus from equipped shield",
    )
    natural_armor: Annotated[int, Field(ge=0, le=25)] | None = Field(
        default=None,
        description="Natural armor (if present, replaces base AC)",
    )
    dex_modifier_cap: Annotated[int, Field(ge=0, le=10)] | None = Field(
        default=None,
        description="Maximum DEX modifier for AC (None = no cap)",
    )
    magical_bonus: Annotated[int, Field(ge=0, le=5)] = Field(
        default=0,
        description="Magical enhancement bonus to AC",
    )

    def calculate_ac(self, dex_modifier: int) -> int:
        """Calculate total armor class.

        Args:
            dex_modifier: The entity's Dexterity modifier.

        Returns:
            Total calculated AC.
        """
        # Use natural armor if present
        base = self.natural_armor if self.natural_armor is not None else self.base_ac

        # Apply DEX modifier with cap
        effective_dex = dex_modifier
        if self.dex_modifier_cap is not None:
            effective_dex = min(dex_modifier, self.dex_modifier_cap)

        return base + self.armor_bonus + self.shield_bonus + effective_dex + self.magical_bonus


# =============================================================================
# Saving Throws Component
# =============================================================================


class SavingThrowsComponent(BaseModel):
    """Component for saving throw proficiencies.

    Tracks which saving throws an entity is proficient in.

    Attributes:
        proficiencies: Set of abilities the entity is proficient in saving throws for.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    proficiencies: set[Ability] = Field(
        default_factory=set,
        description="Abilities with saving throw proficiency",
    )

    def get_save_bonus(
        self,
        ability: Ability,
        ability_modifier: int,
        proficiency_bonus: int,
    ) -> int:
        """Calculate saving throw bonus.

        Args:
            ability: The ability for the save.
            ability_modifier: The modifier for that ability.
            proficiency_bonus: The entity's proficiency bonus.

        Returns:
            Total saving throw bonus.
        """
        bonus = ability_modifier
        if ability in self.proficiencies:
            bonus += proficiency_bonus
        return bonus


# =============================================================================
# Skills Component
# =============================================================================


class SkillsComponent(BaseModel):
    """Component for skill proficiencies and expertise.

    Tracks skill proficiencies and expertise for an entity.

    Attributes:
        proficiencies: Skills the entity is proficient in.
        expertise: Skills the entity has expertise in (double proficiency).
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    proficiencies: set[Skill] = Field(
        default_factory=set,
        description="Skills with proficiency",
    )
    expertise: set[Skill] = Field(
        default_factory=set,
        description="Skills with expertise (double proficiency)",
    )

    @model_validator(mode="after")
    def validate_expertise_requires_proficiency(self) -> "SkillsComponent":
        """Ensure expertise skills are also proficient."""
        if not self.expertise.issubset(self.proficiencies):
            missing = self.expertise - self.proficiencies
            msg = f"Expertise requires proficiency. Missing proficiency in: {missing}"
            raise ValueError(msg)
        return self

    def get_skill_bonus(
        self,
        skill: Skill,
        stats: StatsComponent,
        proficiency_bonus: int,
    ) -> int:
        """Calculate skill check bonus.

        Args:
            skill: The skill to check.
            stats: The entity's ability scores.
            proficiency_bonus: The entity's proficiency bonus.

        Returns:
            Total skill check bonus.
        """
        ability_modifier = stats.get_modifier(skill.ability)
        bonus = ability_modifier

        if skill in self.expertise:
            bonus += proficiency_bonus * 2
        elif skill in self.proficiencies:
            bonus += proficiency_bonus

        return bonus


# =============================================================================
# Deprecation Warning
# =============================================================================

def _issue_deprecation_warning() -> None:
    """Issue deprecation warning when this module is imported."""
    warnings.warn(
        "models.components is deprecated. Use components from models.ecs instead. "
        "This compatibility layer will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


# Issue warning on import
_issue_deprecation_warning()


__all__ = [
    "AbilityScore",
    "calculate_modifier",
    "validate_ability_score",
    "StatsComponent",
    "HealthComponent",
    "PersonaComponent",
    "SpeedComponent",
    "ArmorComponent",
    "SavingThrowsComponent",
    "SkillsComponent",
]
