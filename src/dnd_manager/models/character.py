"""Pydantic V2 schemas for character entities.

This module defines the data models for player characters and NPCs,
including ability scores, classes, and derived statistics.

DEPRECATED: This module is deprecated in favor of the unified ECS architecture
in models/ecs.py. It is maintained for backward compatibility only.
New code should use ActorEntity from models/ecs.py.
"""

from __future__ import annotations

import warnings

from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field


class AbilityScore(StrEnum):
    """D&D 5E ability scores."""

    STRENGTH = "strength"
    DEXTERITY = "dexterity"
    CONSTITUTION = "constitution"
    INTELLIGENCE = "intelligence"
    WISDOM = "wisdom"
    CHARISMA = "charisma"


class Size(StrEnum):
    """D&D 5E creature sizes."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"
    GARGANTUAN = "gargantuan"


class CharacterStats(BaseModel):
    """Ability scores and derived statistics for a character.

    Attributes:
        strength: Strength ability score (1-30).
        dexterity: Dexterity ability score (1-30).
        constitution: Constitution ability score (1-30).
        intelligence: Intelligence ability score (1-30).
        wisdom: Wisdom ability score (1-30).
        charisma: Charisma ability score (1-30).
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    strength: Annotated[int, Field(ge=1, le=30, description="Strength score")]
    dexterity: Annotated[int, Field(ge=1, le=30, description="Dexterity score")]
    constitution: Annotated[int, Field(ge=1, le=30, description="Constitution score")]
    intelligence: Annotated[int, Field(ge=1, le=30, description="Intelligence score")]
    wisdom: Annotated[int, Field(ge=1, le=30, description="Wisdom score")]
    charisma: Annotated[int, Field(ge=1, le=30, description="Charisma score")]

    def get_modifier(self, ability: AbilityScore) -> int:
        """Calculate the ability modifier for a given ability score.

        Args:
            ability: The ability score to get the modifier for.

        Returns:
            The ability modifier (score - 10) // 2.
        """
        score = getattr(self, ability.value)
        return (score - 10) // 2


class CharacterClass(BaseModel):
    """Character class information.

    Attributes:
        name: Class name (e.g., 'Fighter', 'Wizard').
        subclass: Optional subclass name.
        level: Class level (1-20).
        hit_die: Hit die size for the class.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    name: str = Field(min_length=1, max_length=50, description="Class name")
    subclass: str | None = Field(default=None, max_length=100, description="Subclass name")
    level: Annotated[int, Field(ge=1, le=20, description="Class level")]
    hit_die: Annotated[int, Field(description="Hit die size")] = Field(
        default=8,
        ge=6,
        le=12,
    )


class Character(BaseModel):
    """Player character or NPC schema.

    Attributes:
        id: Unique character identifier.
        name: Character name.
        race: Character race.
        size: Character size category.
        stats: Ability scores.
        classes: List of character classes (multiclassing supported).
        max_hit_points: Maximum hit points.
        current_hit_points: Current hit points.
        temporary_hit_points: Temporary hit points.
        armor_class: Armor class.
        speed: Movement speed in feet.
        proficiency_bonus: Proficiency bonus based on total level.
        is_npc: Whether this character is an NPC.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    name: str = Field(min_length=1, max_length=100, description="Character name")
    race: str = Field(min_length=1, max_length=50, description="Character race")
    size: Size = Field(default=Size.MEDIUM, description="Character size")
    stats: CharacterStats = Field(description="Ability scores")
    classes: list[CharacterClass] = Field(
        min_length=1,
        max_length=20,
        description="Character classes",
    )
    max_hit_points: Annotated[int, Field(ge=1, description="Maximum HP")]
    current_hit_points: int = Field(description="Current HP")
    temporary_hit_points: Annotated[int, Field(ge=0, description="Temporary HP")] = 0
    armor_class: Annotated[int, Field(ge=1, le=30, description="Armor class")]
    speed: Annotated[int, Field(ge=0, le=1000, description="Speed in feet")] = 30
    is_npc: bool = Field(default=False, description="Is this an NPC?")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_level(self) -> int:
        """Calculate total character level from all classes.

        Returns:
            Sum of all class levels.
        """
        return sum(c.level for c in self.classes)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def proficiency_bonus(self) -> int:
        """Calculate proficiency bonus based on total level.

        Returns:
            Proficiency bonus (2-6 based on level).
        """
        return (self.total_level - 1) // 4 + 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_conscious(self) -> bool:
        """Check if character is conscious (HP > 0).

        Returns:
            True if current HP is above 0.
        """
        return self.current_hit_points > 0


# =============================================================================
# Deprecation Warning
# =============================================================================

def _issue_deprecation_warning() -> None:
    """Issue deprecation warning when this module is imported."""
    warnings.warn(
        "models.character is deprecated. Use models.ecs.ActorEntity instead. "
        "This compatibility layer will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


# Issue warning on import
_issue_deprecation_warning()


__all__ = [
    "AbilityScore",
    "Size",
    "CharacterStats",
    "CharacterClass",
    "Character",
]
