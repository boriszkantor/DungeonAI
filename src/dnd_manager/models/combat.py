"""Pydantic V2 schemas for combat management.

This module defines the data models for combat encounters, including
initiative tracking, combatants, and combat state management.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field


class CombatPhase(StrEnum):
    """Combat encounter phases."""

    NOT_STARTED = "not_started"
    INITIATIVE = "initiative"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class CombatantType(StrEnum):
    """Types of combatants."""

    PLAYER = "player"
    NPC_ALLY = "npc_ally"
    NPC_ENEMY = "npc_enemy"
    NEUTRAL = "neutral"


class Condition(StrEnum):
    """D&D 5E conditions."""

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


class Initiative(BaseModel):
    """Initiative tracker entry.

    Attributes:
        combatant_id: Reference to the combatant.
        roll: Initiative roll result.
        dexterity_modifier: Dexterity modifier for tiebreaking.
        has_advantage: Whether the combatant has advantage on initiative.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    combatant_id: UUID = Field(description="Reference to combatant")
    roll: int = Field(description="Initiative roll result")
    dexterity_modifier: int = Field(default=0, description="DEX modifier for tiebreaking")
    has_advantage: bool = Field(default=False, description="Has advantage on initiative")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sort_key(self) -> tuple[int, int, bool]:
        """Generate sort key for initiative ordering.

        Returns:
            Tuple of (roll, dex_mod, has_advantage) for sorting.
        """
        return (self.roll, self.dexterity_modifier, self.has_advantage)


class Combatant(BaseModel):
    """Entity participating in combat.

    Attributes:
        id: Unique combatant identifier.
        character_id: Reference to the underlying character.
        name: Display name in combat.
        combatant_type: Type of combatant (player, enemy, ally, neutral).
        current_hp: Current hit points.
        max_hp: Maximum hit points.
        temp_hp: Temporary hit points.
        armor_class: Current armor class.
        conditions: Active conditions on this combatant.
        concentration_spell: Name of spell being concentrated on, if any.
        death_saves_success: Number of successful death saves.
        death_saves_failure: Number of failed death saves.
        notes: DM notes for this combatant.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Unique combatant ID")
    character_id: UUID | None = Field(default=None, description="Reference to character")
    name: str = Field(min_length=1, max_length=100, description="Display name")
    combatant_type: CombatantType = Field(description="Combatant type")
    current_hp: int = Field(description="Current HP")
    max_hp: Annotated[int, Field(ge=1, description="Maximum HP")]
    temp_hp: Annotated[int, Field(ge=0, description="Temporary HP")] = 0
    armor_class: Annotated[int, Field(ge=1, le=30, description="Armor class")]
    conditions: list[Condition] = Field(default_factory=list, description="Active conditions")
    concentration_spell: str | None = Field(default=None, description="Concentrating on spell")
    death_saves_success: Annotated[int, Field(ge=0, le=3)] = 0
    death_saves_failure: Annotated[int, Field(ge=0, le=3)] = 0
    notes: str = Field(default="", max_length=500, description="DM notes")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_conscious(self) -> bool:
        """Check if combatant is conscious.

        Returns:
            True if HP > 0 and not incapacitated/unconscious.
        """
        incapacitating = {Condition.UNCONSCIOUS, Condition.PARALYZED, Condition.PETRIFIED}
        return self.current_hp > 0 and not any(c in incapacitating for c in self.conditions)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_dead(self) -> bool:
        """Check if combatant is dead.

        Returns:
            True if death save failures >= 3 or HP <= negative max HP.
        """
        return self.death_saves_failure >= 3 or self.current_hp <= -self.max_hp

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_stable(self) -> bool:
        """Check if combatant is stable (unconscious but not dying).

        Returns:
            True if death save successes >= 3.
        """
        return self.death_saves_success >= 3 and self.current_hp <= 0


class CombatState(BaseModel):
    """Current state of a combat encounter.

    Attributes:
        id: Unique combat encounter identifier.
        name: Encounter name.
        phase: Current combat phase.
        round_number: Current round number.
        turn_index: Index of current combatant in initiative order.
        combatants: List of combatants in the encounter.
        initiative_order: Sorted initiative entries.
        started_at: When combat started.
        ended_at: When combat ended (if applicable).
        notes: DM notes for the encounter.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Unique encounter ID")
    name: str = Field(
        default="Combat Encounter",
        min_length=1,
        max_length=100,
        description="Encounter name",
    )
    phase: CombatPhase = Field(default=CombatPhase.NOT_STARTED, description="Combat phase")
    round_number: Annotated[int, Field(ge=0, description="Current round")] = 0
    turn_index: Annotated[int, Field(ge=0, description="Current turn index")] = 0
    combatants: list[Combatant] = Field(default_factory=list, description="Combatants")
    initiative_order: list[Initiative] = Field(
        default_factory=list,
        description="Initiative order",
    )
    started_at: datetime | None = Field(default=None, description="Combat start time")
    ended_at: datetime | None = Field(default=None, description="Combat end time")
    notes: str = Field(default="", max_length=2000, description="Encounter notes")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_combatant_id(self) -> UUID | None:
        """Get the ID of the current combatant.

        Returns:
            UUID of current combatant or None if combat not active.
        """
        if self.phase != CombatPhase.ACTIVE or not self.initiative_order:
            return None
        index = self.turn_index % len(self.initiative_order)
        return self.initiative_order[index].combatant_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_combatants(self) -> list[Combatant]:
        """Get list of conscious, active combatants.

        Returns:
            List of combatants who can still act.
        """
        return [c for c in self.combatants if c.is_conscious and not c.is_dead]


__all__ = [
    "CombatPhase",
    "CombatantType",
    "Condition",
    "Initiative",
    "Combatant",
    "CombatState",
]
