"""Pydantic V2 schemas for campaign management.

This module defines the data models for campaigns, sessions, and scenes,
enabling persistent campaign state management.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class SceneType(StrEnum):
    """Types of scenes/encounters."""

    COMBAT = "combat"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    PUZZLE = "puzzle"
    REST = "rest"
    TRAVEL = "travel"
    CUSTOM = "custom"


class Difficulty(StrEnum):
    """Encounter difficulty ratings."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    DEADLY = "deadly"


class Scene(BaseModel):
    """Scene or encounter definition.

    Attributes:
        id: Unique scene identifier.
        name: Scene name.
        scene_type: Type of scene (combat, exploration, etc.).
        description: Scene description for the DM.
        read_aloud_text: Text to read aloud to players.
        difficulty: Scene difficulty rating.
        location: Scene location name.
        npcs: List of NPC identifiers in this scene.
        monsters: List of monster stat block references.
        loot: Potential loot/treasure in this scene.
        notes: DM notes.
        is_completed: Whether the scene has been completed.
        metadata: Additional scene metadata.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Unique scene ID")
    name: str = Field(min_length=1, max_length=200, description="Scene name")
    scene_type: SceneType = Field(default=SceneType.CUSTOM, description="Scene type")
    description: str = Field(default="", max_length=5000, description="DM description")
    read_aloud_text: str = Field(default="", max_length=2000, description="Read-aloud text")
    difficulty: Difficulty | None = Field(default=None, description="Difficulty rating")
    location: str = Field(default="", max_length=200, description="Location name")
    npcs: list[UUID] = Field(default_factory=list, description="NPC references")
    monsters: list[str] = Field(default_factory=list, description="Monster stat block refs")
    loot: list[str] = Field(default_factory=list, description="Potential loot")
    notes: str = Field(default="", max_length=2000, description="DM notes")
    is_completed: bool = Field(default=False, description="Scene completed?")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Session(BaseModel):
    """Game session record.

    Attributes:
        id: Unique session identifier.
        session_number: Sequential session number.
        title: Session title.
        date: When the session occurred.
        duration_minutes: Session duration in minutes.
        summary: Session summary/recap.
        scenes_completed: List of scene IDs completed this session.
        xp_awarded: Experience points awarded this session.
        notes: DM notes from the session.
        player_attendance: List of player character IDs who attended.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Unique session ID")
    session_number: Annotated[int, Field(ge=1, description="Session number")]
    title: str = Field(min_length=1, max_length=200, description="Session title")
    date: datetime = Field(default_factory=datetime.now, description="Session date")
    duration_minutes: Annotated[int, Field(ge=0, description="Duration in minutes")] = 0
    summary: str = Field(default="", max_length=5000, description="Session summary")
    scenes_completed: list[UUID] = Field(default_factory=list, description="Completed scenes")
    xp_awarded: Annotated[int, Field(ge=0, description="XP awarded")] = 0
    notes: str = Field(default="", max_length=5000, description="Session notes")
    player_attendance: list[UUID] = Field(default_factory=list, description="Players present")


class Campaign(BaseModel):
    """Campaign metadata and state.

    Attributes:
        id: Unique campaign identifier.
        name: Campaign name.
        description: Campaign description/premise.
        setting: Campaign setting (e.g., 'Forgotten Realms').
        player_characters: List of player character IDs.
        scenes: List of scenes in the campaign.
        sessions: List of completed sessions.
        current_scene_id: ID of the current active scene.
        created_at: When the campaign was created.
        updated_at: When the campaign was last updated.
        is_active: Whether the campaign is currently active.
        notes: DM campaign notes.
        metadata: Additional campaign metadata.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Unique campaign ID")
    name: str = Field(min_length=1, max_length=200, description="Campaign name")
    description: str = Field(default="", max_length=5000, description="Campaign description")
    setting: str = Field(default="Custom", max_length=200, description="Campaign setting")
    player_characters: list[UUID] = Field(default_factory=list, description="Player characters")
    scenes: list[Scene] = Field(default_factory=list, description="Campaign scenes")
    sessions: list[Session] = Field(default_factory=list, description="Completed sessions")
    current_scene_id: UUID | None = Field(default=None, description="Current scene")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    is_active: bool = Field(default=True, description="Campaign active?")
    notes: str = Field(default="", max_length=10000, description="Campaign notes")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


__all__ = [
    "SceneType",
    "Difficulty",
    "Scene",
    "Session",
    "Campaign",
]
