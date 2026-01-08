"""Game state models for the D&D 5E AI Campaign Manager.

This module defines models for tracking game state including turn order,
scenes, encounters, and the overall game session state.

Models:
    TurnOrder: Queue of combatant IDs for initiative tracking.
    Scene: A game scene with combatants and environment.
    Encounter: A combat encounter with full state tracking.
    GameSession: Overall game session state.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
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

from dnd_manager.models.entities import Combatant, Monster, NPC, PlayerCharacter
from dnd_manager.models.enums import AutonomyLevel, CombatantType


# =============================================================================
# Turn Order
# =============================================================================


class InitiativeEntry(BaseModel):
    """An entry in the initiative order.

    Tracks a combatant's position in the turn order with their
    initiative roll and tiebreaker information.

    Attributes:
        combatant_uid: Reference to the combatant.
        combatant_name: Cached name for display.
        combatant_type: Type discriminator for quick filtering.
        initiative_roll: Total initiative roll result.
        dexterity_score: DEX score for tiebreaking.
        tiebreaker_roll: Additional roll for ties (d20).
        is_active: Whether this combatant can still act.
        has_acted_this_round: Whether they've taken their turn.
        delayed: Whether they are delaying their turn.
        readied_action: Description of readied action, if any.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    combatant_uid: UUID = Field(description="Reference to the combatant entity")
    combatant_name: str = Field(description="Cached name for display")
    combatant_type: CombatantType = Field(description="Type for quick filtering")
    initiative_roll: int = Field(description="Total initiative roll (d20 + DEX mod)")
    dexterity_score: Annotated[int, Field(ge=1, le=30)] = Field(
        default=10,
        description="DEX score for tiebreaking",
    )
    tiebreaker_roll: Annotated[int, Field(ge=1, le=20)] = Field(
        default=10,
        description="Additional d20 roll for remaining ties",
    )
    is_active: bool = Field(default=True, description="Can this combatant still act")
    has_acted_this_round: bool = Field(default=False, description="Has taken turn this round")
    delayed: bool = Field(default=False, description="Delaying turn")
    readied_action: str | None = Field(default=None, description="Readied action description")

    @computed_field(description="Sort key for initiative ordering")
    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Generate sort key (higher is better).

        Returns:
            Tuple of (initiative, dex_score, tiebreaker) for sorting.
        """
        return (self.initiative_roll, self.dexterity_score, self.tiebreaker_roll)


class TurnOrder(BaseModel):
    """Queue of combatant IDs for initiative tracking.

    Manages the turn order for combat, including round tracking,
    current turn index, and initiative modifications.

    Attributes:
        entries: Ordered list of initiative entries.
        current_round: Current combat round (1-based).
        current_index: Index of current combatant in entries.
        round_started_at: Timestamp when current round started.

    Example:
        >>> turn_order = TurnOrder(entries=[...])
        >>> current = turn_order.current_combatant_uid
        >>> turn_order.advance_turn()
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "description": "Initiative order tracking for combat encounters"
        },
    )

    entries: list[InitiativeEntry] = Field(
        default_factory=list,
        description="Ordered list of initiative entries (highest first)",
    )
    current_round: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Current round number (0 = not started, 1+ = active)",
    )
    current_index: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Index of current combatant in entries list",
    )
    round_started_at: datetime | None = Field(
        default=None,
        description="When the current round started",
    )

    @computed_field(description="Whether combat has started")
    @property
    def is_combat_active(self) -> bool:
        """Check if combat is currently active."""
        return self.current_round > 0 and len(self.active_entries) > 0

    @computed_field(description="List of active (non-defeated) entries")
    @property
    def active_entries(self) -> list[InitiativeEntry]:
        """Get only active combatants."""
        return [e for e in self.entries if e.is_active]

    @computed_field(description="UID of current combatant")
    @property
    def current_combatant_uid(self) -> UUID | None:
        """Get the UID of the current combatant."""
        active = self.active_entries
        if not active or self.current_round == 0:
            return None
        idx = self.current_index % len(active)
        return active[idx].combatant_uid

    @computed_field(description="Current combatant's entry")
    @property
    def current_entry(self) -> InitiativeEntry | None:
        """Get the current initiative entry."""
        active = self.active_entries
        if not active or self.current_round == 0:
            return None
        idx = self.current_index % len(active)
        return active[idx]

    def sort_by_initiative(self) -> "TurnOrder":
        """Sort entries by initiative (highest first).

        Returns:
            New TurnOrder with sorted entries.
        """
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.sort_key,
            reverse=True,
        )
        return TurnOrder(
            entries=sorted_entries,
            current_round=self.current_round,
            current_index=self.current_index,
            round_started_at=self.round_started_at,
        )

    def start_combat(self) -> "TurnOrder":
        """Start combat at round 1.

        Returns:
            New TurnOrder with combat started.
        """
        sorted_order = self.sort_by_initiative()
        return TurnOrder(
            entries=sorted_order.entries,
            current_round=1,
            current_index=0,
            round_started_at=datetime.now(),
        )

    def advance_turn(self) -> "TurnOrder":
        """Advance to the next turn.

        Returns:
            New TurnOrder with advanced turn/round.
        """
        active = self.active_entries
        if not active:
            return self

        # Mark current as acted
        new_entries = []
        for i, entry in enumerate(self.entries):
            if entry.combatant_uid == self.current_combatant_uid:
                new_entries.append(
                    InitiativeEntry(
                        **{**entry.model_dump(), "has_acted_this_round": True}
                    )
                )
            else:
                new_entries.append(entry)

        new_index = self.current_index + 1
        new_round = self.current_round
        round_started = self.round_started_at

        # Check if round is complete
        if new_index >= len(active):
            new_index = 0
            new_round += 1
            round_started = datetime.now()
            # Reset has_acted_this_round for new round
            new_entries = [
                InitiativeEntry(**{**e.model_dump(), "has_acted_this_round": False})
                for e in new_entries
            ]

        return TurnOrder(
            entries=new_entries,
            current_round=new_round,
            current_index=new_index,
            round_started_at=round_started,
        )

    def remove_combatant(self, combatant_uid: UUID) -> "TurnOrder":
        """Mark a combatant as inactive (defeated).

        Args:
            combatant_uid: UID of combatant to remove.

        Returns:
            New TurnOrder with combatant marked inactive.
        """
        new_entries = []
        for entry in self.entries:
            if entry.combatant_uid == combatant_uid:
                new_entries.append(
                    InitiativeEntry(**{**entry.model_dump(), "is_active": False})
                )
            else:
                new_entries.append(entry)

        return TurnOrder(
            entries=new_entries,
            current_round=self.current_round,
            current_index=self.current_index,
            round_started_at=self.round_started_at,
        )


# =============================================================================
# Scene
# =============================================================================


class SceneType(StrEnum):
    """Types of scenes in the game."""

    COMBAT = "combat"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    PUZZLE = "puzzle"
    REST = "rest"
    TRAVEL = "travel"
    NARRATIVE = "narrative"


class EnvironmentEffect(BaseModel):
    """An environmental effect active in a scene.

    Attributes:
        name: Effect name.
        description: What the effect does.
        area: Affected area description.
        damage: Damage dealt (if any).
        save_dc: DC for saving throws (if any).
        save_ability: Ability for the save.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100, description="Effect name")
    description: str = Field(max_length=500, description="Effect description")
    area: str = Field(default="", max_length=200, description="Affected area")
    damage: str | None = Field(default=None, description="Damage expression")
    save_dc: Annotated[int, Field(ge=1, le=30)] | None = Field(default=None, description="Save DC")
    save_ability: str | None = Field(default=None, description="Save ability")


class Scene(BaseModel):
    """A game scene containing combatants and environmental context.

    Represents a location or situation in the game with full state
    including all participants, environmental effects, and narrative context.

    Attributes:
        uid: Unique scene identifier.
        name: Scene name.
        description: DM description of the scene.
        read_aloud: Text to read to players.
        scene_type: Type of scene.
        combatants: All entities in the scene.
        turn_order: Initiative tracking (if in combat).
        environment_effects: Active environmental effects.
        objectives: Scene objectives for the party.
        secrets: Hidden information (DM only).
        loot: Available treasure/items.
        notes: DM notes.
        is_active: Whether scene is currently active.
        started_at: When scene started.
        completed_at: When scene completed.

    Example:
        >>> scene = Scene(
        ...     name="Goblin Ambush",
        ...     description="A group of goblins hides in the trees...",
        ...     combatants=[goblin1, goblin2, fighter, wizard],
        ...     scene_type=SceneType.COMBAT,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "description": "A game scene with combatants and environment"
        },
    )

    uid: UUID = Field(default_factory=uuid4, description="Unique scene identifier")
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Scene name for reference",
    )
    description: str = Field(
        default="",
        max_length=5000,
        description="Full DM description of the scene",
    )
    read_aloud: str = Field(
        default="",
        max_length=2000,
        description="Boxed text to read to players",
    )
    scene_type: SceneType = Field(
        default=SceneType.EXPLORATION,
        description="Type of scene (combat, social, etc.)",
    )

    # Participants
    combatants: list[Combatant] = Field(
        default_factory=list,
        description="All entities present in the scene",
    )
    turn_order: TurnOrder = Field(
        default_factory=TurnOrder,
        description="Initiative order for combat",
    )

    # Environment
    environment_effects: list[EnvironmentEffect] = Field(
        default_factory=list,
        description="Active environmental hazards or effects",
    )
    terrain_description: str = Field(
        default="",
        max_length=1000,
        description="Terrain and tactical features",
    )
    lighting: Literal["bright", "dim", "darkness", "magical_darkness"] = Field(
        default="bright",
        description="Lighting conditions",
    )

    # Objectives and secrets
    objectives: list[str] = Field(
        default_factory=list,
        description="Scene objectives for the party",
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="Hidden information (DM only)",
    )
    loot: list[str] = Field(
        default_factory=list,
        description="Available treasure and items",
    )
    notes: str = Field(
        default="",
        max_length=5000,
        description="DM notes about the scene",
    )

    # State tracking
    is_active: bool = Field(default=False, description="Scene currently active")
    is_completed: bool = Field(default=False, description="Scene has been completed")
    started_at: datetime | None = Field(default=None, description="When scene started")
    completed_at: datetime | None = Field(default=None, description="When scene completed")

    @computed_field(description="Index of combatant whose turn it is")
    @property
    def active_turn_index(self) -> int | None:
        """Get the current turn index in combat.

        Returns:
            Index of current combatant, or None if not in combat.
        """
        if not self.turn_order.is_combat_active:
            return None
        return self.turn_order.current_index

    @computed_field(description="Whether scene is in active combat")
    @property
    def is_in_combat(self) -> bool:
        """Check if combat is active in this scene."""
        return self.scene_type == SceneType.COMBAT and self.turn_order.is_combat_active

    @computed_field(description="List of player characters in scene")
    @property
    def player_characters(self) -> list[PlayerCharacter]:
        """Get all player characters in the scene."""
        return [c for c in self.combatants if isinstance(c, PlayerCharacter)]

    @computed_field(description="List of monsters in scene")
    @property
    def monsters(self) -> list[Monster]:
        """Get all monsters in the scene."""
        return [c for c in self.combatants if isinstance(c, Monster)]

    @computed_field(description="List of NPCs in scene")
    @property
    def npcs(self) -> list[NPC]:
        """Get all NPCs in the scene."""
        return [c for c in self.combatants if isinstance(c, NPC)]

    def get_combatant_by_uid(self, uid: UUID) -> Combatant | None:
        """Find a combatant by UID.

        Args:
            uid: The combatant's unique identifier.

        Returns:
            The combatant or None if not found.
        """
        for combatant in self.combatants:
            if combatant.uid == uid:
                return combatant
        return None

    def start_scene(self) -> "Scene":
        """Mark scene as started.

        Returns:
            New Scene with is_active=True and started_at set.
        """
        return Scene(
            **{
                **self.model_dump(),
                "is_active": True,
                "started_at": datetime.now(),
            }
        )

    def complete_scene(self) -> "Scene":
        """Mark scene as completed.

        Returns:
            New Scene with is_completed=True and completed_at set.
        """
        return Scene(
            **{
                **self.model_dump(),
                "is_active": False,
                "is_completed": True,
                "completed_at": datetime.now(),
            }
        )


# =============================================================================
# Game Session
# =============================================================================


class GamePhase(StrEnum):
    """Current phase of the game session."""

    SETUP = "setup"
    EXPLORATION = "exploration"
    COMBAT = "combat"
    SOCIAL = "social"
    REST = "rest"
    DOWNTIME = "downtime"
    PAUSED = "paused"
    ENDED = "ended"


class ChatMessage(BaseModel):
    """A message in the game chat/narrative log.

    Attributes:
        uid: Message identifier.
        timestamp: When message was sent.
        author: Who sent the message.
        author_type: Type of author (player, dm, ai, system).
        content: Message content.
        is_narrative: Whether this is narrative text.
        is_dice_roll: Whether this is a dice roll result.
        metadata: Additional message data.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    uid: UUID = Field(default_factory=uuid4, description="Message ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="When sent")
    author: str = Field(description="Author name")
    author_type: Literal["player", "dm", "ai", "system"] = Field(description="Author type")
    content: str = Field(max_length=10000, description="Message content")
    is_narrative: bool = Field(default=False, description="Narrative text")
    is_dice_roll: bool = Field(default=False, description="Dice roll result")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra data")


class GameSession(BaseModel):
    """Overall game session state.

    Tracks the complete state of a game session including all scenes,
    chat history, and session metadata.

    Attributes:
        uid: Session identifier.
        name: Session name.
        campaign_name: Name of the campaign.
        dm_name: Dungeon Master's name.
        phase: Current game phase.
        current_scene: Active scene (if any).
        completed_scenes: Previously completed scenes.
        party: Player characters in the party.
        chat_history: Narrative and chat log.
        session_number: Session number in campaign.
        started_at: Session start time.
        ended_at: Session end time.
        total_xp_awarded: XP given this session.
        notes: Session notes.

    Example:
        >>> session = GameSession(
        ...     name="Session 5: Into the Dungeon",
        ...     campaign_name="Lost Mines of Phandelver",
        ...     dm_name="Matthew",
        ...     party=[fighter, wizard, cleric, rogue],
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "description": "Complete game session state"
        },
    )

    uid: UUID = Field(default_factory=uuid4, description="Session identifier")
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Session name",
    )
    campaign_name: str = Field(
        default="",
        max_length=200,
        description="Campaign this session belongs to",
    )
    dm_name: str = Field(
        default="Dungeon Master",
        max_length=100,
        description="DM's display name",
    )

    # State
    phase: GamePhase = Field(
        default=GamePhase.SETUP,
        description="Current game phase",
    )
    current_scene: Scene | None = Field(
        default=None,
        description="Currently active scene",
    )
    completed_scenes: list[Scene] = Field(
        default_factory=list,
        description="Scenes completed this session",
    )

    # Party
    party: list[PlayerCharacter] = Field(
        default_factory=list,
        description="Player characters in the party",
    )

    # Chat and narrative
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Session chat and narrative log",
    )

    # Metadata
    session_number: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Session number in the campaign",
    )
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When session started",
    )
    ended_at: datetime | None = Field(
        default=None,
        description="When session ended",
    )
    total_xp_awarded: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Total XP awarded this session",
    )
    notes: str = Field(
        default="",
        max_length=10000,
        description="Session notes and recap",
    )

    @computed_field(description="Duration of session")
    @property
    def duration_minutes(self) -> int | None:
        """Calculate session duration in minutes."""
        if self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return int(delta.total_seconds() / 60)

    @computed_field(description="Whether session is active")
    @property
    def is_active(self) -> bool:
        """Check if session is currently running."""
        return self.phase not in (GamePhase.ENDED, GamePhase.PAUSED)

    def add_chat_message(
        self,
        author: str,
        content: str,
        *,
        author_type: Literal["player", "dm", "ai", "system"] = "player",
        is_narrative: bool = False,
    ) -> "GameSession":
        """Add a message to chat history.

        Args:
            author: Message author name.
            content: Message content.
            author_type: Type of author.
            is_narrative: Whether this is narrative text.

        Returns:
            New GameSession with message added.
        """
        message = ChatMessage(
            author=author,
            author_type=author_type,
            content=content,
            is_narrative=is_narrative,
        )
        return GameSession(
            **{
                **self.model_dump(),
                "chat_history": [*self.chat_history, message],
            }
        )

    def end_session(self) -> "GameSession":
        """Mark session as ended.

        Returns:
            New GameSession with ended state.
        """
        return GameSession(
            **{
                **self.model_dump(),
                "phase": GamePhase.ENDED,
                "ended_at": datetime.now(),
            }
        )


__all__ = [
    # Turn order
    "InitiativeEntry",
    "TurnOrder",
    # Scene
    "SceneType",
    "EnvironmentEffect",
    "Scene",
    # Session
    "GamePhase",
    "ChatMessage",
    "GameSession",
]
