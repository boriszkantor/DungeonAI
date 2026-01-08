"""Main game loop orchestration.

This module provides the core game loop functionality for the
D&D 5E AI Campaign Manager.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from dnd_manager.core.exceptions import GameEngineError, InvalidGameStateError
from dnd_manager.core.logging import get_logger
from dnd_manager.engine.turn_manager import TurnManager


if TYPE_CHECKING:
    from dnd_manager.models.campaign import Campaign, Scene
    from dnd_manager.models.character import Character
    from dnd_manager.models.combat import CombatState

logger = get_logger(__name__)


class GameState(StrEnum):
    """Game state enumeration."""

    IDLE = "idle"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    COMBAT = "combat"
    REST = "rest"
    PAUSED = "paused"


class GameEvent:
    """Base class for game events.

    Attributes:
        event_type: Type of the event.
        data: Event data payload.
        source_id: ID of the event source (character, scene, etc.).
    """

    def __init__(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Initialize a game event.

        Args:
            event_type: Type identifier for the event.
            data: Optional event data payload.
            source_id: Optional ID of the event source.
        """
        self.event_type = event_type
        self.data = data or {}
        self.source_id = source_id


class GameLoop:
    """Main game loop orchestration.

    This class manages the overall game state, coordinating between
    exploration, social encounters, combat, and rest modes.

    Attributes:
        state: Current game state.
        campaign: Active campaign (if any).
        current_scene: Currently active scene.
    """

    def __init__(self) -> None:
        """Initialize the game loop."""
        self._state = GameState.IDLE
        self._campaign: Campaign | None = None
        self._current_scene: Scene | None = None
        self._combat_state: CombatState | None = None
        self._turn_manager: TurnManager | None = None
        self._player_characters: dict[UUID, Character] = {}
        self._event_handlers: dict[str, list[Any]] = {}

        logger.info("GameLoop initialized")

    @property
    def state(self) -> GameState:
        """Get the current game state.

        Returns:
            Current GameState.
        """
        return self._state

    @property
    def campaign(self) -> Campaign | None:
        """Get the active campaign.

        Returns:
            Active Campaign or None.
        """
        return self._campaign

    @property
    def current_scene(self) -> Scene | None:
        """Get the current scene.

        Returns:
            Current Scene or None.
        """
        return self._current_scene

    @property
    def is_in_combat(self) -> bool:
        """Check if currently in combat.

        Returns:
            True if in combat state.
        """
        return self._state == GameState.COMBAT

    def load_campaign(self, campaign: Campaign) -> None:
        """Load a campaign into the game loop.

        Args:
            campaign: The campaign to load.

        Raises:
            GameEngineError: If a campaign is already loaded.
        """
        if self._campaign is not None:
            raise GameEngineError(
                "A campaign is already loaded. Unload it first.",
                details={"current_campaign": str(self._campaign.id)},
            )

        self._campaign = campaign
        self._state = GameState.IDLE

        # Load current scene if specified
        if campaign.current_scene_id:
            for scene in campaign.scenes:
                if scene.id == campaign.current_scene_id:
                    self._current_scene = scene
                    break

        logger.info("Campaign loaded", campaign_id=str(campaign.id), name=campaign.name)

    def unload_campaign(self) -> None:
        """Unload the current campaign.

        Raises:
            GameEngineError: If in combat or invalid state.
        """
        if self._state == GameState.COMBAT:
            raise GameEngineError("Cannot unload campaign during combat")

        campaign_id = self._campaign.id if self._campaign else None
        self._campaign = None
        self._current_scene = None
        self._state = GameState.IDLE

        logger.info("Campaign unloaded", campaign_id=str(campaign_id))

    def add_player_character(self, character: Character) -> None:
        """Add a player character to the game.

        Args:
            character: The character to add.
        """
        self._player_characters[character.id] = character
        logger.info("Player character added", character_id=str(character.id))

    def start_scene(self, scene: Scene) -> None:
        """Start a new scene.

        Args:
            scene: The scene to start.

        Raises:
            InvalidGameStateError: If cannot start scene in current state.
        """
        if self._state == GameState.COMBAT:
            raise InvalidGameStateError(
                "Cannot start new scene during combat",
                current_state=self._state.value,
                expected_states=["idle", "exploration", "social"],
            )

        self._current_scene = scene

        # Set state based on scene type
        from dnd_manager.models.campaign import SceneType

        state_map = {
            SceneType.COMBAT: GameState.COMBAT,
            SceneType.EXPLORATION: GameState.EXPLORATION,
            SceneType.SOCIAL: GameState.SOCIAL,
            SceneType.REST: GameState.REST,
        }
        self._state = state_map.get(scene.scene_type, GameState.EXPLORATION)

        logger.info(
            "Scene started",
            scene_id=str(scene.id),
            scene_type=scene.scene_type,
            state=self._state,
        )

        # Emit scene start event
        self._emit_event(GameEvent("scene_started", {"scene_id": str(scene.id)}))

    def end_scene(self) -> None:
        """End the current scene.

        Raises:
            InvalidGameStateError: If cannot end scene in current state.
        """
        if self._current_scene is None:
            raise InvalidGameStateError(
                "No scene is currently active",
                current_state=self._state.value,
            )

        if self._state == GameState.COMBAT and self._combat_state is not None:
            raise InvalidGameStateError(
                "Cannot end scene: combat is still active",
                current_state=self._state.value,
            )

        scene_id = self._current_scene.id
        self._current_scene.is_completed = True
        self._current_scene = None
        self._state = GameState.IDLE

        logger.info("Scene ended", scene_id=str(scene_id))
        self._emit_event(GameEvent("scene_ended", {"scene_id": str(scene_id)}))

    def start_combat(self) -> TurnManager:
        """Start a combat encounter.

        Returns:
            The TurnManager for the combat.

        Raises:
            InvalidGameStateError: If already in combat.
        """
        if self._state == GameState.COMBAT:
            raise InvalidGameStateError(
                "Already in combat",
                current_state=self._state.value,
            )

        self._state = GameState.COMBAT
        self._turn_manager = TurnManager()

        from datetime import datetime

        from dnd_manager.models.combat import CombatPhase, CombatState

        self._combat_state = CombatState(
            phase=CombatPhase.INITIATIVE,
            started_at=datetime.now(),
        )

        logger.info("Combat started")
        self._emit_event(GameEvent("combat_started"))

        return self._turn_manager

    def end_combat(self) -> None:
        """End the current combat encounter.

        Raises:
            InvalidGameStateError: If not in combat.
        """
        if self._state != GameState.COMBAT:
            raise InvalidGameStateError(
                "Not in combat",
                current_state=self._state.value,
                expected_states=["combat"],
            )

        from datetime import datetime

        from dnd_manager.models.combat import CombatPhase

        if self._combat_state:
            self._combat_state.phase = CombatPhase.ENDED
            self._combat_state.ended_at = datetime.now()

        self._state = GameState.EXPLORATION
        self._turn_manager = None

        logger.info("Combat ended")
        self._emit_event(GameEvent("combat_ended"))

    def pause(self) -> None:
        """Pause the game.

        Raises:
            InvalidGameStateError: If already paused or idle.
        """
        if self._state in (GameState.PAUSED, GameState.IDLE):
            raise InvalidGameStateError(
                f"Cannot pause from {self._state} state",
                current_state=self._state.value,
            )

        self._previous_state = self._state
        self._state = GameState.PAUSED

        logger.info("Game paused", previous_state=self._previous_state.value)

    def resume(self) -> None:
        """Resume the game from pause.

        Raises:
            InvalidGameStateError: If not paused.
        """
        if self._state != GameState.PAUSED:
            raise InvalidGameStateError(
                "Game is not paused",
                current_state=self._state.value,
                expected_states=["paused"],
            )

        self._state = self._previous_state  # type: ignore[has-type]
        logger.info("Game resumed", state=self._state.value)

    def on_event(self, event_type: str, handler: Any) -> None:
        """Register an event handler.

        Args:
            event_type: Type of event to handle.
            handler: Callback function to invoke.
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _emit_event(self, event: GameEvent) -> None:
        """Emit a game event to registered handlers.

        Args:
            event: The event to emit.
        """
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Event handler error",
                    event_type=event.event_type,
                )


__all__ = [
    "GameState",
    "GameEvent",
    "GameLoop",
]
