"""Game loop for turn-based combat management.

This module implements the core game loop that processes combat turns,
handling the branching between player-controlled and AI-controlled
combatants.

The GameLoop is the central coordinator for:
- Turn order management
- AI decision execution for autonomous combatants
- User input requests for player-controlled characters
- State transitions and combat flow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

from dnd_manager.core.exceptions import CombatError, GameEngineError, InvalidGameStateError
from dnd_manager.core.logging import get_logger
from dnd_manager.engine.ai_agent import AgentDecision, DungeonAgent, get_ai_action
from dnd_manager.engine.tools import ToolCall, ToolResult, execute_tool_calls
from dnd_manager.models import AutonomyLevel


if TYPE_CHECKING:
    from dnd_manager.models import Combatant
    from dnd_manager.models.game_state import GameSession, InitiativeEntry, Scene, TurnOrder

logger = get_logger(__name__)


# =============================================================================
# Turn Status
# =============================================================================


class TurnStatus(StrEnum):
    """Status of turn processing."""

    WAITING_FOR_USER = "waiting_for_user"
    """Turn belongs to a player-controlled character, awaiting input."""

    AI_THINKING = "ai_thinking"
    """AI is deciding on an action."""

    AI_EXECUTING = "ai_executing"
    """AI is executing its decided action."""

    TURN_COMPLETED = "turn_completed"
    """Turn has been fully processed."""

    COMBAT_ENDED = "combat_ended"
    """Combat has ended (victory, defeat, or other condition)."""

    NO_COMBAT = "no_combat"
    """No active combat in the scene."""

    ERROR = "error"
    """An error occurred during turn processing."""


@dataclass
class TurnResult:
    """Result of processing a turn.

    Attributes:
        status: The turn status.
        combatant_name: Name of the acting combatant.
        combatant_uid: UID of the acting combatant.
        autonomy: Autonomy level of the combatant.
        ai_decision: AI decision if applicable.
        tool_results: Results of executed tools.
        message: Human-readable status message.
        error: Error message if status is ERROR.
        round_number: Current combat round.
        turn_index: Current turn index in initiative.
    """

    status: TurnStatus
    combatant_name: str = ""
    combatant_uid: UUID | None = None
    autonomy: AutonomyLevel = AutonomyLevel.NONE
    ai_decision: AgentDecision | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    message: str = ""
    error: str = ""
    round_number: int = 0
    turn_index: int = 0


# =============================================================================
# Combat End Conditions
# =============================================================================


@dataclass
class CombatEndCondition:
    """Condition for ending combat.

    Attributes:
        name: Condition name.
        check: Function that checks if condition is met.
        message: Message to display when condition triggers.
    """

    name: str
    check: Callable[["Scene"], bool]
    message: str


def all_enemies_defeated(scene: "Scene") -> bool:
    """Check if all enemies are defeated."""
    for combatant in scene.monsters:
        if combatant.health.is_conscious and not combatant.health.is_dead:
            return False
    return len(scene.monsters) > 0


def all_players_defeated(scene: "Scene") -> bool:
    """Check if all players are defeated."""
    for combatant in scene.player_characters:
        if combatant.health.is_conscious and not combatant.health.is_dead:
            return False
    return len(scene.player_characters) > 0


DEFAULT_END_CONDITIONS = [
    CombatEndCondition(
        name="victory",
        check=all_enemies_defeated,
        message="🎉 Victory! All enemies have been defeated!",
    ),
    CombatEndCondition(
        name="defeat",
        check=all_players_defeated,
        message="💀 Defeat! All player characters have fallen!",
    ),
]


# =============================================================================
# Game Loop
# =============================================================================


class GameLoop:
    """Core game loop for processing combat turns.

    The GameLoop manages the flow of combat, determining when to
    request user input vs. execute AI decisions, and maintaining
    turn order progression.

    Attributes:
        scene: The current combat scene.
        ai_agent: AI agent for autonomous decisions.
        end_conditions: Conditions that end combat.
        turn_callbacks: Callbacks invoked during turn processing.
    """

    def __init__(
        self,
        scene: "Scene",
        *,
        ai_model: str = "google/gemini-2.0-flash-001",
        end_conditions: list[CombatEndCondition] | None = None,
    ) -> None:
        """Initialize the game loop.

        Args:
            scene: The combat scene to manage.
            ai_model: LLM model for AI decisions.
            end_conditions: Custom combat end conditions.
        """
        self._scene = scene
        self._ai_agent = DungeonAgent(model=ai_model)
        self._end_conditions = end_conditions or DEFAULT_END_CONDITIONS
        self._turn_callbacks: list[Callable[[TurnResult], None]] = []
        self._action_log: list[dict[str, Any]] = []

        logger.info(
            "GameLoop initialized",
            scene=scene.name,
            combatants=len(scene.combatants),
        )

    @property
    def scene(self) -> "Scene":
        """Get the current scene (read-only during decision phase)."""
        return self._scene

    @property
    def turn_order(self) -> "TurnOrder":
        """Get the current turn order."""
        return self._scene.turn_order

    @property
    def current_round(self) -> int:
        """Get the current combat round."""
        return self._scene.turn_order.current_round

    @property
    def is_combat_active(self) -> bool:
        """Check if combat is currently active."""
        return self._scene.turn_order.is_combat_active

    @property
    def action_log(self) -> list[dict[str, Any]]:
        """Get the log of actions taken."""
        return self._action_log.copy()

    def add_turn_callback(self, callback: Callable[[TurnResult], None]) -> None:
        """Add a callback to be invoked after each turn.

        Args:
            callback: Function to call with TurnResult.
        """
        self._turn_callbacks.append(callback)

    def _get_current_combatant(self) -> "Combatant | None":
        """Get the combatant whose turn it is.

        Returns:
            Current combatant or None if no active turn.
        """
        current_uid = self.turn_order.current_combatant_uid
        if current_uid is None:
            return None

        return self._scene.get_combatant_by_uid(current_uid)

    def _check_end_conditions(self) -> CombatEndCondition | None:
        """Check if any combat end condition is met.

        Returns:
            The triggered condition or None.
        """
        for condition in self._end_conditions:
            if condition.check(self._scene):
                return condition
        return None

    def _log_action(
        self,
        combatant_name: str,
        action_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log an action to the action log.

        Args:
            combatant_name: Name of the acting combatant.
            action_type: Type of action taken.
            details: Additional action details.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": self.current_round,
            "combatant": combatant_name,
            "action_type": action_type,
            **details,
        }
        self._action_log.append(entry)

    def _invoke_callbacks(self, result: TurnResult) -> None:
        """Invoke all registered turn callbacks.

        Args:
            result: The turn result to pass to callbacks.
        """
        for callback in self._turn_callbacks:
            try:
                callback(result)
            except Exception:
                logger.exception("Turn callback failed")

    def peek_turn(self) -> TurnResult:
        """Peek at the current turn without processing it.

        Returns information about whose turn it is and what kind
        of input is expected.

        Returns:
            TurnResult with status indicating what's needed.
        """
        # Check if combat is active
        if not self.is_combat_active:
            return TurnResult(
                status=TurnStatus.NO_COMBAT,
                message="No active combat. Start combat to begin processing turns.",
            )

        # Check end conditions
        end_condition = self._check_end_conditions()
        if end_condition:
            return TurnResult(
                status=TurnStatus.COMBAT_ENDED,
                message=end_condition.message,
                round_number=self.current_round,
            )

        # Get current combatant
        combatant = self._get_current_combatant()
        if combatant is None:
            return TurnResult(
                status=TurnStatus.ERROR,
                error="Could not find current combatant",
            )

        # Check if combatant can act
        if not combatant.health.is_conscious:
            # Skip unconscious combatants (will be handled in process_turn)
            return TurnResult(
                status=TurnStatus.TURN_COMPLETED,
                combatant_name=combatant.name,
                combatant_uid=combatant.uid,
                message=f"{combatant.name} is unconscious and cannot act.",
                round_number=self.current_round,
                turn_index=self.turn_order.current_index,
            )

        # Determine if AI or user controlled
        autonomy = combatant.persona.autonomy

        if autonomy == AutonomyLevel.NONE:
            return TurnResult(
                status=TurnStatus.WAITING_FOR_USER,
                combatant_name=combatant.name,
                combatant_uid=combatant.uid,
                autonomy=autonomy,
                message=f"Waiting for {combatant.name}'s action (player-controlled).",
                round_number=self.current_round,
                turn_index=self.turn_order.current_index,
            )
        elif autonomy == AutonomyLevel.SUGGESTIVE:
            return TurnResult(
                status=TurnStatus.WAITING_FOR_USER,
                combatant_name=combatant.name,
                combatant_uid=combatant.uid,
                autonomy=autonomy,
                message=f"Waiting for {combatant.name}'s action (AI will suggest).",
                round_number=self.current_round,
                turn_index=self.turn_order.current_index,
            )
        else:
            # ASSISTED or FULL_AUTO - AI will act
            return TurnResult(
                status=TurnStatus.AI_THINKING,
                combatant_name=combatant.name,
                combatant_uid=combatant.uid,
                autonomy=autonomy,
                message=f"{combatant.name} is deciding on an action (AI-controlled).",
                round_number=self.current_round,
                turn_index=self.turn_order.current_index,
            )

    def process_turn(self) -> TurnResult:
        """Process the current turn.

        This is the main entry point for turn processing. It:
        1. Checks if combat is active and end conditions
        2. Determines if current combatant is AI or player controlled
        3. For AI: Gets decision and executes tools
        4. For player: Returns WAITING_FOR_USER status
        5. Advances turn order after completion

        Returns:
            TurnResult with status and any action results.
        """
        # First peek to check state
        peek_result = self.peek_turn()

        if peek_result.status in (
            TurnStatus.NO_COMBAT,
            TurnStatus.COMBAT_ENDED,
            TurnStatus.ERROR,
        ):
            self._invoke_callbacks(peek_result)
            return peek_result

        combatant = self._get_current_combatant()
        if combatant is None:
            result = TurnResult(
                status=TurnStatus.ERROR,
                error="Current combatant not found",
            )
            self._invoke_callbacks(result)
            return result

        # Handle unconscious combatants - skip their turn
        if not combatant.health.is_conscious:
            logger.info(f"{combatant.name} is unconscious, skipping turn")

            # Advance turn order
            self._scene = Scene(
                **{
                    **self._scene.model_dump(),
                    "turn_order": self.turn_order.advance_turn(),
                }
            )

            result = TurnResult(
                status=TurnStatus.TURN_COMPLETED,
                combatant_name=combatant.name,
                combatant_uid=combatant.uid,
                message=f"{combatant.name} is unconscious and cannot act. Turn skipped.",
                round_number=self.current_round,
                turn_index=self.turn_order.current_index,
            )
            self._invoke_callbacks(result)
            return result

        autonomy = combatant.persona.autonomy

        # Branch based on autonomy level
        if autonomy == AutonomyLevel.NONE:
            # Player controlled - return and wait for user input
            result = TurnResult(
                status=TurnStatus.WAITING_FOR_USER,
                combatant_name=combatant.name,
                combatant_uid=combatant.uid,
                autonomy=autonomy,
                message=f"⏳ Waiting for {combatant.name}'s action...",
                round_number=self.current_round,
                turn_index=self.turn_order.current_index,
            )
            self._invoke_callbacks(result)
            return result

        elif autonomy == AutonomyLevel.SUGGESTIVE:
            # AI suggests but player decides
            try:
                decision = self._ai_agent.decide_action(combatant, self._scene)

                result = TurnResult(
                    status=TurnStatus.WAITING_FOR_USER,
                    combatant_name=combatant.name,
                    combatant_uid=combatant.uid,
                    autonomy=autonomy,
                    ai_decision=decision,
                    message=f"💡 AI suggests for {combatant.name}: {decision.reasoning}",
                    round_number=self.current_round,
                    turn_index=self.turn_order.current_index,
                )
                self._invoke_callbacks(result)
                return result

            except Exception as exc:
                logger.exception("AI suggestion failed")
                result = TurnResult(
                    status=TurnStatus.WAITING_FOR_USER,
                    combatant_name=combatant.name,
                    combatant_uid=combatant.uid,
                    autonomy=autonomy,
                    message=f"⏳ Waiting for {combatant.name}'s action (AI suggestion unavailable).",
                    error=str(exc),
                    round_number=self.current_round,
                    turn_index=self.turn_order.current_index,
                )
                self._invoke_callbacks(result)
                return result

        else:
            # ASSISTED or FULL_AUTO - AI takes action
            logger.info(
                f"AI taking action for {combatant.name}",
                autonomy=autonomy.name,
            )

            try:
                # Get AI decision
                decision = self._ai_agent.decide_action(combatant, self._scene)

                # Execute the decision
                tool_results = self._ai_agent.execute_decision(decision)

                # Log the action
                self._log_action(
                    combatant_name=combatant.name,
                    action_type="ai_action",
                    details={
                        "reasoning": decision.reasoning,
                        "tool_calls": [
                            {"name": tc.tool_name, "args": tc.arguments}
                            for tc in decision.tool_calls
                        ],
                        "results": [
                            {"tool": r.tool_name, "success": r.success, "result": r.result}
                            for r in tool_results
                        ],
                    },
                )

                # Build result message
                result_messages = []
                if decision.reasoning:
                    result_messages.append(f"💭 {combatant.name}: {decision.reasoning}")
                for tr in tool_results:
                    if tr.success:
                        result_messages.append(tr.result)
                    else:
                        result_messages.append(f"❌ {tr.tool_name} failed: {tr.error}")

                # Advance turn order
                self._scene = Scene(
                    **{
                        **self._scene.model_dump(),
                        "turn_order": self.turn_order.advance_turn(),
                    }
                )

                result = TurnResult(
                    status=TurnStatus.TURN_COMPLETED,
                    combatant_name=combatant.name,
                    combatant_uid=combatant.uid,
                    autonomy=autonomy,
                    ai_decision=decision,
                    tool_results=tool_results,
                    message="\n".join(result_messages),
                    round_number=self.current_round,
                    turn_index=self.turn_order.current_index,
                )
                self._invoke_callbacks(result)
                return result

            except Exception as exc:
                logger.exception("AI action failed")

                # On failure, skip turn for FULL_AUTO, wait for user for ASSISTED
                if autonomy == AutonomyLevel.FULL_AUTO:
                    self._scene = Scene(
                        **{
                            **self._scene.model_dump(),
                            "turn_order": self.turn_order.advance_turn(),
                        }
                    )

                    result = TurnResult(
                        status=TurnStatus.TURN_COMPLETED,
                        combatant_name=combatant.name,
                        combatant_uid=combatant.uid,
                        autonomy=autonomy,
                        message=f"⚠️ {combatant.name}'s AI action failed. Turn skipped.",
                        error=str(exc),
                        round_number=self.current_round,
                        turn_index=self.turn_order.current_index,
                    )
                else:
                    result = TurnResult(
                        status=TurnStatus.WAITING_FOR_USER,
                        combatant_name=combatant.name,
                        combatant_uid=combatant.uid,
                        autonomy=autonomy,
                        message=f"⚠️ AI action failed for {combatant.name}. Manual input required.",
                        error=str(exc),
                        round_number=self.current_round,
                        turn_index=self.turn_order.current_index,
                    )

                self._invoke_callbacks(result)
                return result

    def submit_user_action(
        self,
        tool_calls: list[ToolCall],
    ) -> TurnResult:
        """Submit a user's action for processing.

        Call this when a player-controlled combatant has decided
        on their action.

        Args:
            tool_calls: The tool calls representing the user's action.

        Returns:
            TurnResult with execution results.

        Raises:
            InvalidGameStateError: If not waiting for user input.
        """
        # Verify we're waiting for user input
        peek = self.peek_turn()
        if peek.status != TurnStatus.WAITING_FOR_USER:
            raise InvalidGameStateError(
                f"Not waiting for user input. Current status: {peek.status}",
                current_state=peek.status.value,
                expected_states=["waiting_for_user"],
            )

        combatant = self._get_current_combatant()
        if combatant is None:
            raise InvalidGameStateError("No current combatant found")

        # Execute the user's tool calls
        tool_results = execute_tool_calls(tool_calls)

        # Log the action
        self._log_action(
            combatant_name=combatant.name,
            action_type="user_action",
            details={
                "tool_calls": [
                    {"name": tc.tool_name, "args": tc.arguments}
                    for tc in tool_calls
                ],
                "results": [
                    {"tool": r.tool_name, "success": r.success, "result": r.result}
                    for r in tool_results
                ],
            },
        )

        # Build result message
        result_messages = []
        for tr in tool_results:
            if tr.success:
                result_messages.append(tr.result)
            else:
                result_messages.append(f"❌ {tr.tool_name} failed: {tr.error}")

        # Advance turn order
        self._scene = Scene(
            **{
                **self._scene.model_dump(),
                "turn_order": self.turn_order.advance_turn(),
            }
        )

        result = TurnResult(
            status=TurnStatus.TURN_COMPLETED,
            combatant_name=combatant.name,
            combatant_uid=combatant.uid,
            autonomy=combatant.persona.autonomy,
            tool_results=tool_results,
            message="\n".join(result_messages),
            round_number=self.current_round,
            turn_index=self.turn_order.current_index,
        )
        self._invoke_callbacks(result)
        return result

    def get_ai_suggestion(self) -> AgentDecision | None:
        """Get an AI suggestion for the current player's turn.

        Useful for SUGGESTIVE autonomy or when player wants help.

        Returns:
            AgentDecision with suggested action, or None on failure.
        """
        combatant = self._get_current_combatant()
        if combatant is None:
            return None

        try:
            return self._ai_agent.decide_action(combatant, self._scene)
        except Exception:
            logger.exception("Failed to get AI suggestion")
            return None

    def end_combat(self, reason: str = "Combat ended") -> TurnResult:
        """Manually end the current combat.

        Args:
            reason: Reason for ending combat.

        Returns:
            TurnResult with COMBAT_ENDED status.
        """
        # Update scene to end combat
        from dnd_manager.models.game_state import CombatPhase

        self._scene = self._scene.complete_scene()

        result = TurnResult(
            status=TurnStatus.COMBAT_ENDED,
            message=reason,
            round_number=self.current_round,
        )
        self._invoke_callbacks(result)
        return result

    def run_until_user_input(self) -> TurnResult:
        """Run the game loop until user input is required.

        Processes all AI turns automatically until:
        - A player-controlled combatant's turn
        - Combat ends
        - An error occurs

        Returns:
            TurnResult requiring user attention.
        """
        while True:
            result = self.process_turn()

            if result.status in (
                TurnStatus.WAITING_FOR_USER,
                TurnStatus.COMBAT_ENDED,
                TurnStatus.NO_COMBAT,
                TurnStatus.ERROR,
            ):
                return result

            # Continue processing AI turns
            logger.debug(
                "Continuing to next turn",
                last_combatant=result.combatant_name,
                status=result.status,
            )


__all__ = [
    "TurnStatus",
    "TurnResult",
    "CombatEndCondition",
    "GameLoop",
    "all_enemies_defeated",
    "all_players_defeated",
    "DEFAULT_END_CONDITIONS",
]
