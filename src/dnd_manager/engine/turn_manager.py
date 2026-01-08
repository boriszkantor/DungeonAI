"""Turn and initiative management for combat encounters.

This module provides functionality for tracking initiative order
and managing combat turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from dnd_manager.core.exceptions import CombatError, TurnManagementError
from dnd_manager.core.logging import get_logger
from dnd_manager.engine.dice import DiceRoller, RollType


if TYPE_CHECKING:
    from dnd_manager.models.combat import Combatant

logger = get_logger(__name__)


@dataclass
class InitiativeEntry:
    """An entry in the initiative tracker.

    Attributes:
        combatant_id: ID of the combatant.
        name: Display name of the combatant.
        roll: Initiative roll result.
        dexterity_modifier: DEX modifier for tiebreaking.
        is_active: Whether this combatant is still active.
    """

    combatant_id: UUID
    name: str
    roll: int
    dexterity_modifier: int = 0
    is_active: bool = True


class InitiativeTracker:
    """Track and manage initiative order.

    This class handles initiative rolls, ordering, and turn progression
    for combat encounters.
    """

    def __init__(self) -> None:
        """Initialize the initiative tracker."""
        self._entries: list[InitiativeEntry] = []
        self._current_index: int = 0
        self._round: int = 0
        self._dice_roller = DiceRoller()
        logger.info("InitiativeTracker initialized")

    @property
    def current_round(self) -> int:
        """Get the current combat round.

        Returns:
            Current round number (0 if combat hasn't started).
        """
        return self._round

    @property
    def current_combatant(self) -> InitiativeEntry | None:
        """Get the current combatant's initiative entry.

        Returns:
            Current InitiativeEntry or None if no combatants.
        """
        active = [e for e in self._entries if e.is_active]
        if not active:
            return None
        return active[self._current_index % len(active)]

    @property
    def initiative_order(self) -> list[InitiativeEntry]:
        """Get the current initiative order.

        Returns:
            List of active initiative entries in order.
        """
        return [e for e in self._entries if e.is_active]

    def roll_initiative(
        self,
        combatant_id: UUID,
        name: str,
        dexterity_modifier: int,
        *,
        roll_type: RollType = RollType.NORMAL,
        fixed_roll: int | None = None,
    ) -> InitiativeEntry:
        """Roll initiative for a combatant.

        Args:
            combatant_id: ID of the combatant.
            name: Display name of the combatant.
            dexterity_modifier: DEX modifier for the roll.
            roll_type: Type of roll (advantage, disadvantage, etc.).
            fixed_roll: Optional fixed roll result (for NPCs with fixed initiative).

        Returns:
            The created InitiativeEntry.
        """
        if fixed_roll is not None:
            roll_result = fixed_roll
        else:
            dice_result = self._dice_roller.roll_initiative(
                dexterity_modifier, roll_type=roll_type
            )
            roll_result = dice_result.total

        entry = InitiativeEntry(
            combatant_id=combatant_id,
            name=name,
            roll=roll_result,
            dexterity_modifier=dexterity_modifier,
        )
        self._entries.append(entry)
        self._sort_initiative()

        logger.info(
            "Initiative rolled",
            combatant=name,
            roll=roll_result,
            dex_mod=dexterity_modifier,
        )

        return entry

    def _sort_initiative(self) -> None:
        """Sort entries by initiative (highest first, DEX as tiebreaker)."""
        self._entries.sort(
            key=lambda e: (e.roll, e.dexterity_modifier),
            reverse=True,
        )

    def start_combat(self) -> InitiativeEntry | None:
        """Start combat and begin round 1.

        Returns:
            The first combatant's entry, or None if no combatants.

        Raises:
            TurnManagementError: If no combatants in initiative.
        """
        if not self._entries:
            raise TurnManagementError(
                "Cannot start combat: no combatants in initiative"
            )

        self._round = 1
        self._current_index = 0

        logger.info("Combat started", round=self._round)

        return self.current_combatant

    def next_turn(self) -> InitiativeEntry | None:
        """Advance to the next turn.

        Returns:
            The next combatant's entry, or None if combat ended.

        Raises:
            TurnManagementError: If combat hasn't started.
        """
        if self._round == 0:
            raise TurnManagementError("Combat hasn't started yet")

        active = [e for e in self._entries if e.is_active]
        if not active:
            return None

        self._current_index += 1
        if self._current_index >= len(active):
            self._current_index = 0
            self._round += 1
            logger.info("New round started", round=self._round)

        current = self.current_combatant
        if current:
            logger.info("Next turn", combatant=current.name, round=self._round)

        return current

    def remove_combatant(self, combatant_id: UUID) -> None:
        """Remove a combatant from initiative.

        Args:
            combatant_id: ID of the combatant to remove.

        Raises:
            TurnManagementError: If combatant not found.
        """
        for entry in self._entries:
            if entry.combatant_id == combatant_id:
                entry.is_active = False
                logger.info("Combatant removed from initiative", combatant=entry.name)
                return

        raise TurnManagementError(
            f"Combatant not found in initiative: {combatant_id}",
            details={"combatant_id": str(combatant_id)},
        )

    def delay_turn(self, combatant_id: UUID, new_position: int) -> None:
        """Delay a combatant's turn to a new position.

        Args:
            combatant_id: ID of the combatant delaying.
            new_position: New position in initiative (0-based).

        Raises:
            TurnManagementError: If combatant not found or invalid position.
        """
        active = [e for e in self._entries if e.is_active]
        if new_position < 0 or new_position >= len(active):
            raise TurnManagementError(
                f"Invalid initiative position: {new_position}",
                details={"max_position": len(active) - 1},
            )

        # Find and remove the entry
        entry = None
        for i, e in enumerate(self._entries):
            if e.combatant_id == combatant_id and e.is_active:
                entry = e
                self._entries.pop(i)
                break

        if entry is None:
            raise TurnManagementError(
                f"Combatant not found: {combatant_id}",
                details={"combatant_id": str(combatant_id)},
            )

        # Find the target position in the full list
        active_count = 0
        insert_index = 0
        for i, e in enumerate(self._entries):
            if e.is_active:
                if active_count == new_position:
                    insert_index = i
                    break
                active_count += 1
        else:
            insert_index = len(self._entries)

        self._entries.insert(insert_index, entry)
        logger.info("Turn delayed", combatant=entry.name, new_position=new_position)

    def reset(self) -> None:
        """Reset the initiative tracker for a new encounter."""
        self._entries.clear()
        self._current_index = 0
        self._round = 0
        logger.info("Initiative tracker reset")


@dataclass
class TurnState:
    """State of the current turn.

    Attributes:
        combatant_id: ID of the combatant whose turn it is.
        has_action: Whether the action is available.
        has_bonus_action: Whether the bonus action is available.
        has_reaction: Whether the reaction is available.
        has_movement: Whether movement is available.
        movement_remaining: Remaining movement in feet.
        concentration_check_pending: Whether a concentration check is needed.
    """

    combatant_id: UUID
    has_action: bool = True
    has_bonus_action: bool = True
    has_reaction: bool = True
    has_movement: bool = True
    movement_remaining: int = 30
    concentration_check_pending: bool = False


class TurnManager:
    """Manage combat turns and actions.

    This class tracks action economy and turn state for each combatant.
    """

    def __init__(self) -> None:
        """Initialize the turn manager."""
        self._initiative_tracker = InitiativeTracker()
        self._turn_states: dict[UUID, TurnState] = {}
        self._combatants: dict[UUID, Combatant] = {}
        logger.info("TurnManager initialized")

    @property
    def initiative_tracker(self) -> InitiativeTracker:
        """Get the initiative tracker.

        Returns:
            The InitiativeTracker instance.
        """
        return self._initiative_tracker

    @property
    def current_turn(self) -> TurnState | None:
        """Get the current turn state.

        Returns:
            Current TurnState or None if no active turn.
        """
        current = self._initiative_tracker.current_combatant
        if current is None:
            return None
        return self._turn_states.get(current.combatant_id)

    def add_combatant(
        self,
        combatant: Combatant,
        *,
        roll_type: RollType = RollType.NORMAL,
        fixed_initiative: int | None = None,
    ) -> InitiativeEntry:
        """Add a combatant to the encounter.

        Args:
            combatant: The combatant to add.
            roll_type: Type of initiative roll.
            fixed_initiative: Optional fixed initiative value.

        Returns:
            The created InitiativeEntry.
        """
        # Store combatant reference
        self._combatants[combatant.id] = combatant

        # Initialize turn state (will be reset at turn start)
        self._turn_states[combatant.id] = TurnState(
            combatant_id=combatant.id,
            movement_remaining=30,  # Default speed
        )

        # Roll initiative (assume DEX mod of 0 if not available)
        dex_mod = 0  # Would normally get from character stats
        return self._initiative_tracker.roll_initiative(
            combatant.id,
            combatant.name,
            dex_mod,
            roll_type=roll_type,
            fixed_roll=fixed_initiative,
        )

    def start_combat(self) -> TurnState | None:
        """Start combat and initialize the first turn.

        Returns:
            The first combatant's turn state.

        Raises:
            CombatError: If combat cannot be started.
        """
        try:
            first = self._initiative_tracker.start_combat()
            if first:
                self._reset_turn_state(first.combatant_id)
                return self._turn_states[first.combatant_id]
            return None
        except TurnManagementError as exc:
            raise CombatError(str(exc)) from exc

    def end_turn(self) -> TurnState | None:
        """End the current turn and start the next.

        Returns:
            The next combatant's turn state, or None if combat ended.
        """
        next_combatant = self._initiative_tracker.next_turn()
        if next_combatant:
            self._reset_turn_state(next_combatant.combatant_id)
            return self._turn_states[next_combatant.combatant_id]
        return None

    def _reset_turn_state(self, combatant_id: UUID) -> None:
        """Reset turn state for a new turn.

        Args:
            combatant_id: ID of the combatant.
        """
        combatant = self._combatants.get(combatant_id)
        speed = combatant.speed if combatant else 30

        self._turn_states[combatant_id] = TurnState(
            combatant_id=combatant_id,
            movement_remaining=speed,
        )

    def use_action(self, combatant_id: UUID) -> None:
        """Mark the action as used.

        Args:
            combatant_id: ID of the combatant.

        Raises:
            CombatError: If action is not available.
        """
        state = self._turn_states.get(combatant_id)
        if state is None:
            raise CombatError(
                "Combatant not in combat",
                combatant_id=str(combatant_id),
            )
        if not state.has_action:
            raise CombatError(
                "Action already used this turn",
                combatant_id=str(combatant_id),
            )
        state.has_action = False
        logger.debug("Action used", combatant_id=str(combatant_id))

    def use_bonus_action(self, combatant_id: UUID) -> None:
        """Mark the bonus action as used.

        Args:
            combatant_id: ID of the combatant.

        Raises:
            CombatError: If bonus action is not available.
        """
        state = self._turn_states.get(combatant_id)
        if state is None:
            raise CombatError(
                "Combatant not in combat",
                combatant_id=str(combatant_id),
            )
        if not state.has_bonus_action:
            raise CombatError(
                "Bonus action already used this turn",
                combatant_id=str(combatant_id),
            )
        state.has_bonus_action = False
        logger.debug("Bonus action used", combatant_id=str(combatant_id))

    def use_reaction(self, combatant_id: UUID) -> None:
        """Mark the reaction as used.

        Args:
            combatant_id: ID of the combatant.

        Raises:
            CombatError: If reaction is not available.
        """
        state = self._turn_states.get(combatant_id)
        if state is None:
            raise CombatError(
                "Combatant not in combat",
                combatant_id=str(combatant_id),
            )
        if not state.has_reaction:
            raise CombatError(
                "Reaction already used this round",
                combatant_id=str(combatant_id),
            )
        state.has_reaction = False
        logger.debug("Reaction used", combatant_id=str(combatant_id))

    def use_movement(self, combatant_id: UUID, feet: int) -> int:
        """Use movement.

        Args:
            combatant_id: ID of the combatant.
            feet: Amount of movement to use in feet.

        Returns:
            Remaining movement after this use.

        Raises:
            CombatError: If not enough movement remaining.
        """
        state = self._turn_states.get(combatant_id)
        if state is None:
            raise CombatError(
                "Combatant not in combat",
                combatant_id=str(combatant_id),
            )
        if feet > state.movement_remaining:
            raise CombatError(
                f"Not enough movement remaining ({state.movement_remaining} feet)",
                combatant_id=str(combatant_id),
            )
        state.movement_remaining -= feet
        logger.debug(
            "Movement used",
            combatant_id=str(combatant_id),
            feet=feet,
            remaining=state.movement_remaining,
        )
        return state.movement_remaining


__all__ = [
    "InitiativeEntry",
    "InitiativeTracker",
    "TurnState",
    "TurnManager",
]
