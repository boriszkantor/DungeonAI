"""Reusable UI components for the Streamlit interface.

This module provides reusable components for the D&D Campaign Manager
web interface, including combat trackers, character sheets, and
dice rolling interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from dnd_manager.core.exceptions import ComponentRenderError
from dnd_manager.core.logging import get_logger


if TYPE_CHECKING:
    from dnd_manager.models.character import Character
    from dnd_manager.models.combat import CombatState

logger = get_logger(__name__)


class BaseComponent(ABC):
    """Abstract base class for UI components.

    All UI components should inherit from this class to ensure
    consistent rendering and error handling patterns.
    """

    @abstractmethod
    def render(self) -> None:
        """Render the component.

        Raises:
            ComponentRenderError: If rendering fails.
        """
        raise NotImplementedError("Subclasses must implement render")


class DiceRollerUI(BaseComponent):
    """Dice rolling interface component.

    Provides a UI for rolling dice with various modifiers and
    advantage/disadvantage options.
    """

    def __init__(self, *, show_history: bool = True, history_limit: int = 10) -> None:
        """Initialize the dice roller UI.

        Args:
            show_history: Whether to show roll history.
            history_limit: Maximum number of history entries to display.
        """
        self.show_history = show_history
        self.history_limit = history_limit

    def render(self) -> None:
        """Render the dice roller component.

        Raises:
            ComponentRenderError: If rendering fails.
        """
        try:
            import streamlit as st

            from dnd_manager.engine.dice import DiceRoller, RollType

            st.subheader("🎲 Dice Roller")

            col1, col2 = st.columns([2, 1])

            with col1:
                expression = st.text_input(
                    "Dice Expression",
                    value="1d20",
                    placeholder="e.g., 1d20+5, 2d6+3",
                    help="Enter dice notation like 1d20, 2d6+3, etc.",
                )

            with col2:
                roll_type = st.selectbox(
                    "Roll Type",
                    options=["Normal", "Advantage", "Disadvantage"],
                )

            if st.button("🎲 Roll!", type="primary", use_container_width=True):
                roller = DiceRoller()
                roll_type_map = {
                    "Normal": RollType.NORMAL,
                    "Advantage": RollType.ADVANTAGE,
                    "Disadvantage": RollType.DISADVANTAGE,
                }
                result = roller.roll(expression, roll_type=roll_type_map[roll_type])

                # Display result
                st.markdown(f"### Result: **{result.total}**")
                st.caption(f"Dice: {result.dice} | Modifier: {result.modifier:+d}")

                if result.is_critical:
                    st.success("🎯 Critical Hit!")
                elif result.is_fumble:
                    st.error("💀 Critical Fumble!")

                # Add to history
                if "dice_history" not in st.session_state:
                    st.session_state.dice_history = []
                st.session_state.dice_history.insert(0, {
                    "expression": expression,
                    "total": result.total,
                    "dice": result.dice,
                })
                st.session_state.dice_history = (
                    st.session_state.dice_history[: self.history_limit]
                )

            # Show history
            if self.show_history and st.session_state.get("dice_history"):
                st.divider()
                st.caption("Recent Rolls")
                for roll in st.session_state.dice_history[:5]:
                    st.text(f"{roll['expression']} = {roll['total']} ({roll['dice']})")

        except ImportError as exc:
            raise ComponentRenderError(
                "Streamlit not installed"
            ) from exc
        except Exception as exc:
            raise ComponentRenderError(
                f"Failed to render dice roller: {exc}"
            ) from exc


class CharacterSheet(BaseComponent):
    """Character sheet display component.

    Provides a comprehensive view of a character's statistics,
    abilities, and equipment.
    """

    def __init__(self, character: Character) -> None:
        """Initialize the character sheet.

        Args:
            character: The character to display.
        """
        self.character = character

    def render(self) -> None:
        """Render the character sheet component.

        Raises:
            ComponentRenderError: If rendering fails.
        """
        try:
            import streamlit as st

            char = self.character

            # Header
            st.header(f"📜 {char.name}")
            st.caption(f"{char.race} | Level {char.total_level}")

            # Core stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                hp_color = "normal" if char.current_hit_points > char.max_hit_points // 2 else "off"
                st.metric(
                    "Hit Points",
                    f"{char.current_hit_points}/{char.max_hit_points}",
                    delta=char.temporary_hit_points if char.temporary_hit_points > 0 else None,
                )
            with col2:
                st.metric("Armor Class", char.armor_class)
            with col3:
                st.metric("Speed", f"{char.speed} ft")
            with col4:
                st.metric("Proficiency", f"+{char.proficiency_bonus}")

            st.divider()

            # Ability scores
            st.subheader("Ability Scores")
            abilities = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
            cols = st.columns(6)

            for i, ability in enumerate(abilities):
                score = getattr(char.stats, ability)
                modifier = (score - 10) // 2
                sign = "+" if modifier >= 0 else ""
                with cols[i]:
                    st.metric(
                        ability[:3].upper(),
                        score,
                        f"{sign}{modifier}",
                    )

            st.divider()

            # Classes
            st.subheader("Classes")
            for char_class in char.classes:
                class_info = f"**{char_class.name}** Level {char_class.level}"
                if char_class.subclass:
                    class_info += f" ({char_class.subclass})"
                st.write(class_info)

        except ImportError as exc:
            raise ComponentRenderError(
                "Streamlit not installed"
            ) from exc
        except Exception as exc:
            raise ComponentRenderError(
                f"Failed to render character sheet: {exc}"
            ) from exc


class CombatTracker(BaseComponent):
    """Combat tracking interface component.

    Provides initiative tracking, turn management, and combat
    status displays.
    """

    def __init__(self, combat_state: CombatState | None = None) -> None:
        """Initialize the combat tracker.

        Args:
            combat_state: Optional initial combat state.
        """
        self.combat_state = combat_state

    def render(self) -> None:
        """Render the combat tracker component.

        Raises:
            ComponentRenderError: If rendering fails.
        """
        try:
            import streamlit as st

            st.subheader("⚔️ Combat Tracker")

            if self.combat_state is None:
                st.info("No active combat. Start an encounter to begin.")
                if st.button("🗡️ Start Combat", type="primary"):
                    st.session_state.show_combat_setup = True
                return

            # Round indicator
            st.markdown(f"### Round {self.combat_state.round_number}")
            st.progress(
                (self.combat_state.turn_index + 1) / max(len(self.combat_state.combatants), 1)
            )

            st.divider()

            # Initiative order
            for i, combatant in enumerate(self.combat_state.combatants):
                is_current = i == self.combat_state.turn_index

                # Status indicators
                if combatant.is_dead:
                    status = "💀"
                elif not combatant.is_conscious:
                    status = "😵"
                elif combatant.conditions:
                    status = "⚠️"
                else:
                    status = "🟢"

                # Build display
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])

                with col1:
                    if is_current:
                        st.markdown("**▶️**")
                    else:
                        st.write(status)

                with col2:
                    name_style = "**" if is_current else ""
                    st.write(f"{name_style}{combatant.name}{name_style}")

                with col3:
                    hp_pct = combatant.current_hp / combatant.max_hp
                    if hp_pct > 0.5:
                        hp_color = "🟢"
                    elif hp_pct > 0.25:
                        hp_color = "🟡"
                    else:
                        hp_color = "🔴"
                    st.write(f"{hp_color} {combatant.current_hp}/{combatant.max_hp}")

                with col4:
                    st.write(f"AC: {combatant.armor_class}")

                # Show conditions if any
                if combatant.conditions:
                    st.caption(f"Conditions: {', '.join(c.value for c in combatant.conditions)}")

            st.divider()

            # Combat controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⏭️ Next Turn", use_container_width=True):
                    st.session_state.advance_turn = True
            with col2:
                if st.button("➕ Add Combatant", use_container_width=True):
                    st.session_state.show_add_combatant = True
            with col3:
                if st.button("🏁 End Combat", use_container_width=True, type="secondary"):
                    st.session_state.end_combat = True

        except ImportError as exc:
            raise ComponentRenderError(
                "Streamlit not installed"
            ) from exc
        except Exception as exc:
            raise ComponentRenderError(
                f"Failed to render combat tracker: {exc}"
            ) from exc


class InitiativeRoller(BaseComponent):
    """Initiative rolling component for combat setup."""

    def __init__(self, characters: list[Any] | None = None) -> None:
        """Initialize the initiative roller.

        Args:
            characters: List of characters to roll initiative for.
        """
        self.characters = characters or []

    def render(self) -> None:
        """Render the initiative roller component.

        Raises:
            ComponentRenderError: If rendering fails.
        """
        try:
            import streamlit as st

            st.subheader("🎲 Roll Initiative")

            if not self.characters:
                st.warning("No characters available. Add characters first.")
                return

            # Roll options
            roll_all = st.checkbox("Roll for all NPCs automatically", value=True)

            st.divider()

            initiatives: dict[str, int] = {}

            for char in self.characters:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{char.name}**")
                with col2:
                    if char.is_npc and roll_all:
                        st.write("Auto")
                    else:
                        initiatives[str(char.id)] = st.number_input(
                            f"Initiative for {char.name}",
                            min_value=1,
                            max_value=30,
                            value=10,
                            label_visibility="collapsed",
                            key=f"init_{char.id}",
                        )

            st.divider()

            if st.button("⚔️ Start Combat", type="primary", use_container_width=True):
                st.session_state.combat_initiatives = initiatives
                st.session_state.start_combat_confirmed = True

        except ImportError as exc:
            raise ComponentRenderError("Streamlit not installed") from exc
        except Exception as exc:
            raise ComponentRenderError(
                f"Failed to render initiative roller: {exc}"
            ) from exc


__all__ = [
    "BaseComponent",
    "DiceRollerUI",
    "CharacterSheet",
    "CombatTracker",
    "InitiativeRoller",
]
