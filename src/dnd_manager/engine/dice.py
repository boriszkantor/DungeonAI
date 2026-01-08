"""Dice rolling mechanics for D&D 5E.

This module provides comprehensive dice rolling functionality using
the d20 library, with support for advantage, disadvantage, and
various D&D-specific modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from dnd_manager.core.exceptions import DiceRollError
from dnd_manager.core.logging import get_logger


logger = get_logger(__name__)


class RollType(StrEnum):
    """Types of dice rolls."""

    NORMAL = "normal"
    ADVANTAGE = "advantage"
    DISADVANTAGE = "disadvantage"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DiceExpression:
    """A parsed dice expression.

    Attributes:
        expression: The original dice expression string.
        total: The total result of the roll.
        dice: Individual dice results.
        modifier: Static modifier applied.
        is_critical: Whether a natural 20 was rolled.
        is_fumble: Whether a natural 1 was rolled.
        roll_type: The type of roll performed.
    """

    expression: str
    total: int
    dice: list[int]
    modifier: int
    is_critical: bool
    is_fumble: bool
    roll_type: RollType


class DiceRoller:
    """Dice rolling with D&D 5E mechanics.

    This class provides dice rolling functionality with support for
    advantage, disadvantage, critical hits, and the d20 library.

    Example:
        >>> roller = DiceRoller()
        >>> result = roller.roll("1d20+5")
        >>> print(f"Total: {result.total}")
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the dice roller.

        Args:
            seed: Optional random seed for reproducible rolls.
        """
        self._seed = seed
        if seed is not None:
            import random

            random.seed(seed)
        logger.info("DiceRoller initialized", seed=seed)

    def roll(
        self,
        expression: str,
        *,
        roll_type: RollType = RollType.NORMAL,
        extra_dice: int = 0,
    ) -> DiceExpression:
        """Roll dice according to the given expression.

        Args:
            expression: Dice expression (e.g., '1d20+5', '2d6+3').
            roll_type: Type of roll (normal, advantage, disadvantage).
            extra_dice: Additional dice to roll (e.g., for critical hits).

        Returns:
            DiceExpression containing roll results.

        Raises:
            DiceRollError: If the expression is invalid.
        """
        if not expression or not expression.strip():
            raise DiceRollError("Empty dice expression", expression=expression)

        logger.debug("Rolling dice", expression=expression, roll_type=roll_type)

        try:
            import d20

            # Handle advantage/disadvantage for d20 rolls
            modified_expression = expression
            if "d20" in expression.lower() and roll_type != RollType.NORMAL:
                if roll_type == RollType.ADVANTAGE:
                    modified_expression = expression.replace("1d20", "2d20kh1").replace(
                        "d20", "2d20kh1"
                    )
                elif roll_type == RollType.DISADVANTAGE:
                    modified_expression = expression.replace("1d20", "2d20kl1").replace(
                        "d20", "2d20kl1"
                    )

            # Parse and roll
            result: d20.RollResult = d20.roll(modified_expression)

            # Extract dice values
            dice_values = self._extract_dice_values(result.expr)

            # Calculate modifier (total - sum of dice)
            dice_sum = sum(dice_values)
            modifier = result.total - dice_sum

            # Check for critical hit/fumble on d20 rolls
            is_critical = False
            is_fumble = False
            if "d20" in expression.lower():
                # Get the actual d20 result (after advantage/disadvantage selection)
                d20_values = [d for d in dice_values if 1 <= d <= 20]
                if d20_values:
                    actual_d20 = d20_values[0]
                    is_critical = actual_d20 == 20
                    is_fumble = actual_d20 == 1

            result_expr = DiceExpression(
                expression=expression,
                total=result.total,
                dice=dice_values,
                modifier=modifier,
                is_critical=is_critical,
                is_fumble=is_fumble,
                roll_type=roll_type,
            )

            logger.info(
                "Dice rolled",
                expression=expression,
                total=result.total,
                is_critical=is_critical,
            )

            return result_expr

        except ImportError as exc:
            raise DiceRollError(
                "d20 library not installed. Install with: pip install d20",
                expression=expression,
            ) from exc
        except Exception as exc:
            raise DiceRollError(
                f"Invalid dice expression: {exc}",
                expression=expression,
            ) from exc

    def _extract_dice_values(self, expr: Any) -> list[int]:
        """Extract individual dice values from a d20 expression.

        Args:
            expr: The d20 expression tree.

        Returns:
            List of individual dice values.
        """
        import d20

        values: list[int] = []

        def traverse(node: Any) -> None:
            if isinstance(node, d20.Dice):
                for die in node.values:
                    if not die.dropped:
                        values.append(die.number)
            elif hasattr(node, "children"):
                for child in node.children:
                    traverse(child)

        traverse(expr)
        return values

    def roll_ability_check(
        self,
        modifier: int,
        *,
        roll_type: RollType = RollType.NORMAL,
    ) -> DiceExpression:
        """Roll an ability check.

        Args:
            modifier: The ability modifier to apply.
            roll_type: Type of roll (normal, advantage, disadvantage).

        Returns:
            DiceExpression containing roll results.
        """
        sign = "+" if modifier >= 0 else ""
        expression = f"1d20{sign}{modifier}"
        return self.roll(expression, roll_type=roll_type)

    def roll_attack(
        self,
        attack_bonus: int,
        *,
        roll_type: RollType = RollType.NORMAL,
    ) -> DiceExpression:
        """Roll an attack roll.

        Args:
            attack_bonus: The attack bonus to apply.
            roll_type: Type of roll (normal, advantage, disadvantage).

        Returns:
            DiceExpression containing roll results.
        """
        sign = "+" if attack_bonus >= 0 else ""
        expression = f"1d20{sign}{attack_bonus}"
        return self.roll(expression, roll_type=roll_type)

    def roll_damage(
        self,
        damage_expression: str,
        *,
        is_critical: bool = False,
        critical_rule: str = "double_dice",
    ) -> DiceExpression:
        """Roll damage.

        Args:
            damage_expression: Damage dice expression (e.g., '2d6+3').
            is_critical: Whether this is a critical hit.
            critical_rule: How to handle critical hits.

        Returns:
            DiceExpression containing damage roll results.
        """
        if is_critical and critical_rule == "double_dice":
            # Double the number of dice
            import re

            def double_dice(match: re.Match[str]) -> str:
                count = int(match.group(1) or 1) * 2
                return f"{count}d{match.group(2)}"

            damage_expression = re.sub(r"(\d*)d(\d+)", double_dice, damage_expression)

        return self.roll(damage_expression)

    def roll_saving_throw(
        self,
        modifier: int,
        *,
        roll_type: RollType = RollType.NORMAL,
    ) -> DiceExpression:
        """Roll a saving throw.

        Args:
            modifier: The saving throw modifier.
            roll_type: Type of roll (normal, advantage, disadvantage).

        Returns:
            DiceExpression containing roll results.
        """
        return self.roll_ability_check(modifier, roll_type=roll_type)

    def roll_initiative(
        self,
        dexterity_modifier: int,
        *,
        roll_type: RollType = RollType.NORMAL,
    ) -> DiceExpression:
        """Roll initiative.

        Args:
            dexterity_modifier: The dexterity modifier.
            roll_type: Type of roll (normal, advantage, disadvantage).

        Returns:
            DiceExpression containing roll results.
        """
        return self.roll_ability_check(dexterity_modifier, roll_type=roll_type)


# Module-level convenience roller
_default_roller: DiceRoller | None = None


def roll(
    expression: str,
    *,
    roll_type: RollType = RollType.NORMAL,
) -> DiceExpression:
    """Convenience function to roll dice.

    Args:
        expression: Dice expression (e.g., '1d20+5').
        roll_type: Type of roll (normal, advantage, disadvantage).

    Returns:
        DiceExpression containing roll results.

    Example:
        >>> result = roll("1d20+5")
        >>> print(result.total)
    """
    global _default_roller  # noqa: PLW0603
    if _default_roller is None:
        _default_roller = DiceRoller()
    return _default_roller.roll(expression, roll_type=roll_type)


__all__ = [
    "RollType",
    "DiceExpression",
    "DiceRoller",
    "roll",
]
