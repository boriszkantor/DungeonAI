"""Dice Rolling - PYTHON TRUTH (NOT LLM).

NEURO-SYMBOLIC PRINCIPLE:
This module implements the ONLY way dice are rolled. LLMs cannot
generate random numbers or override this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dnd_manager.core.exceptions import DiceRollError


# =============================================================================
# Dice Result
# =============================================================================


@dataclass
class DiceResult:
    """Result of a dice roll.
    
    CRITICAL: These values come from the d20 library,
    NEVER from LLM output.
    """

    expression: str
    total: int
    details: str
    is_critical: bool = False
    is_fumble: bool = False
    natural_roll: int | None = None


# =============================================================================
# Dice Rolling Functions
# =============================================================================


def roll_dice(expression: str) -> DiceResult:
    """Roll dice using the d20 library.
    
    NEURO-SYMBOLIC PRINCIPLE:
    This is the ONLY way dice are rolled. LLMs cannot
    generate random numbers or override this.
    
    Args:
        expression: Dice expression (e.g., "1d20+5", "2d6+3").
        
    Returns:
        DiceResult with true random outcome.
    """
    try:
        import d20
    except ImportError as exc:
        raise DiceRollError(
            "d20 library not installed",
            expression=expression,
        ) from exc

    try:
        result = d20.roll(expression)

        # Check for natural 20/1 on d20 rolls
        is_crit = False
        is_fumble = False
        natural = None

        if "d20" in expression.lower():
            # Extract the natural d20 roll
            def find_d20(node: Any) -> int | None:
                if isinstance(node, d20.Dice):
                    if node.num == 1 and node.size == 20:
                        for die in node.values:
                            # Use 'kept' attribute (True if not dropped)
                            if getattr(die, 'kept', True):
                                return die.number
                # Handle the case where it's a single Die object
                if hasattr(node, 'number') and hasattr(node, 'size'):
                    if node.size == 20:
                        return node.number
                if hasattr(node, "children"):
                    for child in node.children:
                        val = find_d20(child)
                        if val is not None:
                            return val
                return None

            natural = find_d20(result.expr)
            if natural == 20:
                is_crit = True
            elif natural == 1:
                is_fumble = True

        return DiceResult(
            expression=expression,
            total=result.total,
            details=str(result),
            is_critical=is_crit,
            is_fumble=is_fumble,
            natural_roll=natural,
        )

    except Exception as exc:
        raise DiceRollError(
            f"Dice roll failed: {exc}",
            expression=expression,
        ) from exc


def roll_check(
    modifier: int,
    dc: int,
    advantage: bool = False,
    disadvantage: bool = False,
) -> tuple[DiceResult, bool]:
    """Roll an ability check or skill check.
    
    Args:
        modifier: Total modifier to add.
        dc: Difficulty class.
        advantage: Roll with advantage.
        disadvantage: Roll with disadvantage.
        
    Returns:
        Tuple of (DiceResult, success).
    """
    if advantage and not disadvantage:
        expr = f"2d20kh1+{modifier}"
    elif disadvantage and not advantage:
        expr = f"2d20kl1+{modifier}"
    else:
        expr = f"1d20+{modifier}"

    result = roll_dice(expr)
    success = result.total >= dc

    # Natural 20 on ability checks doesn't auto-succeed in RAW,
    # but we track it for flavor
    return result, success


def roll_save(
    modifier: int,
    dc: int,
    advantage: bool = False,
    disadvantage: bool = False,
) -> tuple[DiceResult, bool]:
    """Roll a saving throw.
    
    Args:
        modifier: Save modifier.
        dc: Save DC.
        advantage: Roll with advantage.
        disadvantage: Roll with disadvantage.
        
    Returns:
        Tuple of (DiceResult, success).
    """
    if advantage and not disadvantage:
        expr = f"2d20kh1+{modifier}"
    elif disadvantage and not advantage:
        expr = f"2d20kl1+{modifier}"
    else:
        expr = f"1d20+{modifier}"

    result = roll_dice(expr)

    # Natural 1 always fails, natural 20 always succeeds for saves
    if result.is_fumble:
        success = False
    elif result.is_critical:
        success = True
    else:
        success = result.total >= dc

    return result, success


def roll_attack(
    attack_bonus: int,
    target_ac: int,
    advantage: bool = False,
    disadvantage: bool = False,
) -> tuple[DiceResult, bool, bool]:
    """Roll an attack.
    
    Args:
        attack_bonus: Attack modifier.
        target_ac: Target's AC.
        advantage: Roll with advantage.
        disadvantage: Roll with disadvantage.
        
    Returns:
        Tuple of (DiceResult, hit, is_critical_hit).
    """
    if advantage and not disadvantage:
        expr = f"2d20kh1+{attack_bonus}"
    elif disadvantage and not advantage:
        expr = f"2d20kl1+{attack_bonus}"
    else:
        expr = f"1d20+{attack_bonus}"

    result = roll_dice(expr)

    # Critical hit/fumble
    if result.is_fumble:
        hit = False
        is_crit = False
    elif result.is_critical:
        hit = True
        is_crit = True
    else:
        hit = result.total >= target_ac
        is_crit = False

    return result, hit, is_crit


def roll_damage(
    damage_expression: str,
    critical: bool = False,
) -> DiceResult:
    """Roll damage.
    
    Args:
        damage_expression: Damage dice (e.g., "2d6+3").
        critical: Double the dice for critical hit.
        
    Returns:
        DiceResult with damage total.
    """
    if critical:
        # Double the dice portion
        match = re.match(r"(\d+)d(\d+)(.*)", damage_expression)
        if match:
            num_dice = int(match.group(1)) * 2
            die_size = match.group(2)
            rest = match.group(3)
            damage_expression = f"{num_dice}d{die_size}{rest}"

    return roll_dice(damage_expression)
