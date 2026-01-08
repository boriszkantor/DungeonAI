"""Game action tools for AI agents and manual execution.

This module defines the tools (actions) available to combatants during
their turns. Tools are decorated for AI agent binding and return
formatted strings describing outcomes.

All probability resolution uses the d20 library for authentic
D&D 5E dice mechanics.

Tools:
    roll_attack: Make an attack roll against a target AC
    roll_damage: Roll damage dice
    cast_spell: Cast a spell with saving throw or attack
    check_skill: Make a skill check against a DC
    check_saving_throw: Make a saving throw
    move: Move to a new position
    dash: Use action to double movement
    dodge: Take the Dodge action
    disengage: Disengage from combat
    help: Help an ally
    hide: Attempt to hide
    use_item: Use an item
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, TypeVar

from dnd_manager.core.exceptions import DiceRollError, GameEngineError
from dnd_manager.core.logging import get_logger


logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Tool Registry
# =============================================================================


class ToolCategory(StrEnum):
    """Categories of game tools."""

    ATTACK = "attack"
    SPELL = "spell"
    SKILL = "skill"
    MOVEMENT = "movement"
    ACTION = "action"
    BONUS_ACTION = "bonus_action"
    REACTION = "reaction"


@dataclass
class ToolDefinition:
    """Definition of a game tool for AI binding.

    Attributes:
        name: Tool function name.
        description: Human-readable description for AI.
        category: Tool category.
        parameters: JSON schema for parameters.
        function: The actual function to call.
        action_cost: Type of action required (action, bonus_action, etc.)
    """

    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]
    function: Callable[..., str]
    action_cost: str = "action"


# Global tool registry
_tool_registry: dict[str, ToolDefinition] = {}


def tool(
    *,
    description: str,
    category: ToolCategory,
    action_cost: str = "action",
) -> Callable[[F], F]:
    """Decorator to register a function as a game tool.

    Args:
        description: Description for AI agents.
        category: Tool category.
        action_cost: Type of action required.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        # Extract parameter info from function annotations
        import inspect
        sig = inspect.signature(func)
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue

            param_schema: dict[str, Any] = {}

            # Determine type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == str:
                    param_schema["type"] = "string"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                else:
                    param_schema["type"] = "string"
            else:
                param_schema["type"] = "string"

            # Add description from docstring if available
            param_schema["description"] = f"Parameter: {param_name}"

            parameters["properties"][param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        # Register tool
        tool_def = ToolDefinition(
            name=func.__name__,
            description=description,
            category=category,
            parameters=parameters,
            function=func,
            action_cost=action_cost,
        )
        _tool_registry[func.__name__] = tool_def

        # Add metadata to function
        func._tool_definition = tool_def  # type: ignore[attr-defined]

        return func

    return decorator


def get_tool(name: str) -> ToolDefinition | None:
    """Get a tool definition by name."""
    return _tool_registry.get(name)


def get_all_tools() -> list[ToolDefinition]:
    """Get all registered tools."""
    return list(_tool_registry.values())


def get_tools_by_category(category: ToolCategory) -> list[ToolDefinition]:
    """Get tools filtered by category."""
    return [t for t in _tool_registry.values() if t.category == category]


def get_tools_as_openai_schema() -> list[dict[str, Any]]:
    """Get all tools in OpenAI function calling schema format."""
    tools = []
    for tool_def in _tool_registry.values():
        tools.append({
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.parameters,
            },
        })
    return tools


# =============================================================================
# Dice Rolling Utilities
# =============================================================================


def _roll_d20(expression: str) -> tuple[int, str, bool, bool]:
    """Roll dice using the d20 library.

    Args:
        expression: Dice expression (e.g., "1d20+5").

    Returns:
        Tuple of (total, description, is_crit, is_fumble).

    Raises:
        DiceRollError: If roll fails.
    """
    try:
        import d20

        result = d20.roll(expression)

        # Check for critical/fumble on d20 rolls
        is_crit = False
        is_fumble = False

        if "d20" in expression.lower():
            # Find the d20 die result
            def find_d20(node: Any) -> int | None:
                if isinstance(node, d20.Dice):
                    if node.num == 1 and node.size == 20:
                        for die in node.values:
                            if not die.dropped:
                                return die.number
                if hasattr(node, "children"):
                    for child in node.children:
                        val = find_d20(child)
                        if val is not None:
                            return val
                return None

            d20_val = find_d20(result.expr)
            if d20_val == 20:
                is_crit = True
            elif d20_val == 1:
                is_fumble = True

        return result.total, str(result), is_crit, is_fumble

    except ImportError as exc:
        raise DiceRollError(
            "d20 library not installed. Install with: pip install d20",
            expression=expression,
        ) from exc
    except Exception as exc:
        raise DiceRollError(
            f"Dice roll failed: {exc}",
            expression=expression,
        ) from exc


# =============================================================================
# Combat Tools
# =============================================================================


@tool(
    description="Make a melee or ranged attack roll against a target. Returns whether the attack hits and damage if applicable.",
    category=ToolCategory.ATTACK,
    action_cost="action",
)
def roll_attack(
    attack_bonus: int,
    target_ac: int,
    damage_dice: str = "1d8",
    damage_bonus: int = 0,
    advantage: bool = False,
    disadvantage: bool = False,
) -> str:
    """Make an attack roll against a target.

    Args:
        attack_bonus: Total attack bonus (ability mod + proficiency + other).
        target_ac: Target's armor class.
        damage_dice: Damage dice expression (e.g., "1d8", "2d6").
        damage_bonus: Bonus to damage.
        advantage: Whether attacker has advantage.
        disadvantage: Whether attacker has disadvantage.

    Returns:
        Formatted string describing the attack result.
    """
    # Determine roll expression
    if advantage and not disadvantage:
        roll_expr = f"2d20kh1+{attack_bonus}"
        roll_type = "with advantage"
    elif disadvantage and not advantage:
        roll_expr = f"2d20kl1+{attack_bonus}"
        roll_type = "with disadvantage"
    else:
        roll_expr = f"1d20+{attack_bonus}"
        roll_type = ""

    total, roll_str, is_crit, is_fumble = _roll_d20(roll_expr)

    # Determine hit/miss
    if is_fumble:
        result = f"🎲 Attack Roll {roll_type}: {roll_str} = **{total}** vs AC {target_ac}\n"
        result += "💀 **CRITICAL MISS!** The attack automatically fails."
        logger.info("Attack critical miss", roll=total, target_ac=target_ac)
        return result

    if is_crit:
        # Critical hit - double damage dice
        crit_damage_dice = damage_dice.replace("d", "d").split("d")
        if len(crit_damage_dice) == 2:
            num_dice = int(crit_damage_dice[0]) if crit_damage_dice[0] else 1
            crit_expr = f"{num_dice * 2}d{crit_damage_dice[1]}+{damage_bonus}"
        else:
            crit_expr = f"{damage_dice}+{damage_dice}+{damage_bonus}"

        damage_total, damage_str, _, _ = _roll_d20(crit_expr)

        result = f"🎲 Attack Roll {roll_type}: {roll_str} = **{total}** vs AC {target_ac}\n"
        result += f"⚔️ **CRITICAL HIT!** Damage: {damage_str} = **{damage_total}** damage!"
        logger.info("Attack critical hit", damage=damage_total)
        return result

    if total >= target_ac:
        # Normal hit
        damage_expr = f"{damage_dice}+{damage_bonus}" if damage_bonus else damage_dice
        damage_total, damage_str, _, _ = _roll_d20(damage_expr)

        result = f"🎲 Attack Roll {roll_type}: {roll_str} = **{total}** vs AC {target_ac}\n"
        result += f"✅ **HIT!** Damage: {damage_str} = **{damage_total}** damage."
        logger.info("Attack hit", roll=total, damage=damage_total)
        return result
    else:
        result = f"🎲 Attack Roll {roll_type}: {roll_str} = **{total}** vs AC {target_ac}\n"
        result += "❌ **MISS!** The attack fails to connect."
        logger.info("Attack miss", roll=total, target_ac=target_ac)
        return result


@tool(
    description="Roll damage dice for an attack or effect. Use after confirming a hit.",
    category=ToolCategory.ATTACK,
    action_cost="free",
)
def roll_damage(
    damage_dice: str,
    damage_bonus: int = 0,
    damage_type: str = "slashing",
    is_critical: bool = False,
) -> str:
    """Roll damage for an attack.

    Args:
        damage_dice: Damage dice expression (e.g., "2d6").
        damage_bonus: Flat bonus to damage.
        damage_type: Type of damage (slashing, fire, etc.).
        is_critical: Whether this is critical hit damage (doubles dice).

    Returns:
        Formatted damage result string.
    """
    if is_critical:
        # Double the dice
        parts = damage_dice.split("d")
        if len(parts) == 2:
            num = int(parts[0]) if parts[0] else 1
            damage_dice = f"{num * 2}d{parts[1]}"

    expr = f"{damage_dice}+{damage_bonus}" if damage_bonus else damage_dice
    total, roll_str, _, _ = _roll_d20(expr)

    crit_note = " (CRITICAL)" if is_critical else ""
    result = f"💥 Damage{crit_note}: {roll_str} = **{total}** {damage_type} damage"

    logger.info("Damage rolled", total=total, type=damage_type, critical=is_critical)
    return result


# =============================================================================
# Spell Tools
# =============================================================================


@tool(
    description="Cast a spell. Handles attack rolls or saving throws as appropriate.",
    category=ToolCategory.SPELL,
    action_cost="action",
)
def cast_spell(
    spell_name: str,
    spell_level: int,
    is_attack: bool = False,
    spell_attack_bonus: int = 0,
    target_ac: int = 0,
    requires_save: bool = False,
    save_dc: int = 0,
    save_ability: str = "DEX",
    target_save_bonus: int = 0,
    damage_dice: str = "",
    damage_type: str = "magical",
    effect_description: str = "",
) -> str:
    """Cast a spell with full resolution.

    Args:
        spell_name: Name of the spell.
        spell_level: Level the spell is being cast at.
        is_attack: Whether the spell requires an attack roll.
        spell_attack_bonus: Spell attack modifier.
        target_ac: Target AC (if attack spell).
        requires_save: Whether targets must make a saving throw.
        save_dc: Spell save DC.
        save_ability: Ability for the save (DEX, WIS, etc.).
        target_save_bonus: Target's save bonus.
        damage_dice: Damage dice if applicable.
        damage_type: Type of damage.
        effect_description: Description of non-damage effects.

    Returns:
        Formatted spell casting result.
    """
    level_suffix = {1: "st", 2: "nd", 3: "rd"}.get(spell_level, "th")
    result = f"✨ **{spell_name}** ({spell_level}{level_suffix} level)\n"

    if is_attack:
        # Spell attack roll
        total, roll_str, is_crit, is_fumble = _roll_d20(f"1d20+{spell_attack_bonus}")

        result += f"🎲 Spell Attack: {roll_str} = **{total}** vs AC {target_ac}\n"

        if is_fumble:
            result += "💀 **CRITICAL MISS!** The spell goes wide."
            return result

        if is_crit or total >= target_ac:
            hit_type = "CRITICAL HIT" if is_crit else "HIT"
            result += f"✅ **{hit_type}!**\n"

            if damage_dice:
                if is_crit:
                    parts = damage_dice.split("d")
                    if len(parts) == 2:
                        num = int(parts[0]) if parts[0] else 1
                        crit_dice = f"{num * 2}d{parts[1]}"
                    else:
                        crit_dice = damage_dice
                    dmg_total, dmg_str, _, _ = _roll_d20(crit_dice)
                else:
                    dmg_total, dmg_str, _, _ = _roll_d20(damage_dice)
                result += f"💥 Damage: {dmg_str} = **{dmg_total}** {damage_type} damage"
        else:
            result += "❌ **MISS!** The spell fails to hit."

    elif requires_save:
        # Saving throw
        save_total, save_str, save_crit, save_fumble = _roll_d20(f"1d20+{target_save_bonus}")

        result += f"🎲 Target {save_ability} Save: {save_str} = **{save_total}** vs DC {save_dc}\n"

        if save_fumble or save_total < save_dc:
            result += "❌ **FAILED SAVE!**\n"
            if damage_dice:
                dmg_total, dmg_str, _, _ = _roll_d20(damage_dice)
                result += f"💥 Full Damage: {dmg_str} = **{dmg_total}** {damage_type} damage"
            if effect_description:
                result += f"\n📋 Effect: {effect_description}"
        else:
            result += "✅ **SUCCESSFUL SAVE!**\n"
            if damage_dice:
                # Half damage on save (typical)
                dmg_total, dmg_str, _, _ = _roll_d20(damage_dice)
                half_dmg = dmg_total // 2
                result += f"💥 Half Damage: {dmg_str} ÷ 2 = **{half_dmg}** {damage_type} damage"
    else:
        # No attack or save - just describe effect
        if effect_description:
            result += f"📋 Effect: {effect_description}"
        else:
            result += "📋 The spell takes effect."

    logger.info("Spell cast", spell=spell_name, level=spell_level)
    return result


# =============================================================================
# Skill Tools
# =============================================================================


@tool(
    description="Make a skill check against a difficulty class (DC). Returns success/failure and margin.",
    category=ToolCategory.SKILL,
    action_cost="action",
)
def check_skill(
    skill_name: str,
    skill_bonus: int,
    dc: int,
    advantage: bool = False,
    disadvantage: bool = False,
) -> str:
    """Make a skill check.

    Args:
        skill_name: Name of the skill (e.g., "Perception", "Stealth").
        skill_bonus: Total skill modifier.
        dc: Difficulty class to beat.
        advantage: Whether to roll with advantage.
        disadvantage: Whether to roll with disadvantage.

    Returns:
        Formatted skill check result.
    """
    if advantage and not disadvantage:
        roll_expr = f"2d20kh1+{skill_bonus}"
        roll_type = "with advantage"
    elif disadvantage and not advantage:
        roll_expr = f"2d20kl1+{skill_bonus}"
        roll_type = "with disadvantage"
    else:
        roll_expr = f"1d20+{skill_bonus}"
        roll_type = ""

    total, roll_str, is_crit, is_fumble = _roll_d20(roll_expr)

    result = f"🎲 {skill_name} Check {roll_type}: {roll_str} = **{total}** vs DC {dc}\n"

    if is_crit:
        result += "🌟 **NATURAL 20!** "
    elif is_fumble:
        result += "💀 **NATURAL 1!** "

    if total >= dc:
        margin = total - dc
        result += f"✅ **SUCCESS** (by {margin})"
    else:
        margin = dc - total
        result += f"❌ **FAILURE** (by {margin})"

    logger.info("Skill check", skill=skill_name, total=total, dc=dc, success=total >= dc)
    return result


@tool(
    description="Make a saving throw against an effect. Returns success/failure.",
    category=ToolCategory.SKILL,
    action_cost="reaction",
)
def check_saving_throw(
    ability: str,
    save_bonus: int,
    dc: int,
    advantage: bool = False,
    disadvantage: bool = False,
) -> str:
    """Make a saving throw.

    Args:
        ability: Ability for the save (STR, DEX, CON, INT, WIS, CHA).
        save_bonus: Total saving throw modifier.
        dc: Difficulty class to beat.
        advantage: Whether to roll with advantage.
        disadvantage: Whether to roll with disadvantage.

    Returns:
        Formatted saving throw result.
    """
    if advantage and not disadvantage:
        roll_expr = f"2d20kh1+{save_bonus}"
        roll_type = "with advantage"
    elif disadvantage and not advantage:
        roll_expr = f"2d20kl1+{save_bonus}"
        roll_type = "with disadvantage"
    else:
        roll_expr = f"1d20+{save_bonus}"
        roll_type = ""

    total, roll_str, is_crit, is_fumble = _roll_d20(roll_expr)

    result = f"🎲 {ability} Saving Throw {roll_type}: {roll_str} = **{total}** vs DC {dc}\n"

    if is_crit:
        result += "🌟 **NATURAL 20!** Automatic success!\n"
        result += "✅ **SAVE SUCCESSFUL**"
        return result
    elif is_fumble:
        result += "💀 **NATURAL 1!** Automatic failure!\n"
        result += "❌ **SAVE FAILED**"
        return result

    if total >= dc:
        result += "✅ **SAVE SUCCESSFUL**"
    else:
        result += "❌ **SAVE FAILED**"

    logger.info("Saving throw", ability=ability, total=total, dc=dc, success=total >= dc)
    return result


# =============================================================================
# Movement Tools
# =============================================================================


@tool(
    description="Move up to your movement speed to a new position. Specify direction or target location.",
    category=ToolCategory.MOVEMENT,
    action_cost="movement",
)
def move(
    direction: str,
    distance_feet: int,
    movement_remaining: int,
) -> str:
    """Move to a new position.

    Args:
        direction: Direction or target (e.g., "north", "toward goblin", "away from dragon").
        distance_feet: Distance to move in feet.
        movement_remaining: Movement speed remaining this turn.

    Returns:
        Movement result description.
    """
    if distance_feet > movement_remaining:
        return f"❌ Cannot move {distance_feet} ft. Only {movement_remaining} ft movement remaining."

    remaining = movement_remaining - distance_feet
    result = f"🏃 Moved {distance_feet} ft {direction}. ({remaining} ft movement remaining)"

    logger.info("Movement", direction=direction, distance=distance_feet)
    return result


@tool(
    description="Take the Dash action to double your movement speed this turn.",
    category=ToolCategory.ACTION,
    action_cost="action",
)
def dash(current_speed: int) -> str:
    """Take the Dash action.

    Args:
        current_speed: Base movement speed.

    Returns:
        Dash action result.
    """
    result = f"🏃💨 **DASH!** Gained {current_speed} ft additional movement this turn."
    logger.info("Dash action", extra_movement=current_speed)
    return result


@tool(
    description="Take the Dodge action. Attacks against you have disadvantage until your next turn.",
    category=ToolCategory.ACTION,
    action_cost="action",
)
def dodge() -> str:
    """Take the Dodge action.

    Returns:
        Dodge action result.
    """
    result = "🛡️ **DODGE!** Until your next turn:\n"
    result += "• Attack rolls against you have disadvantage\n"
    result += "• You have advantage on DEX saving throws"
    logger.info("Dodge action")
    return result


@tool(
    description="Take the Disengage action. Your movement doesn't provoke opportunity attacks this turn.",
    category=ToolCategory.ACTION,
    action_cost="action",
)
def disengage() -> str:
    """Take the Disengage action.

    Returns:
        Disengage action result.
    """
    result = "🏃🛡️ **DISENGAGE!** Your movement doesn't provoke opportunity attacks this turn."
    logger.info("Disengage action")
    return result


@tool(
    description="Take the Help action to assist an ally. Give advantage on their next ability check or attack.",
    category=ToolCategory.ACTION,
    action_cost="action",
)
def help_action(ally_name: str, help_type: str = "attack") -> str:
    """Take the Help action.

    Args:
        ally_name: Name of the ally to help.
        help_type: Type of help ("attack" or "ability").

    Returns:
        Help action result.
    """
    if help_type == "attack":
        result = f"🤝 **HELP!** {ally_name} has advantage on their next attack roll against a target within 5 ft of you."
    else:
        result = f"🤝 **HELP!** {ally_name} has advantage on their next ability check to perform the task you're helping with."

    logger.info("Help action", ally=ally_name, type=help_type)
    return result


@tool(
    description="Attempt to Hide. Make a Stealth check to become hidden from enemies.",
    category=ToolCategory.ACTION,
    action_cost="action",
)
def hide(stealth_bonus: int, observer_perception: int = 10) -> str:
    """Attempt to hide.

    Args:
        stealth_bonus: Your Stealth skill modifier.
        observer_perception: Passive Perception of observers.

    Returns:
        Hide attempt result.
    """
    total, roll_str, _, is_fumble = _roll_d20(f"1d20+{stealth_bonus}")

    result = f"🫥 Stealth Check: {roll_str} = **{total}** vs Passive Perception {observer_perception}\n"

    if is_fumble:
        result += "💀 **NATURAL 1!** You make a lot of noise and are definitely noticed!"
    elif total >= observer_perception:
        result += "✅ **HIDDEN!** You are now hidden from enemies."
    else:
        result += "❌ **SPOTTED!** Your attempt to hide fails."

    logger.info("Hide attempt", stealth=total, perception=observer_perception)
    return result


@tool(
    description="Use an item from your inventory (potion, scroll, etc.).",
    category=ToolCategory.ACTION,
    action_cost="action",
)
def use_item(item_name: str, effect_description: str = "") -> str:
    """Use an item.

    Args:
        item_name: Name of the item to use.
        effect_description: Description of the item's effect.

    Returns:
        Item use result.
    """
    result = f"🎒 Used **{item_name}**"
    if effect_description:
        result += f"\n📋 Effect: {effect_description}"

    logger.info("Item used", item=item_name)
    return result


# =============================================================================
# Tool Execution
# =============================================================================


@dataclass
class ToolCall:
    """A request to execute a tool.

    Attributes:
        tool_name: Name of the tool to call.
        arguments: Arguments to pass to the tool.
        call_id: Unique identifier for this call.
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    """Result of executing a tool.

    Attributes:
        tool_name: Name of the tool that was called.
        call_id: ID of the original call.
        result: The tool's output string.
        success: Whether execution succeeded.
        error: Error message if failed.
    """

    tool_name: str
    call_id: str
    result: str
    success: bool = True
    error: str = ""


def execute_tool(call: ToolCall) -> ToolResult:
    """Execute a tool call.

    Args:
        call: The tool call to execute.

    Returns:
        ToolResult with outcome.
    """
    tool_def = get_tool(call.tool_name)

    if tool_def is None:
        return ToolResult(
            tool_name=call.tool_name,
            call_id=call.call_id,
            result="",
            success=False,
            error=f"Unknown tool: {call.tool_name}",
        )

    try:
        result = tool_def.function(**call.arguments)
        return ToolResult(
            tool_name=call.tool_name,
            call_id=call.call_id,
            result=result,
            success=True,
        )
    except Exception as exc:
        logger.exception("Tool execution failed", tool=call.tool_name)
        return ToolResult(
            tool_name=call.tool_name,
            call_id=call.call_id,
            result="",
            success=False,
            error=str(exc),
        )


def execute_tool_calls(calls: list[ToolCall]) -> list[ToolResult]:
    """Execute multiple tool calls in sequence.

    Args:
        calls: List of tool calls.

    Returns:
        List of results.
    """
    return [execute_tool(call) for call in calls]


__all__ = [
    # Types
    "ToolCategory",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    # Registry
    "tool",
    "get_tool",
    "get_all_tools",
    "get_tools_by_category",
    "get_tools_as_openai_schema",
    # Execution
    "execute_tool",
    "execute_tool_calls",
    # Tools
    "roll_attack",
    "roll_damage",
    "cast_spell",
    "check_skill",
    "check_saving_throw",
    "move",
    "dash",
    "dodge",
    "disengage",
    "help_action",
    "hide",
    "use_item",
]
