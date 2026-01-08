"""DM Orchestrator - The Neuro-Symbolic Game Loop.

This module implements the core game loop pattern:
1. INPUT: User action → RAG retrieval → Context assembly
2. REASONING: LLM decides what checks/rolls are needed
3. TOOL USE: Python executes dice rolls (d20 library)
4. RESOLUTION: LLM generates narrative → Emits state updates

NEURO-SYMBOLIC PRINCIPLE:
- Python owns TRUTH (GameState, dice rolls, rule validation)
- LLMs handle INTERFACE (narrative, reasoning, context)
- LLMs CANNOT invent dice results or modify state directly
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from dnd_manager.core.config import get_settings
from dnd_manager.core.exceptions import DiceRollError, GameEngineError
from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.universal_loader import (
    ChunkedDocument,
    ChromaStore,
    DocumentType,
    OpenRouterClient,
)
from dnd_manager.models.ecs import (
    ActorEntity,
    Condition,
    DamageType,
    GameState,
    StateUpdateRequest,
    StateUpdateResult,
)


logger = get_logger(__name__)


# =============================================================================
# Dice Rolling (PYTHON TRUTH - NOT LLM)
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


# =============================================================================
# DM Tools (Called by LLM, Executed by Python)
# =============================================================================


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    success: bool
    result: str
    data: dict[str, Any] = Field(default_factory=dict)


class DMTool:
    """A tool that the DM can invoke.
    
    Tools are defined by Python and called by the LLM.
    The LLM provides arguments, Python executes.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., ToolResult],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            return self.handler(**kwargs)
        except Exception as exc:
            logger.exception(f"Tool {self.name} failed")
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=f"Error: {exc}",
            )


def create_dm_tools(
    game_state: GameState,
    chroma_store: ChromaStore | None = None,
) -> list[DMTool]:
    """Create the DM's toolkit.
    
    These tools allow the LLM to:
    - Roll dice (using d20 library)
    - Query game state
    - Request state updates
    - Create entities from RAG data
    
    CRITICAL: LLMs call these tools, Python executes them.
    """

    def handle_roll_dice(expression: str) -> ToolResult:
        """Roll dice using the expression provided."""
        result = roll_dice(expression)
        
        # Build result string with crit/fumble indicators
        result_str = f"🎲 {result.details} = **{result.total}**"
        if result.is_critical:
            result_str += " ⭐ **NATURAL 20!**"
        elif result.is_fumble:
            result_str += " 💀 **NATURAL 1!**"
        
        return ToolResult(
            tool_name="roll_dice",
            success=True,
            result=result_str,
            data={
                "total": result.total,
                "expression": expression,
                "is_critical": result.is_critical,
                "is_fumble": result.is_fumble,
                "natural_roll": result.natural_roll,
            },
        )

    def handle_roll_check(
        character_name: str,
        skill_or_ability: str,
        dc: int,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> ToolResult:
        """Roll a skill check or ability check for a character."""
        # Find the character
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break

        if actor is None:
            return ToolResult(
                tool_name="roll_check",
                success=False,
                result=f"Character '{character_name}' not found",
            )

        # Get modifier (simplified - would need skill mapping)
        ability_map = {
            "strength": actor.stats.str_mod,
            "dexterity": actor.stats.dex_mod,
            "constitution": actor.stats.con_mod,
            "intelligence": actor.stats.int_mod,
            "wisdom": actor.stats.wis_mod,
            "charisma": actor.stats.cha_mod,
        }

        skill_ability_map = {
            "athletics": "strength",
            "acrobatics": "dexterity",
            "stealth": "dexterity",
            "perception": "wisdom",
            "investigation": "intelligence",
            "persuasion": "charisma",
            "intimidation": "charisma",
            "deception": "charisma",
        }

        skill_lower = skill_or_ability.lower()
        if skill_lower in ability_map:
            modifier = ability_map[skill_lower]
        elif skill_lower in skill_ability_map:
            base_ability = skill_ability_map[skill_lower]
            modifier = ability_map[base_ability]
            # Add proficiency if applicable
            prof_mult = actor.stats.skill_proficiencies.get(skill_lower, 0)
            modifier += actor.stats.proficiency_bonus * prof_mult
        else:
            modifier = 0

        result, success = roll_check(modifier, dc, advantage, disadvantage)

        outcome = "✅ SUCCESS" if success else "❌ FAILURE"
        crit_note = ""
        if result.is_critical:
            crit_note = " (Natural 20!)"
        elif result.is_fumble:
            crit_note = " (Natural 1!)"

        return ToolResult(
            tool_name="roll_check",
            success=True,
            result=f"🎲 {actor.name} {skill_or_ability} check: {result.details} vs DC {dc} → {outcome}{crit_note}",
            data={
                "total": result.total,
                "dc": dc,
                "success": success,
                "character": character_name,
                "skill": skill_or_ability,
            },
        )

    def handle_roll_attack(
        attacker_name: str,
        target_name: str,
        attack_bonus: int,
        damage_dice: str,
        damage_type: str = "slashing",
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> ToolResult:
        """Roll an attack from one character to another."""
        # Find attacker and target
        attacker = None
        target = None
        for a in game_state.actors.values():
            if a.name.lower() == attacker_name.lower():
                attacker = a
            if a.name.lower() == target_name.lower():
                target = a

        if attacker is None:
            return ToolResult(
                tool_name="roll_attack",
                success=False,
                result=f"Attacker '{attacker_name}' not found",
            )
        if target is None:
            return ToolResult(
                tool_name="roll_attack",
                success=False,
                result=f"Target '{target_name}' not found",
            )

        # Roll attack
        attack_result, hit, is_crit = roll_attack(
            attack_bonus, target.ac, advantage, disadvantage
        )

        result_lines = [
            f"⚔️ **{attacker.name}** attacks **{target.name}**!",
            f"🎲 Attack: {attack_result.details} vs AC {target.ac}",
        ]

        damage_dealt = 0

        if attack_result.is_fumble:
            result_lines.append("💀 **CRITICAL MISS!** The attack fails!")
        elif is_crit:
            damage_result = roll_damage(damage_dice, critical=True)
            damage_dealt = damage_result.total
            result_lines.append(f"💥 **CRITICAL HIT!** Damage: {damage_result.details} = **{damage_dealt}** {damage_type}")
        elif hit:
            damage_result = roll_damage(damage_dice, critical=False)
            damage_dealt = damage_result.total
            result_lines.append(f"✅ **HIT!** Damage: {damage_result.details} = **{damage_dealt}** {damage_type}")
        else:
            result_lines.append("❌ **MISS!** The attack fails to connect.")

        return ToolResult(
            tool_name="roll_attack",
            success=True,
            result="\n".join(result_lines),
            data={
                "attacker": attacker_name,
                "target": target_name,
                "hit": hit,
                "critical": is_crit,
                "fumble": attack_result.is_fumble,
                "damage": damage_dealt,
                "damage_type": damage_type,
            },
        )

    def handle_roll_save(
        character_name: str,
        ability: str,
        dc: int,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> ToolResult:
        """Roll a saving throw for a character."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break

        if actor is None:
            return ToolResult(
                tool_name="roll_save",
                success=False,
                result=f"Character '{character_name}' not found",
            )

        modifier = actor.stats.get_save_bonus(ability.lower())
        result, success = roll_save(modifier, dc, advantage, disadvantage)

        outcome = "✅ SUCCESS" if success else "❌ FAILURE"

        return ToolResult(
            tool_name="roll_save",
            success=True,
            result=f"🎲 {actor.name} {ability} save: {result.details} vs DC {dc} → {outcome}",
            data={
                "total": result.total,
                "dc": dc,
                "success": success,
                "character": character_name,
            },
        )

    def handle_get_character_status(character_name: str) -> ToolResult:
        """Get the current status of a character."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break

        if actor is None:
            return ToolResult(
                tool_name="get_character_status",
                success=False,
                result=f"Character '{character_name}' not found",
            )

        status_lines = [
            f"**{actor.name}** ({actor.race} {actor.level_display})",
            f"HP: {actor.health.hp_current}/{actor.health.hp_max}",
            f"AC: {actor.ac}",
        ]

        if actor.health.conditions:
            conditions = ", ".join(str(c) for c in actor.health.conditions)
            status_lines.append(f"Conditions: {conditions}")

        if actor.spellbook and actor.spellbook.spell_slots:
            slots = []
            for level, (current, max_val) in actor.spellbook.spell_slots.items():
                slots.append(f"L{level}: {current}/{max_val}")
            status_lines.append(f"Spell Slots: {', '.join(slots)}")
        
        # Include inventory and equipment
        if actor.inventory and actor.inventory.items:
            equipped = [item.name for item in actor.inventory.items if item.equipped]
            other_items = [item.name for item in actor.inventory.items if not item.equipped]
            
            if equipped:
                status_lines.append(f"Equipped: {', '.join(equipped)}")
            if other_items:
                status_lines.append(f"Inventory: {', '.join(other_items[:10])}")  # Limit to 10

        return ToolResult(
            tool_name="get_character_status",
            success=True,
            result="\n".join(status_lines),
            data={
                "name": actor.name,
                "hp_current": actor.health.hp_current,
                "hp_max": actor.health.hp_max,
                "ac": actor.ac,
            },
        )

    def handle_apply_damage(
        character_name: str,
        amount: int,
        damage_type: str = "untyped",
    ) -> ToolResult:
        """Apply damage to a character (creates StateUpdateRequest)."""
        from dnd_manager.models.progression import get_morale_modifier
        
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break

        if actor is None:
            return ToolResult(
                tool_name="apply_damage",
                success=False,
                result=f"Character '{character_name}' not found",
            )

        # Check if this will kill the target
        was_alive = actor.health.hp_current > 0
        old_hp = actor.health.hp_current

        # Create and apply state update
        request = StateUpdateRequest(
            target_uid=actor.uid,
            update_type="damage",
            payload={"amount": amount, "damage_type": damage_type},
        )

        result = game_state.apply_update(request)
        
        result_lines = [f"💔 {actor.name} takes {amount} {damage_type} damage! (HP: {actor.health.hp_current}/{actor.health.hp_max})"]
        morale_effects = []
        
        # Check for morale effects
        actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
        
        if result.success:
            # Check if target died
            if was_alive and actor.health.hp_current <= 0:
                result_lines.append(f"💀 **{actor.name}** is slain!")
                
                # If an enemy died, reduce morale of other enemies
                if actor_type != "player":
                    morale_loss = abs(get_morale_modifier("ally_killed"))
                    for other in game_state.actors.values():
                        other_type = other.type.value if hasattr(other.type, 'value') else str(other.type)
                        if other_type != "player" and other.uid != actor.uid and not other.health.is_dead:
                            old_morale = other.morale
                            new_morale, is_broken = other.reduce_morale(morale_loss)
                            if is_broken and old_morale > 25:
                                morale_effects.append(f"💔 {other.name}'s morale BREAKS!")
                            elif new_morale <= 50 and old_morale > 50:
                                morale_effects.append(f"😰 {other.name} looks shaken...")
                    
            # Check if target is at half HP or quarter HP (morale trigger for enemies)
            elif actor_type != "player" and actor.health.hp_max > 0:
                hp_pct = actor.health.hp_current / actor.health.hp_max
                old_hp_pct = old_hp / actor.health.hp_max
                
                if hp_pct <= 0.25 and old_hp_pct > 0.25:
                    morale_loss = abs(get_morale_modifier("quarter_hp"))
                    actor.reduce_morale(morale_loss)
                    morale_effects.append(f"😨 {actor.name} is gravely wounded!")
                elif hp_pct <= 0.5 and old_hp_pct > 0.5:
                    morale_loss = abs(get_morale_modifier("half_hp"))
                    actor.reduce_morale(morale_loss)
        
        # Add morale effects to result
        if morale_effects:
            result_lines.extend(morale_effects)

        if result.success:
            return ToolResult(
                tool_name="apply_damage",
                success=True,
                result="\n".join(result_lines),
                data={
                    "character": character_name,
                    "damage": amount,
                    "hp_remaining": actor.health.hp_current,
                    "is_dead": actor.health.hp_current <= 0,
                },
            )
        else:
            return ToolResult(
                tool_name="apply_damage",
                success=False,
                result=result.message,
            )

    def handle_apply_healing(character_name: str, amount: int) -> ToolResult:
        """Heal a character."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break

        if actor is None:
            return ToolResult(
                tool_name="apply_healing",
                success=False,
                result=f"Character '{character_name}' not found",
            )

        request = StateUpdateRequest(
            target_uid=actor.uid,
            update_type="healing",
            payload={"amount": amount},
        )

        result = game_state.apply_update(request)

        if result.success:
            return ToolResult(
                tool_name="apply_healing",
                success=True,
                result=f"💚 {actor.name} heals {amount} HP! (HP: {actor.health.hp_current}/{actor.health.hp_max})",
                data={
                    "character": character_name,
                    "healed": amount,
                    "hp_current": actor.health.hp_current,
                },
            )
        else:
            return ToolResult(
                tool_name="apply_healing",
                success=False,
                result=result.message,
            )

    def handle_get_current_turn() -> ToolResult:
        """Get whose turn it currently is in combat, and the full turn order."""
        if not game_state.combat_active:
            return ToolResult(
                tool_name="get_current_turn",
                success=True,
                result="⚔️ **No active combat.** Start combat by spawning enemies with `create_entity`.",
                data={"combat_active": False},
            )
        
        current = game_state.get_current_combatant()
        
        # Build turn order display
        lines = [f"**Round {game_state.combat_round}**"]
        
        if game_state.combat_order:
            lines.append("\n**Turn Order:**")
            for i, uid in enumerate(game_state.combat_order):
                actor = game_state.actors.get(uid)
                if actor:
                    marker = "➡️ " if i == game_state.combat_turn_index else "   "
                    status = "💀" if actor.health.is_dead else ""
                    actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
                    emoji = "🧙" if actor_type == "player" else "👹"
                    lines.append(f"{marker}{emoji} {actor.name} (Init: {actor.initiative}) {status}")
        
        if current:
            lines.append(f"\n🎯 **Current Turn: {current.name}**")
        
        return ToolResult(
            tool_name="get_current_turn",
            success=True,
            result="\n".join(lines),
            data={
                "combat_active": True,
                "round": game_state.combat_round,
                "current_actor": current.name if current else None,
                "turn_index": game_state.combat_turn_index,
            },
        )

    def handle_start_combat() -> ToolResult:
        """Start combat! Rolls initiative for all entities and sets up turn order.
        
        Call this AFTER spawning enemies with create_entity.
        """
        if game_state.combat_active:
            return ToolResult(
                tool_name="start_combat",
                success=False,
                result="Combat is already active! Use end_combat first.",
            )
        
        # Get all actors (party + enemies)
        all_actors = list(game_state.actors.values())
        
        if len(all_actors) < 2:
            return ToolResult(
                tool_name="start_combat",
                success=False,
                result="Need at least 2 combatants! Use create_entity to spawn enemies first.",
            )
        
        # Roll initiative for everyone
        initiative_results: list[tuple[ActorEntity, DiceResult]] = []
        
        for actor in all_actors:
            init_roll = roll_dice(f"1d20+{actor.stats.dex_mod}")
            actor.initiative = init_roll.total
            actor.is_in_combat = True
            initiative_results.append((actor, init_roll))
        
        # Sort by initiative (descending), with DEX as tiebreaker
        initiative_results.sort(
            key=lambda x: (x[1].total, x[0].stats.dexterity),
            reverse=True
        )
        
        # Set up combat state
        game_state.combat_order = [actor.uid for actor, _ in initiative_results]
        game_state.combat_active = True
        game_state.combat_round = 1
        game_state.combat_turn_index = 0
        
        # Build result message
        lines = ["⚔️ **COMBAT BEGINS!** Roll for initiative!\n"]
        for actor, init_roll in initiative_results:
            actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
            emoji = "🧙" if actor_type == "player" else "👹"
            lines.append(f"{emoji} **{actor.name}**: {init_roll.details} = **{init_roll.total}**")
        
        lines.append("")
        first = game_state.get_current_combatant()
        if first:
            lines.append(f"🎯 **{first.name}** acts first!")
        
        return ToolResult(
            tool_name="start_combat",
            success=True,
            result="\n".join(lines),
            data={
                "round": 1,
                "turn_order": [a.name for a, _ in initiative_results],
                "current": first.name if first else None,
            },
        )

    def handle_end_combat() -> ToolResult:
        """End combat and clean up."""
        from dnd_manager.models.ecs import EffectDuration
        
        if not game_state.combat_active:
            return ToolResult(
                tool_name="end_combat",
                success=False,
                result="No combat is active!",
            )
        
        # Reset combat state
        game_state.combat_active = False
        game_state.combat_order = []
        game_state.combat_round = 0
        game_state.combat_turn_index = 0
        
        # Reset actor combat flags and expire combat-duration effects
        expired_effects_summary = []
        for actor in game_state.actors.values():
            actor.is_in_combat = False
            
            # Expire round-based effects (combat is over)
            expired = []
            remaining = []
            for effect in actor.active_effects:
                dur_type = effect.duration_type
                if hasattr(dur_type, 'value'):
                    dur_type = dur_type.value
                
                if dur_type in ("rounds", "until_start_of_next_turn", "until_end_of_next_turn"):
                    expired.append(effect)
                    # Revert AC changes
                    if effect.effect_type == "ac_bonus" and effect.value:
                        actor.defense.ac_bonus -= effect.value
                else:
                    remaining.append(effect)
            
            actor.active_effects = remaining
            if expired:
                expired_effects_summary.append(f"{actor.name}: {', '.join(e.name for e in expired)}")
        
        result_lines = ["🏁 **Combat has ended.** The dust settles..."]
        if expired_effects_summary:
            result_lines.append("\n*Expired effects:*")
            for summary in expired_effects_summary:
                result_lines.append(f"  - {summary}")
        
        return ToolResult(
            tool_name="end_combat",
            success=True,
            result="\n".join(result_lines),
            data={"combat_ended": True},
        )

    def handle_end_turn() -> ToolResult:
        """End the current combatant's turn and advance to the next."""
        if not game_state.combat_active:
            return ToolResult(
                tool_name="end_turn",
                success=False,
                result="No active combat!",
            )
        
        old_actor = game_state.get_current_combatant()
        
        # Expire effects at end of current actor's turn
        end_turn_expired = []
        if old_actor:
            expired = old_actor.expire_effects_on_turn_end(
                game_state.combat_round, 
                game_state.combat_turn_index
            )
            end_turn_expired = [e.name for e in expired]
        
        game_state.advance_turn()
        new_actor = game_state.get_current_combatant()
        
        # Expire effects at start of new actor's turn (e.g., Shield)
        start_turn_expired = []
        if new_actor:
            expired = new_actor.expire_effects_on_turn_start(
                game_state.combat_round,
                game_state.combat_turn_index
            )
            start_turn_expired = [e.name for e in expired]
        
        # Check if we wrapped to a new round
        round_msg = ""
        if game_state.combat_turn_index == 0 and old_actor != new_actor:
            round_msg = f"\n\n**⚔️ ROUND {game_state.combat_round} BEGINS!**"
        
        new_name = new_actor.name if new_actor else "unknown"
        new_type = new_actor.type.value if new_actor and hasattr(new_actor.type, 'value') else str(new_actor.type) if new_actor else "unknown"
        
        result_lines = [f"✅ **{old_actor.name if old_actor else 'Unknown'}**'s turn ends."]
        
        if end_turn_expired:
            result_lines.append(f"*{old_actor.name}: {', '.join(end_turn_expired)} expired*")
        
        if round_msg:
            result_lines.append(round_msg)
        
        result_lines.append(f"\n🎯 **{new_name}**'s turn!")
        
        if start_turn_expired:
            result_lines.append(f"*{new_name}: {', '.join(start_turn_expired)} expired*")
        
        return ToolResult(
            tool_name="end_turn",
            success=True,
            result="\n".join(result_lines),
            data={
                "previous": old_actor.name if old_actor else None,
                "current": new_name,
                "round": game_state.combat_round,
                "is_player_turn": new_type == "player",
                "expired_effects": end_turn_expired + start_turn_expired,
            },
        )

    def handle_list_combatants() -> ToolResult:
        """List all entities currently in the game state."""
        if not game_state.actors:
            return ToolResult(
                tool_name="list_combatants",
                success=True,
                result="No entities in game state. Use create_entity to spawn enemies.",
                data={"combatants": []},
            )
        
        lines = ["**Current Entities:**"]
        party = []
        enemies = []
        
        for actor in game_state.actors.values():
            status = "💀" if actor.health.is_dead else ("😵" if not actor.health.is_conscious else "")
            hp_str = f"{actor.health.hp_current}/{actor.health.hp_max}"
            
            entry = f"- {actor.name} (HP: {hp_str}, AC: {actor.ac}) {status}"
            
            actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
            if actor_type == "player":
                party.append(entry)
            else:
                enemies.append(entry)
        
        if party:
            lines.append("\n**Party:**")
            lines.extend(party)
        if enemies:
            lines.append("\n**Enemies/NPCs:**")
            lines.extend(enemies)
        
        # Add combat info if active
        if game_state.combat_active:
            current = game_state.get_current_combatant()
            lines.append(f"\n⚔️ **Combat Active** - Round {game_state.combat_round}")
            if current:
                lines.append(f"🎯 Current Turn: **{current.name}**")
        
        return ToolResult(
            tool_name="list_combatants",
            success=True,
            result="\n".join(lines),
            data={
                "combatants": [
                    {"name": a.name, "hp": a.health.hp_current, "ac": a.ac, "type": a.type.value if hasattr(a.type, 'value') else str(a.type)}
                    for a in game_state.actors.values()
                ],
                "combat_active": game_state.combat_active,
            },
        )

    def handle_remove_entity(entity_name: str) -> ToolResult:
        """Remove a dead or fleeing entity from the game state."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == entity_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="remove_entity",
                success=False,
                result=f"Entity '{entity_name}' not found",
            )
        
        # Don't allow removing player characters
        actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
        if actor_type == "player":
            return ToolResult(
                tool_name="remove_entity",
                success=False,
                result="Cannot remove player characters from the game.",
            )
        
        game_state.remove_actor(actor.uid)
        
        return ToolResult(
            tool_name="remove_entity",
            success=True,
            result=f"💨 **{actor.name}** has been removed from combat.",
            data={"removed": entity_name},
        )

    def handle_award_xp(amount: int, reason: str = "encounter") -> ToolResult:
        """Award XP to the party after defeating enemies or completing objectives.
        
        The XP is split evenly among party members.
        """
        party = game_state.get_party()
        
        if not party:
            return ToolResult(
                tool_name="award_xp",
                success=False,
                result="No party members to award XP to.",
            )
        
        # Split XP among party
        xp_per_member = amount // len(party)
        
        # Track level ups
        level_ups = []
        
        for actor in party:
            old_level = actor.current_level
            leveled = actor.add_xp(xp_per_member)
            
            if leveled:
                level_ups.append({
                    "name": actor.name,
                    "old_level": old_level,
                    "new_level": actor.current_level,
                })
        
        # Update party XP tracker
        game_state.party_xp += amount
        
        # Build result message
        result_lines = [
            f"✨ **{amount} XP** awarded for {reason}!",
            f"Each party member receives **{xp_per_member} XP**.",
        ]
        
        for actor in party:
            from dnd_manager.models.progression import get_xp_for_next_level
            next_xp = get_xp_for_next_level(actor.current_level)
            if next_xp:
                result_lines.append(f"• {actor.name}: {actor.experience_points} XP (Next level: {next_xp})")
            else:
                result_lines.append(f"• {actor.name}: {actor.experience_points} XP (MAX LEVEL)")
        
        if level_ups:
            result_lines.append("")
            result_lines.append("🎉 **LEVEL UP!**")
            for lu in level_ups:
                result_lines.append(f"• **{lu['name']}** has reached level {lu['new_level']}!")
        
        return ToolResult(
            tool_name="award_xp",
            success=True,
            result="\n".join(result_lines),
            data={
                "total_xp": amount,
                "xp_per_member": xp_per_member,
                "level_ups": level_ups,
            },
        )

    def handle_get_xp_for_cr(challenge_rating: str) -> ToolResult:
        """Get the XP value for a creature based on its CR.
        
        Use this to calculate XP awards after defeating enemies.
        """
        xp_by_cr: dict[str, int] = {
            "0": 10,
            "1/8": 25,
            "1/4": 50,
            "1/2": 100,
            "1": 200,
            "2": 450,
            "3": 700,
            "4": 1100,
            "5": 1800,
            "6": 2300,
            "7": 2900,
            "8": 3900,
            "9": 5000,
            "10": 5900,
            "11": 7200,
            "12": 8400,
            "13": 10000,
            "14": 11500,
            "15": 13000,
            "16": 15000,
            "17": 18000,
            "18": 20000,
            "19": 22000,
            "20": 25000,
            "21": 33000,
            "22": 41000,
            "23": 50000,
            "24": 62000,
            "25": 75000,
            "26": 90000,
            "27": 105000,
            "28": 120000,
            "29": 135000,
            "30": 155000,
        }
        
        xp = xp_by_cr.get(challenge_rating, 0)
        
        if xp == 0 and challenge_rating not in xp_by_cr:
            return ToolResult(
                tool_name="get_xp_for_cr",
                success=False,
                result=f"Unknown CR: {challenge_rating}. Valid CRs: 0, 1/8, 1/4, 1/2, 1-30.",
            )
        
        return ToolResult(
            tool_name="get_xp_for_cr",
            success=True,
            result=f"CR {challenge_rating} = **{xp} XP**",
            data={"cr": challenge_rating, "xp": xp},
        )

    def handle_cast_spell(
        caster_name: str,
        spell_name: str,
        spell_level: int = 0,
        target_name: str | None = None,
    ) -> ToolResult:
        """Cast a spell, using a spell slot if required.
        
        Args:
            caster_name: Name of the character casting the spell.
            spell_name: Name of the spell being cast.
            spell_level: Slot level to use (0 for cantrips).
            target_name: Optional target of the spell.
        """
        # Find caster
        caster = None
        for a in game_state.actors.values():
            if a.name.lower() == caster_name.lower():
                caster = a
                break
        
        if caster is None:
            return ToolResult(
                tool_name="cast_spell",
                success=False,
                result=f"Caster '{caster_name}' not found",
            )
        
        if caster.spellbook is None:
            return ToolResult(
                tool_name="cast_spell",
                success=False,
                result=f"{caster.name} is not a spellcaster!",
            )
        
        # Check if they know the spell
        all_spells = (
            caster.spellbook.cantrips +
            caster.spellbook.spells_known +
            caster.spellbook.spells_prepared
        )
        spell_known = any(s.lower() == spell_name.lower() for s in all_spells)
        
        # For cantrips (level 0), no slot needed
        is_cantrip = spell_name.lower() in [c.lower() for c in caster.spellbook.cantrips]
        
        result_lines = []
        
        if is_cantrip or spell_level == 0:
            # Cantrip - no slot required
            result_lines.append(f"✨ **{caster.name}** casts **{spell_name}** (cantrip)!")
        else:
            # Check for spell slot
            if not caster.spellbook.has_slot(spell_level):
                # Check Warlock pact slots
                if caster.spellbook.pact_slots_current > 0:
                    caster.spellbook.pact_slots_current -= 1
                    result_lines.append(
                        f"✨ **{caster.name}** casts **{spell_name}** "
                        f"(Pact Slot, level {caster.spellbook.pact_slot_level})!"
                    )
                    result_lines.append(
                        f"Pact slots remaining: {caster.spellbook.pact_slots_current}/{caster.spellbook.pact_slots_max}"
                    )
                else:
                    return ToolResult(
                        tool_name="cast_spell",
                        success=False,
                        result=f"{caster.name} has no level {spell_level} spell slots remaining!",
                    )
            else:
                # Use the slot
                caster.spellbook.use_slot(spell_level)
                current, max_slots = caster.spellbook.spell_slots.get(spell_level, (0, 0))
                result_lines.append(
                    f"✨ **{caster.name}** casts **{spell_name}** (level {spell_level})!"
                )
                result_lines.append(f"Level {spell_level} slots remaining: {current}/{max_slots}")
        
        # Handle spell effects based on spell type
        # Define some common auto-hit and auto-damage spells
        AUTO_HIT_SPELLS = {
            "magic missile": {"damage": "1d4+1", "missiles": 3, "type": "force"},  # 3 missiles at base
        }
        ATTACK_ROLL_SPELLS = {
            "fire bolt": {"damage": "1d10", "type": "fire"},
            "ray of frost": {"damage": "1d8", "type": "cold"},
            "eldritch blast": {"damage": "1d10", "type": "force"},
            "scorching ray": {"damage": "2d6", "type": "fire", "rays": 3},
            "chromatic orb": {"damage": "3d8", "type": "varies"},
            "guiding bolt": {"damage": "4d6", "type": "radiant"},
        }
        
        # Buff and utility spells that modify game state
        BUFF_SPELLS = {
            # AC buffs - with duration tracking
            "mage armor": {
                "ac_formula": "13+dex",
                "effect_type": "ac_set",
                "duration_type": "hours",
                "duration_value": 8,
                "concentration": False,
                "description": "AC becomes 13 + DEX modifier for 8 hours",
            },
            "shield": {
                "ac_formula": "+5_reaction",
                "effect_type": "ac_bonus",
                "ac_bonus": 5,
                "duration_type": "until_start_of_next_turn",
                "duration_value": 0,
                "concentration": False,
                "description": "+5 AC until start of next turn",
            },
            "barkskin": {
                "ac_formula": "16_min",
                "effect_type": "ac_set",
                "ac_value": 16,
                "duration_type": "hours",
                "duration_value": 1,
                "concentration": True,
                "description": "AC can't be less than 16 for 1 hour (concentration)",
            },
            "shield of faith": {
                "ac_formula": "+2",
                "effect_type": "ac_bonus",
                "ac_bonus": 2,
                "duration_type": "minutes",
                "duration_value": 10,
                "concentration": True,
                "description": "+2 AC for 10 minutes (concentration)",
            },
            
            # Temp HP - instantaneous but temp HP lasts until depleted or long rest
            "false life": {
                "temp_hp": "1d4+4",
                "effect_type": "temp_hp",
                "duration_type": "hours",
                "duration_value": 1,
                "concentration": False,
                "description": "Gain temporary hit points for 1 hour",
            },
            "armor of agathys": {
                "temp_hp": 5,
                "effect_type": "temp_hp",
                "duration_type": "hours",
                "duration_value": 1,
                "concentration": False,
                "description": "5 temp HP for 1 hour, attackers take 5 cold damage",
            },
            "heroism": {
                "temp_hp": 0,
                "effect_type": "temp_hp",
                "duration_type": "minutes",
                "duration_value": 1,
                "concentration": True,
                "description": "Gain temp HP equal to spellcasting mod each turn (concentration)",
            },
            
            # Healing - instantaneous, no duration tracking needed
            "cure wounds": {"healing": "1d8", "add_mod": True, "duration_type": "instantaneous", "description": "Touch healing"},
            "healing word": {"healing": "1d4", "add_mod": True, "duration_type": "instantaneous", "description": "Bonus action ranged healing"},
            "mass cure wounds": {"healing": "3d8", "add_mod": True, "duration_type": "instantaneous", "description": "Heal up to 6 creatures"},
            "heal": {"healing_flat": 70, "duration_type": "instantaneous", "description": "Restore 70 HP"},
            
            # Utility spells with duration
            "bless": {
                "effect_type": "save_bonus",
                "bonus": "1d4",
                "duration_type": "minutes",
                "duration_value": 1,
                "concentration": True,
                "description": "Up to 3 targets add 1d4 to attacks and saves (concentration)",
            },
            "haste": {
                "effect_type": "custom",
                "ac_bonus": 2,
                "speed_bonus": 2,  # Double speed
                "duration_type": "minutes",
                "duration_value": 1,
                "concentration": True,
                "description": "Double speed, +2 AC, extra action for 1 minute (concentration)",
            },
            "invisibility": {
                "effect_type": "invisibility",
                "duration_type": "hours",
                "duration_value": 1,
                "concentration": True,
                "description": "Target becomes invisible for 1 hour (concentration)",
            },
            "fly": {
                "effect_type": "flying",
                "fly_speed": 60,
                "duration_type": "minutes",
                "duration_value": 10,
                "concentration": True,
                "description": "Target gains 60ft flying speed for 10 minutes (concentration)",
            },
            "mirror image": {
                "effect_type": "custom",
                "duration_type": "minutes",
                "duration_value": 1,
                "concentration": False,
                "description": "3 illusory duplicates protect you for 1 minute",
            },
            "blur": {
                "effect_type": "disadvantage_attacks",
                "duration_type": "minutes",
                "duration_value": 1,
                "concentration": True,
                "description": "Attackers have disadvantage for 1 minute (concentration)",
            },
            "stoneskin": {
                "effect_type": "damage_resistance",
                "damage_types": ["bludgeoning", "piercing", "slashing"],
                "duration_type": "hours",
                "duration_value": 1,
                "concentration": True,
                "description": "Resistance to nonmagical physical damage for 1 hour (concentration)",
            },
            "death ward": {
                "effect_type": "custom",
                "duration_type": "hours",
                "duration_value": 8,
                "concentration": False,
                "description": "First time reduced to 0 HP, drop to 1 HP instead (8 hours)",
            },
        }
        
        spell_lower = spell_name.lower()
        damage_dealt = 0
        target = None
        
        # Find target if provided
        if target_name:
            for a in game_state.actors.values():
                if a.name.lower() == target_name.lower():
                    target = a
                    break
        
        # Handle auto-hit spells (Magic Missile)
        if spell_lower in AUTO_HIT_SPELLS:
            spell_info = AUTO_HIT_SPELLS[spell_lower]
            missiles = spell_info.get("missiles", 3) + (spell_level - 1 if spell_level > 1 else 0)
            
            # Roll damage for each missile
            total_damage = 0
            damage_rolls = []
            for i in range(missiles):
                dmg_roll = roll_dice(spell_info["damage"])
                total_damage += dmg_roll.total
                damage_rolls.append(str(dmg_roll.total))
            
            result_lines.append(f"🎯 **AUTO-HIT!** {missiles} missiles strike!")
            result_lines.append(f"💥 Damage: {' + '.join(damage_rolls)} = **{total_damage}** {spell_info['type']}")
            
            if target:
                # Apply damage
                old_hp = target.health.hp_current
                target.health.hp_current = max(0, target.health.hp_current - total_damage)
                result_lines.append(f"⚔️ **{target.name}**: {old_hp} → {target.health.hp_current} HP")
                
                if target.health.hp_current <= 0:
                    target.health.is_dead = True
                    result_lines.append(f"💀 **{target.name}** is slain!")
                
                damage_dealt = total_damage
            else:
                result_lines.append("⚠️ No target specified - describe where the missiles go!")
        
        # Handle attack roll spells
        elif spell_lower in ATTACK_ROLL_SPELLS and target:
            spell_info = ATTACK_ROLL_SPELLS[spell_lower]
            attack_roll = roll_dice(f"1d20+{caster.spellbook.spell_attack_bonus}")
            
            result_lines.append(f"🎲 Spell Attack: {attack_roll.details} + {caster.spellbook.spell_attack_bonus} = **{attack_roll.total}** vs AC {target.ac}")
            
            if attack_roll.total >= target.ac:
                dmg_roll = roll_dice(spell_info["damage"])
                result_lines.append(f"✅ **HIT!** 💥 Damage: {dmg_roll.details} = **{dmg_roll.total}** {spell_info['type']}")
                
                old_hp = target.health.hp_current
                target.health.hp_current = max(0, target.health.hp_current - dmg_roll.total)
                result_lines.append(f"⚔️ **{target.name}**: {old_hp} → {target.health.hp_current} HP")
                
                if target.health.hp_current <= 0:
                    target.health.is_dead = True
                    result_lines.append(f"💀 **{target.name}** is slain!")
                
                damage_dealt = dmg_roll.total
            else:
                result_lines.append("❌ **MISS!** The spell fizzles past the target.")
        
        # Handle buff/utility spells
        elif spell_lower in BUFF_SPELLS:
            from dnd_manager.models.ecs import ActiveEffect, EffectType, EffectDuration
            
            buff_info = BUFF_SPELLS[spell_lower]
            buff_target = target or caster  # Default to self if no target
            
            effect_applied = False
            created_effect: ActiveEffect | None = None
            
            # Get duration info
            duration_type_str = buff_info.get("duration_type", "instantaneous")
            duration_value = buff_info.get("duration_value", 0)
            concentration = buff_info.get("concentration", False)
            
            # Map duration string to enum
            duration_type_map = {
                "instantaneous": EffectDuration.INSTANTANEOUS,
                "until_start_of_next_turn": EffectDuration.UNTIL_START_OF_NEXT_TURN,
                "rounds": EffectDuration.ROUNDS,
                "minutes": EffectDuration.MINUTES,
                "hours": EffectDuration.HOURS,
                "until_short_rest": EffectDuration.UNTIL_SHORT_REST,
                "until_long_rest": EffectDuration.UNTIL_LONG_REST,
                "until_dispelled": EffectDuration.UNTIL_DISPELLED,
                "concentration": EffectDuration.CONCENTRATION,
            }
            duration_type = duration_type_map.get(duration_type_str, EffectDuration.INSTANTANEOUS)
            
            # If concentration, override duration type
            if concentration and duration_type != EffectDuration.INSTANTANEOUS:
                # Keep original duration type but mark as concentration
                pass
            
            if buff_info.get("ac_formula"):
                # AC-modifying spells like Mage Armor, Shield
                formula = buff_info["ac_formula"]
                effect_type = EffectType.AC_BONUS
                effect_value = 0
                
                if formula == "13+dex":
                    new_ac = 13 + buff_target.stats.dex_mod
                    if buff_target.defense.ac_base < new_ac:
                        buff_target.defense.ac_base = new_ac
                        buff_target.defense.ac_bonus = 0  # Reset bonus, mage armor replaces base
                        result_lines.append(f"🛡️ **{buff_target.name}**'s AC becomes **{new_ac}** (13 + DEX)")
                        effect_type = EffectType.AC_SET
                        effect_value = new_ac
                        effect_applied = True
                    else:
                        result_lines.append(f"⚠️ {buff_target.name}'s current AC ({buff_target.ac}) is already higher!")
                elif formula == "+5_reaction":
                    # Shield spell - +5 AC until start of next turn
                    buff_target.defense.ac_bonus += 5
                    result_lines.append(f"🛡️ **{buff_target.name}**'s AC increases by **+5** until next turn! (Now {buff_target.ac})")
                    effect_type = EffectType.AC_BONUS
                    effect_value = 5
                    effect_applied = True
                elif formula == "+2":
                    buff_target.defense.ac_bonus += 2
                    result_lines.append(f"🛡️ **{buff_target.name}**'s AC increases by **+2**! (Now {buff_target.ac})")
                    effect_type = EffectType.AC_BONUS
                    effect_value = 2
                    effect_applied = True
                elif formula == "16_min":
                    if buff_target.ac < 16:
                        # Barkskin - AC minimum 16
                        buff_target.defense.ac_base = 16
                        buff_target.defense.ac_bonus = 0
                        result_lines.append(f"🛡️ **{buff_target.name}**'s AC becomes **16** (Barkskin minimum)")
                        effect_type = EffectType.AC_SET
                        effect_value = 16
                        effect_applied = True
                    else:
                        result_lines.append(f"⚠️ {buff_target.name}'s AC is already 16 or higher!")
                
                # Create the active effect for AC spells
                if effect_applied and duration_type != EffectDuration.INSTANTANEOUS:
                    created_effect = ActiveEffect(
                        name=spell_name.title(),
                        source="spell",
                        source_actor=caster.name,
                        effect_type=effect_type,
                        value=effect_value,
                        description=buff_info.get("description", ""),
                        duration_type=duration_type,
                        duration_value=duration_value,
                        concentration=concentration,
                    )
            
            if buff_info.get("temp_hp"):
                # Temp HP spells like False Life
                temp_hp = buff_info["temp_hp"]
                if isinstance(temp_hp, str):
                    hp_roll = roll_dice(temp_hp)
                    temp_hp = hp_roll.total
                    result_lines.append(f"🎲 Temp HP: {hp_roll.details} = **{temp_hp}**")
                buff_target.health.hp_temp = max(buff_target.health.hp_temp, temp_hp)
                result_lines.append(f"💚 **{buff_target.name}** gains **{temp_hp}** temporary HP!")
                effect_applied = True
                
                # Create effect for temp HP tracking
                if duration_type != EffectDuration.INSTANTANEOUS:
                    created_effect = ActiveEffect(
                        name=spell_name.title(),
                        source="spell",
                        source_actor=caster.name,
                        effect_type=EffectType.TEMP_HP,
                        value=temp_hp,
                        description=buff_info.get("description", ""),
                        duration_type=duration_type,
                        duration_value=duration_value,
                        concentration=concentration,
                    )
            
            if buff_info.get("healing") or buff_info.get("healing_flat"):
                # Healing spells - instantaneous, no effect tracking
                if buff_info.get("healing_flat"):
                    heal_total = buff_info["healing_flat"]
                    result_lines.append(f"💚 Healing: **{heal_total}** HP")
                else:
                    heal_dice = buff_info["healing"]
                    # Add spellcasting modifier if applicable
                    spell_mod = 0
                    if buff_info.get("add_mod") and caster.spellbook.spellcasting_ability:
                        ability = caster.spellbook.spellcasting_ability
                        if hasattr(ability, 'value'):
                            ability = ability.value
                        ability_lower = ability.lower()
                        stat_map = {
                            "strength": caster.stats.strength,
                            "dexterity": caster.stats.dexterity,
                            "constitution": caster.stats.constitution,
                            "intelligence": caster.stats.intelligence,
                            "wisdom": caster.stats.wisdom,
                            "charisma": caster.stats.charisma,
                        }
                        spell_mod = (stat_map.get(ability_lower, 10) - 10) // 2
                    
                    heal_roll = roll_dice(heal_dice)
                    heal_total = heal_roll.total + spell_mod
                    if spell_mod > 0:
                        result_lines.append(f"💚 Healing: {heal_roll.details} + {spell_mod} = **{heal_total}** HP")
                    else:
                        result_lines.append(f"💚 Healing: {heal_roll.details} = **{heal_total}** HP")
                
                old_hp = buff_target.health.hp_current
                buff_target.health.hp_current = min(
                    buff_target.health.hp_max,
                    buff_target.health.hp_current + heal_total
                )
                result_lines.append(f"**{buff_target.name}**: {old_hp} → {buff_target.health.hp_current}/{buff_target.health.hp_max}")
                effect_applied = True
            
            # Handle utility spells that just create effects
            if buff_info.get("effect_type") and not created_effect and duration_type != EffectDuration.INSTANTANEOUS:
                effect_type_str = buff_info.get("effect_type", "custom")
                effect_type_map = {
                    "ac_set": EffectType.AC_SET,
                    "ac_bonus": EffectType.AC_BONUS,
                    "temp_hp": EffectType.TEMP_HP,
                    "damage_resistance": EffectType.DAMAGE_RESISTANCE,
                    "save_bonus": EffectType.SAVE_BONUS,
                    "invisibility": EffectType.INVISIBILITY,
                    "flying": EffectType.FLYING,
                    "disadvantage_attacks": EffectType.DISADVANTAGE_ATTACKS,
                    "custom": EffectType.CUSTOM,
                }
                effect_type = effect_type_map.get(effect_type_str, EffectType.CUSTOM)
                
                created_effect = ActiveEffect(
                    name=spell_name.title(),
                    source="spell",
                    source_actor=caster.name,
                    effect_type=effect_type,
                    value=buff_info.get("ac_bonus") or buff_info.get("fly_speed"),
                    description=buff_info.get("description", ""),
                    duration_type=duration_type,
                    duration_value=duration_value,
                    concentration=concentration,
                )
                result_lines.append(f"✨ {buff_info.get('description', f'{spell_name.title()} is now active!')}")
                effect_applied = True
            
            # Add the effect to the target
            if created_effect:
                current_round = game_state.combat_round if game_state.combat_active else 0
                current_turn = game_state.combat_turn_index if game_state.combat_active else 0
                buff_target.add_effect(created_effect, current_round, current_turn)
                
                # Add duration info to output
                if duration_type == EffectDuration.UNTIL_START_OF_NEXT_TURN:
                    result_lines.append("⏱️ *Expires at the start of your next turn*")
                elif duration_type == EffectDuration.ROUNDS:
                    result_lines.append(f"⏱️ *Duration: {duration_value} round(s)*")
                elif duration_type == EffectDuration.MINUTES:
                    result_lines.append(f"⏱️ *Duration: {duration_value} minute(s)*")
                elif duration_type == EffectDuration.HOURS:
                    result_lines.append(f"⏱️ *Duration: {duration_value} hour(s)*")
                
                if concentration:
                    result_lines.append("🎯 *Concentration required*")
            
            if buff_info.get("description") and not effect_applied:
                result_lines.append(f"✨ {buff_info['description']}")
        
        # For other spells, just provide info
        else:
            if target:
                result_lines.append(f"Target: **{target.name}** (AC {target.ac}, HP {target.health.hp_current}/{target.health.hp_max})")
            result_lines.append(f"Spell Save DC: {caster.spellbook.spell_save_dc}")
            result_lines.append(f"Spell Attack Bonus: +{caster.spellbook.spell_attack_bonus}")
            result_lines.append("📝 Use `apply_damage` or `roll_save` to resolve effects.")
        
        return ToolResult(
            tool_name="cast_spell",
            success=True,
            result="\n".join(result_lines),
            data={
                "caster": caster.name,
                "spell": spell_name,
                "level": spell_level,
                "target": target_name,
                "is_cantrip": is_cantrip,
                "damage_dealt": damage_dealt,
                "spell_save_dc": caster.spellbook.spell_save_dc,
                "spell_attack_bonus": caster.spellbook.spell_attack_bonus,
            },
        )

    def handle_lookup_spell(spell_name: str) -> ToolResult:
        """Look up spell details from the indexed rulebooks.
        
        Use this to find out what a spell does, its range, components, etc.
        """
        if chroma_store is None:
            return ToolResult(
                tool_name="lookup_spell",
                success=False,
                result="No rulebooks indexed. Cannot look up spell details.",
            )
        
        # Search for the spell
        search_queries = [
            f"{spell_name} spell",
            f"{spell_name} casting time range",
        ]
        
        results = []
        for query in search_queries:
            docs = chroma_store.search(query, n_results=3)
            results.extend(docs)
        
        if not results:
            return ToolResult(
                tool_name="lookup_spell",
                success=False,
                result=f"Could not find spell '{spell_name}' in indexed rulebooks.",
            )
        
        # Find the best match (content mentioning the spell name)
        best_content = None
        for doc in results:
            if spell_name.lower() in doc.content.lower():
                best_content = doc.content[:800]
                break
        
        if not best_content:
            best_content = results[0].content[:800]
        
        # Clean up the content
        import re
        best_content = re.sub(r'\s+', ' ', best_content).strip()
        
        return ToolResult(
            tool_name="lookup_spell",
            success=True,
            result=f"📖 **{spell_name}**\n\n{best_content}",
            data={"spell_name": spell_name, "content": best_content},
        )

    def handle_get_character_spells(character_name: str) -> ToolResult:
        """Get the spells known/prepared and spell slots for a character."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="get_character_spells",
                success=False,
                result=f"Character '{character_name}' not found",
            )
        
        if actor.spellbook is None:
            return ToolResult(
                tool_name="get_character_spells",
                success=False,
                result=f"{actor.name} is not a spellcaster.",
            )
        
        sb = actor.spellbook
        lines = [f"**{actor.name}'s Spellcasting**"]
        lines.append(f"Spell Save DC: {sb.spell_save_dc} | Spell Attack: +{sb.spell_attack_bonus}")
        lines.append("")
        
        # Spell slots
        if sb.spell_slots:
            slots = []
            for level in sorted(sb.spell_slots.keys()):
                current, max_val = sb.spell_slots[level]
                filled = "●" * current
                empty = "○" * (max_val - current)
                slots.append(f"L{level}: {filled}{empty}")
            lines.append(f"**Spell Slots:** {' | '.join(slots)}")
        
        # Pact slots (Warlock)
        if sb.pact_slots_max > 0:
            filled = "●" * sb.pact_slots_current
            empty = "○" * (sb.pact_slots_max - sb.pact_slots_current)
            lines.append(f"**Pact Slots (L{sb.pact_slot_level}):** {filled}{empty}")
        
        lines.append("")
        
        # Cantrips
        if sb.cantrips:
            lines.append(f"**Cantrips:** {', '.join(sb.cantrips)}")
        
        # Spells known/prepared
        if sb.spells_prepared:
            lines.append(f"**Prepared Spells:** {', '.join(sb.spells_prepared)}")
        elif sb.spells_known:
            lines.append(f"**Spells Known:** {', '.join(sb.spells_known)}")
        
        return ToolResult(
            tool_name="get_character_spells",
            success=True,
            result="\n".join(lines),
            data={
                "cantrips": sb.cantrips,
                "spells_known": sb.spells_known,
                "spells_prepared": sb.spells_prepared,
                "spell_slots": {str(k): v for k, v in sb.spell_slots.items()},
                "spell_save_dc": sb.spell_save_dc,
                "spell_attack_bonus": sb.spell_attack_bonus,
            },
        )

    def handle_use_feature(
        character_name: str,
        feature_name: str,
    ) -> ToolResult:
        """Use a class or racial feature/ability.
        
        For limited-use features (like Rage, Action Surge), this tracks usage.
        Returns a description of what the feature does.
        """
        from dnd_manager.models.progression import get_feature_resource_info
        
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="use_feature",
                success=False,
                result=f"Character '{character_name}' not found",
            )
        
        if actor.class_features is None:
            return ToolResult(
                tool_name="use_feature",
                success=False,
                result=f"{actor.name} has no class features.",
            )
        
        # First, check tracked features (new system with cooldowns)
        tracked_feature = actor.class_features.get_feature(feature_name)
        
        if tracked_feature:
            # Use the tracked feature
            if tracked_feature.use():
                result_lines = [f"⚡ **{actor.name}** uses **{tracked_feature.name}**!"]
                if tracked_feature.uses_max:
                    result_lines.append(f"Uses remaining: {tracked_feature.uses_display}")
                    recharge_val = tracked_feature.recharge.value if hasattr(tracked_feature.recharge, 'value') else str(tracked_feature.recharge)
                    recharge_str = recharge_val.replace("_", " ")
                    result_lines.append(f"*(Recharges on {recharge_str})*")
                if tracked_feature.description:
                    result_lines.append(f"\n*{tracked_feature.description}*")
                
                return ToolResult(
                    tool_name="use_feature",
                    success=True,
                    result="\n".join(result_lines),
                    data={
                        "character": actor.name,
                        "feature": tracked_feature.name,
                        "uses_remaining": tracked_feature.uses_current,
                        "uses_max": tracked_feature.uses_max,
                    },
                )
            else:
                recharge_val = tracked_feature.recharge.value if hasattr(tracked_feature.recharge, 'value') else str(tracked_feature.recharge)
                recharge_str = recharge_val.replace("_", " ")
                return ToolResult(
                    tool_name="use_feature",
                    success=False,
                    result=f"❌ {actor.name} has no uses of **{tracked_feature.name}** remaining!\n*(Recharges on {recharge_str})*",
                )
        
        # Check legacy simple features list
        feature_lower = feature_name.lower()
        has_feature = any(
            feature_lower in f.lower()
            for f in actor.class_features.features
        )
        
        if not has_feature:
            # Check legacy resources dict
            resource_key = None
            for key in actor.class_features.resources:
                if feature_lower in key.lower() or key.lower() in feature_lower:
                    resource_key = key
                    break
            
            if not resource_key:
                return ToolResult(
                    tool_name="use_feature",
                    success=False,
                    result=f"{actor.name} doesn't have the feature '{feature_name}'.",
                )
        
        # Check for limited-use features in legacy resources
        resources = actor.class_features.resources
        resource_key = None
        for key in resources:
            if feature_lower in key.lower() or key.lower() in feature_lower:
                resource_key = key
                break
        
        result_lines = [f"⚡ **{actor.name}** uses **{feature_name}**!"]
        
        if resource_key:
            current, max_val = resources[resource_key]
            if current <= 0:
                return ToolResult(
                    tool_name="use_feature",
                    success=False,
                    result=f"❌ {actor.name} has no uses of {feature_name} remaining! *(Recharges on rest)*",
                )
            
            # Use the resource
            actor.class_features.use_resource(resource_key)
            new_current = current - 1
            result_lines.append(f"Uses remaining: {new_current}/{max_val}")
        
        # Get feature info from progression data
        level = actor.class_features.total_level if actor.class_features else 1
        stats_dict = {
            "strength": actor.stats.strength,
            "dexterity": actor.stats.dexterity,
            "constitution": actor.stats.constitution,
            "intelligence": actor.stats.intelligence,
            "wisdom": actor.stats.wisdom,
            "charisma": actor.stats.charisma,
        }
        
        feature_info = get_feature_resource_info(feature_name, level, stats_dict)
        if feature_info and feature_info.get("description"):
            result_lines.append(f"\n*{feature_info['description']}*")
        
        return ToolResult(
            tool_name="use_feature",
            success=True,
            result="\n".join(result_lines),
            data={
                "character": actor.name,
                "feature": feature_name,
                "resource_used": resource_key is not None,
            },
        )

    def handle_short_rest(character_name: str | None = None) -> ToolResult:
        """Take a short rest (1+ hours). Restores short-rest features, hit dice for HP.
        
        If character_name is None, rests the entire party.
        """
        if character_name:
            # Rest single character
            actor = None
            for a in game_state.actors.values():
                if a.name.lower() == character_name.lower():
                    actor = a
                    break
            
            if actor is None:
                return ToolResult(
                    tool_name="short_rest",
                    success=False,
                    result=f"Character '{character_name}' not found",
                )
            
            actors_to_rest = [actor]
        else:
            # Rest entire party
            actors_to_rest = game_state.get_party()
        
        result_lines = ["☕ **Short Rest**"]
        
        for actor in actors_to_rest:
            restored = []
            expired_effects = []
            
            # Restore class features
            if actor.class_features:
                restored.extend(actor.class_features.short_rest())
            
            # Warlock pact slots restore on short rest
            if actor.spellbook and actor.spellbook.pact_slots_max > 0:
                actor.spellbook.pact_slots_current = actor.spellbook.pact_slots_max
                restored.append(f"Pact Slots ({actor.spellbook.pact_slots_max})")
            
            # Expire effects that end on short rest
            expired = actor.expire_effects_on_rest("short")
            expired_effects.extend([e.name for e in expired])
            
            if restored:
                result_lines.append(f"\n**{actor.name}** restored: {', '.join(restored)}")
            else:
                result_lines.append(f"\n**{actor.name}** rested (no features to restore)")
            
            if expired_effects:
                result_lines.append(f"  *Expired effects: {', '.join(expired_effects)}*")
        
        result_lines.append("\n*Characters may spend Hit Dice to recover HP.*")
        
        return ToolResult(
            tool_name="short_rest",
            success=True,
            result="\n".join(result_lines),
            data={"rested": [a.name for a in actors_to_rest]},
        )

    def handle_long_rest(character_name: str | None = None) -> ToolResult:
        """Take a long rest (8 hours). Restores all features, spell slots, and HP.
        
        If character_name is None, rests the entire party.
        """
        if character_name:
            actor = None
            for a in game_state.actors.values():
                if a.name.lower() == character_name.lower():
                    actor = a
                    break
            
            if actor is None:
                return ToolResult(
                    tool_name="long_rest",
                    success=False,
                    result=f"Character '{character_name}' not found",
                )
            
            actors_to_rest = [actor]
        else:
            actors_to_rest = game_state.get_party()
        
        result_lines = ["🌙 **Long Rest**"]
        
        for actor in actors_to_rest:
            restored = []
            expired_effects = []
            
            # Restore HP
            old_hp = actor.health.hp_current
            actor.health.hp_current = actor.health.hp_max
            if old_hp < actor.health.hp_max:
                restored.append(f"HP ({old_hp} → {actor.health.hp_max})")
            
            # Clear temporary HP (resets on long rest)
            if actor.health.hp_temp > 0:
                actor.health.hp_temp = 0
                restored.append("Temp HP cleared")
            
            # Restore all class features
            if actor.class_features:
                features_restored = actor.class_features.long_rest()
                if features_restored:
                    restored.extend(features_restored)
            
            # Restore all spell slots
            if actor.spellbook:
                actor.spellbook.restore_slots()  # Restores all when level=None
                if actor.spellbook.pact_slots_max > 0:
                    actor.spellbook.pact_slots_current = actor.spellbook.pact_slots_max
                restored.append("All spell slots")
            
            # Expire effects that end on long rest (and short rest effects too)
            expired = actor.expire_effects_on_rest("long")
            expired_effects.extend([e.name for e in expired])
            
            # Clear ALL remaining effects on long rest (8 hours clears most buffs)
            remaining_expired = []
            for effect in list(actor.active_effects):
                dur_type = effect.duration_type.value if hasattr(effect.duration_type, 'value') else str(effect.duration_type)
                if dur_type in ("hours", "minutes", "rounds"):
                    # These would have expired during 8 hours
                    remaining_expired.append(effect.name)
            actor.active_effects = [
                e for e in actor.active_effects 
                if (e.duration_type.value if hasattr(e.duration_type, 'value') else str(e.duration_type)) not in ("hours", "minutes", "rounds")
            ]
            expired_effects.extend(remaining_expired)
            
            if restored:
                result_lines.append(f"\n**{actor.name}** restored: {', '.join(restored)}")
            else:
                result_lines.append(f"\n**{actor.name}** fully rested")
            
            if expired_effects:
                result_lines.append(f"  *Expired effects: {', '.join(set(expired_effects))}*")
        
        result_lines.append("\n*The party is refreshed and ready for adventure!*")
        
        return ToolResult(
            tool_name="long_rest",
            success=True,
            result="\n".join(result_lines),
            data={"rested": [a.name for a in actors_to_rest]},
        )

    def handle_get_character_features(character_name: str) -> ToolResult:
        """Get all features and abilities for a character."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="get_character_features",
                success=False,
                result=f"Character '{character_name}' not found",
            )
        
        lines = [f"**{actor.name}'s Features & Abilities**"]
        
        if actor.class_features:
            # Class info
            classes = ", ".join(
                f"{c.class_name} {c.level}" + (f" ({c.subclass})" if c.subclass else "")
                for c in actor.class_features.classes
            )
            lines.append(f"Classes: {classes}")
            lines.append("")
            
            # Resources
            if actor.class_features.resources:
                lines.append("**Resources:**")
                for name, (current, max_val) in actor.class_features.resources.items():
                    filled = "●" * current
                    empty = "○" * (max_val - current)
                    lines.append(f"• {name.title()}: {filled}{empty}")
                lines.append("")
            
            # Features
            if actor.class_features.features:
                lines.append("**Features:**")
                for feature in actor.class_features.features:
                    lines.append(f"• {feature}")
        else:
            lines.append("No class features.")
        
        return ToolResult(
            tool_name="get_character_features",
            success=True,
            result="\n".join(lines),
            data={
                "features": actor.class_features.features if actor.class_features else [],
                "resources": {k: v for k, v in (actor.class_features.resources.items() if actor.class_features else [])},
            },
        )

    def handle_list_effects(character_name: str | None = None) -> ToolResult:
        """List all active effects on a character or all characters.
        
        Args:
            character_name: Character to check, or None for all.
        """
        if character_name:
            actor = None
            for a in game_state.actors.values():
                if a.name.lower() == character_name.lower():
                    actor = a
                    break
            
            if actor is None:
                return ToolResult(
                    tool_name="list_effects",
                    success=False,
                    result=f"Character '{character_name}' not found",
                )
            
            actors_to_check = [actor]
        else:
            actors_to_check = list(game_state.actors.values())
        
        lines = ["**Active Effects:**"]
        total_effects = 0
        
        for actor in actors_to_check:
            if actor.active_effects:
                lines.append(f"\n**{actor.name}:**")
                for effect in actor.active_effects:
                    dur_type = effect.duration_type
                    if hasattr(dur_type, 'value'):
                        dur_type = dur_type.value
                    
                    # Build duration string
                    if dur_type == "until_start_of_next_turn":
                        dur_str = "until next turn"
                    elif dur_type == "rounds" and effect.expires_round:
                        rounds_left = effect.expires_round - game_state.combat_round
                        dur_str = f"{rounds_left} round(s) left"
                    elif dur_type == "minutes":
                        dur_str = f"{effect.duration_value} min"
                    elif dur_type == "hours":
                        dur_str = f"{effect.duration_value} hr"
                    elif dur_type == "concentration":
                        dur_str = "concentration"
                    else:
                        dur_str = dur_type
                    
                    conc_marker = " 🎯" if effect.concentration else ""
                    lines.append(f"  • **{effect.name}** ({dur_str}){conc_marker}")
                    if effect.description:
                        lines.append(f"    *{effect.description}*")
                    total_effects += 1
        
        if total_effects == 0:
            lines.append("\nNo active effects.")
        
        return ToolResult(
            tool_name="list_effects",
            success=True,
            result="\n".join(lines),
            data={
                "total_effects": total_effects,
                "by_character": {
                    a.name: [{"name": e.name, "type": str(e.effect_type), "duration": str(e.duration_type)} 
                             for e in a.active_effects]
                    for a in actors_to_check if a.active_effects
                },
            },
        )

    def handle_dispel_effect(character_name: str, effect_name: str) -> ToolResult:
        """Remove an active effect from a character (for dispelling, concentration loss, etc.).
        
        Args:
            character_name: Character with the effect.
            effect_name: Name of the effect to remove.
        """
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="dispel_effect",
                success=False,
                result=f"Character '{character_name}' not found",
            )
        
        removed = actor.remove_effect_by_name(effect_name)
        
        if removed:
            effect_names = ", ".join(e.name for e in removed)
            
            # Describe what was reversed
            reversals = []
            for effect in removed:
                eff_type = effect.effect_type
                if hasattr(eff_type, 'value'):
                    eff_type = eff_type.value
                    
                if eff_type == "ac_bonus" and effect.value:
                    reversals.append(f"AC -{effect.value}")
                elif eff_type == "ac_set":
                    reversals.append("AC returns to normal")
            
            reversal_str = f" ({', '.join(reversals)})" if reversals else ""
            
            return ToolResult(
                tool_name="dispel_effect",
                success=True,
                result=f"✨ **{effect_names}** removed from **{actor.name}**{reversal_str}",
                data={
                    "character": actor.name,
                    "removed": [e.name for e in removed],
                    "current_ac": actor.ac,
                },
            )
        else:
            # List current effects for reference
            current = ", ".join(e.name for e in actor.active_effects) if actor.active_effects else "none"
            return ToolResult(
                tool_name="dispel_effect",
                success=False,
                result=f"No effect named '{effect_name}' found on {actor.name}. Current effects: {current}",
            )

    def handle_break_concentration(character_name: str) -> ToolResult:
        """Break concentration for a character, ending their concentration spells.
        
        Args:
            character_name: The character who loses concentration.
        """
        caster = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                caster = a
                break
        
        if caster is None:
            return ToolResult(
                tool_name="break_concentration",
                success=False,
                result=f"Character '{character_name}' not found",
            )
        
        # Find and remove concentration effects from all actors that this caster is maintaining
        removed_effects = []
        for actor in game_state.actors.values():
            expired = actor.clear_concentration_effects(caster.name)
            for effect in expired:
                removed_effects.append((actor.name, effect))
        
        if removed_effects:
            lines = [f"💥 **{caster.name}** loses concentration!"]
            for target_name, effect in removed_effects:
                lines.append(f"  • **{effect.name}** ends on {target_name}")
            
            return ToolResult(
                tool_name="break_concentration",
                success=True,
                result="\n".join(lines),
                data={
                    "caster": caster.name,
                    "ended_effects": [(t, e.name) for t, e in removed_effects],
                },
            )
        else:
            return ToolResult(
                tool_name="break_concentration",
                success=True,
                result=f"**{caster.name}** was not concentrating on any spells.",
                data={"caster": caster.name, "ended_effects": []},
            )

    # =========================================================================
    # Morale & Flee Tools
    # =========================================================================

    def handle_check_morale(entity_name: str | None = None) -> ToolResult:
        """Check morale status of enemies. Use when enemies take heavy losses or are intimidated.
        
        Args:
            entity_name: Specific entity to check, or None for all enemies.
        """
        from dnd_manager.models.progression import get_creature_morale
        
        def get_actor_type(a):
            return a.type.value if hasattr(a.type, 'value') else str(a.type)
        
        if entity_name:
            actor = None
            for a in game_state.actors.values():
                if a.name.lower() == entity_name.lower():
                    actor = a
                    break
            
            if actor is None:
                return ToolResult(
                    tool_name="check_morale",
                    success=False,
                    result=f"Entity '{entity_name}' not found",
                )
            
            entities = [actor]
        else:
            # Check all non-player entities
            entities = [a for a in game_state.actors.values() if get_actor_type(a) != "player"]
        
        if not entities:
            return ToolResult(
                tool_name="check_morale",
                success=True,
                result="No enemies to check morale for.",
                data={"entities": []},
            )
        
        result_lines = ["**Morale Check:**"]
        morale_data = []
        
        for entity in entities:
            should_flee, reason = entity.check_morale()
            hp_pct = int((entity.health.hp_current / max(entity.health.hp_max, 1)) * 100)
            
            # Determine status emoji
            if entity.has_surrendered:
                emoji = "🏳️"
                status = "SURRENDERED"
            elif entity.is_fleeing:
                emoji = "🏃"
                status = "FLEEING"
            elif should_flee:
                emoji = "😰"
                status = f"BREAKING ({reason})"
            elif entity.morale <= 25:
                emoji = "😨"
                status = "BROKEN"
            elif entity.morale <= 50:
                emoji = "😟"
                status = "SHAKEN"
            else:
                emoji = "😤"
                status = "STEADY"
            
            result_lines.append(f"{emoji} **{entity.name}**: {status}")
            result_lines.append(f"   Morale: {entity.morale}% | HP: {hp_pct}%")
            
            morale_data.append({
                "name": entity.name,
                "morale": entity.morale,
                "hp_percent": hp_pct,
                "should_flee": should_flee,
                "is_fleeing": entity.is_fleeing,
                "has_surrendered": entity.has_surrendered,
            })
        
        return ToolResult(
            tool_name="check_morale",
            success=True,
            result="\n".join(result_lines),
            data={"entities": morale_data},
        )

    def handle_intimidate(
        intimidator_name: str,
        target_name: str,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> ToolResult:
        """Roll Intimidation against an enemy. On success, reduces their morale significantly.
        
        Args:
            intimidator_name: Character making the Intimidation check.
            target_name: Enemy being intimidated.
            advantage: Roll with advantage.
            disadvantage: Roll with disadvantage.
        """
        from dnd_manager.models.progression import get_creature_morale
        
        intimidator = None
        target = None
        
        for a in game_state.actors.values():
            if a.name.lower() == intimidator_name.lower():
                intimidator = a
            if a.name.lower() == target_name.lower():
                target = a
        
        if intimidator is None:
            return ToolResult(
                tool_name="intimidate",
                success=False,
                result=f"Character '{intimidator_name}' not found",
            )
        
        if target is None:
            return ToolResult(
                tool_name="intimidate",
                success=False,
                result=f"Target '{target_name}' not found",
            )
        
        # Get creature morale data for DC modifier
        creature_morale = get_creature_morale(target.name)
        dc_mod = creature_morale.get("intimidation_dc_mod", 0)
        
        # If creature is immune to intimidation (constructs, mindless undead)
        if dc_mod >= 100:
            return ToolResult(
                tool_name="intimidate",
                success=False,
                result=f"⚠️ **{target.name}** is immune to Intimidation (mindless or construct)!",
                data={"immune": True},
            )
        
        # DC = 10 + target's Wisdom modifier + creature type modifier
        target_wis_mod = (target.stats.wisdom - 10) // 2
        dc = 10 + target_wis_mod + dc_mod
        dc = max(5, min(dc, 25))  # Clamp to 5-25
        
        # Build Intimidation roll
        cha_mod = (intimidator.stats.charisma - 10) // 2
        prof_mult = intimidator.stats.skill_proficiencies.get("intimidation", 0)
        skill_bonus = cha_mod + (intimidator.stats.proficiency_bonus * prof_mult)
        
        if advantage and not disadvantage:
            expr = f"2d20kh1+{skill_bonus}"
        elif disadvantage and not advantage:
            expr = f"2d20kl1+{skill_bonus}"
        else:
            expr = f"1d20+{skill_bonus}"
        
        result = roll_dice(expr)
        success = result.total >= dc
        
        result_lines = [
            f"🗣️ **{intimidator.name}** attempts to intimidate **{target.name}**!",
            f"🎲 Intimidation: {result.details} = **{result.total}** vs DC **{dc}**",
        ]
        
        if success:
            # Reduce morale significantly
            morale_loss = 30 + (result.total - dc) * 2  # 30 base + 2 per point over DC
            old_morale = target.morale
            new_morale, is_broken = target.reduce_morale(morale_loss)
            
            result_lines.append(f"✅ **SUCCESS!** {target.name}'s morale drops from {old_morale}% to {new_morale}%!")
            
            if is_broken:
                result_lines.append(f"💀 **{target.name}'s morale is BROKEN!** They may flee or surrender!")
            elif new_morale <= 50:
                result_lines.append(f"😰 {target.name} looks shaken and uncertain...")
        else:
            result_lines.append(f"❌ **FAILED!** {target.name} remains defiant!")
            # Slight morale boost for resisting
            target.restore_morale(5)
        
        return ToolResult(
            tool_name="intimidate",
            success=True,
            result="\n".join(result_lines),
            data={
                "intimidator": intimidator.name,
                "target": target.name,
                "roll": result.total,
                "dc": dc,
                "success": success,
                "target_morale": target.morale,
            },
        )

    def handle_enemy_flees(entity_name: str) -> ToolResult:
        """Mark an enemy as fleeing and remove them from combat.
        
        Args:
            entity_name: The enemy that is fleeing.
        """
        from dnd_manager.models.progression import get_morale_modifier
        
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == entity_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="enemy_flees",
                success=False,
                result=f"Entity '{entity_name}' not found",
            )
        
        actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
        if actor_type == "player":
            return ToolResult(
                tool_name="enemy_flees",
                success=False,
                result="Players control their own flee actions!",
            )
        
        # Mark as fleeing
        actor.is_fleeing = True
        actor.is_in_combat = False
        
        # Remove from combat order if in combat
        if game_state.combat_active and actor.uid in game_state.combat_order:
            game_state.combat_order.remove(actor.uid)
        
        # Reduce morale of remaining allies
        result_lines = [f"🏃 **{actor.name}** flees from combat!"]
        
        allies_affected = []
        for other in game_state.actors.values():
            other_type = other.type.value if hasattr(other.type, 'value') else str(other.type)
            if other_type != "player" and other.uid != actor.uid and not other.is_fleeing and not other.has_surrendered:
                morale_loss = abs(get_morale_modifier("ally_fled"))
                other.reduce_morale(morale_loss)
                allies_affected.append(other.name)
        
        if allies_affected:
            result_lines.append(f"😰 Remaining enemies' morale drops: {', '.join(allies_affected)}")
        
        return ToolResult(
            tool_name="enemy_flees",
            success=True,
            result="\n".join(result_lines),
            data={
                "fled": actor.name,
                "allies_affected": allies_affected,
            },
        )

    def handle_enemy_surrenders(entity_name: str) -> ToolResult:
        """Mark an enemy as surrendered.
        
        Args:
            entity_name: The enemy that is surrendering.
        """
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == entity_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="enemy_surrenders",
                success=False,
                result=f"Entity '{entity_name}' not found",
            )
        
        actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
        if actor_type == "player":
            return ToolResult(
                tool_name="enemy_surrenders",
                success=False,
                result="Players control their own surrender actions!",
            )
        
        # Mark as surrendered
        actor.has_surrendered = True
        actor.is_in_combat = False
        actor.is_fleeing = False  # Not fleeing if surrendered
        
        # Remove from combat order if in combat
        if game_state.combat_active and actor.uid in game_state.combat_order:
            game_state.combat_order.remove(actor.uid)
        
        result_lines = [
            f"🏳️ **{actor.name}** surrenders!",
            f"*{actor.name} drops their weapons and pleads for mercy...*",
        ]
        
        return ToolResult(
            tool_name="enemy_surrenders",
            success=True,
            result="\n".join(result_lines),
            data={"surrendered": actor.name},
        )

    def handle_reduce_morale(entity_name: str, amount: int, reason: str = "") -> ToolResult:
        """Manually reduce an enemy's morale (for ally deaths, critical hits, etc.).
        
        Args:
            entity_name: The enemy whose morale to reduce.
            amount: How much to reduce morale by.
            reason: Why morale is being reduced.
        """
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == entity_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="reduce_morale",
                success=False,
                result=f"Entity '{entity_name}' not found",
            )
        
        old_morale = actor.morale
        new_morale, is_broken = actor.reduce_morale(amount, reason)
        
        reason_str = f" ({reason})" if reason else ""
        result_lines = [f"😰 **{actor.name}**'s morale: {old_morale}% → {new_morale}%{reason_str}"]
        
        if is_broken and old_morale > 25:
            result_lines.append(f"💀 **{actor.name}'s morale is BROKEN!** They may flee or surrender!")
        
        return ToolResult(
            tool_name="reduce_morale",
            success=True,
            result="\n".join(result_lines),
            data={
                "entity": actor.name,
                "old_morale": old_morale,
                "new_morale": new_morale,
                "is_broken": is_broken,
            },
        )

    def handle_roll_skill(
        character_name: str,
        skill_name: str,
        dc: int | None = None,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> ToolResult:
        """Roll a skill check for a character."""
        actor = None
        for a in game_state.actors.values():
            if a.name.lower() == character_name.lower():
                actor = a
                break
        
        if actor is None:
            return ToolResult(
                tool_name="roll_skill",
                success=False,
                result=f"Character '{character_name}' not found",
            )
        
        # Normalize skill name
        from dnd_manager.models.progression import SKILL_ABILITIES
        skill_key = skill_name.lower().replace(" ", "_")
        
        if skill_key not in SKILL_ABILITIES:
            return ToolResult(
                tool_name="roll_skill",
                success=False,
                result=f"Unknown skill: {skill_name}. Valid skills: {', '.join(SKILL_ABILITIES.keys())}",
            )
        
        ability = SKILL_ABILITIES[skill_key]
        stats = actor.stats
        
        # Get ability modifier
        ability_mod = getattr(stats, f"{ability[:3]}_mod", 0)
        
        # Get proficiency multiplier
        prof_mult = stats.skill_proficiencies.get(skill_key, 0)
        skill_bonus = ability_mod + (stats.proficiency_bonus * prof_mult)
        
        # Build roll expression
        if advantage and not disadvantage:
            expr = f"2d20kh1+{skill_bonus}"
            roll_type = "advantage"
        elif disadvantage and not advantage:
            expr = f"2d20kl1+{skill_bonus}"
            roll_type = "disadvantage"
        else:
            expr = f"1d20+{skill_bonus}"
            roll_type = "normal"
        
        result = roll_dice(expr)
        
        # Determine success if DC provided
        skill_display = skill_name.replace("_", " ").title()
        prof_marker = "★★" if prof_mult == 2 else ("★" if prof_mult == 1 else "")
        
        result_lines = [
            f"🎲 **{actor.name}** rolls {skill_display} {prof_marker}",
            f"{result.details} = **{result.total}**",
        ]
        
        if roll_type != "normal":
            result_lines[0] += f" ({roll_type})"
        
        success = None
        if dc is not None:
            success = result.total >= dc
            outcome = "✅ SUCCESS" if success else "❌ FAILURE"
            result_lines.append(f"vs DC {dc}: {outcome}")
        
        if result.is_critical:
            result_lines.append("⭐ **Natural 20!**")
        elif result.is_fumble:
            result_lines.append("💀 **Natural 1!**")
        
        return ToolResult(
            tool_name="roll_skill",
            success=True,
            result="\n".join(result_lines),
            data={
                "character": actor.name,
                "skill": skill_key,
                "total": result.total,
                "dc": dc,
                "passed": success,
                "natural_roll": result.natural_roll,
            },
        )

    def handle_create_entity(
        entity_name: str,
        count: int = 1,
        entity_type: str = "monster",
    ) -> ToolResult:
        """Create one or more entities (monsters/NPCs) using RAG data from rulebooks.
        
        NEURO-SYMBOLIC PRINCIPLE:
        Entity stats come from indexed rulebooks, not LLM invention.
        """
        # Warn if creating during combat (for reinforcements this is valid)
        combat_warning = ""
        if game_state.combat_active:
            combat_warning = "⚠️ Note: Combat is active. New entities will need initiative rolls. "
        
        if chroma_store is None:
            return ToolResult(
                tool_name="create_entity",
                success=False,
                result="No rulebooks indexed. Cannot create entity without stat data.",
            )
        
        # Clamp count to reasonable limits
        count = max(1, min(count, 10))
        
        # Search RAG for entity stats
        search_queries = [
            f"{entity_name} stats monster",
            f"{entity_name} stat block",
            f"{entity_name} creature statistics",
        ]
        
        rag_results: list[ChunkedDocument] = []
        for query in search_queries:
            results = chroma_store.search(query, n_results=3)
            rag_results.extend(results)
        
        # Default stats (used if RAG doesn't have the creature)
        stats = {
            "strength": 10,
            "dexterity": 10,
            "constitution": 10,
            "intelligence": 10,
            "wisdom": 10,
            "charisma": 10,
        }
        hp = 9  # Default for CR 1/8 humanoid
        ac = 12
        cr = "1/8"
        size = "Medium"
        attack_bonus = 3
        damage_dice = "1d6+1"
        
        # Parse stats from RAG content if found
        if rag_results:
            combined_content = "\n\n".join(doc.content for doc in rag_results[:3])
            
            # Try to extract ability scores
            ability_pattern = r"(?i)(STR|DEX|CON|INT|WIS|CHA)\s*(\d+)"
            for match in re.finditer(ability_pattern, combined_content):
                ability_name = match.group(1).upper()
                value = int(match.group(2))
                if ability_name == "STR":
                    stats["strength"] = value
                elif ability_name == "DEX":
                    stats["dexterity"] = value
                elif ability_name == "CON":
                    stats["constitution"] = value
                elif ability_name == "INT":
                    stats["intelligence"] = value
                elif ability_name == "WIS":
                    stats["wisdom"] = value
                elif ability_name == "CHA":
                    stats["charisma"] = value
            
            # Try to extract HP
            hp_pattern = r"(?i)hit\s*points?\s*(\d+)"
            hp_match = re.search(hp_pattern, combined_content)
            if hp_match:
                hp = int(hp_match.group(1))
            
            # Try to extract AC
            ac_pattern = r"(?i)armor\s*class\s*(\d+)"
            ac_match = re.search(ac_pattern, combined_content)
            if ac_match:
                ac = int(ac_match.group(1))
            
            # Try to extract CR
            cr_pattern = r"(?i)challenge\s*(?:rating)?\s*([\d/]+)"
            cr_match = re.search(cr_pattern, combined_content)
            if cr_match:
                cr = cr_match.group(1)
            
            # Try to extract size
            size_pattern = r"(?i)(tiny|small|medium|large|huge|gargantuan)"
            size_match = re.search(size_pattern, combined_content)
            if size_match:
                size = size_match.group(1).title()
            
            # Try to extract attack bonus
            attack_pattern = r"(?i)(?:melee|ranged).*?\+(\d+)\s*to hit"
            attack_match = re.search(attack_pattern, combined_content)
            if attack_match:
                attack_bonus = int(attack_match.group(1))
            
            # Try to extract damage
            damage_pattern = r"(?i)hit:.*?(\d+d\d+(?:\s*[+\-]\s*\d+)?)"
            damage_match = re.search(damage_pattern, combined_content)
            if damage_match:
                damage_dice = damage_match.group(1).replace(" ", "")
        
        # Import entity creation
        from dnd_manager.models.ecs import (
            ActorEntity,
            ActorType,
            DefenseComponent,
            HealthComponent,
            StatsComponent,
            JournalComponent,
        )
        
        # Determine actor type
        actor_type = ActorType.MONSTER
        if entity_type.lower() in ["npc", "npc_ally", "ally"]:
            actor_type = ActorType.NPC_ALLY
        
        # Count existing entities of this type for numbering
        base_name = entity_name.title()
        existing_count = sum(
            1 for a in game_state.actors.values()
            if a.name.startswith(base_name)
        )
        
        created_entities = []
        
        for i in range(count):
            # Number entities if multiple OR if others already exist
            if count > 1 or existing_count > 0:
                entity_number = existing_count + i + 1
                display_name = f"{base_name} {entity_number}"
            else:
                display_name = base_name
            
            # Get creature morale data
            from dnd_manager.models.progression import get_creature_morale
            creature_morale_data = get_creature_morale(base_name)
            
            # Create the entity
            entity = ActorEntity(
                name=display_name,
                type=actor_type,
                race=base_name,
                size=size,
                alignment="Neutral Evil" if actor_type == ActorType.MONSTER else "Neutral",
                challenge_rating=cr,
                stats=StatsComponent(
                    strength=stats["strength"],
                    dexterity=stats["dexterity"],
                    constitution=stats["constitution"],
                    intelligence=stats["intelligence"],
                    wisdom=stats["wisdom"],
                    charisma=stats["charisma"],
                ),
                health=HealthComponent(
                    hp_current=hp,
                    hp_max=hp,
                ),
                defense=DefenseComponent(
                    ac_base=ac,
                    uses_dex=False,  # Use fixed AC from stat block
                ),
                journal=JournalComponent(
                    goals=["Fight enemies", "Serve masters"],
                ),
                # Apply creature morale data
                morale=creature_morale_data["morale_base"],
                morale_base=creature_morale_data["morale_base"],
                flee_threshold=creature_morale_data["flee_threshold"],
            )
            
            # Store attack info in journal for DM reference
            entity.journal.personality_traits = [f"Attack: +{attack_bonus}, Damage: {damage_dice}"]
            
            # Add to game state
            game_state.add_actor(entity)
            created_entities.append(entity)
        
        # Build result message
        if count == 1:
            entity = created_entities[0]
            result_msg = f"👹 **{entity.name}** enters the fray!\n"
            result_msg += f"HP: {hp} | AC: {ac} | CR {cr}\n"
            result_msg += f"Attack: +{attack_bonus} | Damage: {damage_dice}"
        else:
            names = [e.name for e in created_entities]
            result_msg = f"👹 **{count}x {base_name}** enter the fray!\n"
            result_msg += f"({', '.join(names)})\n"
            result_msg += f"Each: HP {hp} | AC {ac} | CR {cr}\n"
            result_msg += f"Attack: +{attack_bonus} | Damage: {damage_dice}"
        
        # Add combat warning if applicable
        if combat_warning:
            result_msg = combat_warning + "\n" + result_msg
        
        return ToolResult(
            tool_name="create_entity",
            success=True,
            result=result_msg,
            data={
                "entities": [
                    {
                        "uid": str(e.uid),
                        "name": e.name,
                        "hp": hp,
                        "ac": ac,
                    }
                    for e in created_entities
                ],
                "count": count,
                "base_name": base_name,
                "cr": cr,
                "attack_bonus": attack_bonus,
                "damage_dice": damage_dice,
            },
        )

    # Build tool list
    tools = [
        DMTool(
            name="roll_dice",
            description="Roll dice using standard notation (e.g., '1d20+5', '2d6+3', '4d6kh3'). Use this for any random roll.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Dice expression like '1d20+5' or '2d6+3'",
                    },
                },
                "required": ["expression"],
            },
            handler=handle_roll_dice,
        ),
        DMTool(
            name="roll_check",
            description="Roll a skill check or ability check for a character against a DC.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character making the check"},
                    "skill_or_ability": {"type": "string", "description": "Skill (perception, athletics) or ability (strength, dexterity)"},
                    "dc": {"type": "integer", "description": "Difficulty class"},
                    "advantage": {"type": "boolean", "description": "Roll with advantage", "default": False},
                    "disadvantage": {"type": "boolean", "description": "Roll with disadvantage", "default": False},
                },
                "required": ["character_name", "skill_or_ability", "dc"],
            },
            handler=handle_roll_check,
        ),
        DMTool(
            name="roll_attack",
            description="Roll an attack from one character against another, including damage on hit.",
            parameters={
                "type": "object",
                "properties": {
                    "attacker_name": {"type": "string", "description": "Name of the attacker"},
                    "target_name": {"type": "string", "description": "Name of the target"},
                    "attack_bonus": {"type": "integer", "description": "Total attack bonus"},
                    "damage_dice": {"type": "string", "description": "Damage dice expression (e.g., '1d8+3')"},
                    "damage_type": {"type": "string", "description": "Type of damage (slashing, fire, etc.)"},
                    "advantage": {"type": "boolean", "default": False},
                    "disadvantage": {"type": "boolean", "default": False},
                },
                "required": ["attacker_name", "target_name", "attack_bonus", "damage_dice"],
            },
            handler=handle_roll_attack,
        ),
        DMTool(
            name="roll_save",
            description="Roll a saving throw for a character.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character"},
                    "ability": {"type": "string", "description": "Ability for the save (strength, dexterity, etc.)"},
                    "dc": {"type": "integer", "description": "Save DC"},
                    "advantage": {"type": "boolean", "default": False},
                    "disadvantage": {"type": "boolean", "default": False},
                },
                "required": ["character_name", "ability", "dc"],
            },
            handler=handle_roll_save,
        ),
        DMTool(
            name="get_character_status",
            description="Get the current status of a character (HP, AC, conditions, spell slots).",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character"},
                },
                "required": ["character_name"],
            },
            handler=handle_get_character_status,
        ),
        DMTool(
            name="apply_damage",
            description="Apply damage to a character. Use after confirming a hit.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character taking damage"},
                    "amount": {"type": "integer", "description": "Amount of damage"},
                    "damage_type": {"type": "string", "description": "Type of damage", "default": "untyped"},
                },
                "required": ["character_name", "amount"],
            },
            handler=handle_apply_damage,
        ),
        DMTool(
            name="apply_healing",
            description="Heal a character.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character to heal"},
                    "amount": {"type": "integer", "description": "Amount of healing"},
                },
                "required": ["character_name", "amount"],
            },
            handler=handle_apply_healing,
        ),
        DMTool(
            name="create_entity",
            description="SPAWN enemies/NPCs into the game state. Prefer spawning BEFORE combat. Creates monsters using stats from indexed rulebooks. Use count > 1 for multiple enemies (they get numbered: 'Goblin 1', 'Goblin 2', etc.). IMPORTANT: Check existing entities with list_combatants before creating new ones!",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Creature type (e.g., 'cultist', 'kobold', 'goblin', 'bandit', 'guard')"},
                    "count": {"type": "integer", "description": "Number to create (1-10). Use for groups like '3 goblins'.", "default": 1},
                    "entity_type": {"type": "string", "description": "Type: 'monster' for enemies, 'npc_ally' for friendly NPCs", "default": "monster"},
                },
                "required": ["entity_name"],
            },
            handler=handle_create_entity,
        ),
        DMTool(
            name="list_combatants",
            description="List all entities currently in the game (party members and enemies). Use this to see exact names for targeting attacks. Shows combat status if active.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=handle_list_combatants,
        ),
        DMTool(
            name="get_current_turn",
            description="Get whose turn it is in combat and the full initiative order. Use this to track turn order!",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=handle_get_current_turn,
        ),
        DMTool(
            name="start_combat",
            description="⚔️ START COMBAT! Call this AFTER spawning enemies. Rolls initiative for everyone and sets up turn order. REQUIRED to enable combat tracking.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=handle_start_combat,
        ),
        DMTool(
            name="end_turn",
            description="End the current combatant's turn and advance to the next in initiative order. Call this after each combatant finishes their turn.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=handle_end_turn,
        ),
        DMTool(
            name="end_combat",
            description="End combat when all enemies are defeated or combat otherwise ends. Cleans up combat state.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=handle_end_combat,
        ),
        DMTool(
            name="remove_entity",
            description="Remove a defeated or fleeing enemy/NPC from the game. Use after an enemy dies or flees.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Exact name of the entity to remove (e.g., 'Cultist 1')"},
                },
                "required": ["entity_name"],
            },
            handler=handle_remove_entity,
        ),
        DMTool(
            name="award_xp",
            description="Award experience points to the party. Use after defeating enemies or completing objectives. XP is split evenly among party members. Will announce if anyone levels up.",
            parameters={
                "type": "object",
                "properties": {
                    "amount": {"type": "integer", "description": "Total XP to award (will be split among party)"},
                    "reason": {"type": "string", "description": "Brief reason (e.g., 'defeating the cultists', 'rescuing the villagers')", "default": "encounter"},
                },
                "required": ["amount"],
            },
            handler=handle_award_xp,
        ),
        DMTool(
            name="get_xp_for_cr",
            description="Look up the XP value for a creature's Challenge Rating. Use this to calculate how much XP to award.",
            parameters={
                "type": "object",
                "properties": {
                    "challenge_rating": {"type": "string", "description": "CR value (e.g., '1/4', '1', '5')"},
                },
                "required": ["challenge_rating"],
            },
            handler=handle_get_xp_for_cr,
        ),
        DMTool(
            name="cast_spell",
            description="Cast a spell, consuming a spell slot if required. For cantrips, use spell_level=0. Returns caster's spell save DC and attack bonus for resolving effects.",
            parameters={
                "type": "object",
                "properties": {
                    "caster_name": {"type": "string", "description": "Name of the spellcaster"},
                    "spell_name": {"type": "string", "description": "Name of the spell (e.g., 'Magic Missile', 'Fireball')"},
                    "spell_level": {"type": "integer", "description": "Spell slot level to use (0 for cantrips)", "default": 0},
                    "target_name": {"type": "string", "description": "Optional target of the spell"},
                },
                "required": ["caster_name", "spell_name"],
            },
            handler=handle_cast_spell,
        ),
        DMTool(
            name="lookup_spell",
            description="Look up a spell's details from indexed rulebooks. Use to find casting time, range, components, duration, and effects.",
            parameters={
                "type": "object",
                "properties": {
                    "spell_name": {"type": "string", "description": "Name of the spell to look up"},
                },
                "required": ["spell_name"],
            },
            handler=handle_lookup_spell,
        ),
        DMTool(
            name="get_character_spells",
            description="Get a character's known spells, prepared spells, and available spell slots.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character"},
                },
                "required": ["character_name"],
            },
            handler=handle_get_character_spells,
        ),
        DMTool(
            name="use_feature",
            description="Use a class or racial feature/ability. Tracks limited uses for features like Rage, Action Surge, Second Wind. Returns remaining uses.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character"},
                    "feature_name": {"type": "string", "description": "Name of the feature (e.g., 'Rage', 'Second Wind', 'Sneak Attack')"},
                },
                "required": ["character_name", "feature_name"],
            },
            handler=handle_use_feature,
        ),
        DMTool(
            name="short_rest",
            description="Take a short rest (1+ hours). Restores: Warlock pact slots, features that recharge on short rest (Second Wind, Action Surge, Channel Divinity, Wild Shape). Can target one character or entire party.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character to rest, or omit for entire party"},
                },
                "required": [],
            },
            handler=handle_short_rest,
        ),
        DMTool(
            name="long_rest",
            description="Take a long rest (8 hours). Restores: All HP, all spell slots, all features and abilities. Can target one character or entire party.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character to rest, or omit for entire party"},
                },
                "required": [],
            },
            handler=handle_long_rest,
        ),
        DMTool(
            name="get_character_features",
            description="Get all features, abilities, and class resources for a character.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character"},
                },
                "required": ["character_name"],
            },
            handler=handle_get_character_features,
        ),
        DMTool(
            name="roll_skill",
            description="Roll a skill check for a character. Automatically applies proficiency and expertise bonuses.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Name of the character"},
                    "skill_name": {"type": "string", "description": "Skill name (e.g., 'Perception', 'Stealth', 'Athletics')"},
                    "dc": {"type": "integer", "description": "Difficulty Class (optional - if provided, reports success/failure)"},
                    "advantage": {"type": "boolean", "description": "Roll with advantage", "default": False},
                    "disadvantage": {"type": "boolean", "description": "Roll with disadvantage", "default": False},
                },
                "required": ["character_name", "skill_name"],
            },
            handler=handle_roll_skill,
        ),
        DMTool(
            name="list_effects",
            description="List all active buff/debuff effects on a character or all characters. Use to check what spells/abilities are affecting them.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character to check, or omit for all characters"},
                },
                "required": [],
            },
            handler=handle_list_effects,
        ),
        DMTool(
            name="dispel_effect",
            description="Remove an active effect from a character. Use for dispel magic, concentration loss, or effect ending.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character with the effect"},
                    "effect_name": {"type": "string", "description": "Name of the effect to remove (e.g., 'Mage Armor', 'Bless')"},
                },
                "required": ["character_name", "effect_name"],
            },
            handler=handle_dispel_effect,
        ),
        DMTool(
            name="break_concentration",
            description="Break a character's concentration, ending all spells they are concentrating on. Use when a character takes damage and fails concentration save, or is incapacitated.",
            parameters={
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character who loses concentration"},
                },
                "required": ["character_name"],
            },
            handler=handle_break_concentration,
        ),
        # Morale tools
        DMTool(
            name="check_morale",
            description="Check morale status of enemies. Use when enemies take heavy losses, are intimidated, or situation changes dramatically. Returns whether enemies might flee/surrender.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Specific enemy to check, or omit for all enemies"},
                },
                "required": [],
            },
            handler=handle_check_morale,
        ),
        DMTool(
            name="intimidate",
            description="Roll Intimidation against an enemy. On success, significantly reduces their morale. Use when a player tries to scare, threaten, or cow an enemy.",
            parameters={
                "type": "object",
                "properties": {
                    "intimidator_name": {"type": "string", "description": "Character making the Intimidation attempt"},
                    "target_name": {"type": "string", "description": "Enemy being intimidated"},
                    "advantage": {"type": "boolean", "description": "Roll with advantage", "default": False},
                    "disadvantage": {"type": "boolean", "description": "Roll with disadvantage", "default": False},
                },
                "required": ["intimidator_name", "target_name"],
            },
            handler=handle_intimidate,
        ),
        DMTool(
            name="enemy_flees",
            description="Mark an enemy as fleeing from combat. Removes them from initiative order. Use when an enemy's morale breaks or they choose to run.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The enemy that is fleeing"},
                },
                "required": ["entity_name"],
            },
            handler=handle_enemy_flees,
        ),
        DMTool(
            name="enemy_surrenders",
            description="Mark an enemy as surrendered. Removes them from initiative. Use when an enemy gives up and begs for mercy.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The enemy that is surrendering"},
                },
                "required": ["entity_name"],
            },
            handler=handle_enemy_surrenders,
        ),
        DMTool(
            name="reduce_morale",
            description="Manually reduce an enemy's morale. Use when allies die, critical hits land, or other demoralizing events occur.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The enemy whose morale to reduce"},
                    "amount": {"type": "integer", "description": "How much to reduce morale (1-50)"},
                    "reason": {"type": "string", "description": "Why morale is dropping (e.g., 'ally killed', 'critical hit')"},
                },
                "required": ["entity_name", "amount"],
            },
            handler=handle_reduce_morale,
        ),
    ]

    return tools


# =============================================================================
# DM System Prompt
# =============================================================================


DM_SYSTEM_PROMPT = """You are the Dungeon Master for "{campaign_name}". Stay fully in character at all times.

## VOICE & STYLE

- **STAY IN CHARACTER**: You ARE the Dungeon Master. Never break the fourth wall. Never use phrases like "based on the module", "as written", "canonically", "Note:", or any meta-commentary.
- **IMMERSIVE NARRATION**: Describe the world vividly. Use sensory details. Make the players feel like they're there.
- **NO META-TALK**: Don't mention game mechanics in your narration unless the player asks. Don't explain why you're doing something. Just DO it.
- **DIRECT ADDRESS**: Speak directly to the character(s), not about them.

## INFORMATION SECRECY (CRITICAL!)

- **ONLY DESCRIBE WHAT THE CHARACTER PERCEIVES**: Don't reveal enemy names, faction identities, or plot details the character hasn't discovered yet.
- **NO SPOILERS**: If the adventure says "these are Cult of the Dragon members", describe them as "robed figures with purple insignias" until the character LEARNS who they are.
- **SECRETS STAY SECRET**: Don't explain the significance of symbols, items, or NPCs until the character investigates or is told in-game.
- **DESCRIBE, DON'T IDENTIFY**: Say "a tall man in dark robes" not "the cult leader Mondath" unless the character knows their name.
- **LET PLAYERS DISCOVER**: The joy of D&D is discovery. Reveal information through gameplay, NPC dialogue, found documents, or successful checks - not narration.

## ⚠️ ABSOLUTE RULES - NEVER VIOLATE THESE ⚠️

**YOU MUST USE TOOLS FOR ALL GAME MECHANICS. NEVER NARRATE OUTCOMES WITHOUT TOOL CALLS.**

If you need to:
- Roll dice → CALL `roll_dice` FIRST, then narrate the result
- Attack → CALL `roll_attack` FIRST, then narrate hit/miss based on result
- Cast a spell → CALL `cast_spell` FIRST, then narrate
- Deal damage → CALL `apply_damage` to update HP
- Make a skill check → CALL `roll_skill` or `roll_dice` FIRST

**NEVER write things like "the goblin attacks and deals 5 damage" without calling tools!**
**NEVER invent dice results like "rolls a 15" without calling roll_dice!**
**NEVER say "the spell hits" without calling cast_spell or roll_attack!**

If you narrate combat actions without tool calls, the game state becomes corrupted and the player loses trust.

## CRITICAL RULES

1. **FOLLOW THE ADVENTURE**: Use the plot, NPCs, locations, and events from the adventure context below. Adapt to player choices while guiding them through key story beats.

2. **NEVER FUDGE DICE**: Use roll_dice, roll_check, roll_attack, or roll_save tools for ALL random outcomes. Never invent dice results.

3. **CRITICAL HITS & FUMBLES**:
   - Natural 20 on attack = CRITICAL HIT (double damage dice)
   - Natural 1 on attack = CRITICAL MISS (automatic failure, describe something dramatic)
   - Natural 20 on ability checks = exceptional success (describe impressive outcome)
   - Natural 1 on ability checks = embarrassing failure (describe comical/dramatic mishap)

4. **YOU KNOW THE CHARACTERS**: The game state below shows each character's HP, AC, equipment, abilities, and inventory. NEVER ask the player what weapons they have or what they can do - you already know! Use get_character_status if you need more detail.

5. **APPLY CHANGES**: After attacks/effects, use apply_damage or apply_healing to update state.

6. **USE 5E RULES**: Advantage/disadvantage, proper DCs (Easy 10, Medium 15, Hard 20, Very Hard 25).

7. **⚠️ SPAWN BEFORE COMBAT (CRITICAL!)**: 
   - Before ANY combat, you MUST call `create_entity` to spawn enemies into the game state
   - Example: "3 cultists attack" → call `create_entity(entity_name="cultist", count=3)`
   - This creates "Cultist 1", "Cultist 2", "Cultist 3" that can be attacked
   - If you try to attack without spawning, the attack will fail with "Target not found"
   - ALWAYS spawn creatures FIRST, then narrate combat, then roll attacks

8. **⚔️ COMBAT RULES (STRICT!)**:

   **SETUP PHASE (do this ONCE when combat starts):**
   1. Call `create_entity` to spawn ALL enemies (with count for groups)
   2. Call `start_combat` - this rolls initiative for EVERYONE and creates the turn order
   3. The turn order will be shown - proceed with combat
   
   **EACH TURN - FOLLOW THIS EXACT PATTERN:**
   1. Check whose turn it is (use `get_current_turn` or check game state)
   2. If it's the PLAYER's turn:
      - Wait for player input
      - When they act, CALL THE APPROPRIATE TOOL (roll_attack, cast_spell, roll_skill)
      - Narrate the result AFTER the tool returns
      - Call `end_turn`
   3. If it's an ENEMY's turn:
      - Call `roll_attack` for that ONE enemy
      - If hit, call `apply_damage` to update the player's HP
      - Narrate AFTER the tool calls
      - Call `end_turn`
   
   **⚠️ FORBIDDEN ACTIONS (will corrupt the game):**
   - ❌ NEVER narrate "the goblin attacks and deals 5 damage" without calling roll_attack AND apply_damage
   - ❌ NEVER have multiple enemies attack at once - each gets ONE turn
   - ❌ NEVER invent dice results - always call roll_dice/roll_attack
   - ❌ NEVER skip turns or change initiative order
   
   **✅ REQUIRED ACTIONS:**
   - ✅ Call `start_combat` ONCE after spawning enemies (auto-rolls initiative)
   - ✅ Use EXACT entity names (e.g., "Goblin 1", not "Goblin")
   - ✅ Call `roll_attack(attacker, target)` for EVERY attack
   - ✅ Call `apply_damage(target, amount)` for EVERY damage dealt
   - ✅ Call `end_turn` after EVERY combatant finishes their turn
   - ✅ Call `end_combat` when all enemies are dead or combat ends

9. **AWARD XP AFTER ENCOUNTERS**:
   - When combat ends, use get_xp_for_cr to calculate total XP
   - Use award_xp to give XP to the party (auto-splits among members)
   - Also award XP for non-combat achievements (roleplay, puzzles, objectives)

10. **🏃 ENEMY MORALE & FLEEING (IMPORTANT!)**:
   
   **Morale is a GUIDE, not a rule. YOU (the DM) have final say on whether enemies flee!**
   
   The morale system helps you track enemy willpower, but CONTEXT matters:
   - A goblin protecting its young might fight to the death despite broken morale
   - An enemy cornered with no escape route CAN'T flee
   - A fanatical cultist might NEVER surrender
   - A creature magically compelled or enslaved must fight
   - Mindless creatures (undead, constructs) don't have morale
   
   **USE MORALE AS GUIDANCE:**
   - Low morale = enemy is SCARED, not forced to flee
   - Consider: Can they escape? Do they have reason to fight? What's their personality?
   - The `check_morale` tool tells you their mental state; YOU decide what they do
   
   **MORALE TIERS (advisory):**
   - 75-100%: CONFIDENT - likely fights aggressively
   - 50-74%: STEADY - fights normally
   - 25-49%: SHAKEN - might parley, retreat, or fight desperately
   - 0-24%: BROKEN - strongly inclined to flee/surrender (but context matters!)
   
   **CREATURE TENDENCIES (typical, not absolute):**
   - 🐀 COWARDS (goblins, kobolds): Usually flee at first sign of trouble
   - ⚔️ AVERAGE (orcs, bandits): Follow standard morale patterns
   - 🦁 BRAVE (knights, veterans): Usually fight until gravely wounded
   - 💀 FEARLESS (undead, constructs, dragons): Typically fight to the death
   
   **WHEN PLAYERS TRY INTIMIDATION:**
   1. Call `intimidate(intimidator, target)` to roll Charisma (Intimidation) vs target's resolve
   2. If successful, morale drops significantly
   3. Based on context, decide: Does enemy flee? Surrender? Fight desperately? Negotiate?
   4. Use `enemy_flees` or `enemy_surrenders` if appropriate
   
   **REASONS AN ENEMY MIGHT FIGHT DESPITE BROKEN MORALE:**
   - No escape route available
   - Protecting something/someone precious
   - Religious/fanatical devotion
   - More afraid of their master than the player
   - Berserker rage or magical compulsion
   - Cornered with nothing to lose
   
   **GIVE PLAYERS OPTIONS!** Combat doesn't have to end in death. Fleeing enemies can:
   - Be captured and interrogated
   - Escape and return with reinforcements
   - Provide plot information if spared

## RESPONSE FORMAT

1. Narrate what happens (in character, immersively)
2. Call dice tools if needed
3. Narrate outcomes based on results
4. Apply state changes silently

## CURRENT GAME STATE
{game_state_context}

## ADVENTURE CONTENT (USE THIS FOR YOUR NARRATION)
{rag_context}

Use the adventure content above to describe locations, NPCs, and events accurately. The story comes from this module - you bring it to life."""


# =============================================================================
# DM Response
# =============================================================================


@dataclass
class DMResponse:
    """Response from the DM orchestrator."""

    narrative: str
    """The story/narrative response."""

    tool_results: list[ToolResult] = field(default_factory=list)
    """Results from any tools called."""

    state_changes: list[StateUpdateResult] = field(default_factory=list)
    """State changes applied."""

    raw_llm_response: str = ""
    """Raw LLM output for debugging."""


# =============================================================================
# DM Orchestrator
# =============================================================================


class DMOrchestrator:
    """The Neuro-Symbolic DM - orchestrates the game loop.
    
    ARCHITECTURE:
    1. User input → Context assembly (state + RAG)
    2. LLM reasoning → Tool calls (dice, state queries)
    3. Python execution → Results back to LLM
    4. LLM narrative → Response to user
    """

    def __init__(
        self,
        game_state: GameState,
        chroma_store: ChromaStore | None = None,
        openrouter_client: OpenRouterClient | None = None,
        model: str = "google/gemini-3-pro-preview",
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the DM orchestrator.
        
        Args:
            game_state: The game state (source of truth).
            chroma_store: Vector store for RAG retrieval.
            openrouter_client: OpenRouter client.
            model: LLM model to use for reasoning.
            conversation_history: Existing conversation history to continue from.
        """
        self.game_state = game_state
        self.chroma_store = chroma_store or ChromaStore()
        self.client = openrouter_client or OpenRouterClient()
        self.model = model
        self.tools = create_dm_tools(game_state, chroma_store=self.chroma_store)
        self.conversation_history: list[dict[str, Any]] = conversation_history or []

        logger.info(f"DMOrchestrator initialized with model {model}, history={len(self.conversation_history)} messages")

    def _clean_rag_content(self, content: str) -> str:
        """Clean RAG content of OCR artifacts and formatting issues.
        
        Args:
            content: Raw content from RAG.
            
        Returns:
            Cleaned content string.
        """
        import re
        
        # Remove bracketed artifacts [like this]
        content = re.sub(r'\s*\[.*?\]\s*', ' ', content)
        
        # Remove bullet points and weird characters
        content = re.sub(r'\s*[·•■□▪▫◦‣⁃]\s*', ' ', content)
        
        # Remove non-ASCII (OCR garbage)
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        
        # Fix multiple spaces/newlines
        content = re.sub(r' {2,}', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove lines that are mostly punctuation/symbols (OCR artifacts)
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            # Skip lines that are too short or mostly non-alphanumeric
            if len(line.strip()) < 3:
                continue
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in line) / max(len(line), 1)
            if alpha_ratio > 0.7:  # At least 70% alphanumeric
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()

    def _get_rag_context(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant context from RAG store.
        
        Prioritizes adventure content for the current campaign.
        
        Args:
            query: User's input to find relevant info for.
            n_results: Number of results to retrieve.
            
        Returns:
            Formatted context string.
        """
        try:
            campaign_name = self.game_state.campaign_name
            context_parts = []
            
            # First, search for adventure-specific content
            adventure_queries = [
                f"{campaign_name} {query}",
                f"{campaign_name} location scene",
                query,
            ]
            
            from dnd_manager.ingestion.universal_loader import DocumentType
            
            adventure_results = []
            for q in adventure_queries:
                results = self.chroma_store.search(q, n_results=3, doc_type=DocumentType.ADVENTURE)
                adventure_results.extend(results)
            
            # Deduplicate by chunk_id
            seen_ids = set()
            unique_adventure = []
            for doc in adventure_results:
                if doc.chunk_id not in seen_ids:
                    seen_ids.add(doc.chunk_id)
                    unique_adventure.append(doc)
            
            # Add adventure content (more text for story context)
            for doc in unique_adventure[:4]:
                title = doc.metadata.get("title", "Adventure")
                # Clean the content before adding
                clean_content = self._clean_rag_content(doc.content[:1200])
                if clean_content:
                    context_parts.append(f"**[ADVENTURE - {title}]**:\n{clean_content}")
            
            # Also search for rules/sourcebook content
            rules_results = self.chroma_store.search(query, n_results=2)
            for doc in rules_results:
                if doc.chunk_id not in seen_ids:
                    title = doc.metadata.get("title", "Rules")
                    source = doc.source
                    clean_content = self._clean_rag_content(doc.content[:600])
                    if clean_content:
                        context_parts.append(f"**[RULES - {title}]** ({source}):\n{clean_content}")

            if not context_parts:
                return "No relevant adventure or rules content found."

            return "\n\n---\n\n".join(context_parts)

        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return "RAG retrieval unavailable."

    def _build_system_prompt(self, rag_context: str) -> str:
        """Build the system prompt with current context."""
        game_context = self.game_state.to_ai_context()
        return DM_SYSTEM_PROMPT.format(
            campaign_name=self.game_state.campaign_name,
            game_state_context=game_context,
            rag_context=rag_context,
        )

    def _execute_tool_call(self, tool_call: Any) -> ToolResult:
        """Execute a tool call from the LLM.
        
        Args:
            tool_call: OpenAI tool call object.
            
        Returns:
            ToolResult from execution.
        """
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}

        # Find and execute the tool
        for tool in self.tools:
            if tool.name == tool_name:
                logger.info(f"Executing tool: {tool_name}", args=arguments)
                return tool.execute(**arguments)

        return ToolResult(
            tool_name=tool_name,
            success=False,
            result=f"Unknown tool: {tool_name}",
        )

    def process_input(self, user_input: str) -> DMResponse:
        """Process user input and generate DM response.
        
        This is the main game loop entry point.
        
        Args:
            user_input: The player's action or speech.
            
        Returns:
            DMResponse with narrative and results.
        """
        from openai import OpenAI

        logger.info("Processing user input", input_preview=user_input[:100])

        # Step 1: Get RAG context
        rag_context = self._get_rag_context(user_input)

        # Step 2: Build messages
        system_prompt = self._build_system_prompt(rag_context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history (last 10 exchanges)
        messages.extend(self.conversation_history[-20:])

        # Add combat reminder if in combat
        if self.game_state.combat_active:
            current = self.game_state.get_current_combatant()
            current_name = current.name if current else "unknown"
            combat_reminder = (
                f"\n\n⚔️ COMBAT IS ACTIVE - Round {self.game_state.combat_round}\n"
                f"Current turn: {current_name}\n"
                f"⚠️ YOU MUST USE TOOLS: roll_attack, apply_damage, cast_spell, end_turn\n"
                f"DO NOT narrate attacks or damage without calling tools first!"
            )
            messages.append({
                "role": "system", 
                "content": combat_reminder
            })

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Step 3: Get tool schemas
        tools_schema = [tool.to_openai_schema() for tool in self.tools]

        # Step 4: Call LLM
        client = self.client._get_client()

        tool_results: list[ToolResult] = []
        state_changes: list[StateUpdateResult] = []
        final_narrative = ""

        try:
            # Force tool usage during combat to prevent hallucinated dice/damage
            tool_choice_setting = "required" if self.game_state.combat_active else "auto"
            
            # Initial call
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema,
                tool_choice=tool_choice_setting,
                temperature=0.7,
                max_tokens=2048,
            )

            message = response.choices[0].message

            # Handle tool calls in a loop
            while message.tool_calls:
                # Execute each tool call
                tool_messages = []
                for tool_call in message.tool_calls:
                    result = self._execute_tool_call(tool_call)
                    tool_results.append(result)

                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.result,
                    })

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Add tool results
                messages.extend(tool_messages)

                # Continue conversation - use auto for follow-up calls to allow final narrative
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",  # Allow narrative after tool calls
                    temperature=0.7,
                    max_tokens=2048,
                )

                message = response.choices[0].message

            # Final narrative
            final_narrative = message.content or ""

            # Update conversation history - INCLUDE tool results so DM remembers its actions!
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Build assistant message that includes tool call summaries
            assistant_content = final_narrative
            if tool_results:
                # Add a summary of tool actions to help DM remember what happened
                tool_summary_lines = ["\n\n[TOOL ACTIONS TAKEN THIS TURN:]"]
                for tr in tool_results:
                    # Include key information from tool results
                    tool_summary_lines.append(f"- {tr.tool_name}: {tr.result[:300]}...")
                assistant_content = final_narrative + "\n".join(tool_summary_lines)
            
            self.conversation_history.append({"role": "assistant", "content": assistant_content})

            # Keep history manageable
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

        except Exception as exc:
            logger.exception("DM processing failed")
            final_narrative = f"*The DM's crystal ball clouds over...* (Error: {exc})"

        return DMResponse(
            narrative=final_narrative,
            tool_results=tool_results,
            state_changes=state_changes,
            raw_llm_response=final_narrative,
        )

    def start_combat(self, enemies: list[ActorEntity]) -> DMResponse:
        """Start a combat encounter.
        
        Args:
            enemies: List of enemy actors to add to combat.
            
        Returns:
            DMResponse announcing combat start.
        """
        # Add enemies to game state
        for enemy in enemies:
            self.game_state.add_actor(enemy)

        # Roll initiative for everyone
        initiative_results = []

        all_combatants = list(self.game_state.get_party()) + enemies

        for actor in all_combatants:
            init_roll = roll_dice(f"1d20+{actor.stats.dex_mod}")
            actor.initiative = init_roll.total
            actor.is_in_combat = True
            initiative_results.append((actor, init_roll))

        # Sort by initiative (descending)
        initiative_results.sort(key=lambda x: (x[1].total, x[0].stats.dexterity), reverse=True)

        # Set combat order
        self.game_state.combat_order = [actor.uid for actor, _ in initiative_results]
        self.game_state.combat_active = True
        self.game_state.combat_round = 1
        self.game_state.combat_turn_index = 0

        # Build narrative
        lines = ["⚔️ **COMBAT BEGINS!** Roll for initiative!\n"]
        for actor, roll in initiative_results:
            actor_type = actor.type.value if hasattr(actor.type, 'value') else str(actor.type)
            emoji = "🧙" if actor_type == "player" else "👹"
            lines.append(f"{emoji} {actor.name}: {roll.details} = **{roll.total}**")

        current = self.game_state.get_current_combatant()
        if current:
            lines.append(f"\n**{current.name}** acts first!")

        return DMResponse(
            narrative="\n".join(lines),
            tool_results=[],
        )

    def end_combat(self) -> DMResponse:
        """End the current combat encounter."""
        self.game_state.combat_active = False
        self.game_state.combat_order = []
        self.game_state.combat_round = 0
        self.game_state.combat_turn_index = 0

        # Reset combat flags on actors
        for actor in self.game_state.actors.values():
            actor.is_in_combat = False

        return DMResponse(
            narrative="🏁 **Combat has ended.** The dust settles...",
            tool_results=[],
        )

    def next_turn(self) -> str:
        """Advance to the next turn in combat.
        
        Returns:
            Name of the combatant whose turn it now is.
        """
        if not self.game_state.combat_active:
            return ""

        self.game_state.advance_turn()
        current = self.game_state.get_current_combatant()

        if current:
            return current.name
        return ""


__all__ = [
    # Dice rolling
    "DiceResult",
    "roll_dice",
    "roll_check",
    "roll_save",
    "roll_attack",
    "roll_damage",
    # Tools
    "ToolResult",
    "DMTool",
    "create_dm_tools",
    # Orchestrator
    "DMResponse",
    "DMOrchestrator",
    "DM_SYSTEM_PROMPT",
]
