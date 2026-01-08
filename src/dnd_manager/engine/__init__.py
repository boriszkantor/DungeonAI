"""Game engine module for the D&D 5E AI Campaign Manager.

This module provides the core game engine functionality including
the game loop, turn management, dice rolling, AI decision-making,
and combat action tools.

Submodules:
    dice: Dice rolling with D&D 5E mechanics (d20 library)
    tools: Game action tools for combat (attack, cast, move, etc.)
    ai_agent: AI agent for autonomous combatant decisions
    turn_manager: Initiative and turn order tracking
    game_loop: Overall game state management
    loop: Combat turn loop with AI/user branching

Example:
    >>> from dnd_manager.engine import (
    ...     GameLoop, DungeonAgent, roll_attack, TurnStatus
    ... )
    >>>
    >>> # Create game loop for a scene
    >>> loop = GameLoop(scene)
    >>>
    >>> # Process turns until user input needed
    >>> result = loop.run_until_user_input()
    >>> if result.status == TurnStatus.WAITING_FOR_USER:
    ...     print(f"Waiting for {result.combatant_name}")
"""

from __future__ import annotations

# =============================================================================
# Dice Rolling
# =============================================================================
from dnd_manager.engine.dice import (
    DiceExpression,
    DiceRoller,
    RollType,
    roll,
)

# =============================================================================
# Game Actions (Tools)
# =============================================================================
from dnd_manager.engine.tools import (
    ToolCall,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    cast_spell,
    check_saving_throw,
    check_skill,
    dash,
    disengage,
    dodge,
    execute_tool,
    execute_tool_calls,
    get_all_tools,
    get_tool,
    get_tools_as_openai_schema,
    get_tools_by_category,
    help_action,
    hide,
    move,
    roll_attack,
    roll_damage,
    tool,
    use_item,
)

# =============================================================================
# AI Agent
# =============================================================================
from dnd_manager.engine.ai_agent import (
    DUNGEON_AGENT_SYSTEM_PROMPT,
    AgentDecision,
    BattlefieldContext,
    DungeonAgent,
    build_battlefield_context,
    build_directive_guidelines,
    build_persona_context,
    get_ai_action,
)

# =============================================================================
# Turn Management
# =============================================================================
from dnd_manager.engine.turn_manager import (
    InitiativeEntry,
    InitiativeTracker,
    TurnManager,
    TurnState,
)

# =============================================================================
# Game Loop
# =============================================================================
from dnd_manager.engine.game_loop import (
    GameEvent,
    GameLoop as GameLoopLegacy,
    GameState,
)

from dnd_manager.engine.loop import (
    CombatEndCondition,
    DEFAULT_END_CONDITIONS,
    GameLoop,
    TurnResult,
    TurnStatus,
    all_enemies_defeated,
    all_players_defeated,
)


__all__ = [
    # Dice Rolling
    "DiceExpression",
    "DiceRoller",
    "RollType",
    "roll",
    # Tools
    "ToolCategory",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "tool",
    "get_tool",
    "get_all_tools",
    "get_tools_by_category",
    "get_tools_as_openai_schema",
    "execute_tool",
    "execute_tool_calls",
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
    # AI Agent
    "BattlefieldContext",
    "build_battlefield_context",
    "build_persona_context",
    "build_directive_guidelines",
    "AgentDecision",
    "DungeonAgent",
    "get_ai_action",
    "DUNGEON_AGENT_SYSTEM_PROMPT",
    # Turn Management
    "InitiativeEntry",
    "InitiativeTracker",
    "TurnState",
    "TurnManager",
    # Game Loop (Legacy)
    "GameState",
    "GameEvent",
    "GameLoopLegacy",
    # Game Loop (Combat)
    "TurnStatus",
    "TurnResult",
    "CombatEndCondition",
    "GameLoop",
    "all_enemies_defeated",
    "all_players_defeated",
    "DEFAULT_END_CONDITIONS",
]
