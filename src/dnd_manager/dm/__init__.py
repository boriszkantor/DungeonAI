"""Dungeon Master module for DungeonAI.

This module provides the AI Dungeon Master functionality:
- Game loop orchestration
- LLM-based reasoning and narrative
- Tool execution (dice rolling, state queries)
- RAG context retrieval

NEURO-SYMBOLIC PRINCIPLE:
The DM module uses LLMs for reasoning and narrative,
but all game mechanics are executed by Python.
Dice rolls use the d20 library - LLMs NEVER generate random numbers.
"""

from __future__ import annotations

from dnd_manager.dm.orchestrator import (
    DMOrchestrator,
    DMResponse,
    DMTool,
    ToolResult,
    roll_dice,
    roll_check,
    roll_save,
    roll_attack,
    roll_damage,
)


__all__ = [
    "DMOrchestrator",
    "DMResponse",
    "DMTool",
    "ToolResult",
    "roll_dice",
    "roll_check",
    "roll_save",
    "roll_attack",
    "roll_damage",
]
