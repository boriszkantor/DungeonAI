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

from .dice import (
    DiceResult,
    roll_attack,
    roll_check,
    roll_damage,
    roll_dice,
    roll_save,
)
from .orchestrator import DMOrchestrator, DMResponse
from .tools.base import DMTool, ToolResult

__all__ = [
    "DMOrchestrator",
    "DMResponse",
    "DMTool",
    "ToolResult",
    "DiceResult",
    "roll_dice",
    "roll_check",
    "roll_save",
    "roll_attack",
    "roll_damage",
]
