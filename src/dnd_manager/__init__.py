"""DungeonAI - Neuro-Symbolic D&D 5E Engine.

A modular monolith for AI-powered tabletop RPG management.

NEURO-SYMBOLIC ARCHITECTURE:
- Python owns TRUTH (GameState, dice rolls via d20, rule validation)
- LLMs handle INTERFACE (narrative, reasoning, vision extraction)
- LLMs NEVER directly mutate state or generate random numbers

Example:
    >>> from dnd_manager import GameState, create_player_character, DMOrchestrator
    >>>
    >>> # Create game state (THE SOURCE OF TRUTH)
    >>> state = GameState(campaign_name="Lost Mines")
    >>>
    >>> # Add a player
    >>> hero = create_player_character("Thorin", race="Dwarf", class_name="Fighter", level=3)
    >>> state.add_actor(hero)
    >>> state.party_uids.append(hero.uid)
    >>>
    >>> # Create DM (uses LLM for narrative, Python for dice)
    >>> dm = DMOrchestrator(state)
    >>> response = dm.process_input("I search the room for traps")
    >>> print(response.narrative)

Modules:
    core: Configuration, logging, and base exceptions.
    models: Pydantic V2 schemas and ECS components.
    models.ecs: Entity Component System (GameState, ActorEntity, Components).
    ingestion: Vision pipeline (pdf2image), markdown extraction, ChromaDB.
    dm: AI Dungeon Master orchestrator with tool use.
    engine: Game loop, turn management, and dice logic.
    ui: Mobile-first Streamlit interface.
"""

from __future__ import annotations

# Core
from dnd_manager.core.config import Settings, get_settings
from dnd_manager.core.exceptions import DndManagerError
from dnd_manager.core.logging import configure_logging, get_logger

# ECS Models (The Source of Truth)
from dnd_manager.models.ecs import (
    ActorEntity,
    ActorType,
    GameState,
    StateUpdateRequest,
    StateUpdateResult,
    StatsComponent,
    HealthComponent,
    DefenseComponent,
    InventoryComponent,
    SpellbookComponent,
    JournalComponent,
    ClassFeatureComponent,
    create_player_character,
    create_monster,
)

# DM Orchestrator
from dnd_manager.dm.orchestrator import (
    DMOrchestrator,
    DMResponse,
    roll_dice,
    roll_check,
    roll_save,
    roll_attack,
    roll_damage,
)

# Ingestion
from dnd_manager.ingestion.universal_loader import (
    UniversalIngestor,
    CharacterExtractor,
    ChromaStore,
    DocumentType,
)


__version__ = "0.1.0"
__author__ = "DungeonAI Team"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core
    "DndManagerError",
    "Settings",
    "get_settings",
    "configure_logging",
    "get_logger",
    # ECS Models
    "ActorEntity",
    "ActorType",
    "GameState",
    "StateUpdateRequest",
    "StateUpdateResult",
    "StatsComponent",
    "HealthComponent",
    "DefenseComponent",
    "InventoryComponent",
    "SpellbookComponent",
    "JournalComponent",
    "ClassFeatureComponent",
    "create_player_character",
    "create_monster",
    # DM
    "DMOrchestrator",
    "DMResponse",
    "roll_dice",
    "roll_check",
    "roll_save",
    "roll_attack",
    "roll_damage",
    # Ingestion
    "UniversalIngestor",
    "CharacterExtractor",
    "ChromaStore",
    "DocumentType",
]
