"""
D&D 5E Campaign Manager - Data Layer
=====================================

A modular monolith architecture for managing D&D 5E campaigns
with an AI Dungeon Master.
"""

from .models import (
    # Enums
    AbilityName,
    CharacterType,
    ConditionType,
    DamageType,
    MessageRole,
    # Components
    ClassFeature,
    CoreStats,
    DMNotes,
    InventoryItem,
    Location,
    RoleplayData,
    Spell,
    SpellSlots,
    StatusEffect,
    Vitals,
    # Entities
    Character,
    CombatState,
    GameState,
    Message,
    # Factories
    create_character,
    create_game_state,
)

__all__ = [
    # Enums
    "AbilityName",
    "CharacterType",
    "ConditionType",
    "DamageType",
    "MessageRole",
    # Components
    "ClassFeature",
    "CoreStats",
    "DMNotes",
    "InventoryItem",
    "Location",
    "RoleplayData",
    "Spell",
    "SpellSlots",
    "StatusEffect",
    "Vitals",
    # Entities
    "Character",
    "CombatState",
    "GameState",
    "Message",
    # Factories
    "create_character",
    "create_game_state",
]

__version__ = "0.1.0"
