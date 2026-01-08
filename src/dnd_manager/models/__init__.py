"""Pydantic V2 schemas for the D&D 5E AI Campaign Manager.

This module provides the complete data model layer for the application,
including entity definitions, components, game state, and enumerations.
All models use Pydantic V2 with strict typing and comprehensive validation.

Submodules:
    enums: Enumeration types (Ability, Skill, Alignment, AutonomyLevel, etc.)
    components: Reusable components (Stats, Health, Persona, etc.)
    entities: Game entities (PlayerCharacter, Monster, NPC, Combatant)
    game_state: Game state tracking (TurnOrder, Scene, GameSession)

Example:
    >>> from dnd_manager.models import (
    ...     PlayerCharacter, Monster, StatsComponent, HealthComponent,
    ...     PersonaComponent, AutonomyLevel, Ability
    ... )
    >>> stats = StatsComponent(strength=18, dexterity=14)
    >>> health = HealthComponent(current_hp=45, max_hp=45)
    >>> persona = PersonaComponent(name="Thorin", autonomy=AutonomyLevel.NONE)
"""

from __future__ import annotations

# =============================================================================
# Enumerations
# =============================================================================
from dnd_manager.models.enums import (
    Ability,
    ActionType,
    Alignment,
    AutonomyLevel,
    CombatantType,
    Condition,
    CreatureType,
    DamageType,
    RestType,
    Size,
    Skill,
    SpellSchool,
)

# =============================================================================
# Components
# =============================================================================
from dnd_manager.models.components import (
    AbilityScore,
    ArmorComponent,
    HealthComponent,
    PersonaComponent,
    SavingThrowsComponent,
    SkillsComponent,
    SpeedComponent,
    StatsComponent,
    calculate_modifier,
    validate_ability_score,
)

# =============================================================================
# Entities
# =============================================================================
from dnd_manager.models.entities import (
    ChallengeRating,
    CharacterLevel,
    ClassLevel,
    Combatant,
    Entity,
    InventoryItem,
    LegendaryAction,
    Monster,
    MonsterAction,
    NPC,
    PlayerCharacter,
    SpellSlots,
    cr_to_float,
    cr_to_proficiency_bonus,
    create_monster,
    create_player_character,
)

# =============================================================================
# Game State
# =============================================================================
from dnd_manager.models.game_state import (
    ChatMessage,
    EnvironmentEffect,
    GamePhase,
    GameSession,
    InitiativeEntry,
    Scene,
    SceneType,
    TurnOrder,
)


__all__ = [
    # === Enumerations ===
    "Ability",
    "Skill",
    "Alignment",
    "AutonomyLevel",
    "DamageType",
    "Condition",
    "Size",
    "CreatureType",
    "ActionType",
    "CombatantType",
    "SpellSchool",
    "RestType",
    # === Components ===
    "AbilityScore",
    "calculate_modifier",
    "validate_ability_score",
    "StatsComponent",
    "HealthComponent",
    "PersonaComponent",
    "SpeedComponent",
    "ArmorComponent",
    "SavingThrowsComponent",
    "SkillsComponent",
    # === Entities ===
    "CharacterLevel",
    "ChallengeRating",
    "cr_to_float",
    "cr_to_proficiency_bonus",
    "Entity",
    "InventoryItem",
    "ClassLevel",
    "SpellSlots",
    "LegendaryAction",
    "MonsterAction",
    "PlayerCharacter",
    "Monster",
    "NPC",
    "Combatant",
    "create_player_character",
    "create_monster",
    # === Game State ===
    "InitiativeEntry",
    "TurnOrder",
    "SceneType",
    "EnvironmentEffect",
    "Scene",
    "GamePhase",
    "ChatMessage",
    "GameSession",
]
