"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests
in the D&D Campaign Manager test suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest


if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Generator[None, None, None]:
    """Reset the settings cache before and after each test."""
    from dnd_manager.core.config import clear_settings_cache

    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up mock environment variables for testing.

    Returns:
        Dictionary of environment variables that were set.
    """
    env_vars = {
        "DND_MANAGER_GEMINI_API_KEY": "test-gemini-key",
        "DND_MANAGER_OPENAI_API_KEY": "test-openai-key",
        "DND_MANAGER_DEBUG": "true",
        "DND_MANAGER_LOG_LEVEL": "DEBUG",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def sample_character_stats() -> dict[str, int]:
    """Provide sample character ability scores.

    Returns:
        Dictionary of ability scores.
    """
    return {
        "strength": 16,
        "dexterity": 14,
        "constitution": 15,
        "intelligence": 10,
        "wisdom": 12,
        "charisma": 8,
    }


@pytest.fixture
def sample_character_data(sample_character_stats: dict[str, int]) -> dict[str, Any]:
    """Provide sample character data for testing.

    Args:
        sample_character_stats: Character ability scores.

    Returns:
        Dictionary of character data.
    """
    return {
        "name": "Test Fighter",
        "race": "Human",
        "stats": sample_character_stats,
        "classes": [
            {"name": "Fighter", "level": 5, "subclass": "Champion", "hit_die": 10}
        ],
        "max_hit_points": 44,
        "current_hit_points": 44,
        "armor_class": 18,
        "speed": 30,
        "is_npc": False,
    }


@pytest.fixture
def sample_character(sample_character_data: dict[str, Any]) -> Any:
    """Create a sample ActorEntity instance for testing.

    Args:
        sample_character_data: Character data dictionary.

    Returns:
        ActorEntity instance.
    """
    from dnd_manager.models.ecs import (
        ActorEntity,
        ActorType,
        ClassFeatureComponent,
        ClassLevel,
        DefenseComponent,
        HealthComponent,
        StatsComponent,
    )

    # Create stats component
    stats = StatsComponent(**sample_character_data["stats"])
    
    # Create health component
    health = HealthComponent(
        hp_max=sample_character_data["max_hit_points"],
        hp_current=sample_character_data["current_hit_points"],
    )
    
    # Create defense component (armor)
    defense = DefenseComponent(
        base_armor=sample_character_data["armor_class"],
    )
    
    # Create class features component
    class_levels = []
    for cls_data in sample_character_data["classes"]:
        class_levels.append(
            ClassLevel(
                class_name=cls_data["name"],
                level=cls_data["level"],
                subclass_name=cls_data.get("subclass"),
                hit_die=cls_data.get("hit_die", 10),
            )
        )
    
    class_features = ClassFeatureComponent(
        classes=class_levels,
        features=[],  # Can be populated with specific features if needed
    )
    
    # Create the actor entity
    return ActorEntity(
        name=sample_character_data["name"],
        race=sample_character_data["race"],
        type=ActorType.NPC_ALLY if sample_character_data.get("is_npc", False) else ActorType.PLAYER,
        stats=stats,
        health=health,
        defense=defense,
        class_features=class_features,
    )


@pytest.fixture
def sample_combatant_data() -> dict[str, Any]:
    """Provide sample combatant data for testing.

    Returns:
        Dictionary of combatant data.
    """
    from dnd_manager.models.combat import CombatantType

    return {
        "name": "Goblin",
        "combatant_type": CombatantType.NPC_ENEMY,
        "current_hp": 7,
        "max_hp": 7,
        "armor_class": 15,
    }


@pytest.fixture
def sample_combatant(sample_combatant_data: dict[str, Any]) -> Any:
    """Create a sample Combatant instance.

    Args:
        sample_combatant_data: Combatant data dictionary.

    Returns:
        Combatant instance.
    """
    from dnd_manager.models.combat import Combatant

    return Combatant(**sample_combatant_data)


# =============================================================================
# Engine Fixtures
# =============================================================================


@pytest.fixture
def dice_roller() -> Any:
    """Create a DiceRoller with a fixed seed for reproducible tests.

    Returns:
        DiceRoller instance with fixed seed.
    """
    from dnd_manager.engine.dice import DiceRoller

    return DiceRoller(seed=42)


@pytest.fixture
def turn_manager() -> Any:
    """Create a TurnManager instance for testing.

    Returns:
        TurnManager instance.
    """
    from dnd_manager.engine.turn_manager import TurnManager

    return TurnManager()


@pytest.fixture
def game_loop() -> Any:
    """Create a GameLoop instance for testing.

    Returns:
        GameLoop instance.
    """
    from dnd_manager.engine.game_loop import GameLoop

    return GameLoop()


# =============================================================================
# Campaign Fixtures
# =============================================================================


@pytest.fixture
def sample_scene_data() -> dict[str, Any]:
    """Provide sample scene data for testing.

    Returns:
        Dictionary of scene data.
    """
    from dnd_manager.models.campaign import Difficulty, SceneType

    return {
        "name": "Goblin Ambush",
        "scene_type": SceneType.COMBAT,
        "description": "A group of goblins lies in wait along the forest road.",
        "read_aloud_text": "As you round the bend, arrows fly from the treeline!",
        "difficulty": Difficulty.MEDIUM,
        "location": "Forest Road",
        "monsters": ["Goblin", "Goblin", "Goblin Boss"],
    }


@pytest.fixture
def sample_scene(sample_scene_data: dict[str, Any]) -> Any:
    """Create a sample Scene instance.

    Args:
        sample_scene_data: Scene data dictionary.

    Returns:
        Scene instance.
    """
    from dnd_manager.models.campaign import Scene

    return Scene(**sample_scene_data)


@pytest.fixture
def sample_campaign_data(sample_scene: Any) -> dict[str, Any]:
    """Provide sample campaign data for testing.

    Args:
        sample_scene: Sample scene instance.

    Returns:
        Dictionary of campaign data.
    """
    return {
        "name": "Lost Mine of Phandelver",
        "description": "A classic D&D 5E adventure for levels 1-5.",
        "setting": "Forgotten Realms",
        "scenes": [sample_scene],
    }


@pytest.fixture
def sample_campaign(sample_campaign_data: dict[str, Any]) -> Any:
    """Create a sample Campaign instance.

    Args:
        sample_campaign_data: Campaign data dictionary.

    Returns:
        Campaign instance.
    """
    from dnd_manager.models.campaign import Campaign

    return Campaign(**sample_campaign_data)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path: Any) -> Any:
    """Create a temporary data directory for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to temporary data directory.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "scenes").mkdir()
    (data_dir / "assets").mkdir()
    (data_dir / "cache").mkdir()
    return data_dir
