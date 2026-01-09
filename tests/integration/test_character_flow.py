"""Integration tests for character lifecycle.

Tests the complete character flow: create, edit, save, load, and use in session.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from dnd_manager.models.ecs import (
    ActorEntity,
    ActorType,
    ClassFeatureComponent,
    ClassLevel,
    DefenseComponent,
    HealthComponent,
    StatsComponent,
)
from dnd_manager.storage.database import get_database


class TestCharacterFlow:
    """Test character creation, editing, saving, and loading."""

    def test_create_character(self) -> None:
        """Create a character with all required fields."""
        # Create a character using ECS
        character = ActorEntity(
            uid=uuid4(),
            name="Test Wizard",
            type=ActorType.PLAYER,
            race="Human",
            alignment="Neutral Good",
            stats=StatsComponent(
                strength=10,
                dexterity=14,
                constitution=13,
                intelligence=16,
                wisdom=12,
                charisma=11,
                proficiency_bonus=2,
            ),
            health=HealthComponent(
                hp_current=8,
                hp_max=8,
            ),
            defense=DefenseComponent(
                ac_base=10,
                uses_dex=True,
            ),
            class_features=ClassFeatureComponent(
                classes=[
                    ClassLevel(
                        class_name="Wizard",
                        level=1,
                    )
                ],
                features=["Spellcasting", "Arcane Recovery"],
            ),
        )

        # Verify character was created correctly
        assert character.name == "Test Wizard"
        assert character.type == ActorType.PLAYER
        assert character.race == "Human"
        assert character.stats.intelligence == 16
        assert character.health.hp_max == 8
        assert len(character.class_features.classes) == 1
        assert character.class_features.classes[0].class_name == "Wizard"

    def test_edit_character_stats(self, sample_character: ActorEntity) -> None:
        """Edit character ability scores and verify changes persist."""
        original_strength = sample_character.stats.strength
        original_hp = sample_character.health.hp_max

        # Edit ability scores
        sample_character.stats.strength = 18
        sample_character.stats.dexterity = 16

        # Edit HP
        sample_character.health.hp_max = 50
        sample_character.health.hp_current = 50

        # Verify changes
        assert sample_character.stats.strength == 18
        assert sample_character.stats.strength != original_strength
        assert sample_character.stats.dexterity == 16
        assert sample_character.health.hp_max == 50
        assert sample_character.health.hp_max != original_hp

    def test_save_and_load_character(self, sample_character: ActorEntity) -> None:
        """Save character to database, reload, verify integrity."""
        db = get_database()

        # Save character to a test session
        session_id = "test_char_flow_session"
        char_uid = str(sample_character.uid)

        db.save_session(
            name="Test Session",
            campaign_name="Test Campaign",
            game_state_dict={char_uid: sample_character.model_dump(mode="json")},
            chat_history=[],
            session_id=session_id,
        )

        # Load the session back
        loaded_session = db.get_session(session_id)
        assert loaded_session is not None

        # Parse game state
        import json
        game_state_dict = json.loads(loaded_session.game_state_json)

        # Verify character data
        assert char_uid in game_state_dict
        char_data = game_state_dict[char_uid]

        # Reconstruct character
        loaded_char = ActorEntity.model_validate(char_data)

        # Verify all fields match
        assert loaded_char.name == sample_character.name
        assert loaded_char.race == sample_character.race
        assert loaded_char.stats.strength == sample_character.stats.strength
        assert loaded_char.stats.dexterity == sample_character.stats.dexterity
        assert loaded_char.health.hp_max == sample_character.health.hp_max
        assert loaded_char.defense.base_armor == sample_character.defense.base_armor

    def test_character_in_session(self, sample_character: ActorEntity) -> None:
        """Add character to game session and verify it works."""
        from dnd_manager.models.ecs import GameState

        # Create a game state with the character
        game_state = GameState(
            round=1,
            turn_order=[str(sample_character.uid)],
            current_turn_index=0,
            entities={str(sample_character.uid): sample_character},
        )

        # Verify character is in the game state
        assert str(sample_character.uid) in game_state.entities
        assert game_state.entities[str(sample_character.uid)].name == sample_character.name

        # Verify turn order
        assert len(game_state.turn_order) == 1
        assert game_state.turn_order[0] == str(sample_character.uid)

        # Test getting current turn
        current_entity_uid = game_state.turn_order[game_state.current_turn_index]
        current_entity = game_state.entities[current_entity_uid]
        assert current_entity.name == sample_character.name
