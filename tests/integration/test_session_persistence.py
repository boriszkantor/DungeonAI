"""Integration tests for session persistence.

Tests session save/load and state integrity.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from dnd_manager.models.ecs import (
    ActorEntity,
    ActorType,
    DefenseComponent,
    GameState,
    HealthComponent,
    StatsComponent,
)
from dnd_manager.storage.database import get_database


class TestSessionPersistence:
    """Test session state persistence."""

    def test_save_session(self, sample_character: ActorEntity) -> None:
        """Save complete session state."""
        db = get_database()

        # Create a game state with the character
        game_state = GameState(
            round=3,
            turn_order=[str(sample_character.uid)],
            current_turn_index=0,
            in_combat=False,
            entities={str(sample_character.uid): sample_character},
        )

        # Create chat history
        chat_history = [
            {"role": "user", "content": "I search the room."},
            {"role": "assistant", "content": "You find a hidden door."},
        ]

        # Save session
        session_id = "test_persistence_session"
        db.save_session(
            name="Test Adventure",
            campaign_name="Test Campaign",
            game_state_dict=game_state.model_dump(mode="json"),
            chat_history=chat_history,
            session_id=session_id,
        )

        # Verify session was saved
        saved_session = db.get_session(session_id)
        assert saved_session is not None
        assert saved_session.name == "Test Adventure"
        assert saved_session.campaign_name == "Test Campaign"

    def test_load_session(self, sample_character: ActorEntity) -> None:
        """Load session and verify all state."""
        db = get_database()

        # Create and save a session
        game_state = GameState(
            round=5,
            turn_order=[str(sample_character.uid)],
            current_turn_index=0,
            in_combat=True,
            entities={str(sample_character.uid): sample_character},
        )

        chat_history = [
            {"role": "user", "content": "I attack the goblin."},
            {"role": "assistant", "content": "Roll for attack!"},
        ]

        session_id = "test_load_session"
        db.save_session(
            name="Combat Session",
            campaign_name="Test Campaign",
            game_state_dict=game_state.model_dump(mode="json"),
            chat_history=chat_history,
            session_id=session_id,
        )

        # Load the session
        loaded_session = db.get_session(session_id)
        assert loaded_session is not None

        # Verify basic session data
        assert loaded_session.name == "Combat Session"
        assert loaded_session.campaign_name == "Test Campaign"

        # Parse and verify game state
        game_state_dict = json.loads(loaded_session.game_state_json)
        loaded_game_state = GameState.model_validate(game_state_dict)

        assert loaded_game_state.round == 5
        assert loaded_game_state.in_combat is True
        assert len(loaded_game_state.turn_order) == 1
        assert str(sample_character.uid) in loaded_game_state.entities

        # Verify character data
        loaded_char = loaded_game_state.entities[str(sample_character.uid)]
        assert loaded_char.name == sample_character.name
        assert loaded_char.health.hp_current == sample_character.health.hp_current

        # Verify chat history
        loaded_chat = json.loads(loaded_session.chat_history_json)
        assert len(loaded_chat) == 2
        assert loaded_chat[0]["role"] == "user"

    def test_session_with_combat_state(self) -> None:
        """Save/load session mid-combat."""
        db = get_database()

        # Create combatants
        player = ActorEntity(
            uid=uuid4(),
            name="Ranger",
            type=ActorType.PLAYER,
            stats=StatsComponent(proficiency_bonus=2),
            health=HealthComponent(hp_current=15, hp_max=20),
            defense=DefenseComponent(ac_base=16),
        )

        enemy1 = ActorEntity(
            uid=uuid4(),
            name="Orc 1",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(proficiency_bonus=0),
            health=HealthComponent(hp_current=15, hp_max=15),
            defense=DefenseComponent(ac_base=13),
        )

        enemy2 = ActorEntity(
            uid=uuid4(),
            name="Orc 2",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(proficiency_bonus=0),
            health=HealthComponent(hp_current=8, hp_max=15),  # Damaged
            defense=DefenseComponent(ac_base=13),
        )

        # Create combat state (middle of combat)
        game_state = GameState(
            round=2,
            turn_order=[str(player.uid), str(enemy1.uid), str(enemy2.uid)],
            current_turn_index=1,  # Enemy 1's turn
            in_combat=True,
            entities={
                str(player.uid): player,
                str(enemy1.uid): enemy1,
                str(enemy2.uid): enemy2,
            },
        )

        # Save mid-combat
        session_id = "test_combat_persistence"
        db.save_session(
            name="Orc Battle",
            campaign_name="Test Campaign",
            game_state_dict=game_state.model_dump(mode="json"),
            chat_history=[],
            session_id=session_id,
        )

        # Load and verify combat state
        loaded_session = db.get_session(session_id)
        assert loaded_session is not None

        game_state_dict = json.loads(loaded_session.game_state_json)
        loaded_state = GameState.model_validate(game_state_dict)

        # Verify combat details
        assert loaded_state.round == 2
        assert loaded_state.in_combat is True
        assert loaded_state.current_turn_index == 1
        assert len(loaded_state.entities) == 3

        # Verify damaged enemy
        enemy2_loaded = loaded_state.entities[str(enemy2.uid)]
        assert enemy2_loaded.health.hp_current == 8
        assert enemy2_loaded.health.hp_max == 15

    def test_session_integrity(self, sample_character: ActorEntity) -> None:
        """Verify no data loss on round-trip."""
        db = get_database()

        # Create complex game state
        original_state = GameState(
            round=10,
            turn_order=[str(sample_character.uid)],
            current_turn_index=0,
            in_combat=False,
            entities={str(sample_character.uid): sample_character},
        )

        # Save
        session_id = "test_integrity"
        db.save_session(
            name="Integrity Test",
            campaign_name="Test",
            game_state_dict=original_state.model_dump(mode="json"),
            chat_history=[],
            session_id=session_id,
        )

        # Load
        loaded_session = db.get_session(session_id)
        assert loaded_session is not None

        game_state_dict = json.loads(loaded_session.game_state_json)
        loaded_state = GameState.model_validate(game_state_dict)

        # Verify exact match
        assert loaded_state.round == original_state.round
        assert loaded_state.in_combat == original_state.in_combat
        assert loaded_state.current_turn_index == original_state.current_turn_index
        assert len(loaded_state.entities) == len(original_state.entities)

        # Verify character details
        orig_char = original_state.entities[str(sample_character.uid)]
        loaded_char = loaded_state.entities[str(sample_character.uid)]

        assert loaded_char.name == orig_char.name
        assert loaded_char.race == orig_char.race
        assert loaded_char.stats.strength == orig_char.stats.strength
        assert loaded_char.stats.dexterity == orig_char.stats.dexterity
        assert loaded_char.health.hp_current == orig_char.health.hp_current
        assert loaded_char.health.hp_max == orig_char.health.hp_max
        assert loaded_char.defense.base_armor == orig_char.defense.base_armor
