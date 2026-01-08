"""Tests for game state models."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from dnd_manager.models import (
    AutonomyLevel,
    CombatantType,
    HealthComponent,
    Monster,
    PersonaComponent,
    PlayerCharacter,
    StatsComponent,
    ClassLevel,
    create_monster,
    create_player_character,
)
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


class TestInitiativeEntry:
    """Tests for InitiativeEntry model."""

    def test_basic_entry(self) -> None:
        """Test basic initiative entry creation."""
        entry = InitiativeEntry(
            combatant_uid=uuid4(),
            combatant_name="Fighter",
            combatant_type=CombatantType.PLAYER_CHARACTER,
            initiative_roll=18,
            dexterity_score=14,
        )
        assert entry.combatant_name == "Fighter"
        assert entry.initiative_roll == 18
        assert entry.is_active is True

    def test_sort_key(self) -> None:
        """Test sort key generation."""
        entry = InitiativeEntry(
            combatant_uid=uuid4(),
            combatant_name="Test",
            combatant_type=CombatantType.PLAYER_CHARACTER,
            initiative_roll=15,
            dexterity_score=14,
            tiebreaker_roll=10,
        )
        assert entry.sort_key == (15, 14, 10)


class TestTurnOrder:
    """Tests for TurnOrder model."""

    @pytest.fixture
    def sample_entries(self) -> list[InitiativeEntry]:
        """Create sample initiative entries."""
        return [
            InitiativeEntry(
                combatant_uid=uuid4(),
                combatant_name="Fighter",
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=20,
                dexterity_score=14,
            ),
            InitiativeEntry(
                combatant_uid=uuid4(),
                combatant_name="Wizard",
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=15,
                dexterity_score=16,
            ),
            InitiativeEntry(
                combatant_uid=uuid4(),
                combatant_name="Goblin",
                combatant_type=CombatantType.MONSTER,
                initiative_roll=12,
                dexterity_score=14,
            ),
        ]

    def test_empty_turn_order(self) -> None:
        """Test empty turn order."""
        turn_order = TurnOrder()
        assert turn_order.is_combat_active is False
        assert turn_order.current_combatant_uid is None

    def test_sort_by_initiative(self, sample_entries: list[InitiativeEntry]) -> None:
        """Test initiative sorting."""
        turn_order = TurnOrder(entries=sample_entries)
        sorted_order = turn_order.sort_by_initiative()

        # Fighter (20) should be first
        assert sorted_order.entries[0].combatant_name == "Fighter"
        # Wizard (15) second
        assert sorted_order.entries[1].combatant_name == "Wizard"
        # Goblin (12) last
        assert sorted_order.entries[2].combatant_name == "Goblin"

    def test_start_combat(self, sample_entries: list[InitiativeEntry]) -> None:
        """Test starting combat."""
        turn_order = TurnOrder(entries=sample_entries)
        started = turn_order.start_combat()

        assert started.current_round == 1
        assert started.is_combat_active is True
        assert started.round_started_at is not None

    def test_advance_turn(self, sample_entries: list[InitiativeEntry]) -> None:
        """Test advancing turns."""
        turn_order = TurnOrder(entries=sample_entries).start_combat()

        # First turn - Fighter
        assert turn_order.current_entry.combatant_name == "Fighter"

        # Advance to Wizard
        turn_order = turn_order.advance_turn()
        assert turn_order.current_entry.combatant_name == "Wizard"

        # Advance to Goblin
        turn_order = turn_order.advance_turn()
        assert turn_order.current_entry.combatant_name == "Goblin"

    def test_advance_to_new_round(self, sample_entries: list[InitiativeEntry]) -> None:
        """Test that advancing past last combatant starts new round."""
        turn_order = TurnOrder(entries=sample_entries).start_combat()

        # Advance through all combatants
        for _ in range(3):
            turn_order = turn_order.advance_turn()

        # Should be round 2, back to first combatant
        assert turn_order.current_round == 2
        assert turn_order.current_index == 0

    def test_remove_combatant(self, sample_entries: list[InitiativeEntry]) -> None:
        """Test removing a combatant."""
        turn_order = TurnOrder(entries=sample_entries).start_combat()
        goblin_uid = sample_entries[2].combatant_uid

        updated = turn_order.remove_combatant(goblin_uid)

        assert len(updated.active_entries) == 2
        assert goblin_uid not in [e.combatant_uid for e in updated.active_entries]


class TestScene:
    """Tests for Scene model."""

    @pytest.fixture
    def sample_combatants(self) -> list[PlayerCharacter | Monster]:
        """Create sample combatants."""
        return [
            create_player_character("Fighter", level=5),
            create_player_character("Wizard", level=5),
            create_monster("Goblin", challenge_rating="1/4"),
            create_monster("Goblin", challenge_rating="1/4"),
        ]

    def test_basic_scene(self, sample_combatants: list) -> None:
        """Test basic scene creation."""
        scene = Scene(
            name="Goblin Ambush",
            description="Goblins attack on the road.",
            scene_type=SceneType.COMBAT,
            combatants=sample_combatants,
        )
        assert scene.name == "Goblin Ambush"
        assert scene.scene_type == SceneType.COMBAT
        assert len(scene.combatants) == 4

    def test_player_characters_property(self, sample_combatants: list) -> None:
        """Test filtering for player characters."""
        scene = Scene(
            name="Test",
            combatants=sample_combatants,
        )
        pcs = scene.player_characters
        assert len(pcs) == 2
        assert all(isinstance(c, PlayerCharacter) for c in pcs)

    def test_monsters_property(self, sample_combatants: list) -> None:
        """Test filtering for monsters."""
        scene = Scene(
            name="Test",
            combatants=sample_combatants,
        )
        monsters = scene.monsters
        assert len(monsters) == 2
        assert all(isinstance(c, Monster) for c in monsters)

    def test_get_combatant_by_uid(self, sample_combatants: list) -> None:
        """Test finding combatant by UID."""
        scene = Scene(name="Test", combatants=sample_combatants)
        target_uid = sample_combatants[0].uid

        found = scene.get_combatant_by_uid(target_uid)
        assert found is not None
        assert found.uid == target_uid

    def test_get_combatant_not_found(self, sample_combatants: list) -> None:
        """Test finding non-existent combatant."""
        scene = Scene(name="Test", combatants=sample_combatants)

        found = scene.get_combatant_by_uid(uuid4())
        assert found is None

    def test_start_scene(self) -> None:
        """Test starting a scene."""
        scene = Scene(name="Test")
        started = scene.start_scene()

        assert started.is_active is True
        assert started.started_at is not None

    def test_complete_scene(self) -> None:
        """Test completing a scene."""
        scene = Scene(name="Test").start_scene()
        completed = scene.complete_scene()

        assert completed.is_active is False
        assert completed.is_completed is True
        assert completed.completed_at is not None

    def test_is_in_combat(self, sample_combatants: list) -> None:
        """Test combat detection."""
        scene = Scene(
            name="Combat",
            scene_type=SceneType.COMBAT,
            combatants=sample_combatants,
        )
        # Not in combat until turn order is started
        assert scene.is_in_combat is False

        # Start turn order
        entries = [
            InitiativeEntry(
                combatant_uid=c.uid,
                combatant_name=c.name,
                combatant_type=c.type,
                initiative_roll=10,
            )
            for c in sample_combatants
        ]
        scene_with_turns = Scene(
            name="Combat",
            scene_type=SceneType.COMBAT,
            combatants=sample_combatants,
            turn_order=TurnOrder(entries=entries).start_combat(),
        )
        assert scene_with_turns.is_in_combat is True


class TestGameSession:
    """Tests for GameSession model."""

    def test_basic_session(self) -> None:
        """Test basic session creation."""
        session = GameSession(
            name="Session 1",
            campaign_name="Test Campaign",
            dm_name="Test DM",
        )
        assert session.name == "Session 1"
        assert session.phase == GamePhase.SETUP
        assert session.is_active is True

    def test_add_chat_message(self) -> None:
        """Test adding chat messages."""
        session = GameSession(name="Test")
        updated = session.add_chat_message(
            "Player1",
            "I attack the goblin!",
            author_type="player",
        )

        assert len(updated.chat_history) == 1
        assert updated.chat_history[0].content == "I attack the goblin!"
        assert updated.chat_history[0].author == "Player1"

    def test_end_session(self) -> None:
        """Test ending a session."""
        session = GameSession(name="Test")
        ended = session.end_session()

        assert ended.phase == GamePhase.ENDED
        assert ended.ended_at is not None
        assert ended.is_active is False

    def test_duration_minutes(self) -> None:
        """Test duration calculation."""
        session = GameSession(name="Test")
        # Initially None (not ended)
        assert session.duration_minutes is None

        ended = session.end_session()
        assert ended.duration_minutes is not None
        assert ended.duration_minutes >= 0


class TestEnvironmentEffect:
    """Tests for EnvironmentEffect model."""

    def test_basic_effect(self) -> None:
        """Test basic environment effect."""
        effect = EnvironmentEffect(
            name="Bonfire",
            description="A roaring fire that deals damage to adjacent creatures.",
            area="5-foot square",
            damage="1d6",
            save_dc=12,
            save_ability="DEX",
        )
        assert effect.name == "Bonfire"
        assert effect.damage == "1d6"
        assert effect.save_dc == 12


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_basic_message(self) -> None:
        """Test basic chat message."""
        message = ChatMessage(
            author="DM",
            author_type="dm",
            content="The door creaks open...",
            is_narrative=True,
        )
        assert message.author == "DM"
        assert message.is_narrative is True
        assert message.timestamp is not None
