"""Tests for the game loop."""

from __future__ import annotations

from uuid import uuid4

import pytest

from dnd_manager.models import (
    AutonomyLevel,
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
    InitiativeEntry,
    Scene,
    SceneType,
    TurnOrder,
)
from dnd_manager.models.enums import CombatantType
from dnd_manager.engine.loop import (
    TurnStatus,
    TurnResult,
    GameLoop,
    all_enemies_defeated,
    all_players_defeated,
)
from dnd_manager.engine.tools import ToolCall


class TestTurnStatus:
    """Tests for TurnStatus enum."""

    def test_waiting_for_user(self) -> None:
        """Test WAITING_FOR_USER status value."""
        assert TurnStatus.WAITING_FOR_USER == "waiting_for_user"

    def test_turn_completed(self) -> None:
        """Test TURN_COMPLETED status value."""
        assert TurnStatus.TURN_COMPLETED == "turn_completed"

    def test_combat_ended(self) -> None:
        """Test COMBAT_ENDED status value."""
        assert TurnStatus.COMBAT_ENDED == "combat_ended"


class TestEndConditions:
    """Tests for combat end condition checks."""

    def test_all_enemies_defeated_true(self) -> None:
        """Test detecting when all enemies are defeated."""
        # Create scene with only defeated monsters
        monster = create_monster("Goblin")
        monster = Monster(
            **{
                **monster.model_dump(),
                "health": HealthComponent(current_hp=0, max_hp=7),
            }
        )

        player = create_player_character("Hero", level=5)

        scene = Scene(
            name="Test Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
        )

        assert all_enemies_defeated(scene) is True

    def test_all_enemies_defeated_false(self) -> None:
        """Test when enemies are still alive."""
        monster = create_monster("Goblin")  # Full HP
        player = create_player_character("Hero", level=5)

        scene = Scene(
            name="Test Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
        )

        assert all_enemies_defeated(scene) is False

    def test_all_players_defeated_true(self) -> None:
        """Test detecting when all players are defeated."""
        player = PlayerCharacter(
            name="Fallen Hero",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=0, max_hp=30),
            persona=PersonaComponent(name="Fallen Hero"),
            classes=[ClassLevel(class_name="Fighter", level=5)],
            race="Human",
        )
        monster = create_monster("Goblin")

        scene = Scene(
            name="Test Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
        )

        assert all_players_defeated(scene) is True

    def test_all_players_defeated_false(self) -> None:
        """Test when players are still alive."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin")

        scene = Scene(
            name="Test Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
        )

        assert all_players_defeated(scene) is False


class TestGameLoopInitialization:
    """Tests for GameLoop initialization."""

    @pytest.fixture
    def basic_scene(self) -> Scene:
        """Create a basic combat scene."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin", challenge_rating="1/4")

        return Scene(
            name="Test Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
        )

    def test_game_loop_creation(self, basic_scene: Scene) -> None:
        """Test creating a game loop."""
        loop = GameLoop(basic_scene)

        assert loop.scene == basic_scene
        assert loop.is_combat_active is False  # Not started yet

    def test_game_loop_with_turn_order(self) -> None:
        """Test game loop with active turn order."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin")

        entries = [
            InitiativeEntry(
                combatant_uid=player.uid,
                combatant_name=player.name,
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=15,
            ),
            InitiativeEntry(
                combatant_uid=monster.uid,
                combatant_name=monster.name,
                combatant_type=CombatantType.MONSTER,
                initiative_roll=10,
            ),
        ]
        turn_order = TurnOrder(entries=entries).start_combat()

        scene = Scene(
            name="Active Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
            turn_order=turn_order,
        )

        loop = GameLoop(scene)
        assert loop.is_combat_active is True
        assert loop.current_round == 1


class TestGameLoopPeekTurn:
    """Tests for GameLoop.peek_turn()."""

    def test_peek_no_combat(self) -> None:
        """Test peeking when no combat is active."""
        player = create_player_character("Hero", level=5)
        scene = Scene(
            name="No Combat",
            combatants=[player],
            scene_type=SceneType.EXPLORATION,
        )

        loop = GameLoop(scene)
        result = loop.peek_turn()

        assert result.status == TurnStatus.NO_COMBAT

    def test_peek_player_turn(self) -> None:
        """Test peeking when it's a player's turn."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin")

        entries = [
            InitiativeEntry(
                combatant_uid=player.uid,
                combatant_name=player.name,
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=20,  # Player goes first
            ),
            InitiativeEntry(
                combatant_uid=monster.uid,
                combatant_name=monster.name,
                combatant_type=CombatantType.MONSTER,
                initiative_roll=10,
            ),
        ]
        turn_order = TurnOrder(entries=entries).start_combat()

        scene = Scene(
            name="Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
            turn_order=turn_order,
        )

        loop = GameLoop(scene)
        result = loop.peek_turn()

        assert result.status == TurnStatus.WAITING_FOR_USER
        assert result.combatant_name == "Hero"
        assert result.autonomy == AutonomyLevel.NONE

    def test_peek_ai_turn(self) -> None:
        """Test peeking when it's an AI-controlled combatant's turn."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin")

        entries = [
            InitiativeEntry(
                combatant_uid=monster.uid,
                combatant_name=monster.name,
                combatant_type=CombatantType.MONSTER,
                initiative_roll=20,  # Monster goes first
            ),
            InitiativeEntry(
                combatant_uid=player.uid,
                combatant_name=player.name,
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=10,
            ),
        ]
        turn_order = TurnOrder(entries=entries).start_combat()

        scene = Scene(
            name="Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
            turn_order=turn_order,
        )

        loop = GameLoop(scene)
        result = loop.peek_turn()

        # Monster has FULL_AUTO autonomy
        assert result.status == TurnStatus.AI_THINKING
        assert result.combatant_name == "Goblin"
        assert result.autonomy == AutonomyLevel.FULL_AUTO


class TestGameLoopProcessTurn:
    """Tests for GameLoop.process_turn()."""

    def test_process_player_turn_waits(self) -> None:
        """Test that processing a player turn returns WAITING_FOR_USER."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin")

        entries = [
            InitiativeEntry(
                combatant_uid=player.uid,
                combatant_name=player.name,
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=20,
            ),
            InitiativeEntry(
                combatant_uid=monster.uid,
                combatant_name=monster.name,
                combatant_type=CombatantType.MONSTER,
                initiative_roll=10,
            ),
        ]
        turn_order = TurnOrder(entries=entries).start_combat()

        scene = Scene(
            name="Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
            turn_order=turn_order,
        )

        loop = GameLoop(scene)
        result = loop.process_turn()

        # Should not advance, just wait for user
        assert result.status == TurnStatus.WAITING_FOR_USER
        assert loop.turn_order.current_index == 0  # Still on first combatant

    def test_process_skips_unconscious(self) -> None:
        """Test that unconscious combatants are skipped."""
        player = PlayerCharacter(
            name="Fallen Hero",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=0, max_hp=30),
            persona=PersonaComponent(name="Fallen Hero"),
            classes=[ClassLevel(class_name="Fighter", level=5)],
            race="Human",
        )
        monster = create_monster("Goblin")

        entries = [
            InitiativeEntry(
                combatant_uid=player.uid,
                combatant_name=player.name,
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=20,
            ),
            InitiativeEntry(
                combatant_uid=monster.uid,
                combatant_name=monster.name,
                combatant_type=CombatantType.MONSTER,
                initiative_roll=10,
            ),
        ]
        turn_order = TurnOrder(entries=entries).start_combat()

        scene = Scene(
            name="Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
            turn_order=turn_order,
        )

        loop = GameLoop(scene)
        result = loop.process_turn()

        # Should skip unconscious player
        assert result.status == TurnStatus.TURN_COMPLETED
        assert "unconscious" in result.message.lower()


class TestGameLoopEndCombat:
    """Tests for GameLoop combat ending."""

    def test_end_combat_manually(self) -> None:
        """Test manually ending combat."""
        player = create_player_character("Hero", level=5)
        monster = create_monster("Goblin")

        entries = [
            InitiativeEntry(
                combatant_uid=player.uid,
                combatant_name=player.name,
                combatant_type=CombatantType.PLAYER_CHARACTER,
                initiative_roll=20,
            ),
            InitiativeEntry(
                combatant_uid=monster.uid,
                combatant_name=monster.name,
                combatant_type=CombatantType.MONSTER,
                initiative_roll=10,
            ),
        ]
        turn_order = TurnOrder(entries=entries).start_combat()

        scene = Scene(
            name="Combat",
            combatants=[player, monster],
            scene_type=SceneType.COMBAT,
            turn_order=turn_order,
        )

        loop = GameLoop(scene)
        result = loop.end_combat("The enemies flee!")

        assert result.status == TurnStatus.COMBAT_ENDED
        assert "flee" in result.message


class TestTurnResult:
    """Tests for TurnResult dataclass."""

    def test_turn_result_creation(self) -> None:
        """Test creating a TurnResult."""
        result = TurnResult(
            status=TurnStatus.TURN_COMPLETED,
            combatant_name="Test Fighter",
            combatant_uid=uuid4(),
            autonomy=AutonomyLevel.NONE,
            message="Attack hit for 10 damage",
            round_number=2,
            turn_index=1,
        )

        assert result.status == TurnStatus.TURN_COMPLETED
        assert result.combatant_name == "Test Fighter"
        assert result.round_number == 2

    def test_turn_result_with_tool_results(self) -> None:
        """Test TurnResult with tool execution results."""
        from dnd_manager.engine.tools import ToolResult

        tool_results = [
            ToolResult(
                tool_name="roll_attack",
                call_id="1",
                result="Hit for 10 damage",
                success=True,
            ),
        ]

        result = TurnResult(
            status=TurnStatus.TURN_COMPLETED,
            combatant_name="Fighter",
            tool_results=tool_results,
        )

        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True
