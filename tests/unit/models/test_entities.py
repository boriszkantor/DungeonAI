"""Tests for entity models."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from dnd_manager.models import (
    AutonomyLevel,
    ClassLevel,
    CombatantType,
    Combatant,
    CreatureType,
    HealthComponent,
    Monster,
    NPC,
    PersonaComponent,
    PlayerCharacter,
    Size,
    StatsComponent,
    create_monster,
    create_player_character,
    cr_to_float,
    cr_to_proficiency_bonus,
)


class TestChallengeRating:
    """Tests for challenge rating utilities."""

    @pytest.mark.parametrize(
        "cr,expected",
        [
            ("0", 0.0),
            ("1/8", 0.125),
            ("1/4", 0.25),
            ("1/2", 0.5),
            ("1", 1.0),
            ("5", 5.0),
            ("20", 20.0),
        ],
    )
    def test_cr_to_float(self, cr: str, expected: float) -> None:
        """Test CR string to float conversion."""
        assert cr_to_float(cr) == expected

    @pytest.mark.parametrize(
        "cr,expected",
        [
            ("0", 2),
            ("1/4", 2),
            ("4", 2),
            ("5", 3),
            ("8", 3),
            ("9", 4),
            ("13", 5),
            ("17", 6),
            ("21", 7),
            ("25", 8),
            ("29", 9),
        ],
    )
    def test_cr_to_proficiency_bonus(self, cr: str, expected: int) -> None:
        """Test CR to proficiency bonus calculation."""
        assert cr_to_proficiency_bonus(cr) == expected


class TestPlayerCharacter:
    """Tests for PlayerCharacter model."""

    @pytest.fixture
    def basic_pc(self) -> PlayerCharacter:
        """Create a basic player character."""
        return PlayerCharacter(
            name="Test Fighter",
            stats=StatsComponent(strength=16, dexterity=14, constitution=15),
            health=HealthComponent(current_hp=44, max_hp=44),
            persona=PersonaComponent(name="Test Fighter"),
            classes=[ClassLevel(class_name="Fighter", level=5)],
            race="Human",
        )

    def test_basic_creation(self, basic_pc: PlayerCharacter) -> None:
        """Test basic PC creation."""
        assert basic_pc.name == "Test Fighter"
        assert basic_pc.type == CombatantType.PLAYER_CHARACTER
        assert basic_pc.race == "Human"

    def test_has_uid(self, basic_pc: PlayerCharacter) -> None:
        """Test that PC has a UUID."""
        assert isinstance(basic_pc.uid, UUID)

    def test_total_level(self, basic_pc: PlayerCharacter) -> None:
        """Test total level calculation."""
        assert basic_pc.total_level == 5

    def test_multiclass_total_level(self) -> None:
        """Test total level with multiclassing."""
        pc = PlayerCharacter(
            name="Multiclass Hero",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=50, max_hp=50),
            persona=PersonaComponent(name="Multiclass Hero"),
            classes=[
                ClassLevel(class_name="Fighter", level=5),
                ClassLevel(class_name="Wizard", level=3),
            ],
            race="Elf",
        )
        assert pc.total_level == 8

    def test_proficiency_bonus(self, basic_pc: PlayerCharacter) -> None:
        """Test proficiency bonus calculation."""
        # Level 5: (5-1)//4 + 2 = 3
        assert basic_pc.proficiency_bonus == 3

    def test_armor_class(self, basic_pc: PlayerCharacter) -> None:
        """Test AC calculation."""
        # Base 10 + DEX mod (2) = 12
        assert basic_pc.armor_class == 12

    def test_initiative_modifier(self, basic_pc: PlayerCharacter) -> None:
        """Test initiative modifier."""
        # DEX 14 = +2
        assert basic_pc.initiative_modifier == 2

    def test_default_persona_autonomy(self, basic_pc: PlayerCharacter) -> None:
        """Test that PC persona defaults to NONE autonomy."""
        assert basic_pc.persona.autonomy == AutonomyLevel.NONE

    def test_level_validation_min(self) -> None:
        """Test that level must be at least 1."""
        with pytest.raises(ValidationError):
            PlayerCharacter(
                name="Invalid",
                stats=StatsComponent(),
                health=HealthComponent(current_hp=10, max_hp=10),
                persona=PersonaComponent(name="Invalid"),
                classes=[ClassLevel(class_name="Fighter", level=0)],
                race="Human",
            )

    def test_level_validation_max(self) -> None:
        """Test that level cannot exceed 20."""
        with pytest.raises(ValidationError):
            PlayerCharacter(
                name="Invalid",
                stats=StatsComponent(),
                health=HealthComponent(current_hp=10, max_hp=10),
                persona=PersonaComponent(name="Invalid"),
                classes=[ClassLevel(class_name="Fighter", level=21)],
                race="Human",
            )


class TestMonster:
    """Tests for Monster model."""

    @pytest.fixture
    def basic_monster(self) -> Monster:
        """Create a basic monster."""
        return Monster(
            name="Goblin",
            stats=StatsComponent(
                strength=8, dexterity=14, constitution=10,
                intelligence=10, wisdom=8, charisma=8,
            ),
            health=HealthComponent(current_hp=7, max_hp=7),
            persona=PersonaComponent(
                name="Goblin",
                autonomy=AutonomyLevel.FULL_AUTO,
                directives=["Attack the nearest enemy"],
            ),
            challenge_rating="1/4",
            creature_type=CreatureType.HUMANOID,
            size=Size.SMALL,
            armor_class_override=15,
        )

    def test_basic_creation(self, basic_monster: Monster) -> None:
        """Test basic monster creation."""
        assert basic_monster.name == "Goblin"
        assert basic_monster.type == CombatantType.MONSTER
        assert basic_monster.challenge_rating == "1/4"

    def test_proficiency_bonus_from_cr(self, basic_monster: Monster) -> None:
        """Test proficiency bonus calculation from CR."""
        # CR 1/4 = proficiency +2
        assert basic_monster.proficiency_bonus == 2

    def test_armor_class_override(self, basic_monster: Monster) -> None:
        """Test that AC override is used."""
        assert basic_monster.armor_class == 15

    def test_experience_points(self, basic_monster: Monster) -> None:
        """Test XP calculation from CR."""
        # CR 1/4 = 50 XP
        assert basic_monster.experience_points == 50

    def test_is_legendary(self) -> None:
        """Test legendary creature detection."""
        non_legendary = Monster(
            name="Goblin",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=7, max_hp=7),
            persona=PersonaComponent(name="Goblin", autonomy=AutonomyLevel.FULL_AUTO),
        )
        assert non_legendary.is_legendary is False

        legendary = Monster(
            name="Dragon",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=200, max_hp=200),
            persona=PersonaComponent(name="Dragon", autonomy=AutonomyLevel.FULL_AUTO),
            legendary_action_count=3,
        )
        assert legendary.is_legendary is True

    def test_cr_validation(self) -> None:
        """Test that invalid CR is rejected."""
        with pytest.raises(ValidationError):
            Monster(
                name="Invalid",
                stats=StatsComponent(),
                health=HealthComponent(current_hp=10, max_hp=10),
                persona=PersonaComponent(name="Invalid", autonomy=AutonomyLevel.FULL_AUTO),
                challenge_rating="invalid",
            )


class TestNPC:
    """Tests for NPC model."""

    def test_basic_npc(self) -> None:
        """Test basic NPC creation."""
        npc = NPC(
            name="Bartender Bob",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=10, max_hp=10),
            persona=PersonaComponent(
                name="Bartender Bob",
                autonomy=AutonomyLevel.SUGGESTIVE,
                biography="A friendly innkeeper.",
            ),
            occupation="Innkeeper",
            attitude="friendly",
        )
        assert npc.name == "Bartender Bob"
        assert npc.type == CombatantType.NPC
        assert npc.attitude == "friendly"

    def test_npc_proficiency_bonus(self) -> None:
        """Test NPC has flat +2 proficiency."""
        npc = NPC(
            name="Guard",
            stats=StatsComponent(),
            health=HealthComponent(current_hp=10, max_hp=10),
            persona=PersonaComponent(name="Guard", autonomy=AutonomyLevel.FULL_AUTO),
        )
        assert npc.proficiency_bonus == 2


class TestCombatantUnion:
    """Tests for the Combatant discriminated union."""

    def test_deserialize_player_character(self) -> None:
        """Test deserializing a PlayerCharacter from dict."""
        data = {
            "type": "player_character",
            "name": "Hero",
            "stats": {"strength": 16},
            "health": {"current_hp": 30, "max_hp": 30},
            "persona": {"name": "Hero"},
            "classes": [{"class_name": "Fighter", "level": 5}],
            "race": "Human",
        }
        adapter = TypeAdapter(Combatant)
        combatant = adapter.validate_python(data)

        assert isinstance(combatant, PlayerCharacter)
        assert combatant.name == "Hero"

    def test_deserialize_monster(self) -> None:
        """Test deserializing a Monster from dict."""
        data = {
            "type": "monster",
            "name": "Goblin",
            "stats": {},
            "health": {"current_hp": 7, "max_hp": 7},
            "persona": {"name": "Goblin", "autonomy": 3},
        }
        adapter = TypeAdapter(Combatant)
        combatant = adapter.validate_python(data)

        assert isinstance(combatant, Monster)
        assert combatant.name == "Goblin"

    def test_deserialize_npc(self) -> None:
        """Test deserializing an NPC from dict."""
        data = {
            "type": "npc",
            "name": "Guard",
            "stats": {},
            "health": {"current_hp": 10, "max_hp": 10},
            "persona": {"name": "Guard", "autonomy": 3},
        }
        adapter = TypeAdapter(Combatant)
        combatant = adapter.validate_python(data)

        assert isinstance(combatant, NPC)
        assert combatant.name == "Guard"


class TestFactoryFunctions:
    """Tests for entity factory functions."""

    def test_create_player_character(self) -> None:
        """Test create_player_character factory."""
        pc = create_player_character(
            "Test Hero",
            class_name="Wizard",
            level=5,
            race="Elf",
        )
        assert pc.name == "Test Hero"
        assert pc.classes[0].class_name == "Wizard"
        assert pc.total_level == 5
        assert pc.race == "Elf"
        assert pc.persona.autonomy == AutonomyLevel.NONE

    def test_create_monster(self) -> None:
        """Test create_monster factory."""
        monster = create_monster(
            "Test Goblin",
            challenge_rating="1/4",
            creature_type=CreatureType.HUMANOID,
            directives=["Attack!", "Flee at low HP"],
        )
        assert monster.name == "Test Goblin"
        assert monster.challenge_rating == "1/4"
        assert monster.persona.autonomy == AutonomyLevel.FULL_AUTO
        assert "Attack!" in monster.persona.directives
