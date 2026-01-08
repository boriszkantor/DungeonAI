"""Tests for component models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dnd_manager.models import (
    Ability,
    AutonomyLevel,
    Condition,
    HealthComponent,
    PersonaComponent,
    SkillsComponent,
    StatsComponent,
    calculate_modifier,
)
from dnd_manager.models.enums import Skill


class TestCalculateModifier:
    """Tests for the calculate_modifier function."""

    def test_modifier_at_10(self) -> None:
        """Score of 10 gives modifier of 0."""
        assert calculate_modifier(10) == 0

    def test_modifier_at_18(self) -> None:
        """Score of 18 gives modifier of +4."""
        assert calculate_modifier(18) == 4

    def test_modifier_at_7(self) -> None:
        """Score of 7 gives modifier of -2."""
        assert calculate_modifier(7) == -2

    def test_modifier_at_1(self) -> None:
        """Score of 1 gives modifier of -5."""
        assert calculate_modifier(1) == -5

    def test_modifier_at_30(self) -> None:
        """Score of 30 gives modifier of +10."""
        assert calculate_modifier(30) == 10

    @pytest.mark.parametrize(
        "score,expected",
        [
            (1, -5), (2, -4), (3, -4), (4, -3), (5, -3),
            (6, -2), (7, -2), (8, -1), (9, -1), (10, 0),
            (11, 0), (12, 1), (13, 1), (14, 2), (15, 2),
            (16, 3), (17, 3), (18, 4), (19, 4), (20, 5),
        ],
    )
    def test_modifier_table(self, score: int, expected: int) -> None:
        """Test modifier calculation against D&D 5E table."""
        assert calculate_modifier(score) == expected


class TestStatsComponent:
    """Tests for StatsComponent."""

    def test_default_values(self) -> None:
        """Test default ability scores are 10."""
        stats = StatsComponent()
        assert stats.strength == 10
        assert stats.dexterity == 10
        assert stats.constitution == 10
        assert stats.intelligence == 10
        assert stats.wisdom == 10
        assert stats.charisma == 10

    def test_computed_modifiers(self) -> None:
        """Test that modifiers are computed correctly."""
        stats = StatsComponent(
            strength=16,
            dexterity=14,
            constitution=15,
            intelligence=8,
            wisdom=12,
            charisma=10,
        )
        assert stats.strength_modifier == 3
        assert stats.dexterity_modifier == 2
        assert stats.constitution_modifier == 2
        assert stats.intelligence_modifier == -1
        assert stats.wisdom_modifier == 1
        assert stats.charisma_modifier == 0

    def test_get_score(self) -> None:
        """Test get_score method."""
        stats = StatsComponent(strength=18, dexterity=14)
        assert stats.get_score(Ability.STR) == 18
        assert stats.get_score(Ability.DEX) == 14

    def test_get_modifier(self) -> None:
        """Test get_modifier method."""
        stats = StatsComponent(strength=18, wisdom=7)
        assert stats.get_modifier(Ability.STR) == 4
        assert stats.get_modifier(Ability.WIS) == -2

    def test_score_validation_min(self) -> None:
        """Test that scores below 1 are rejected."""
        with pytest.raises(ValidationError):
            StatsComponent(strength=0)

    def test_score_validation_max(self) -> None:
        """Test that scores above 30 are rejected."""
        with pytest.raises(ValidationError):
            StatsComponent(dexterity=31)

    def test_immutability(self) -> None:
        """Test that StatsComponent is frozen."""
        stats = StatsComponent(strength=16)
        with pytest.raises(ValidationError):
            stats.strength = 18  # type: ignore[misc]


class TestHealthComponent:
    """Tests for HealthComponent."""

    def test_basic_health(self) -> None:
        """Test basic health tracking."""
        health = HealthComponent(current_hp=25, max_hp=30)
        assert health.current_hp == 25
        assert health.max_hp == 30
        assert health.temp_hp == 0

    def test_effective_hp(self) -> None:
        """Test effective HP includes temp HP."""
        health = HealthComponent(current_hp=20, max_hp=30, temp_hp=10)
        assert health.effective_hp == 30

    def test_hp_percentage(self) -> None:
        """Test HP percentage calculation."""
        health = HealthComponent(current_hp=15, max_hp=30)
        assert health.hp_percentage == 50.0

    def test_is_conscious(self) -> None:
        """Test consciousness check."""
        healthy = HealthComponent(current_hp=10, max_hp=30)
        assert healthy.is_conscious is True

        unconscious = HealthComponent(current_hp=0, max_hp=30)
        assert unconscious.is_conscious is False

    def test_is_conscious_with_condition(self) -> None:
        """Test that certain conditions make entity unconscious."""
        health = HealthComponent(
            current_hp=10, max_hp=30, conditions=[Condition.UNCONSCIOUS]
        )
        assert health.is_conscious is False

    def test_is_dying(self) -> None:
        """Test dying state detection."""
        dying = HealthComponent(current_hp=0, max_hp=30)
        assert dying.is_dying is True

        stable = HealthComponent(
            current_hp=0, max_hp=30, death_save_successes=3
        )
        assert stable.is_dying is False

    def test_is_dead(self) -> None:
        """Test death detection."""
        dead_saves = HealthComponent(
            current_hp=0, max_hp=30, death_save_failures=3
        )
        assert dead_saves.is_dead is True

        massive_damage = HealthComponent(current_hp=-30, max_hp=30)
        assert massive_damage.is_dead is True

    def test_take_damage(self) -> None:
        """Test damage application."""
        health = HealthComponent(current_hp=20, max_hp=30, temp_hp=5)
        damaged = health.take_damage(10)

        # 5 damage absorbed by temp HP, 5 goes to current
        assert damaged.temp_hp == 0
        assert damaged.current_hp == 15

    def test_take_damage_temp_only(self) -> None:
        """Test damage absorbed entirely by temp HP."""
        health = HealthComponent(current_hp=20, max_hp=30, temp_hp=10)
        damaged = health.take_damage(5)

        assert damaged.temp_hp == 5
        assert damaged.current_hp == 20

    def test_heal(self) -> None:
        """Test healing."""
        health = HealthComponent(current_hp=10, max_hp=30)
        healed = health.heal(15)

        assert healed.current_hp == 25

    def test_heal_cannot_exceed_max(self) -> None:
        """Test that healing cannot exceed max HP."""
        health = HealthComponent(current_hp=25, max_hp=30)
        healed = health.heal(20)

        assert healed.current_hp == 30


class TestPersonaComponent:
    """Tests for PersonaComponent."""

    def test_default_autonomy(self) -> None:
        """Test default autonomy is NONE."""
        persona = PersonaComponent(name="Hero")
        assert persona.autonomy == AutonomyLevel.NONE

    def test_full_auto_persona(self) -> None:
        """Test fully autonomous persona."""
        persona = PersonaComponent(
            name="Goblin",
            autonomy=AutonomyLevel.FULL_AUTO,
            directives=["Attack the nearest enemy", "Flee when below 5 HP"],
        )
        assert persona.is_ai_controlled is True
        assert persona.is_fully_autonomous is True

    def test_suggestive_persona(self) -> None:
        """Test suggestive autonomy."""
        persona = PersonaComponent(
            name="Ally NPC",
            autonomy=AutonomyLevel.SUGGESTIVE,
        )
        assert persona.is_ai_controlled is True
        assert persona.is_fully_autonomous is False

    def test_ai_prompt_context(self) -> None:
        """Test AI prompt context generation."""
        persona = PersonaComponent(
            name="Grimjaw",
            biography="A battle-hardened orc warrior.",
            personality_traits=["Brave", "Loyal"],
            directives=["Protect the tribe"],
            voice_style="Gruff",
        )
        context = persona.get_ai_prompt_context()

        assert "Grimjaw" in context
        assert "battle-hardened orc" in context
        assert "Brave" in context
        assert "Protect the tribe" in context
        assert "Gruff" in context


class TestSkillsComponent:
    """Tests for SkillsComponent."""

    def test_proficiency_bonus(self) -> None:
        """Test skill bonus with proficiency."""
        skills = SkillsComponent(proficiencies={Skill.ATHLETICS})
        stats = StatsComponent(strength=16)

        bonus = skills.get_skill_bonus(Skill.ATHLETICS, stats, proficiency_bonus=3)
        # STR mod (3) + proficiency (3) = 6
        assert bonus == 6

    def test_expertise_bonus(self) -> None:
        """Test skill bonus with expertise."""
        skills = SkillsComponent(
            proficiencies={Skill.STEALTH},
            expertise={Skill.STEALTH},
        )
        stats = StatsComponent(dexterity=16)

        bonus = skills.get_skill_bonus(Skill.STEALTH, stats, proficiency_bonus=3)
        # DEX mod (3) + expertise (6) = 9
        assert bonus == 9

    def test_no_proficiency_bonus(self) -> None:
        """Test skill bonus without proficiency."""
        skills = SkillsComponent()
        stats = StatsComponent(charisma=14)

        bonus = skills.get_skill_bonus(Skill.PERSUASION, stats, proficiency_bonus=3)
        # CHA mod (2) only
        assert bonus == 2

    def test_expertise_requires_proficiency(self) -> None:
        """Test that expertise requires proficiency."""
        with pytest.raises(ValidationError):
            SkillsComponent(expertise={Skill.STEALTH})  # No proficiency
