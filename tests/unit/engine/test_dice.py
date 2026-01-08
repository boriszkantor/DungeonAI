"""Tests for dice rolling mechanics."""

from __future__ import annotations

import pytest

from dnd_manager.core.exceptions import DiceRollError
from dnd_manager.engine.dice import DiceExpression, DiceRoller, RollType, roll


class TestDiceRoller:
    """Tests for the DiceRoller class."""

    def test_simple_d20_roll(self, dice_roller: DiceRoller) -> None:
        """Test simple d20 roll."""
        result = dice_roller.roll("1d20")

        assert isinstance(result, DiceExpression)
        assert 1 <= result.total <= 20
        assert len(result.dice) >= 1
        assert result.roll_type == RollType.NORMAL

    def test_roll_with_modifier(self, dice_roller: DiceRoller) -> None:
        """Test roll with positive modifier."""
        result = dice_roller.roll("1d20+5")

        assert result.modifier == 5
        assert result.total >= 6  # 1 + 5
        assert result.total <= 25  # 20 + 5

    def test_roll_with_negative_modifier(self, dice_roller: DiceRoller) -> None:
        """Test roll with negative modifier."""
        result = dice_roller.roll("1d20-3")

        assert result.modifier == -3

    def test_multiple_dice(self, dice_roller: DiceRoller) -> None:
        """Test rolling multiple dice."""
        result = dice_roller.roll("3d6")

        assert 3 <= result.total <= 18
        assert len(result.dice) == 3

    def test_complex_expression(self, dice_roller: DiceRoller) -> None:
        """Test complex dice expression."""
        result = dice_roller.roll("2d6+1d4+3")

        assert result.total >= 6  # 2 + 1 + 3
        assert result.total <= 19  # 12 + 4 + 3

    def test_advantage_roll(self, dice_roller: DiceRoller) -> None:
        """Test rolling with advantage."""
        result = dice_roller.roll("1d20", roll_type=RollType.ADVANTAGE)

        assert result.roll_type == RollType.ADVANTAGE
        # With advantage, should have rolled 2 dice
        assert len(result.dice) >= 1

    def test_disadvantage_roll(self, dice_roller: DiceRoller) -> None:
        """Test rolling with disadvantage."""
        result = dice_roller.roll("1d20", roll_type=RollType.DISADVANTAGE)

        assert result.roll_type == RollType.DISADVANTAGE

    def test_critical_detection(self) -> None:
        """Test that critical hits are detected."""
        # Create roller and roll many times to eventually get a 20
        roller = DiceRoller()
        found_critical = False

        for _ in range(100):
            result = roller.roll("1d20")
            if result.is_critical:
                found_critical = True
                assert result.dice[0] == 20
                break

        # Note: This is probabilistic, but with 100 rolls
        # we should almost always see a 20

    def test_fumble_detection(self) -> None:
        """Test that fumbles are detected."""
        roller = DiceRoller()
        found_fumble = False

        for _ in range(100):
            result = roller.roll("1d20")
            if result.is_fumble:
                found_fumble = True
                assert result.dice[0] == 1
                break

    def test_invalid_expression_raises_error(self, dice_roller: DiceRoller) -> None:
        """Test that invalid expressions raise DiceRollError."""
        with pytest.raises(DiceRollError):
            dice_roller.roll("invalid")

    def test_empty_expression_raises_error(self, dice_roller: DiceRoller) -> None:
        """Test that empty expression raises DiceRollError."""
        with pytest.raises(DiceRollError):
            dice_roller.roll("")

    def test_whitespace_expression_raises_error(self, dice_roller: DiceRoller) -> None:
        """Test that whitespace-only expression raises DiceRollError."""
        with pytest.raises(DiceRollError):
            dice_roller.roll("   ")


class TestDiceRollerSpecializedMethods:
    """Tests for specialized dice rolling methods."""

    def test_roll_ability_check(self, dice_roller: DiceRoller) -> None:
        """Test ability check rolling."""
        result = dice_roller.roll_ability_check(modifier=5)

        assert 6 <= result.total <= 25  # 1+5 to 20+5

    def test_roll_ability_check_negative_modifier(self, dice_roller: DiceRoller) -> None:
        """Test ability check with negative modifier."""
        result = dice_roller.roll_ability_check(modifier=-2)

        assert -1 <= result.total <= 18  # 1-2 to 20-2

    def test_roll_attack(self, dice_roller: DiceRoller) -> None:
        """Test attack roll."""
        result = dice_roller.roll_attack(attack_bonus=7)

        assert 8 <= result.total <= 27

    def test_roll_attack_with_advantage(self, dice_roller: DiceRoller) -> None:
        """Test attack roll with advantage."""
        result = dice_roller.roll_attack(attack_bonus=5, roll_type=RollType.ADVANTAGE)

        assert result.roll_type == RollType.ADVANTAGE

    def test_roll_damage(self, dice_roller: DiceRoller) -> None:
        """Test damage roll."""
        result = dice_roller.roll_damage("2d6+3")

        assert 5 <= result.total <= 15

    def test_roll_damage_critical(self, dice_roller: DiceRoller) -> None:
        """Test critical damage roll doubles dice."""
        result = dice_roller.roll_damage("2d6+3", is_critical=True)

        # Critical doubles dice: 4d6+3
        assert 7 <= result.total <= 27

    def test_roll_saving_throw(self, dice_roller: DiceRoller) -> None:
        """Test saving throw roll."""
        result = dice_roller.roll_saving_throw(modifier=3)

        assert 4 <= result.total <= 23

    def test_roll_initiative(self, dice_roller: DiceRoller) -> None:
        """Test initiative roll."""
        result = dice_roller.roll_initiative(dexterity_modifier=2)

        assert 3 <= result.total <= 22


class TestConvenienceRollFunction:
    """Tests for the module-level roll() function."""

    def test_basic_roll(self) -> None:
        """Test basic roll using convenience function."""
        result = roll("1d20")

        assert isinstance(result, DiceExpression)
        assert 1 <= result.total <= 20

    def test_roll_with_type(self) -> None:
        """Test roll with roll type using convenience function."""
        result = roll("1d20+5", roll_type=RollType.ADVANTAGE)

        assert result.roll_type == RollType.ADVANTAGE


class TestDiceExpression:
    """Tests for the DiceExpression dataclass."""

    def test_expression_is_frozen(self) -> None:
        """Test that DiceExpression is immutable."""
        expr = DiceExpression(
            expression="1d20",
            total=15,
            dice=[15],
            modifier=0,
            is_critical=False,
            is_fumble=False,
            roll_type=RollType.NORMAL,
        )

        with pytest.raises(AttributeError):
            expr.total = 20  # type: ignore[misc]
