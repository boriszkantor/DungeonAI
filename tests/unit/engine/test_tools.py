"""Tests for game action tools."""

from __future__ import annotations

import pytest

from dnd_manager.engine.tools import (
    ToolCall,
    ToolCategory,
    ToolResult,
    check_skill,
    dash,
    disengage,
    dodge,
    execute_tool,
    get_all_tools,
    get_tool,
    get_tools_as_openai_schema,
    get_tools_by_category,
    move,
    roll_attack,
    roll_damage,
)


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_get_all_tools(self) -> None:
        """Test that we can get all registered tools."""
        tools = get_all_tools()
        assert len(tools) > 0
        assert all(hasattr(t, "name") for t in tools)

    def test_get_tool_by_name(self) -> None:
        """Test getting a specific tool by name."""
        tool = get_tool("roll_attack")
        assert tool is not None
        assert tool.name == "roll_attack"
        assert tool.category == ToolCategory.ATTACK

    def test_get_nonexistent_tool(self) -> None:
        """Test getting a tool that doesn't exist."""
        tool = get_tool("nonexistent_tool")
        assert tool is None

    def test_get_tools_by_category(self) -> None:
        """Test filtering tools by category."""
        attack_tools = get_tools_by_category(ToolCategory.ATTACK)
        assert len(attack_tools) > 0
        assert all(t.category == ToolCategory.ATTACK for t in attack_tools)

    def test_tools_as_openai_schema(self) -> None:
        """Test generating OpenAI function calling schema."""
        schema = get_tools_as_openai_schema()
        assert len(schema) > 0
        
        for tool_schema in schema:
            assert tool_schema["type"] == "function"
            assert "function" in tool_schema
            assert "name" in tool_schema["function"]
            assert "description" in tool_schema["function"]
            assert "parameters" in tool_schema["function"]


class TestRollAttack:
    """Tests for the roll_attack tool."""

    def test_attack_returns_string(self) -> None:
        """Test that attack returns a formatted string."""
        result = roll_attack(
            attack_bonus=5,
            target_ac=15,
            damage_dice="1d8",
            damage_bonus=3,
        )
        assert isinstance(result, str)
        assert "Attack Roll" in result
        assert "vs AC 15" in result

    def test_attack_with_advantage(self) -> None:
        """Test attack roll with advantage."""
        result = roll_attack(
            attack_bonus=5,
            target_ac=10,
            advantage=True,
        )
        assert "advantage" in result.lower()

    def test_attack_with_disadvantage(self) -> None:
        """Test attack roll with disadvantage."""
        result = roll_attack(
            attack_bonus=5,
            target_ac=10,
            disadvantage=True,
        )
        assert "disadvantage" in result.lower()

    def test_attack_result_contains_outcome(self) -> None:
        """Test that attack result indicates hit or miss."""
        result = roll_attack(
            attack_bonus=5,
            target_ac=15,
        )
        # Result should contain either HIT, MISS, or CRITICAL
        assert any(word in result.upper() for word in ["HIT", "MISS", "CRITICAL"])


class TestRollDamage:
    """Tests for the roll_damage tool."""

    def test_damage_returns_string(self) -> None:
        """Test that damage returns a formatted string."""
        result = roll_damage(
            damage_dice="2d6",
            damage_bonus=3,
            damage_type="slashing",
        )
        assert isinstance(result, str)
        assert "Damage" in result
        assert "slashing" in result

    def test_critical_damage_doubles_dice(self) -> None:
        """Test that critical damage indication is shown."""
        result = roll_damage(
            damage_dice="2d6",
            damage_bonus=3,
            is_critical=True,
        )
        assert "CRITICAL" in result


class TestCheckSkill:
    """Tests for the check_skill tool."""

    def test_skill_check_returns_string(self) -> None:
        """Test that skill check returns formatted string."""
        result = check_skill(
            skill_name="Perception",
            skill_bonus=5,
            dc=15,
        )
        assert isinstance(result, str)
        assert "Perception" in result
        assert "vs DC 15" in result

    def test_skill_check_shows_success_or_failure(self) -> None:
        """Test that skill check shows outcome."""
        result = check_skill(
            skill_name="Stealth",
            skill_bonus=10,
            dc=10,
        )
        assert any(word in result.upper() for word in ["SUCCESS", "FAILURE"])

    def test_skill_check_with_advantage(self) -> None:
        """Test skill check with advantage."""
        result = check_skill(
            skill_name="Athletics",
            skill_bonus=5,
            dc=15,
            advantage=True,
        )
        assert "advantage" in result.lower()


class TestMovementTools:
    """Tests for movement-related tools."""

    def test_move_within_speed(self) -> None:
        """Test moving within movement allowance."""
        result = move(
            direction="north",
            distance_feet=20,
            movement_remaining=30,
        )
        assert "Moved 20 ft" in result
        assert "10 ft movement remaining" in result

    def test_move_exceeds_speed(self) -> None:
        """Test moving beyond movement allowance."""
        result = move(
            direction="north",
            distance_feet=40,
            movement_remaining=30,
        )
        assert "Cannot move" in result

    def test_dash(self) -> None:
        """Test the dash action."""
        result = dash(current_speed=30)
        assert "DASH" in result
        assert "30" in result

    def test_dodge(self) -> None:
        """Test the dodge action."""
        result = dodge()
        assert "DODGE" in result
        assert "disadvantage" in result.lower()

    def test_disengage(self) -> None:
        """Test the disengage action."""
        result = disengage()
        assert "DISENGAGE" in result
        assert "opportunity attacks" in result.lower()


class TestToolExecution:
    """Tests for tool execution."""

    def test_execute_valid_tool(self) -> None:
        """Test executing a valid tool call."""
        call = ToolCall(
            tool_name="roll_attack",
            arguments={
                "attack_bonus": 5,
                "target_ac": 15,
            },
            call_id="test_1",
        )
        result = execute_tool(call)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "roll_attack"
        assert result.call_id == "test_1"
        assert len(result.result) > 0

    def test_execute_invalid_tool(self) -> None:
        """Test executing a non-existent tool."""
        call = ToolCall(
            tool_name="nonexistent_tool",
            arguments={},
            call_id="test_2",
        )
        result = execute_tool(call)

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_execute_tool_with_bad_arguments(self) -> None:
        """Test executing a tool with invalid arguments."""
        call = ToolCall(
            tool_name="roll_attack",
            arguments={
                "invalid_param": "value",
            },
            call_id="test_3",
        )
        result = execute_tool(call)

        # Should fail due to missing required arguments
        assert result.success is False
