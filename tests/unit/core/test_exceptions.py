"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from dnd_manager.core.exceptions import (
    AIConnectionError,
    AIContextLimitError,
    AIControlError,
    AIRateLimitError,
    CombatError,
    ConfigurationError,
    DiceRollError,
    DndManagerError,
    GameEngineError,
    IngestionError,
    PDFParseError,
    ValidationError,
)


class TestDndManagerError:
    """Tests for the base DndManagerError exception."""

    def test_basic_message(self) -> None:
        """Test exception with basic message."""
        exc = DndManagerError("Test error message")
        assert exc.message == "Test error message"
        assert exc.details == {}
        assert str(exc) == "Test error message"

    def test_with_details(self) -> None:
        """Test exception with additional details."""
        exc = DndManagerError(
            "Test error",
            details={"key": "value", "count": 42},
        )
        assert exc.details == {"key": "value", "count": 42}
        assert "key='value'" in str(exc)
        assert "count=42" in str(exc)

    def test_repr(self) -> None:
        """Test exception repr output."""
        exc = DndManagerError("Test", details={"x": 1})
        repr_str = repr(exc)
        assert "DndManagerError" in repr_str
        assert "Test" in repr_str
        assert "x" in repr_str


class TestIngestionExceptions:
    """Tests for ingestion-related exceptions."""

    def test_ingestion_error_with_source_file(self) -> None:
        """Test IngestionError with source file."""
        exc = IngestionError("Parse failed", source_file="test.pdf")
        assert exc.details["source_file"] == "test.pdf"

    def test_pdf_parse_error_with_page(self) -> None:
        """Test PDFParseError with page number."""
        exc = PDFParseError(
            "Invalid page",
            source_file="doc.pdf",
            page_number=5,
        )
        assert exc.details["source_file"] == "doc.pdf"
        assert exc.details["page_number"] == 5

    def test_inheritance(self) -> None:
        """Test exception inheritance chain."""
        exc = PDFParseError("Error")
        assert isinstance(exc, IngestionError)
        assert isinstance(exc, DndManagerError)
        assert isinstance(exc, Exception)


class TestGameEngineExceptions:
    """Tests for game engine exceptions."""

    def test_combat_error(self) -> None:
        """Test CombatError with combat context."""
        exc = CombatError(
            "Invalid attack",
            combatant_id="fighter-1",
            round_number=3,
        )
        assert exc.details["combatant_id"] == "fighter-1"
        assert exc.details["round_number"] == 3

    def test_dice_roll_error(self) -> None:
        """Test DiceRollError with expression."""
        exc = DiceRollError("Invalid dice", expression="1d0+5")
        assert exc.details["expression"] == "1d0+5"

    def test_game_engine_inheritance(self) -> None:
        """Test game engine exception inheritance."""
        exc = CombatError("Error")
        assert isinstance(exc, GameEngineError)
        assert isinstance(exc, DndManagerError)


class TestAIControlExceptions:
    """Tests for AI control exceptions."""

    def test_ai_control_error_with_provider(self) -> None:
        """Test AIControlError with model and provider."""
        exc = AIControlError(
            "API failed",
            model="gpt-4",
            provider="openai",
        )
        assert exc.details["model"] == "gpt-4"
        assert exc.details["provider"] == "openai"

    def test_rate_limit_error(self) -> None:
        """Test AIRateLimitError with retry timing."""
        exc = AIRateLimitError(
            "Rate limited",
            retry_after_seconds=30.5,
            provider="gemini",
        )
        assert exc.details["retry_after_seconds"] == 30.5
        assert exc.details["provider"] == "gemini"

    def test_context_limit_error(self) -> None:
        """Test AIContextLimitError with token counts."""
        exc = AIContextLimitError(
            "Context exceeded",
            token_count=150000,
            max_tokens=128000,
            model="gpt-4-turbo",
        )
        assert exc.details["token_count"] == 150000
        assert exc.details["max_tokens"] == 128000


class TestConfigurationExceptions:
    """Tests for configuration exceptions."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError with config key."""
        exc = ConfigurationError(
            "Missing API key",
            config_key="gemini_api_key",
        )
        assert exc.details["config_key"] == "gemini_api_key"

    def test_validation_error(self) -> None:
        """Test ValidationError with field info."""
        exc = ValidationError(
            "Invalid value",
            field_name="hit_points",
            invalid_value=-5,
        )
        assert exc.details["field_name"] == "hit_points"
        assert exc.details["invalid_value"] == -5


class TestExceptionChaining:
    """Tests for exception chaining behavior."""

    def test_raise_from(self) -> None:
        """Test that exceptions can be properly chained."""
        original = ValueError("Original error")

        with pytest.raises(IngestionError) as exc_info:
            try:
                raise original
            except ValueError as e:
                raise IngestionError("Wrapped error") from e

        assert exc_info.value.__cause__ is original
