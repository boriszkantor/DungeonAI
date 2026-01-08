"""Custom exception hierarchy for the D&D 5E AI Campaign Manager.

This module defines a comprehensive exception hierarchy that provides
granular error handling across all domains of the application. All
exceptions inherit from DndManagerError, enabling unified error handling
at the application boundary while preserving domain-specific context.

Example:
    >>> from dnd_manager.core.exceptions import IngestionError
    >>> raise IngestionError("Failed to parse PDF", source_file="monster_manual.pdf")
"""

from __future__ import annotations

from typing import Any


class DndManagerError(Exception):
    """Base exception for all D&D Campaign Manager errors.

    All custom exceptions in this application inherit from this class,
    enabling unified error handling at the application boundary.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary containing additional error context.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Initialize the base exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary containing additional error context.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the exception message with optional details.

        Returns:
            Formatted error message including any provided details.
        """
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message

    def __repr__(self) -> str:
        """Return a detailed string representation of the exception.

        Returns:
            String representation suitable for debugging.
        """
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# =============================================================================
# Ingestion Domain Exceptions
# =============================================================================


class IngestionError(DndManagerError):
    """Base exception for all document ingestion errors.

    Raised when there are issues with parsing, processing, or indexing
    documents for the RAG (Retrieval-Augmented Generation) pipeline.
    """

    def __init__(
        self,
        message: str,
        *,
        source_file: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ingestion error with source file context.

        Args:
            message: Human-readable error description.
            source_file: Path to the file that caused the error.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if source_file:
            combined_details["source_file"] = source_file
        super().__init__(message, details=combined_details)


class PDFParseError(IngestionError):
    """Raised when a PDF document cannot be parsed.

    This typically occurs when the PDF is corrupted, password-protected,
    or uses an unsupported format.
    """

    def __init__(
        self,
        message: str,
        *,
        source_file: str | None = None,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize PDF parse error with page context.

        Args:
            message: Human-readable error description.
            source_file: Path to the PDF file.
            page_number: Page number where the error occurred.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if page_number is not None:
            combined_details["page_number"] = page_number
        super().__init__(message, source_file=source_file, details=combined_details)


class OCRError(IngestionError):
    """Raised when OCR (Optical Character Recognition) processing fails.

    This typically occurs when an image cannot be processed or the OCR
    engine encounters an unrecoverable error.
    """


class EmbeddingError(IngestionError):
    """Raised when text embedding generation fails.

    This typically occurs when the embedding model is unavailable or
    the input text exceeds model limits.
    """

    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize embedding error with model context.

        Args:
            message: Human-readable error description.
            model_name: Name of the embedding model that failed.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if model_name:
            combined_details["model_name"] = model_name
        super().__init__(message, details=combined_details)


class VectorStoreError(IngestionError):
    """Raised when vector store operations fail.

    This includes errors with FAISS or ChromaDB operations such as
    indexing, searching, or persistence.
    """


# =============================================================================
# Game Engine Domain Exceptions
# =============================================================================


class GameEngineError(DndManagerError):
    """Base exception for all game engine errors.

    Raised when there are issues with game state management, combat
    resolution, or rules processing.
    """


class InvalidGameStateError(GameEngineError):
    """Raised when the game enters an invalid or inconsistent state.

    This typically occurs when state transitions violate game rules
    or invariants.
    """

    def __init__(
        self,
        message: str,
        *,
        current_state: str | None = None,
        expected_states: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize invalid game state error with state context.

        Args:
            message: Human-readable error description.
            current_state: The current invalid state identifier.
            expected_states: List of valid states that were expected.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if current_state:
            combined_details["current_state"] = current_state
        if expected_states:
            combined_details["expected_states"] = expected_states
        super().__init__(message, details=combined_details)


class CombatError(GameEngineError):
    """Raised when combat resolution encounters an error.

    This includes issues with initiative tracking, action resolution,
    or damage calculation.
    """

    def __init__(
        self,
        message: str,
        *,
        combatant_id: str | None = None,
        round_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize combat error with combat context.

        Args:
            message: Human-readable error description.
            combatant_id: Identifier of the combatant involved.
            round_number: Current combat round when error occurred.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if combatant_id:
            combined_details["combatant_id"] = combatant_id
        if round_number is not None:
            combined_details["round_number"] = round_number
        super().__init__(message, details=combined_details)


class DiceRollError(GameEngineError):
    """Raised when dice rolling operations fail.

    This typically occurs when parsing invalid dice notation or when
    roll modifiers exceed reasonable bounds.
    """

    def __init__(
        self,
        message: str,
        *,
        expression: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize dice roll error with expression context.

        Args:
            message: Human-readable error description.
            expression: The dice expression that caused the error.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if expression:
            combined_details["expression"] = expression
        super().__init__(message, details=combined_details)


class TurnManagementError(GameEngineError):
    """Raised when turn or initiative management fails.

    This includes errors with initiative order, turn skipping, or
    delay/ready action handling.
    """


# =============================================================================
# AI Control Domain Exceptions
# =============================================================================


class AIControlError(DndManagerError):
    """Base exception for all AI-related errors.

    Raised when there are issues with AI model interactions, including
    API calls, response parsing, or content generation.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize AI control error with model context.

        Args:
            message: Human-readable error description.
            model: Name of the AI model involved.
            provider: Name of the AI provider (e.g., 'gemini', 'openai').
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if model:
            combined_details["model"] = model
        if provider:
            combined_details["provider"] = provider
        super().__init__(message, details=combined_details)


class AIConnectionError(AIControlError):
    """Raised when connection to an AI service fails.

    This typically occurs due to network issues, invalid API keys,
    or service unavailability.
    """


class AIResponseError(AIControlError):
    """Raised when an AI response cannot be processed.

    This includes issues with response parsing, unexpected formats,
    or content policy violations.
    """


class AIRateLimitError(AIControlError):
    """Raised when AI API rate limits are exceeded.

    This exception includes retry timing information when available.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        model: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize rate limit error with retry context.

        Args:
            message: Human-readable error description.
            retry_after_seconds: Seconds to wait before retrying.
            model: Name of the AI model involved.
            provider: Name of the AI provider.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if retry_after_seconds is not None:
            combined_details["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, model=model, provider=provider, details=combined_details)


class AIContextLimitError(AIControlError):
    """Raised when AI context window limits are exceeded.

    This typically occurs when prompts or conversation history exceed
    the model's maximum token limit.
    """

    def __init__(
        self,
        message: str,
        *,
        token_count: int | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize context limit error with token context.

        Args:
            message: Human-readable error description.
            token_count: Number of tokens in the request.
            max_tokens: Maximum allowed tokens for the model.
            model: Name of the AI model involved.
            provider: Name of the AI provider.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if token_count is not None:
            combined_details["token_count"] = token_count
        if max_tokens is not None:
            combined_details["max_tokens"] = max_tokens
        super().__init__(message, model=model, provider=provider, details=combined_details)


# =============================================================================
# Configuration & Validation Exceptions
# =============================================================================


class ConfigurationError(DndManagerError):
    """Raised when application configuration is invalid.

    This includes missing required settings, invalid values, or
    incompatible configuration combinations.
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration error with config key context.

        Args:
            message: Human-readable error description.
            config_key: The configuration key that caused the error.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if config_key:
            combined_details["config_key"] = config_key
        super().__init__(message, details=combined_details)


class ValidationError(DndManagerError):
    """Raised when data validation fails.

    This includes schema validation errors, constraint violations,
    or type mismatches in user input or external data.
    """

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        invalid_value: Any | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize validation error with field context.

        Args:
            message: Human-readable error description.
            field_name: Name of the field that failed validation.
            invalid_value: The value that failed validation.
            details: Optional dictionary containing additional error context.
        """
        combined_details = details or {}
        if field_name:
            combined_details["field_name"] = field_name
        if invalid_value is not None:
            combined_details["invalid_value"] = invalid_value
        super().__init__(message, details=combined_details)


# =============================================================================
# UI Domain Exceptions
# =============================================================================


class UIError(DndManagerError):
    """Base exception for all UI-related errors.

    Raised when there are issues with the Streamlit interface, including
    rendering failures, session state issues, or component errors.
    """


class SessionStateError(UIError):
    """Raised when session state operations fail.

    This typically occurs when accessing uninitialized state or when
    state becomes corrupted.
    """


class ComponentRenderError(UIError):
    """Raised when a UI component fails to render.

    This includes errors in custom components or Streamlit widget failures.
    """


__all__ = [
    # Base exception
    "DndManagerError",
    # Ingestion exceptions
    "IngestionError",
    "PDFParseError",
    "OCRError",
    "EmbeddingError",
    "VectorStoreError",
    # Game engine exceptions
    "GameEngineError",
    "InvalidGameStateError",
    "CombatError",
    "DiceRollError",
    "TurnManagementError",
    # AI control exceptions
    "AIControlError",
    "AIConnectionError",
    "AIResponseError",
    "AIRateLimitError",
    "AIContextLimitError",
    # Configuration exceptions
    "ConfigurationError",
    "ValidationError",
    # UI exceptions
    "UIError",
    "SessionStateError",
    "ComponentRenderError",
]
