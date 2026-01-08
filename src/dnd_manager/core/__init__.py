"""Core module providing configuration, logging, and base exceptions.

This module serves as the foundation for the D&D 5E AI Campaign Manager,
providing essential infrastructure components used throughout the application.

Exports:
    Exceptions:
        DndManagerError: Base exception for all application errors.
        ConfigurationError: Configuration-related errors.
        ValidationError: Data validation errors.

    Configuration:
        Settings: Main application settings class.
        get_settings: Get the settings singleton.
        clear_settings_cache: Force settings reload.

    Logging:
        configure_logging: Set up application logging.
        get_logger: Get a configured logger instance.
        bind_context: Add context to log entries.
        clear_context: Clear logging context.
"""

from __future__ import annotations

from dnd_manager.core.config import (
    AIProviderSettings,
    GameSettings,
    Settings,
    StorageSettings,
    UISettings,
    VectorStoreSettings,
    clear_settings_cache,
    get_settings,
)
from dnd_manager.core.exceptions import (
    AIConnectionError,
    AIContextLimitError,
    AIControlError,
    AIRateLimitError,
    AIResponseError,
    CombatError,
    ComponentRenderError,
    ConfigurationError,
    DiceRollError,
    DndManagerError,
    EmbeddingError,
    GameEngineError,
    IngestionError,
    InvalidGameStateError,
    OCRError,
    PDFParseError,
    SessionStateError,
    TurnManagementError,
    UIError,
    ValidationError,
    VectorStoreError,
)
from dnd_manager.core.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)


__all__ = [
    # Base exception
    "DndManagerError",
    # Configuration exceptions
    "ConfigurationError",
    "ValidationError",
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
    # UI exceptions
    "UIError",
    "SessionStateError",
    "ComponentRenderError",
    # Configuration
    "Settings",
    "AIProviderSettings",
    "StorageSettings",
    "VectorStoreSettings",
    "GameSettings",
    "UISettings",
    "get_settings",
    "clear_settings_cache",
    # Logging
    "configure_logging",
    "get_logger",
    "bind_context",
    "clear_context",
]
