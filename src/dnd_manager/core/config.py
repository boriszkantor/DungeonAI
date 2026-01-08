"""Configuration management for the D&D 5E AI Campaign Manager.

This module provides centralized configuration management using pydantic-settings,
supporting environment variables, .env files, and runtime configuration overrides.
All sensitive values (API keys) are handled securely using SecretStr.

Example:
    >>> from dnd_manager.core.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.app_name)
    'D&D 5E AI Campaign Manager'

Environment Variables:
    DND_MANAGER_GEMINI_API_KEY: Google Gemini API key
    DND_MANAGER_OPENAI_API_KEY: OpenAI API key
    DND_MANAGER_SCENE_STORAGE_PATH: Path to scene storage directory
    DND_MANAGER_ASSET_PATH: Path to asset storage directory
    DND_MANAGER_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dnd_manager.core.exceptions import ConfigurationError


class AIProviderSettings(BaseSettings):
    """Configuration for AI provider connections.

    Attributes:
        openrouter_api_key: OpenRouter API key (primary, used for all AI operations).
        gemini_api_key: Google Gemini API key for AI operations.
        openai_api_key: OpenAI API key for alternative AI provider.
        default_provider: The default AI provider to use.
        gemini_model: Default Gemini model identifier.
        openai_model: Default OpenAI model identifier.
        max_retries: Maximum number of API retry attempts.
        timeout_seconds: API request timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="DND_MANAGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key (primary)",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key",
    )
    default_provider: Literal["openrouter", "gemini", "openai"] = Field(
        default="openrouter",
        description="Default AI provider to use",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Default Gemini model",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="Default OpenAI model",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum API retry attempts",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
        description="API request timeout",
    )

    @model_validator(mode="after")
    def validate_api_key_for_provider(self) -> "AIProviderSettings":
        """Ensure the default provider has a valid API key configured.

        Returns:
            Self if validation passes.

        Raises:
            ConfigurationError: If the default provider's API key is not set.
        """
        # OpenRouter is the primary provider - no validation needed if key exists
        if self.default_provider == "openrouter" and not self.openrouter_api_key:
            # Don't raise error - allow app to start, Settings page will prompt for key
            pass
        elif self.default_provider == "gemini" and not self.gemini_api_key:
            raise ConfigurationError(
                "Gemini is set as default provider but GEMINI_API_KEY is not configured",
                config_key="gemini_api_key",
            )
        elif self.default_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "OpenAI is set as default provider but OPENAI_API_KEY is not configured",
                config_key="openai_api_key",
            )
        return self


class StorageSettings(BaseSettings):
    """Configuration for file storage paths.

    Attributes:
        scene_storage_path: Directory for scene data storage.
        asset_path: Directory for asset files (images, maps, etc.).
        cache_path: Directory for cached data (embeddings, etc.).
        database_path: Path to the SQLite database file.
    """

    model_config = SettingsConfigDict(
        env_prefix="DND_MANAGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    scene_storage_path: Path = Field(
        default=Path("data/scenes"),
        description="Directory for scene data storage",
    )
    asset_path: Path = Field(
        default=Path("data/assets"),
        description="Directory for asset files",
    )
    cache_path: Path = Field(
        default=Path("data/cache"),
        description="Directory for cached data",
    )
    database_path: Path = Field(
        default=Path("data/dnd_manager.db"),
        description="Path to SQLite database",
    )

    @field_validator("scene_storage_path", "asset_path", "cache_path", mode="after")
    @classmethod
    def ensure_directory_exists(cls, value: Path) -> Path:
        """Ensure storage directories exist, creating them if necessary.

        Args:
            value: The path to validate and potentially create.

        Returns:
            The validated path.
        """
        value.mkdir(parents=True, exist_ok=True)
        return value


class VectorStoreSettings(BaseSettings):
    """Configuration for vector store (RAG) operations.

    Attributes:
        backend: Vector store backend to use ('faiss' or 'chromadb').
        embedding_model: Model to use for text embeddings.
        chunk_size: Size of text chunks for embedding.
        chunk_overlap: Overlap between consecutive chunks.
        similarity_top_k: Number of similar results to return.
    """

    model_config = SettingsConfigDict(
        env_prefix="DND_MANAGER_VECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend: Literal["faiss", "chromadb"] = Field(
        default="faiss",
        description="Vector store backend",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use",
    )
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=8000,
        description="Text chunk size for embedding",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between consecutive chunks",
    )
    similarity_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar results to return",
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "VectorStoreSettings":
        """Ensure chunk overlap is less than chunk size.

        Returns:
            Self if validation passes.

        Raises:
            ConfigurationError: If chunk_overlap >= chunk_size.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})",
                config_key="chunk_overlap",
            )
        return self


class GameSettings(BaseSettings):
    """Configuration for game engine behavior.

    Attributes:
        default_ruleset: Default D&D ruleset version.
        auto_roll_npc: Automatically roll for NPCs.
        initiative_mode: Initiative rolling mode.
        critical_hit_rule: Critical hit damage calculation rule.
    """

    model_config = SettingsConfigDict(
        env_prefix="DND_MANAGER_GAME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_ruleset: Literal["5e", "5e_2024"] = Field(
        default="5e",
        description="Default D&D ruleset version",
    )
    auto_roll_npc: bool = Field(
        default=True,
        description="Automatically roll for NPCs",
    )
    initiative_mode: Literal["standard", "group", "side"] = Field(
        default="standard",
        description="Initiative rolling mode",
    )
    critical_hit_rule: Literal["double_dice", "double_damage", "max_plus_roll"] = Field(
        default="double_dice",
        description="Critical hit damage calculation",
    )


class UISettings(BaseSettings):
    """Configuration for the Streamlit UI.

    Attributes:
        theme: UI theme ('light', 'dark', or 'auto').
        page_title: Browser page title.
        sidebar_default_state: Default sidebar state.
        enable_dev_mode: Enable development mode features.
    """

    model_config = SettingsConfigDict(
        env_prefix="DND_MANAGER_UI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    theme: Literal["light", "dark", "auto"] = Field(
        default="dark",
        description="UI theme",
    )
    page_title: str = Field(
        default="D&D 5E AI Campaign Manager",
        description="Browser page title",
    )
    sidebar_default_state: Literal["expanded", "collapsed"] = Field(
        default="expanded",
        description="Default sidebar state",
    )
    enable_dev_mode: bool = Field(
        default=False,
        description="Enable development mode",
    )


class Settings(BaseSettings):
    """Main application settings aggregating all configuration domains.

    This class serves as the single source of truth for all application
    configuration, aggregating domain-specific settings classes.

    Attributes:
        app_name: Application name.
        app_version: Application version string.
        debug: Enable debug mode.
        log_level: Application logging level.
        ai: AI provider settings.
        storage: File storage settings.
        vector_store: Vector store settings.
        game: Game engine settings.
        ui: UI settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="DND_MANAGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Application metadata
    app_name: str = Field(
        default="D&D 5E AI Campaign Manager",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Nested settings - using Field with default_factory
    ai: AIProviderSettings = Field(default_factory=AIProviderSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    game: GameSettings = Field(default_factory=GameSettings)
    ui: UISettings = Field(default_factory=UISettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode.

        Returns:
            True if not in debug mode.
        """
        return not self.debug


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the application settings singleton.

    This function returns a cached Settings instance, ensuring that
    configuration is only loaded once and shared across the application.

    Returns:
        The application Settings instance.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.

    Example:
        >>> settings = get_settings()
        >>> print(settings.app_name)
        'D&D 5E AI Campaign Manager'
    """
    try:
        return Settings()
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to load application settings: {exc}",
            details={"original_error": str(exc)},
        ) from exc


def clear_settings_cache() -> None:
    """Clear the settings cache, forcing a reload on next access.

    This is primarily useful for testing or when environment variables
    have changed at runtime.

    Example:
        >>> clear_settings_cache()
        >>> settings = get_settings()  # Reloads from environment
    """
    get_settings.cache_clear()


__all__ = [
    "AIProviderSettings",
    "StorageSettings",
    "VectorStoreSettings",
    "GameSettings",
    "UISettings",
    "Settings",
    "get_settings",
    "clear_settings_cache",
]
