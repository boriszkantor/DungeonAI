"""Tests for configuration management."""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_manager.core.config import (
    Settings,
    StorageSettings,
    VectorStoreSettings,
    clear_settings_cache,
    get_settings,
)
from dnd_manager.core.exceptions import ConfigurationError


class TestStorageSettings:
    """Tests for StorageSettings configuration."""

    def test_default_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default storage paths are created."""
        monkeypatch.chdir(tmp_path)

        settings = StorageSettings()

        assert settings.scene_storage_path.exists()
        assert settings.asset_path.exists()
        assert settings.cache_path.exists()

    def test_custom_paths(self, tmp_path: Path) -> None:
        """Test custom storage paths."""
        custom_scene = tmp_path / "custom_scenes"
        custom_assets = tmp_path / "custom_assets"
        custom_cache = tmp_path / "custom_cache"

        settings = StorageSettings(
            scene_storage_path=custom_scene,
            asset_path=custom_assets,
            cache_path=custom_cache,
        )

        assert settings.scene_storage_path == custom_scene
        assert custom_scene.exists()


class TestVectorStoreSettings:
    """Tests for VectorStoreSettings configuration."""

    def test_default_values(self) -> None:
        """Test default vector store settings."""
        settings = VectorStoreSettings()

        assert settings.backend == "faiss"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.similarity_top_k == 5

    def test_chunk_overlap_validation(self) -> None:
        """Test that chunk_overlap must be less than chunk_size."""
        with pytest.raises(ConfigurationError) as exc_info:
            VectorStoreSettings(chunk_size=500, chunk_overlap=600)

        assert "chunk_overlap" in str(exc_info.value)


class TestSettings:
    """Tests for main Settings configuration."""

    def test_default_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test default settings initialization."""
        # Set required env var and change to temp dir
        monkeypatch.setenv("DND_MANAGER_GEMINI_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)

        settings = Settings()

        assert settings.app_name == "D&D 5E AI Campaign Manager"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_debug_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test debug mode setting."""
        monkeypatch.setenv("DND_MANAGER_DEBUG", "true")
        monkeypatch.setenv("DND_MANAGER_GEMINI_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)

        settings = Settings()

        assert settings.debug is True
        assert settings.is_production is False

    def test_is_production_property(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test is_production property."""
        monkeypatch.setenv("DND_MANAGER_DEBUG", "false")
        monkeypatch.setenv("DND_MANAGER_GEMINI_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)

        settings = Settings()

        assert settings.is_production is True


class TestGetSettings:
    """Tests for get_settings singleton function."""

    def test_returns_settings_instance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that get_settings returns a Settings instance."""
        monkeypatch.setenv("DND_MANAGER_GEMINI_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        clear_settings_cache()

        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_caching(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that settings are cached."""
        monkeypatch.setenv("DND_MANAGER_GEMINI_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        clear_settings_cache()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_cache_clear(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that cache can be cleared."""
        monkeypatch.setenv("DND_MANAGER_GEMINI_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        clear_settings_cache()

        settings1 = get_settings()
        clear_settings_cache()
        settings2 = get_settings()

        # After cache clear, should be different instances
        # (though with same values)
        assert settings1 is not settings2
