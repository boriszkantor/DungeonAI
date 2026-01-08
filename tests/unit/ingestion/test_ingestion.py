"""Tests for the ingestion module."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from dnd_manager.ingestion.rules_loader import (
    Document,
    DocumentType,
    SemanticTextSplitter,
)
from dnd_manager.ingestion.module_loader import (
    MarkdownHeaderSplitter,
    ModuleScene,
    SceneCategory,
)
from dnd_manager.ingestion.rag_store import (
    ContentType,
    IndexedContent,
)


class TestSemanticTextSplitter:
    """Tests for SemanticTextSplitter."""

    def test_split_short_text(self) -> None:
        """Test that short text is not split."""
        splitter = SemanticTextSplitter(chunk_size=500)
        text = "This is a short text."

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self) -> None:
        """Test splitting of long text."""
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph even longer."

        chunks = splitter.split_text(text)

        assert len(chunks) >= 1
        # Each chunk should be within size limit (approximately)

    def test_split_on_paragraphs(self) -> None:
        """Test that splitting prefers paragraph boundaries."""
        splitter = SemanticTextSplitter(chunk_size=200)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = splitter.split_text(text)

        # Should split at paragraph boundaries
        assert all("\n\n" not in chunk or len(chunk) < 200 for chunk in chunks)

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        splitter = SemanticTextSplitter()
        
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []


class TestMarkdownHeaderSplitter:
    """Tests for MarkdownHeaderSplitter."""

    def test_split_by_headers(self) -> None:
        """Test splitting markdown by headers."""
        splitter = MarkdownHeaderSplitter()
        markdown = """# Chapter 1
Content for chapter 1.

## Section 1.1
Content for section 1.1.

## Section 1.2
Content for section 1.2.

# Chapter 2
Content for chapter 2.
"""
        sections = splitter.split(markdown)

        assert len(sections) >= 2
        # Check that headers are extracted
        headers = [s["header"] for s in sections]
        assert "Chapter 1" in headers
        assert "Chapter 2" in headers

    def test_preserves_hierarchy(self) -> None:
        """Test that parent headers are tracked."""
        splitter = MarkdownHeaderSplitter()
        markdown = """# Main
## Sub
Content
"""
        sections = splitter.split(markdown)

        # Find the "Sub" section
        sub_section = next((s for s in sections if s["header"] == "Sub"), None)
        assert sub_section is not None
        assert "Main" in sub_section["parent_headers"]

    def test_content_before_headers(self) -> None:
        """Test handling of content before first header."""
        splitter = MarkdownHeaderSplitter()
        markdown = """Some introductory content.

# First Header
Content under header.
"""
        sections = splitter.split(markdown)

        # Should have introduction section
        assert any(s["header"] == "Introduction" for s in sections)


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self) -> None:
        """Test basic document creation."""
        doc = Document(
            content="Test content",
            doc_id="",  # Auto-generate
            source="Test Source",
            doc_type=DocumentType.RULE,
            title="Test Rule",
        )

        assert doc.content == "Test content"
        assert doc.source == "Test Source"
        assert doc.doc_type == DocumentType.RULE
        assert doc.doc_id  # Should be auto-generated

    def test_document_types(self) -> None:
        """Test all document types are valid."""
        for doc_type in DocumentType:
            doc = Document(
                content="Content",
                doc_id="test",
                source="Source",
                doc_type=doc_type,
            )
            assert doc.doc_type == doc_type


class TestModuleScene:
    """Tests for ModuleScene dataclass."""

    def test_scene_creation(self) -> None:
        """Test basic scene creation."""
        scene = ModuleScene(
            title="Goblin Ambush",
            content="The goblins attack from the trees!",
            source_module="Test Adventure",
            category=SceneCategory.COMBAT,
        )

        assert scene.title == "Goblin Ambush"
        assert scene.category == SceneCategory.COMBAT
        assert isinstance(scene.uid, UUID)

    def test_scene_id_generation(self) -> None:
        """Test scene ID is generated correctly."""
        scene = ModuleScene(
            title="The Dark Cave",
            content="A mysterious cave entrance.",
            source_module="Adventure Module",
        )

        scene_id = scene.scene_id
        assert "Adventure Module" in scene_id
        assert "dark_cave" in scene_id.lower()

    def test_all_scene_categories(self) -> None:
        """Test all scene categories are valid."""
        for category in SceneCategory:
            scene = ModuleScene(
                title="Test",
                content="Content",
                source_module="Module",
                category=category,
            )
            assert scene.category == category


class TestIndexedContent:
    """Tests for IndexedContent dataclass."""

    def test_from_document(self) -> None:
        """Test creating IndexedContent from Document."""
        doc = Document(
            content="Fireball spell description",
            doc_id="phb_spell_fireball",
            source="Player's Handbook",
            doc_type=DocumentType.SPELL,
            title="Fireball",
            category="Evocation",
        )

        indexed = IndexedContent.from_document(doc)

        assert indexed.content_id == doc.doc_id
        assert indexed.content_type == ContentType.RULE_DOCUMENT
        assert indexed.text == doc.content
        assert indexed.source == doc.source
        assert indexed.title == doc.title
        assert indexed.metadata["doc_type"] == "spell"

    def test_from_scene(self) -> None:
        """Test creating IndexedContent from ModuleScene."""
        scene = ModuleScene(
            title="Dragon's Lair",
            content="The ancient dragon sleeps...",
            source_module="Dragon Quest",
            category=SceneCategory.COMBAT,
            read_aloud="You see a massive dragon!",
            monsters_mentioned=["Ancient Red Dragon"],
        )

        indexed = IndexedContent.from_scene(scene)

        assert indexed.content_type == ContentType.ADVENTURE_SCENE
        assert indexed.text == scene.content
        assert indexed.source == scene.source_module
        assert indexed.metadata["category"] == "combat"
        assert indexed.metadata["has_read_aloud"] is True
        assert "Ancient Red Dragon" in indexed.metadata["monsters"]


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_all_types_have_values(self) -> None:
        """Test all document types have string values."""
        for doc_type in DocumentType:
            assert isinstance(doc_type.value, str)
            assert len(doc_type.value) > 0


class TestSceneCategory:
    """Tests for SceneCategory enum."""

    def test_all_categories_have_values(self) -> None:
        """Test all scene categories have string values."""
        for category in SceneCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types(self) -> None:
        """Test content type values."""
        assert ContentType.RULE_DOCUMENT == "rule_document"
        assert ContentType.ADVENTURE_SCENE == "adventure_scene"
        assert ContentType.CHARACTER == "character"
        assert ContentType.CUSTOM == "custom"
