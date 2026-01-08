"""Document ingestion module for the D&D 5E AI Campaign Manager.

This module provides comprehensive document ingestion capabilities including:
- Vision-based character sheet extraction using OpenRouter LLMs
- Rule book parsing and semantic chunking for RAG
- Adventure module scene extraction
- Vector store indexing and retrieval

Submodules:
    vision_parser: Extract character data from character sheet images/PDFs
    rules_loader: Parse and chunk D&D rule books for knowledge base
    module_loader: Extract scenes from adventure modules
    rag_store: Vector store for indexing and semantic search

Example:
    >>> from dnd_manager.ingestion import (
    ...     extract_character_from_pdf,
    ...     ingest_rule_book,
    ...     ingest_adventure_module,
    ...     RAGStore,
    ... )
    >>>
    >>> # Extract character from sheet
    >>> with open("character.pdf", "rb") as f:
    ...     character = extract_character_from_pdf(f.read())
    >>>
    >>> # Build knowledge base
    >>> documents = ingest_rule_book("phb.pdf", source_name="Player's Handbook")
    >>> scenes = ingest_adventure_module("adventure.pdf")
    >>>
    >>> # Index and search
    >>> store = RAGStore()
    >>> store.index_documents(documents)
    >>> store.index_scenes(scenes)
    >>> results = store.retrieve_relevant_context("fireball spell")
"""

from __future__ import annotations

# =============================================================================
# Vision Parser (Character Sheets)
# =============================================================================
from dnd_manager.ingestion.vision_parser import (
    CHARACTER_EXTRACTION_SYSTEM_PROMPT,
    CharacterSheetExtractor,
    extract_character_from_image_file,
    extract_character_from_pdf,
    pdf_page_to_base64_image,
)

# =============================================================================
# Rules Loader (Rule Books & Expansions)
# =============================================================================
from dnd_manager.ingestion.rules_loader import (
    Document,
    DocumentType,
    RuleBookParser,
    SemanticTextSplitter,
    ingest_rule_book,
    ingest_spell_list,
)

# =============================================================================
# Module Loader (Adventure Modules)
# =============================================================================
from dnd_manager.ingestion.module_loader import (
    AdventureModuleParser,
    MarkdownHeaderSplitter,
    ModuleScene,
    SceneCategory,
    extract_combat_encounters,
    extract_npcs,
    ingest_adventure_module,
)

# =============================================================================
# RAG Store (Vector Store & Retrieval)
# =============================================================================
from dnd_manager.ingestion.rag_store import (
    BaseVectorStore,
    ContentType,
    EmbeddingProvider,
    FAISSVectorStore,
    IndexedContent,
    RAGStore,
    SearchResult,
)

# =============================================================================
# Legacy Exports (for backwards compatibility)
# =============================================================================
from dnd_manager.ingestion.pdf_parser import (
    PDFParser,
    TextChunk,
    parse_pdf,
)
from dnd_manager.ingestion.ocr import (
    OCRProcessor,
    OCRResult,
    SUPPORTED_FORMATS,
)


__all__ = [
    # Vision Parser
    "CharacterSheetExtractor",
    "extract_character_from_pdf",
    "extract_character_from_image_file",
    "pdf_page_to_base64_image",
    "CHARACTER_EXTRACTION_SYSTEM_PROMPT",
    # Rules Loader
    "DocumentType",
    "Document",
    "SemanticTextSplitter",
    "RuleBookParser",
    "ingest_rule_book",
    "ingest_spell_list",
    # Module Loader
    "SceneCategory",
    "ModuleScene",
    "MarkdownHeaderSplitter",
    "AdventureModuleParser",
    "ingest_adventure_module",
    "extract_combat_encounters",
    "extract_npcs",
    # RAG Store
    "ContentType",
    "IndexedContent",
    "SearchResult",
    "EmbeddingProvider",
    "BaseVectorStore",
    "FAISSVectorStore",
    "RAGStore",
    # Legacy/PDF parsing
    "PDFParser",
    "TextChunk",
    "parse_pdf",
    "OCRProcessor",
    "OCRResult",
    "SUPPORTED_FORMATS",
]
