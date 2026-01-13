"""Document ingestion module for the D&D 5E AI Campaign Manager.

This module provides comprehensive document ingestion capabilities including:
- Vision-based character sheet extraction using OpenRouter LLMs
- Rule book parsing and semantic chunking for RAG
- Adventure module scene extraction
- Vector store indexing and retrieval
- Knowledge graph construction for hybrid graph-vector RAG

Submodules:
    vision_parser: Extract character data from character sheet images/PDFs
    rules_loader: Parse and chunk D&D rule books for knowledge base
    module_loader: Extract scenes from adventure modules
    rag_store: Vector store for indexing and semantic search
    ontology: D&D 5E ontology schema and triple validation
    entity_resolver: Canonicalization of entity names
    knowledge_graph: NetworkX-based knowledge graph for rule relationships
    hybrid_retriever: Combined graph-vector search

Example:
    >>> from dnd_manager.ingestion import (
    ...     extract_character_from_pdf,
    ...     ingest_rule_book,
    ...     ingest_adventure_module,
    ...     RAGStore,
    ...     HybridRetriever,
    ...     DnDKnowledgeGraph,
    ... )
    >>>
    >>> # Extract character from sheet
    >>> with open("character.pdf", "rb") as f:
    ...     character = extract_character_from_pdf(f.read())
    >>>
    >>> # Build knowledge base with graph
    >>> ingestor = UniversalIngestor(enable_knowledge_graph=True)
    >>> ingestor.ingest_rulebook("phb.pdf")
    >>> 
    >>> # Hybrid search finds related rules (e.g., Evasion for Fireball + Monk)
    >>> result = ingestor.hybrid_search("Monk hit by Fireball", character=monk)
    >>> print(result.graph_context)
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
# Knowledge Graph & Hybrid RAG
# =============================================================================
from dnd_manager.ingestion.ontology import (
    EntityType,
    OntologySchema,
    Predicate,
    PredicateConstraint,
    SemanticTriple,
    ValidationResult,
)
from dnd_manager.ingestion.entity_resolver import (
    CanonicalEntity,
    EntityResolver,
    ResolutionResult,
)
from dnd_manager.ingestion.knowledge_graph import (
    DnDKnowledgeGraph,
    GraphEdge,
    GraphNode,
    TraversalResult,
)
from dnd_manager.ingestion.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    RelatedRule,
    create_hybrid_retriever,
)
from dnd_manager.ingestion.triple_extractor import (
    ExtractionResult,
    ExtractionStats,
    TripleExtractor,
    extract_triples_from_chunks,
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
    # Ontology
    "EntityType",
    "Predicate",
    "PredicateConstraint",
    "SemanticTriple",
    "ValidationResult",
    "OntologySchema",
    # Entity Resolution
    "CanonicalEntity",
    "ResolutionResult",
    "EntityResolver",
    # Knowledge Graph
    "GraphNode",
    "GraphEdge",
    "TraversalResult",
    "DnDKnowledgeGraph",
    # Triple Extraction
    "ExtractionResult",
    "ExtractionStats",
    "TripleExtractor",
    "extract_triples_from_chunks",
    # Hybrid Retriever
    "RelatedRule",
    "HybridSearchResult",
    "HybridRetriever",
    "create_hybrid_retriever",
    # Legacy/PDF parsing
    "PDFParser",
    "TextChunk",
    "parse_pdf",
    "OCRProcessor",
    "OCRResult",
    "SUPPORTED_FORMATS",
]
