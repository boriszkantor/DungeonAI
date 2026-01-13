"""Vector store for RAG (Retrieval-Augmented Generation).

This module provides a unified vector store interface for indexing and
retrieving D&D content including rule documents, adventure scenes, and
other game materials.

Supports both FAISS and ChromaDB backends with metadata filtering.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from dnd_manager.core.config import get_settings
from dnd_manager.core.exceptions import EmbeddingError, IngestionError, VectorStoreError
from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.module_loader import ModuleScene
from dnd_manager.ingestion.rules_loader import Document


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ContentType(StrEnum):
    """Types of content in the vector store."""

    RULE_DOCUMENT = "rule_document"
    ADVENTURE_SCENE = "adventure_scene"
    CHARACTER = "character"
    CUSTOM = "custom"


@dataclass
class EnhancedMetadata:
    """Enhanced metadata for game-aware retrieval filtering.
    
    This enables smart filtering based on character capabilities:
    - "Can I cast Shield?" filters by class and spell level
    - "Can I use Sneak Attack?" filters by class features
    - "Can I run up this wall?" considers character level for Monk features
    
    Attributes:
        doc_subtype: Specific type (spell, feat, class_feature, item, general_rule).
        classes: List of classes that can use this (e.g., ["wizard", "sorcerer"]).
        level_requirement: Minimum character level required.
        prerequisites: Text description of prerequisites.
        tags: Additional searchable tags.
    """
    
    doc_subtype: str | None = None  # spell, feat, class_feature, item, general_rule
    classes: list[str] = field(default_factory=list)  # ["wizard", "sorcerer", "bard"]
    level_requirement: int | None = None  # Minimum level needed
    prerequisites: list[str] = field(default_factory=list)  # ["Dexterity 13 or higher"]
    tags: list[str] = field(default_factory=list)  # ["combat", "utility", "ritual"]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for storage."""
        return {
            "doc_subtype": self.doc_subtype,
            "classes": self.classes,
            "level_requirement": self.level_requirement,
            "prerequisites": self.prerequisites,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhancedMetadata":
        """Create from dict."""
        return cls(
            doc_subtype=data.get("doc_subtype"),
            classes=data.get("classes", []),
            level_requirement=data.get("level_requirement"),
            prerequisites=data.get("prerequisites", []),
            tags=data.get("tags", []),
        )


@dataclass
class IndexedContent:
    """Content stored in the vector store.

    Attributes:
        content_id: Unique identifier.
        content_type: Type of content.
        text: The text content.
        embedding: Vector embedding (set after indexing).
        source: Source document/module name.
        title: Content title.
        metadata: Additional searchable metadata.
    """

    content_id: str
    content_type: ContentType
    text: str
    source: str
    title: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(cls, doc: Document) -> "IndexedContent":
        """Create IndexedContent from a rule Document."""
        return cls(
            content_id=doc.doc_id,
            content_type=ContentType.RULE_DOCUMENT,
            text=doc.content,
            source=doc.source,
            title=doc.title,
            metadata={
                "doc_type": doc.doc_type.value,
                "category": doc.category,
                "page_number": doc.page_number,
            },
        )

    @classmethod
    def from_scene(cls, scene: ModuleScene) -> "IndexedContent":
        """Create IndexedContent from an adventure ModuleScene."""
        return cls(
            content_id=scene.scene_id,
            content_type=ContentType.ADVENTURE_SCENE,
            text=scene.content,
            source=scene.source_module,
            title=scene.title,
            metadata={
                "category": scene.category.value,
                "chapter": scene.chapter,
                "location": scene.location,
                "has_read_aloud": bool(scene.read_aloud),
                "monsters": scene.monsters_mentioned,
                "npcs": scene.npcs_mentioned,
            },
        )


@dataclass
class SearchResult:
    """A search result from the vector store.

    Attributes:
        content: The indexed content.
        score: Similarity score (higher is better).
        rank: Position in results (1-indexed).
    """

    content: IndexedContent
    score: float
    rank: int


# =============================================================================
# Embedding Provider
# =============================================================================


class EmbeddingProvider:
    """Generate text embeddings using OpenAI API (or compatible).

    Supports OpenAI and OpenRouter for embedding generation.
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        use_openrouter: bool = False,
    ) -> None:
        """Initialize the embedding provider.

        Args:
            model: Embedding model name.
            use_openrouter: Whether to use OpenRouter API.
        """
        self.model = model
        self.use_openrouter = use_openrouter
        self._client = None
        self._dimension: int | None = None

        logger.info(
            "EmbeddingProvider initialized",
            model=model,
            use_openrouter=use_openrouter,
        )

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                settings = get_settings()

                if self.use_openrouter:
                    api_key = settings.ai.openrouter_api_key  # Fixed: use openrouter_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
                else:
                    api_key = settings.ai.openai_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(api_key=api_key)

            except ImportError as exc:
                raise EmbeddingError(
                    "openai package not installed. Install with: pip install openai"
                ) from exc

        return self._client

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for the model."""
        if self._dimension is None:
            # Known dimensions for common models
            dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dimension = dimensions.get(self.model, 1536)
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            from openai import APIConnectionError, APIStatusError, RateLimitError

            client = self._get_client()

            # Clean texts (remove empty, truncate long ones)
            clean_texts = []
            for text in texts:
                text = text.strip()
                if text:
                    # Truncate to ~8000 tokens (rough estimate)
                    if len(text) > 30000:
                        text = text[:30000]
                    clean_texts.append(text)

            if not clean_texts:
                return []

            response = client.embeddings.create(
                model=self.model,
                input=clean_texts,
            )

            embeddings = [item.embedding for item in response.data]

            logger.debug(
                "Generated embeddings",
                count=len(embeddings),
                model=self.model,
            )

            return embeddings

        except RateLimitError as exc:
            raise EmbeddingError(
                f"Embedding rate limit exceeded: {exc}",
                model_name=self.model,
            ) from exc
        except APIConnectionError as exc:
            raise EmbeddingError(
                f"Failed to connect for embeddings: {exc}",
                model_name=self.model,
            ) from exc
        except APIStatusError as exc:
            raise EmbeddingError(
                f"Embedding API error: {exc}",
                model_name=self.model,
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed: {exc}",
                model_name=self.model,
            ) from exc


# =============================================================================
# Vector Store Interface
# =============================================================================


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, contents: list[IndexedContent]) -> None:
        """Add content to the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar content."""
        ...

    @abstractmethod
    def delete(self, content_ids: list[str]) -> None:
        """Delete content by ID."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist store to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load store from disk."""
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of items in the store."""
        ...


# =============================================================================
# FAISS Vector Store
# =============================================================================


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation.

    Uses Facebook AI Similarity Search for fast similarity search.
    Supports metadata filtering through post-search filtering.
    """

    def __init__(self, dimension: int = 1536) -> None:
        """Initialize the FAISS store.

        Args:
            dimension: Embedding dimension.
        """
        self._dimension = dimension
        self._index: Any = None
        self._contents: list[IndexedContent] = []

        logger.info("FAISSVectorStore initialized", dimension=dimension)

    def _ensure_index(self) -> None:
        """Ensure FAISS index is created."""
        if self._index is None:
            try:
                import faiss

                # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
                self._index = faiss.IndexFlatIP(self._dimension)

            except ImportError as exc:
                raise VectorStoreError(
                    "faiss-cpu not installed. Install with: pip install faiss-cpu"
                ) from exc

    def add(self, contents: list[IndexedContent]) -> None:
        """Add content to the FAISS index.

        Args:
            contents: List of content to add.

        Raises:
            VectorStoreError: If adding fails.
        """
        if not contents:
            return

        self._ensure_index()

        try:
            import faiss
            import numpy as np

            # Filter to contents with embeddings
            valid_contents = [c for c in contents if c.embedding]

            if not valid_contents:
                logger.warning("No contents with embeddings to add")
                return

            # Build embedding array
            embeddings: npt.NDArray[np.float32] = np.array(
                [c.embedding for c in valid_contents],
                dtype=np.float32,
            )

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to index
            self._index.add(embeddings)
            self._contents.extend(valid_contents)

            logger.info(f"Added {len(valid_contents)} items to FAISS index")

        except Exception as exc:
            raise VectorStoreError(f"Failed to add to FAISS: {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar content.

        Args:
            query_embedding: Query vector.
            k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects.
        """
        self._ensure_index()

        if self._index.ntotal == 0:
            return []

        try:
            import faiss
            import numpy as np

            # Prepare query
            query: npt.NDArray[np.float32] = np.array(
                [query_embedding], dtype=np.float32
            )
            faiss.normalize_L2(query)

            # Search with extra results for filtering
            search_k = k * 3 if filters else k
            search_k = min(search_k, len(self._contents))

            scores, indices = self._index.search(query, search_k)

            # Build results with filtering
            results: list[SearchResult] = []
            rank = 1

            for score, idx in zip(scores[0], indices[0], strict=True):
                if idx < 0 or idx >= len(self._contents):
                    continue

                content = self._contents[idx]

                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key == "content_type":
                            if content.content_type != value:
                                match = False
                                break
                        elif key == "source":
                            if content.source != value:
                                match = False
                                break
                        elif key in content.metadata:
                            if content.metadata[key] != value:
                                match = False
                                break
                    if not match:
                        continue

                results.append(SearchResult(
                    content=content,
                    score=float(score),
                    rank=rank,
                ))
                rank += 1

                if len(results) >= k:
                    break

            return results

        except Exception as exc:
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

    def delete(self, content_ids: list[str]) -> None:
        """Delete content by ID.

        Note: FAISS doesn't support deletion well, so we rebuild the index.

        Args:
            content_ids: IDs to delete.
        """
        if not content_ids:
            return

        id_set = set(content_ids)
        remaining = [c for c in self._contents if c.content_id not in id_set]

        if len(remaining) == len(self._contents):
            return  # Nothing to delete

        # Rebuild index
        self._contents = []
        self._index = None
        self._ensure_index()

        if remaining:
            self.add(remaining)

        logger.info(f"Deleted {len(content_ids)} items from FAISS index")

    def save(self, path: Path) -> None:
        """Save the index to disk.

        Args:
            path: Directory to save to.
        """
        self._ensure_index()

        try:
            import faiss

            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self._index, str(path / "index.faiss"))

            # Save contents as JSON
            contents_data = []
            for content in self._contents:
                contents_data.append({
                    "content_id": content.content_id,
                    "content_type": content.content_type.value,
                    "text": content.text,
                    "source": content.source,
                    "title": content.title,
                    "embedding": content.embedding,
                    "metadata": content.metadata,
                })

            with open(path / "contents.json", "w", encoding="utf-8") as f:
                json.dump(contents_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved FAISS index to {path}")

        except Exception as exc:
            raise VectorStoreError(f"Failed to save FAISS index: {exc}") from exc

    def load(self, path: Path) -> None:
        """Load the index from disk.

        Args:
            path: Directory to load from.
        """
        try:
            import faiss

            path = Path(path)

            # Load FAISS index
            self._index = faiss.read_index(str(path / "index.faiss"))

            # Load contents
            with open(path / "contents.json", encoding="utf-8") as f:
                contents_data = json.load(f)

            self._contents = []
            for data in contents_data:
                self._contents.append(IndexedContent(
                    content_id=data["content_id"],
                    content_type=ContentType(data["content_type"]),
                    text=data["text"],
                    source=data["source"],
                    title=data.get("title", ""),
                    embedding=data.get("embedding", []),
                    metadata=data.get("metadata", {}),
                ))

            logger.info(f"Loaded FAISS index from {path} ({len(self._contents)} items)")

        except Exception as exc:
            raise VectorStoreError(f"Failed to load FAISS index: {exc}") from exc

    @property
    def count(self) -> int:
        """Number of items in the store."""
        return len(self._contents)


# =============================================================================
# RAG Store (High-Level Interface)
# =============================================================================


class RAGStore:
    """High-level RAG store for D&D content.

    Provides a unified interface for indexing and retrieving D&D content
    with automatic embedding generation and metadata handling.
    """

    def __init__(
        self,
        *,
        backend: Literal["faiss", "chromadb"] = "faiss",
        embedding_model: str = "text-embedding-3-small",
        use_openrouter: bool = False,
    ) -> None:
        """Initialize the RAG store.

        Args:
            backend: Vector store backend to use.
            embedding_model: Embedding model name.
            use_openrouter: Whether to use OpenRouter for embeddings.
        """
        self.embedding_provider = EmbeddingProvider(
            model=embedding_model,
            use_openrouter=use_openrouter,
        )

        if backend == "faiss":
            self.vector_store: BaseVectorStore = FAISSVectorStore(
                dimension=self.embedding_provider.dimension
            )
        else:
            raise VectorStoreError(f"Unsupported backend: {backend}")

        logger.info(
            "RAGStore initialized",
            backend=backend,
            embedding_model=embedding_model,
        )

    def index_documents(self, documents: list[Document]) -> int:
        """Index rule documents.

        Args:
            documents: Documents to index.

        Returns:
            Number of documents indexed.
        """
        if not documents:
            return 0

        # Convert to IndexedContent
        contents = [IndexedContent.from_document(doc) for doc in documents]

        # Generate embeddings
        texts = [c.text for c in contents]
        embeddings = self.embedding_provider.embed_texts(texts)

        # Attach embeddings
        for content, embedding in zip(contents, embeddings, strict=True):
            content.embedding = embedding

        # Add to store
        self.vector_store.add(contents)

        logger.info(f"Indexed {len(documents)} rule documents")
        return len(documents)

    def index_scenes(self, scenes: list[ModuleScene]) -> int:
        """Index adventure scenes.

        Args:
            scenes: Scenes to index.

        Returns:
            Number of scenes indexed.
        """
        if not scenes:
            return 0

        # Convert to IndexedContent
        contents = [IndexedContent.from_scene(scene) for scene in scenes]

        # Generate embeddings
        texts = [c.text for c in contents]
        embeddings = self.embedding_provider.embed_texts(texts)

        # Attach embeddings
        for content, embedding in zip(contents, embeddings, strict=True):
            content.embedding = embedding

        # Add to store
        self.vector_store.add(contents)

        logger.info(f"Indexed {len(scenes)} adventure scenes")
        return len(scenes)

    def retrieve_relevant_context(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        content_type: ContentType | None = None,
        source: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant context for a query.

        Args:
            query: Search query.
            k: Number of results to return.
            filters: Additional metadata filters.
            content_type: Filter by content type.
            source: Filter by source name.

        Returns:
            List of SearchResult objects.

        Example:
            >>> results = store.retrieve_relevant_context(
            ...     "How does fireball work?",
            ...     k=3,
            ...     content_type=ContentType.RULE_DOCUMENT,
            ... )
            >>> for result in results:
            ...     print(f"{result.content.title}: {result.score:.2f}")
        """
        # Build filters
        combined_filters = filters.copy() if filters else {}
        if content_type:
            combined_filters["content_type"] = content_type
        if source:
            combined_filters["source"] = source

        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query)

        # Search
        results = self.vector_store.search(
            query_embedding,
            k=k,
            filters=combined_filters if combined_filters else None,
        )

        logger.info(
            "Retrieved context",
            query=query[:50],
            results=len(results),
            filters=combined_filters,
        )

        return results

    def retrieve_rules(
        self,
        query: str,
        *,
        k: int = 3,
        doc_type: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant rules for a query.

        Args:
            query: Search query.
            k: Number of results.
            doc_type: Filter by document type (spell, feat, rule, etc.).

        Returns:
            List of SearchResult objects.
        """
        filters = {}
        if doc_type:
            filters["doc_type"] = doc_type

        return self.retrieve_relevant_context(
            query,
            k=k,
            filters=filters,
            content_type=ContentType.RULE_DOCUMENT,
        )

    def retrieve_scenes(
        self,
        query: str,
        *,
        k: int = 3,
        module: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant adventure scenes.

        Args:
            query: Search query.
            k: Number of results.
            module: Filter by adventure module name.
            category: Filter by scene category.

        Returns:
            List of SearchResult objects.
        """
        filters = {}
        if category:
            filters["category"] = category

        return self.retrieve_relevant_context(
            query,
            k=k,
            filters=filters,
            content_type=ContentType.ADVENTURE_SCENE,
            source=module,
        )

    def save(self, path: Path | str) -> None:
        """Save the store to disk.

        Args:
            path: Directory to save to.
        """
        self.vector_store.save(Path(path))

    def load(self, path: Path | str) -> None:
        """Load the store from disk.

        Args:
            path: Directory to load from.
        """
        self.vector_store.load(Path(path))

    @property
    def document_count(self) -> int:
        """Total number of indexed documents."""
        return self.vector_store.count
    
    def retrieve_with_game_context(
        self,
        query: str,
        *,
        character: Any = None,  # ActorEntity type (avoid circular import)
        game_state: Any = None,  # GameState type
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Smart retrieval that filters by character capabilities and game state.
        
        This solves the "context confusion" problem:
        - "Can I cast Shield?" → Filters by character class (Wizard = spell, Fighter = item)
        - "Can I use Sneak Attack?" → Filters by class features
        - "Can I run up this wall?" → Considers character level for Monk features
        
        Args:
            query: Search query.
            character: Optional character to filter by capabilities.
            game_state: Optional game state for additional context.
            k: Number of results.
            filters: Base filters (will be enhanced with game context).
            
        Returns:
            List of SearchResult objects filtered by game context.
            
        Example:
            >>> # Wizard asking about Shield
            >>> results = store.retrieve_with_game_context(
            ...     "Can I cast Shield?",
            ...     character=wizard_char,
            ... )
            >>> # Returns the Shield SPELL, not the item
        """
        enhanced_filters = filters.copy() if filters else {}
        query_lower = query.lower()
        
        if character:
            # Extract character info
            char_classes = []
            char_level = 1
            has_spellcasting = False
            
            # Try to get character class and level
            try:
                if hasattr(character, 'class_levels'):
                    for cls_lvl in character.class_levels:
                        char_classes.append(cls_lvl.class_name.lower())
                        char_level = max(char_level, cls_lvl.level)
                elif hasattr(character, 'primary_class'):
                    char_classes.append(character.primary_class.lower())
                
                if hasattr(character, 'level'):
                    char_level = character.level
                    
                if hasattr(character, 'spellbook') and character.spellbook:
                    has_spellcasting = True
            except Exception:
                pass  # Fallback to no filtering if character structure different
            
            # Smart filtering based on query type
            if has_spellcasting and any(word in query_lower for word in ["cast", "spell", "magic"]):
                # Character can cast spells, prioritize spell results
                enhanced_filters["doc_subtype"] = "spell"
                logger.debug("Filtering for spells based on character spellcasting")
            
            elif "feat" in query_lower and char_classes:
                # Filter by class-appropriate feats
                enhanced_filters["doc_subtype"] = "feat"
            
            elif any(word in query_lower for word in ["feature", "ability", "can i"]):
                # Query about character abilities - filter by class
                if char_classes and "classes" not in enhanced_filters:
                    # Don't override if already specified
                    # This will be used in post-filtering
                    enhanced_filters["_char_classes"] = char_classes
                    enhanced_filters["_char_level"] = char_level
        
        # Perform retrieval
        results = self.retrieve_relevant_context(
            query,
            k=k * 2,  # Get more results for filtering
            filters=enhanced_filters,
        )
        
        # Post-filter by character class and level if needed
        if "_char_classes" in enhanced_filters:
            char_classes = enhanced_filters["_char_classes"]
            char_level = enhanced_filters.get("_char_level", 1)
            
            filtered_results = []
            for result in results:
                # Check if content is relevant to character's classes
                content_meta = result.content.metadata
                
                # Extract enhanced metadata if available
                if "classes" in content_meta:
                    content_classes = content_meta.get("classes", [])
                    # Allow if no class restriction or matches character class
                    if not content_classes or any(c in content_classes for c in char_classes):
                        # Check level requirement
                        level_req = content_meta.get("level_requirement")
                        if level_req is None or char_level >= level_req:
                            filtered_results.append(result)
                else:
                    # No class filtering in metadata, include it
                    filtered_results.append(result)
            
            results = filtered_results[:k]
        else:
            results = results[:k]
        
        return results


# =============================================================================
# HyDE (Hypothetical Document Embeddings) Retriever
# =============================================================================


class HyDERetriever:
    """Hypothetical Document Embeddings for semantic rule lookup.
    
    Instead of embedding the query directly, HyDE generates a hypothetical
    document that would answer the query, embeds that, and searches.
    
    This solves the "vocabulary mismatch" problem where players describe
    outcomes ("I want to jump on the ogre") rather than rule names
    ("Falling onto a Creature - Tasha's Cauldron").
    
    Example:
        Query: "Can I run up this wall?"
        Hypothetical: "When a monk reaches 9th level, they gain Unarmored
                       Movement which allows them to run up vertical surfaces..."
        Search: Finds Monk class features, not just general movement rules.
    """

    def __init__(
        self,
        rag_store: RAGStore | Any,  # Accept RAGStore or ChromaStore
        *,
        model: str = "google/gemini-2.0-flash-001",
        use_openrouter: bool = True,
    ) -> None:
        """Initialize the HyDE retriever.
        
        Args:
            rag_store: The underlying RAG store to search (RAGStore or ChromaStore).
            model: LLM model for hypothesis generation.
            use_openrouter: Whether to use OpenRouter API.
        """
        self.rag_store = rag_store
        self.model = model
        self.use_openrouter = use_openrouter
        self._client: Any = None
        
        # Detect store type
        self._is_chroma_store = hasattr(rag_store, 'search') and not hasattr(rag_store, 'retrieve_relevant_context')
        
        logger.info(
            "HyDERetriever initialized",
            model=model,
            store_type="ChromaStore" if self._is_chroma_store else "RAGStore",
        )
    
    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                
                settings = get_settings()
                
                if self.use_openrouter:
                    api_key = settings.ai.openrouter_api_key  # Fixed: use openrouter_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers={
                            "HTTP-Referer": "https://github.com/dnd-campaign-manager",
                            "X-Title": "D&D Campaign Manager",
                        },
                    )
                else:
                    api_key = settings.ai.openai_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(api_key=api_key)
                    
            except ImportError as exc:
                raise EmbeddingError(
                    "openai package not installed",
                    model_name=self.model,
                ) from exc
        
        return self._client
    
    def generate_hypothetical_rule(self, query: str) -> str:
        """Generate a hypothetical D&D rule that would answer this query.
        
        Args:
            query: The player's question or action description.
            
        Returns:
            A hypothetical rule text that answers the query.
            
        Example:
            >>> query = "Can I jump on the ogre from above?"
            >>> hyp = generate_hypothetical_rule(query)
            >>> print(hyp)
            "When a creature falls onto another creature, both take falling
             damage. The creature being fallen upon can make a DC 15 Dexterity
             saving throw to avoid the damage..."
        """
        system_prompt = """You are a D&D 5E rules expert. Given a player's question or action,
write a hypothetical D&D rule excerpt that would answer their question.

Write as if you're quoting from the Player's Handbook or Dungeon Master's Guide.
Be specific and use D&D terminology (saving throws, advantage, actions, etc.).
Keep it concise (2-3 sentences)."""

        user_prompt = f"""Player question/action: "{query}"

Write a hypothetical D&D 5E rule that would answer this:"""

        try:
            from openai import APIConnectionError, APIStatusError, RateLimitError
            
            client = self._get_client()
            
            # Log to both structured logger and console for visibility
            print(f"\n🔍 HyDE: Generating hypothetical with model: {self.model}")
            print(f"   Query: {query[:80]}{'...' if len(query) > 80 else ''}")
            
            logger.info(
                "Generating HyDE hypothetical",
                model=self.model,
                query_preview=query[:50],
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more focused rules
                max_tokens=200,
            )
            
            hypothetical = response.choices[0].message.content
            
            # Handle empty or None responses
            if not hypothetical or hypothetical.strip() == "":
                print(f"   ⚠️  WARNING: LLM returned empty hypothetical, using original query")
                logger.warning(
                    "LLM returned empty hypothetical, using original query",
                    model=self.model,
                    response_object=str(response)[:200],
                )
                hypothetical = query
            else:
                print(f"   ✅ Generated: {hypothetical[:80]}{'...' if len(hypothetical) > 80 else ''}\n")
                logger.info(
                    "Generated hypothetical rule successfully",
                    model=self.model,
                    query=query[:50],
                    hypothetical=hypothetical[:100],
                )
            
            return hypothetical
            
        except RateLimitError as exc:
            logger.warning(
                "Rate limited during HyDE generation, falling back to query",
                model=self.model,
                error=str(exc),
            )
            return query
        except (APIConnectionError, APIStatusError) as exc:
            logger.warning(
                f"API error during HyDE generation: {exc}, falling back to query",
                model=self.model,
                error_type=type(exc).__name__,
            )
            return query
        except Exception as exc:
            logger.exception(
                "Failed to generate hypothetical rule",
                model=self.model,
                error_type=type(exc).__name__,
            )
            return query
    
    def retrieve_with_hyde(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        fallback_to_direct: bool = True,
    ) -> list[SearchResult]:
        """Retrieve documents using HyDE approach.
        
        Args:
            query: The player's question or action.
            k: Number of results to return.
            filters: Optional metadata filters.
            fallback_to_direct: If True, also try direct query and merge results.
            
        Returns:
            List of SearchResult objects.
        """
        # Generate hypothetical document
        hypothetical = self.generate_hypothetical_rule(query)
        
        # Search with hypothetical (adapt to store type)
        if self._is_chroma_store:
            # Use ChromaStore.search()
            from dnd_manager.ingestion.universal_loader import DocumentType
            doc_type = DocumentType.SOURCEBOOK if filters and filters.get("content_type") == "rules" else None
            
            hyde_docs = self.rag_store.search(hypothetical, n_results=k, doc_type=doc_type)
            
            # Convert to SearchResult format
            hyde_results = [
                SearchResult(
                    content=IndexedContent(
                        content_id=doc.chunk_id,
                        content_type=ContentType.RULE_DOCUMENT if doc.doc_type.value == "sourcebook" else ContentType.ADVENTURE_SCENE,
                        text=doc.content,
                        source=doc.source,
                        title=doc.metadata.get("title", ""),
                        metadata=doc.metadata,
                    ),
                    score=1.0,  # ChromaDB doesn't return scores in this format
                    rank=i + 1,  # 1-indexed rank
                )
                for i, doc in enumerate(hyde_docs)
            ]
        else:
            # Use RAGStore.retrieve_relevant_context()
            hyde_results = self.rag_store.retrieve_relevant_context(
                hypothetical,
                k=k,
                filters=filters,
            )
        
        # Optionally also do direct search and merge
        if fallback_to_direct and hypothetical != query:
            if self._is_chroma_store:
                direct_docs = self.rag_store.search(query, n_results=k // 2, doc_type=doc_type)
                direct_results = [
                    SearchResult(
                        content=IndexedContent(
                            content_id=doc.chunk_id,
                            content_type=ContentType.RULE_DOCUMENT if doc.doc_type.value == "sourcebook" else ContentType.ADVENTURE_SCENE,
                            text=doc.content,
                            source=doc.source,
                            title=doc.metadata.get("title", ""),
                            metadata=doc.metadata,
                        ),
                        score=1.0,
                        rank=i + 1,  # 1-indexed rank
                    )
                    for i, doc in enumerate(direct_docs)
                ]
            else:
                direct_results = self.rag_store.retrieve_relevant_context(
                    query,
                    k=k // 2,  # Get fewer direct results
                    filters=filters,
                )
            
            # Merge and deduplicate by content_id
            seen_ids = set()
            merged = []
            
            # Prioritize HyDE results (better semantic match)
            for result in hyde_results:
                if result.content.content_id not in seen_ids:
                    merged.append(result)
                    seen_ids.add(result.content.content_id)
            
            # Add unique direct results
            for result in direct_results:
                if result.content.content_id not in seen_ids:
                    merged.append(result)
                    seen_ids.add(result.content.content_id)
            
            # Re-rank by score and limit to k
            merged.sort(key=lambda r: r.score, reverse=True)
            return merged[:k]
        
        return hyde_results


def extract_enhanced_metadata(content: str, title: str) -> EnhancedMetadata:
    """Extract enhanced metadata from content text.
    
    Uses pattern matching and heuristics to identify:
    - Document subtype (spell, feat, class feature, etc.)
    - Relevant classes
    - Level requirements
    - Prerequisites
    
    Args:
        content: The text content.
        title: The content title.
        
    Returns:
        EnhancedMetadata with extracted information.
    """
    import re
    
    meta = EnhancedMetadata()
    content_lower = content.lower()
    title_lower = title.lower()
    
    # Detect document subtype
    if any(word in title_lower for word in ["spell", "cantrip"]):
        meta.doc_subtype = "spell"
    elif any(word in title_lower for word in ["feat"]):
        meta.doc_subtype = "feat"
    elif any(word in content_lower for word in ["class feature", "class ability"]):
        meta.doc_subtype = "class_feature"
    elif any(word in title_lower for word in ["weapon", "armor", "item", "equipment"]):
        meta.doc_subtype = "item"
    else:
        meta.doc_subtype = "general_rule"
    
    # Extract class associations
    all_classes = [
        "barbarian", "bard", "cleric", "druid", "fighter", "monk",
        "paladin", "ranger", "rogue", "sorcerer", "warlock", "wizard"
    ]
    
    for cls in all_classes:
        # Look for class mentions in content or title
        if cls in content_lower or cls in title_lower:
            meta.classes.append(cls)
    
    # Extract level requirements
    level_patterns = [
        r"(\d+)(?:st|nd|rd|th)[\s-]level",
        r"level (\d+)",
        r"at (\d+)(?:st|nd|rd|th) level",
        r"when you reach (\d+)(?:st|nd|rd|th) level",
    ]
    
    for pattern in level_patterns:
        match = re.search(pattern, content_lower)
        if match:
            level = int(match.group(1))
            if meta.level_requirement is None or level < meta.level_requirement:
                meta.level_requirement = level
            break
    
    # Extract spell level for spells
    if meta.doc_subtype == "spell":
        spell_level_match = re.search(r"(\d)(?:st|nd|rd|th)?[\s-]level\s+(?:spell|conjuration|evocation|abjuration|divination|enchantment|illusion|necromancy|transmutation)", content_lower)
        if spell_level_match:
            spell_level = int(spell_level_match.group(1))
            meta.level_requirement = spell_level
        elif "cantrip" in content_lower or "cantrip" in title_lower:
            meta.level_requirement = 0
    
    # Extract prerequisites
    prereq_match = re.search(r"prerequisite[s]?:([^\n\.]+)", content_lower, re.IGNORECASE)
    if prereq_match:
        prereq_text = prereq_match.group(1).strip()
        meta.prerequisites.append(prereq_text)
    
    # Add tags based on content
    tag_keywords = {
        "combat": ["attack", "damage", "hit points", "armor class", "weapon"],
        "utility": ["skill", "tool", "proficiency"],
        "social": ["persuasion", "deception", "intimidation", "performance"],
        "exploration": ["perception", "investigation", "survival"],
        "magic": ["spell", "cantrip", "magic", "arcane", "divine"],
        "healing": ["heal", "restore", "hit points"],
        "movement": ["speed", "fly", "swim", "climb", "jump"],
    }
    
    for tag, keywords in tag_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            meta.tags.append(tag)
    
    return meta


__all__ = [
    "ContentType",
    "EnhancedMetadata",
    "IndexedContent",
    "SearchResult",
    "EmbeddingProvider",
    "BaseVectorStore",
    "FAISSVectorStore",
    "RAGStore",
    "HyDERetriever",
    "extract_enhanced_metadata",
]
