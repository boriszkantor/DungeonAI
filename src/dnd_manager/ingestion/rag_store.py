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
                    api_key = settings.ai.openai_api_key
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


__all__ = [
    "ContentType",
    "IndexedContent",
    "SearchResult",
    "EmbeddingProvider",
    "BaseVectorStore",
    "FAISSVectorStore",
    "RAGStore",
]
