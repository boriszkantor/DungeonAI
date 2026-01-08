"""RAG (Retrieval-Augmented Generation) pipeline.

This module provides the core RAG functionality for querying
D&D source materials and campaign content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dnd_manager.core.exceptions import EmbeddingError, VectorStoreError
from dnd_manager.core.logging import get_logger


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A single search result from the vector store.

    Attributes:
        content: The text content of the result.
        score: Similarity score (higher is better).
        metadata: Additional metadata about the result.
        source_file: Path to the source document.
    """

    content: str
    score: float
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector store implementations.

    This class defines the interface for vector stores used in
    the RAG pipeline, supporting both FAISS and ChromaDB backends.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of document texts.
            embeddings: List of embedding vectors.
            metadata: Optional list of metadata dicts.

        Raises:
            VectorStoreError: If adding documents fails.
        """
        raise NotImplementedError("Subclasses must implement add_documents")

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects.

        Raises:
            VectorStoreError: If search fails.
        """
        raise NotImplementedError("Subclasses must implement search")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the vector store to disk.

        Args:
            path: Path to save the store.

        Raises:
            VectorStoreError: If saving fails.
        """
        raise NotImplementedError("Subclasses must implement save")

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the vector store from disk.

        Args:
            path: Path to load the store from.

        Raises:
            VectorStoreError: If loading fails.
        """
        raise NotImplementedError("Subclasses must implement load")


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation.

    This class provides a fast, efficient vector store using
    Facebook AI Similarity Search (FAISS).
    """

    def __init__(self, dimension: int = 1536) -> None:
        """Initialize the FAISS vector store.

        Args:
            dimension: Embedding dimension size.
        """
        self.dimension = dimension
        self._index: Any | None = None
        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        logger.info("FAISSVectorStore initialized", dimension=dimension)

    def _ensure_index(self) -> None:
        """Ensure the FAISS index is initialized."""
        if self._index is None:
            try:
                import faiss

                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim
            except ImportError as exc:
                raise VectorStoreError(
                    "faiss-cpu not installed. Install with: pip install faiss-cpu"
                ) from exc

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the FAISS index.

        Args:
            documents: List of document texts.
            embeddings: List of embedding vectors.
            metadata: Optional list of metadata dicts.

        Raises:
            VectorStoreError: If adding documents fails.
        """
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                f"Document count ({len(documents)}) doesn't match "
                f"embedding count ({len(embeddings)})"
            )

        self._ensure_index()

        try:
            import numpy as np

            # Normalize embeddings for cosine similarity
            embeddings_array: npt.NDArray[np.float32] = np.array(
                embeddings, dtype=np.float32
            )
            faiss_module = __import__("faiss")
            faiss_module.normalize_L2(embeddings_array)

            self._index.add(embeddings_array)  # type: ignore[union-attr]
            self._documents.extend(documents)
            self._metadata.extend(metadata or [{} for _ in documents])

            logger.info("Documents added to FAISS index", count=len(documents))

        except Exception as exc:
            raise VectorStoreError(f"Failed to add documents: {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects.

        Raises:
            VectorStoreError: If search fails.
        """
        self._ensure_index()

        if self._index is None or self._index.ntotal == 0:  # type: ignore[union-attr]
            return []

        try:
            import numpy as np

            # Normalize query for cosine similarity
            query_array: npt.NDArray[np.float32] = np.array(
                [query_embedding], dtype=np.float32
            )
            faiss_module = __import__("faiss")
            faiss_module.normalize_L2(query_array)

            # Search
            scores, indices = self._index.search(query_array, min(top_k, len(self._documents)))  # type: ignore[union-attr]

            results = []
            for score, idx in zip(scores[0], indices[0], strict=True):
                if idx >= 0:  # FAISS returns -1 for missing results
                    meta = self._metadata[idx]
                    results.append(
                        SearchResult(
                            content=self._documents[idx],
                            score=float(score),
                            source_file=meta.get("source_file", "unknown"),
                            metadata=meta,
                        )
                    )

            return results

        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

    def save(self, path: Path) -> None:
        """Save the FAISS index to disk.

        Args:
            path: Path to save the index.

        Raises:
            VectorStoreError: If saving fails.
        """
        self._ensure_index()

        try:
            import json

            import faiss

            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self._index, str(path / "index.faiss"))  # type: ignore[arg-type]

            # Save documents and metadata
            with open(path / "documents.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"documents": self._documents, "metadata": self._metadata},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info("FAISS index saved", path=str(path))

        except Exception as exc:
            raise VectorStoreError(f"Failed to save index: {exc}") from exc

    def load(self, path: Path) -> None:
        """Load the FAISS index from disk.

        Args:
            path: Path to load the index from.

        Raises:
            VectorStoreError: If loading fails.
        """
        try:
            import json

            import faiss

            path = Path(path)

            # Load FAISS index
            self._index = faiss.read_index(str(path / "index.faiss"))

            # Load documents and metadata
            with open(path / "documents.json", encoding="utf-8") as f:
                data = json.load(f)
                self._documents = data["documents"]
                self._metadata = data["metadata"]

            logger.info("FAISS index loaded", path=str(path))

        except Exception as exc:
            raise VectorStoreError(f"Failed to load index: {exc}") from exc


class RAGPipeline:
    """Main RAG orchestration class.

    This class coordinates document ingestion, embedding generation,
    and retrieval for the D&D Campaign Manager.

    Attributes:
        vector_store: The vector store backend.
        embedding_model: Name of the embedding model to use.
    """

    def __init__(
        self,
        *,
        backend: Literal["faiss", "chromadb"] = "faiss",
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            backend: Vector store backend to use.
            embedding_model: Name of the embedding model.
            dimension: Embedding dimension size.
        """
        self.embedding_model = embedding_model
        self.dimension = dimension

        if backend == "faiss":
            self.vector_store: VectorStore = FAISSVectorStore(dimension=dimension)
        else:
            raise VectorStoreError(f"Unsupported backend: {backend}")

        logger.info(
            "RAGPipeline initialized",
            backend=backend,
            embedding_model=embedding_model,
        )

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            import openai

            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding

        except ImportError as exc:
            raise EmbeddingError(
                "openai not installed. Install with: pip install openai",
                model_name=self.embedding_model,
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed: {exc}",
                model_name=self.embedding_model,
            ) from exc

    async def add_texts(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add texts to the vector store.

        Args:
            texts: List of texts to add.
            metadata: Optional metadata for each text.

        Raises:
            EmbeddingError: If embedding generation fails.
            VectorStoreError: If adding to store fails.
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)

        self.vector_store.add_documents(texts, embeddings, metadata)
        logger.info("Texts added to RAG pipeline", count=len(texts))

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Query the RAG pipeline for relevant documents.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects.

        Raises:
            EmbeddingError: If embedding generation fails.
            VectorStoreError: If search fails.
        """
        query_embedding = await self.generate_embedding(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.info("RAG query completed", query=query[:50], results=len(results))
        return results


__all__ = [
    "SearchResult",
    "VectorStore",
    "FAISSVectorStore",
    "RAGPipeline",
]
