"""Entity Resolution (Canonicalization) for Knowledge Graph.

This module implements Phase 3 of the knowledge graph pipeline:
Entity Resolution - unifying different mentions of the same entity
into a single canonical node.

The Problem:
- Chunk A: "The Wizard class..."
- Chunk B: "Wizards gain..."
- Chunk C: "A wizard's spellbook..."

Without resolution, these become 3 disconnected nodes.

The Solution:
1. Exact match against canonical registry
2. Fuzzy string matching (Levenshtein distance)
3. Semantic similarity matching (embeddings)
4. Register new canonical entities when no match found

NEURO-SYMBOLIC PRINCIPLE:
Entity resolution ensures graph consistency by mapping all variations
of an entity name to a single canonical representation, enabling
proper relationship traversal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.ontology import OntologySchema, SemanticTriple


logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CanonicalEntity:
    """A canonical entity in the knowledge graph.

    Attributes:
        canonical_name: The standardized name.
        entity_type: The entity type from the ontology.
        aliases: All known aliases for this entity.
        chunk_ids: Source chunks where this entity appears.
        attributes: Additional entity attributes.
        embedding: Optional embedding vector for semantic matching.
    """

    canonical_name: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def add_alias(self, alias: str) -> None:
        """Add an alias for this entity."""
        self.aliases.add(alias.lower())

    def add_chunk_id(self, chunk_id: str) -> None:
        """Add a source chunk ID."""
        self.chunk_ids.add(chunk_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": list(self.aliases),
            "chunk_ids": list(self.chunk_ids),
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalEntity":
        """Create from dictionary."""
        return cls(
            canonical_name=data["canonical_name"],
            entity_type=data["entity_type"],
            aliases=set(data.get("aliases", [])),
            chunk_ids=set(data.get("chunk_ids", [])),
            attributes=data.get("attributes", {}),
        )


@dataclass
class ResolutionResult:
    """Result of entity resolution.

    Attributes:
        original_name: The original entity name.
        canonical_name: The resolved canonical name.
        entity_type: The entity type.
        match_type: How the match was found (exact, fuzzy, semantic, new).
        confidence: Match confidence score (0.0-1.0).
        entity: The CanonicalEntity if found.
    """

    original_name: str
    canonical_name: str
    entity_type: str
    match_type: str  # "exact", "fuzzy", "semantic", "new"
    confidence: float
    entity: CanonicalEntity | None = None


# =============================================================================
# Entity Resolver
# =============================================================================


class EntityResolver:
    """Resolves entity mentions to canonical names.

    This class implements a multi-stage resolution process:
    1. Exact match in canonical registry
    2. Fuzzy string matching using rapidfuzz
    3. Semantic similarity matching using embeddings
    4. New entity registration

    Example:
        >>> resolver = EntityResolver.from_schema(schema)
        >>> result = resolver.resolve("Dex save", "Save")
        >>> print(result.canonical_name)  # "DEX"
        >>> print(result.match_type)  # "exact"
    """

    def __init__(
        self,
        *,
        fuzzy_threshold: float = 0.85,
        semantic_threshold: float = 0.90,
        use_embeddings: bool = False,
    ) -> None:
        """Initialize the entity resolver.

        Args:
            fuzzy_threshold: Minimum ratio for fuzzy matching (0.0-1.0).
            semantic_threshold: Minimum cosine similarity for semantic match.
            use_embeddings: Whether to use embedding-based matching.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.use_embeddings = use_embeddings

        # Registry: entity_type -> canonical_name -> CanonicalEntity
        self._registry: dict[str, dict[str, CanonicalEntity]] = {}

        # Reverse lookup: alias (lowercase) -> (canonical_name, entity_type)
        self._alias_index: dict[str, tuple[str, str]] = {}

        # Embedding provider for semantic matching
        self._embedding_provider: Any = None

        logger.info(
            "EntityResolver initialized",
            fuzzy_threshold=fuzzy_threshold,
            semantic_threshold=semantic_threshold,
            use_embeddings=use_embeddings,
        )

    @classmethod
    def from_schema(
        cls,
        schema: OntologySchema,
        *,
        fuzzy_threshold: float = 0.85,
        semantic_threshold: float = 0.90,
        use_embeddings: bool = False,
    ) -> "EntityResolver":
        """Create resolver pre-populated from ontology schema.

        Args:
            schema: Ontology schema with canonical entities.
            fuzzy_threshold: Minimum ratio for fuzzy matching.
            semantic_threshold: Minimum cosine similarity.
            use_embeddings: Whether to use embeddings.

        Returns:
            EntityResolver populated with schema entities.
        """
        resolver = cls(
            fuzzy_threshold=fuzzy_threshold,
            semantic_threshold=semantic_threshold,
            use_embeddings=use_embeddings,
        )

        # Load canonical entities from schema
        for entity_type, entities in schema.canonical_entities.items():
            for canonical_name, entity_data in entities.items():
                entity = CanonicalEntity(
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    aliases=set(
                        a.lower() for a in entity_data.get("aliases", [])
                    ),
                    attributes=entity_data.get("attributes", {}),
                )

                resolver.register_entity(entity)

        logger.info(
            "EntityResolver loaded from schema",
            entity_types=len(resolver._registry),
            total_entities=sum(
                len(v) for v in resolver._registry.values()
            ),
        )

        return resolver

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching.

        Args:
            name: Raw entity name.

        Returns:
            Normalized name (lowercase, stripped, no extra spaces).
        """
        # Remove common prefixes/articles
        name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)

        # Normalize whitespace
        name = " ".join(name.split())

        return name.lower().strip()

    def _get_fuzzy_ratio(self, name1: str, name2: str) -> float:
        """Calculate fuzzy match ratio between two names.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            Match ratio (0.0-1.0).
        """
        try:
            from rapidfuzz import fuzz

            return fuzz.ratio(name1, name2) / 100.0
        except ImportError:
            # Fallback to simple comparison if rapidfuzz not available
            if name1 == name2:
                return 1.0
            elif name1 in name2 or name2 in name1:
                return 0.8
            else:
                return 0.0

    def _get_semantic_similarity(
        self, name: str, entity: CanonicalEntity
    ) -> float:
        """Calculate semantic similarity using embeddings.

        Args:
            name: Entity name to match.
            entity: Canonical entity to compare against.

        Returns:
            Cosine similarity (0.0-1.0).
        """
        if not self.use_embeddings or entity.embedding is None:
            return 0.0

        try:
            if self._embedding_provider is None:
                from dnd_manager.ingestion.rag_store import EmbeddingProvider

                self._embedding_provider = EmbeddingProvider()

            # Get embedding for the name
            name_embedding = self._embedding_provider.embed_text(name)

            # Calculate cosine similarity
            import numpy as np

            a = np.array(name_embedding)
            b = np.array(entity.embedding)
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            return float(similarity)

        except Exception as exc:
            logger.debug(f"Semantic matching failed: {exc}")
            return 0.0

    def register_entity(self, entity: CanonicalEntity) -> None:
        """Register a canonical entity.

        Args:
            entity: The entity to register.
        """
        entity_type = entity.entity_type

        # Add to type registry
        if entity_type not in self._registry:
            self._registry[entity_type] = {}

        self._registry[entity_type][entity.canonical_name] = entity

        # Index canonical name
        key = self._normalize_name(entity.canonical_name)
        self._alias_index[key] = (entity.canonical_name, entity_type)

        # Index all aliases
        for alias in entity.aliases:
            alias_key = self._normalize_name(alias)
            if alias_key not in self._alias_index:
                self._alias_index[alias_key] = (
                    entity.canonical_name,
                    entity_type,
                )

    def resolve(
        self,
        name: str,
        entity_type: str | None = None,
        *,
        auto_register: bool = True,
    ) -> ResolutionResult:
        """Resolve an entity name to its canonical form.

        Args:
            name: The entity name to resolve.
            entity_type: Optional type hint to narrow search.
            auto_register: If True, register new entities automatically.

        Returns:
            ResolutionResult with canonical name and match info.
        """
        normalized = self._normalize_name(name)

        # Stage 1: Exact match in alias index
        if normalized in self._alias_index:
            canonical_name, found_type = self._alias_index[normalized]

            # Check type constraint
            if entity_type is None or entity_type == found_type:
                entity = self._registry.get(found_type, {}).get(canonical_name)
                return ResolutionResult(
                    original_name=name,
                    canonical_name=canonical_name,
                    entity_type=found_type,
                    match_type="exact",
                    confidence=1.0,
                    entity=entity,
                )

        # Stage 2: Fuzzy matching
        best_fuzzy_match: tuple[str, str, float] | None = None

        types_to_search = (
            [entity_type] if entity_type else list(self._registry.keys())
        )

        for search_type in types_to_search:
            if search_type not in self._registry:
                continue

            for canonical_name, entity in self._registry[search_type].items():
                # Check canonical name
                ratio = self._get_fuzzy_ratio(
                    normalized, self._normalize_name(canonical_name)
                )
                if ratio >= self.fuzzy_threshold:
                    if (
                        best_fuzzy_match is None
                        or ratio > best_fuzzy_match[2]
                    ):
                        best_fuzzy_match = (canonical_name, search_type, ratio)

                # Check aliases
                for alias in entity.aliases:
                    ratio = self._get_fuzzy_ratio(normalized, alias)
                    if ratio >= self.fuzzy_threshold:
                        if (
                            best_fuzzy_match is None
                            or ratio > best_fuzzy_match[2]
                        ):
                            best_fuzzy_match = (
                                canonical_name,
                                search_type,
                                ratio,
                            )

        if best_fuzzy_match:
            canonical_name, found_type, confidence = best_fuzzy_match
            entity = self._registry[found_type][canonical_name]

            # Add as new alias
            entity.add_alias(normalized)
            self._alias_index[normalized] = (canonical_name, found_type)

            return ResolutionResult(
                original_name=name,
                canonical_name=canonical_name,
                entity_type=found_type,
                match_type="fuzzy",
                confidence=confidence,
                entity=entity,
            )

        # Stage 3: Semantic matching (if enabled)
        if self.use_embeddings:
            best_semantic_match: tuple[str, str, float] | None = None

            for search_type in types_to_search:
                if search_type not in self._registry:
                    continue

                for canonical_name, entity in self._registry[search_type].items():
                    similarity = self._get_semantic_similarity(name, entity)
                    if similarity >= self.semantic_threshold:
                        if (
                            best_semantic_match is None
                            or similarity > best_semantic_match[2]
                        ):
                            best_semantic_match = (
                                canonical_name,
                                search_type,
                                similarity,
                            )

            if best_semantic_match:
                canonical_name, found_type, confidence = best_semantic_match
                entity = self._registry[found_type][canonical_name]

                # Add as new alias
                entity.add_alias(normalized)
                self._alias_index[normalized] = (canonical_name, found_type)

                return ResolutionResult(
                    original_name=name,
                    canonical_name=canonical_name,
                    entity_type=found_type,
                    match_type="semantic",
                    confidence=confidence,
                    entity=entity,
                )

        # Stage 4: New entity registration
        canonical_name = name.title().strip()

        if auto_register and entity_type:
            new_entity = CanonicalEntity(
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases={normalized},
            )
            self.register_entity(new_entity)

            logger.debug(
                f"Registered new entity: {canonical_name} ({entity_type})"
            )

            return ResolutionResult(
                original_name=name,
                canonical_name=canonical_name,
                entity_type=entity_type,
                match_type="new",
                confidence=0.5,
                entity=new_entity,
            )

        # No match and no auto-register
        return ResolutionResult(
            original_name=name,
            canonical_name=canonical_name,
            entity_type=entity_type or "Unknown",
            match_type="new",
            confidence=0.5,
            entity=None,
        )

    def resolve_triple(
        self,
        triple: SemanticTriple,
        *,
        auto_register: bool = True,
    ) -> SemanticTriple:
        """Resolve entity names in a triple to canonical forms.

        Args:
            triple: The triple to resolve.
            auto_register: If True, register new entities.

        Returns:
            New SemanticTriple with canonical names.
        """
        # Resolve subject
        subject_result = self.resolve(
            triple.subject,
            triple.subject_type,
            auto_register=auto_register,
        )

        # Resolve object
        object_result = self.resolve(
            triple.object,
            triple.object_type,
            auto_register=auto_register,
        )

        # Track chunk ID in entities
        if subject_result.entity and triple.chunk_id:
            subject_result.entity.add_chunk_id(triple.chunk_id)
        if object_result.entity and triple.chunk_id:
            object_result.entity.add_chunk_id(triple.chunk_id)

        return SemanticTriple(
            subject=subject_result.canonical_name,
            subject_type=subject_result.entity_type,
            predicate=triple.predicate,
            object=object_result.canonical_name,
            object_type=object_result.entity_type,
            metadata=triple.metadata,
            chunk_id=triple.chunk_id,
            confidence=triple.confidence
            * min(subject_result.confidence, object_result.confidence),
        )

    def resolve_triples(
        self,
        triples: list[SemanticTriple],
        *,
        auto_register: bool = True,
    ) -> list[SemanticTriple]:
        """Resolve entity names in multiple triples.

        Args:
            triples: List of triples to resolve.
            auto_register: If True, register new entities.

        Returns:
            List of triples with canonical names.
        """
        resolved = []
        for triple in triples:
            resolved.append(
                self.resolve_triple(triple, auto_register=auto_register)
            )
        return resolved

    def get_entity(
        self, canonical_name: str, entity_type: str
    ) -> CanonicalEntity | None:
        """Get a canonical entity by name and type.

        Args:
            canonical_name: The canonical entity name.
            entity_type: The entity type.

        Returns:
            CanonicalEntity if found, None otherwise.
        """
        return self._registry.get(entity_type, {}).get(canonical_name)

    def get_all_entities(
        self, entity_type: str | None = None
    ) -> list[CanonicalEntity]:
        """Get all registered entities.

        Args:
            entity_type: Optional filter by type.

        Returns:
            List of CanonicalEntity objects.
        """
        if entity_type:
            return list(self._registry.get(entity_type, {}).values())

        all_entities = []
        for entities in self._registry.values():
            all_entities.extend(entities.values())
        return all_entities

    def get_stats(self) -> dict[str, int]:
        """Get resolver statistics.

        Returns:
            Dictionary of statistics.
        """
        total_entities = sum(len(v) for v in self._registry.values())
        total_aliases = len(self._alias_index)

        return {
            "entity_types": len(self._registry),
            "total_entities": total_entities,
            "total_aliases": total_aliases,
        }

    def export_registry(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Export the registry for serialization.

        Returns:
            Nested dict of entity_type -> canonical_name -> entity_data.
        """
        result: dict[str, dict[str, dict[str, Any]]] = {}

        for entity_type, entities in self._registry.items():
            result[entity_type] = {}
            for canonical_name, entity in entities.items():
                result[entity_type][canonical_name] = entity.to_dict()

        return result

    def import_registry(
        self, data: dict[str, dict[str, dict[str, Any]]]
    ) -> None:
        """Import registry from serialized data.

        Args:
            data: Nested dict of entity_type -> canonical_name -> entity_data.
        """
        for entity_type, entities in data.items():
            for canonical_name, entity_data in entities.items():
                entity = CanonicalEntity.from_dict(entity_data)
                self.register_entity(entity)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "CanonicalEntity",
    "ResolutionResult",
    "EntityResolver",
]
