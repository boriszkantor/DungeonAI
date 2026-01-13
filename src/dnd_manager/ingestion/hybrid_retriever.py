"""Hybrid Graph-Vector Retriever for D&D 5E.

This module implements the Query Pipeline that combines vector search
with knowledge graph traversal for comprehensive rule retrieval.

The Query Workflow:
1. Vector Search: Find primary chunks matching the query
2. Entity Extraction: Identify entities mentioned in results
3. Graph Traversal: Find related rules (overrides, requirements, etc.)
4. Character Filtering: Filter by character's class/level/features
5. Context Assembly: Combine vector text + graph relationships

This solves the "specific beats general" problem where a standard vector
search for "Fireball damage" would miss the Monk's Evasion feature.

NEURO-SYMBOLIC PRINCIPLE:
Vector search finds WHAT (text content).
Graph traversal finds WHY (rule relationships).
Character filtering ensures RELEVANCE (applicable rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.knowledge_graph import (
    DnDKnowledgeGraph,
    GraphEdge,
    GraphNode,
)
from dnd_manager.ingestion.ontology import OntologySchema


if TYPE_CHECKING:
    from dnd_manager.ingestion.rag_store import RAGStore, SearchResult
    from dnd_manager.ingestion.universal_loader import ChromaStore, ChunkedDocument
    from dnd_manager.models.ecs import ActorEntity

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RelatedRule:
    """A related rule discovered via graph traversal.

    Attributes:
        node: The graph node representing the rule.
        relationship: How this rule relates to the query.
        predicate: The predicate connecting this rule.
        chunk_ids: Chunk IDs containing the rule text.
        explanation: Human-readable explanation of the relationship.
        relevance_score: How relevant this rule is (0.0-1.0).
    """

    node: GraphNode
    relationship: str  # "overrides", "requires", "grants", etc.
    predicate: str
    chunk_ids: list[str] = field(default_factory=list)
    explanation: str = ""
    relevance_score: float = 1.0


@dataclass
class HybridSearchResult:
    """Result of a hybrid graph-vector search.

    Attributes:
        query: The original search query.
        vector_results: Direct vector search results.
        related_rules: Rules found via graph traversal.
        graph_context: Human-readable graph relationship summary.
        total_chunks: Total unique chunks to retrieve.
        character_filtered: Whether results were filtered by character.
    """

    query: str
    vector_results: list[Any] = field(default_factory=list)  # SearchResult or ChunkedDocument
    related_rules: list[RelatedRule] = field(default_factory=list)
    graph_context: str = ""
    total_chunks: list[str] = field(default_factory=list)
    character_filtered: bool = False

    def get_combined_context(self) -> str:
        """Get combined context from vector results and graph relationships.

        Returns:
            Combined context string for LLM consumption.
        """
        parts = []

        # Add vector search results
        if self.vector_results:
            parts.append("## Retrieved Rules\n")
            for i, result in enumerate(self.vector_results, 1):
                if hasattr(result, "content"):
                    # SearchResult from RAGStore
                    content = result.content.text if hasattr(result.content, "text") else str(result.content)
                    title = result.content.title if hasattr(result.content, "title") else ""
                else:
                    # ChunkedDocument
                    content = result.content if hasattr(result, "content") else str(result)
                    title = result.metadata.get("title", "") if hasattr(result, "metadata") else ""

                if title:
                    parts.append(f"### {title}\n{content}\n")
                else:
                    parts.append(f"### Result {i}\n{content}\n")

        # Add graph relationship context
        if self.graph_context:
            parts.append("\n## Related Rules (from Knowledge Graph)\n")
            parts.append(self.graph_context)

        return "\n".join(parts)


# =============================================================================
# Hybrid Retriever
# =============================================================================


class HybridRetriever:
    """Combines vector search with knowledge graph traversal.

    This class implements the full query pipeline:
    1. Vector search for initial results
    2. Entity extraction from results
    3. Graph traversal for related rules
    4. Character-aware filtering
    5. Context assembly

    Example:
        >>> retriever = HybridRetriever(chroma_store, knowledge_graph)
        >>> result = retriever.search(
        ...     "The Monk is hit by Fireball",
        ...     character=monk_entity,
        ... )
        >>> print(result.graph_context)
        # Shows Evasion feature that overrides DEX save behavior
    """

    def __init__(
        self,
        vector_store: "ChromaStore | RAGStore",
        knowledge_graph: DnDKnowledgeGraph,
        schema: OntologySchema | None = None,
        *,
        max_vector_results: int = 5,
        max_graph_hops: int = 2,
        include_hyde: bool = True,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_store: ChromaStore or RAGStore for vector search.
            knowledge_graph: Knowledge graph for relationship traversal.
            schema: Ontology schema (uses graph's schema if None).
            max_vector_results: Maximum vector search results.
            max_graph_hops: Maximum graph traversal depth.
            include_hyde: Whether to use HyDE for vector search.
        """
        self.vector_store = vector_store
        self.graph = knowledge_graph
        self.schema = schema or knowledge_graph.schema
        self.max_vector_results = max_vector_results
        self.max_graph_hops = max_graph_hops
        self.include_hyde = include_hyde

        # Detect vector store type
        self._is_chroma = hasattr(vector_store, "search") and not hasattr(
            vector_store, "retrieve_relevant_context"
        )

        logger.info(
            "HybridRetriever initialized",
            vector_store_type="ChromaStore" if self._is_chroma else "RAGStore",
            max_vector_results=max_vector_results,
            max_graph_hops=max_graph_hops,
        )

    def _vector_search(
        self, query: str, n_results: int | None = None
    ) -> list[Any]:
        """Perform vector search.

        Args:
            query: Search query.
            n_results: Number of results (uses default if None).

        Returns:
            List of search results.
        """
        n = n_results or self.max_vector_results

        if self._is_chroma:
            return self.vector_store.search(query, n_results=n)
        else:
            # RAGStore
            if self.include_hyde and hasattr(self.vector_store, "hyde_retriever"):
                return self.vector_store.hyde_retriever.retrieve_with_hyde(
                    query, k=n
                )
            else:
                return self.vector_store.retrieve_relevant_context(query, k=n)

    def _extract_entities_from_query(self, query: str) -> list[tuple[str, str]]:
        """Extract entity mentions from query text.

        Args:
            query: The search query.

        Returns:
            List of (entity_name, entity_type) tuples.
        """
        entities = []
        query_lower = query.lower()

        # Check canonical entities
        for entity_type, type_entities in self.schema.canonical_entities.items():
            for canonical_name, entity_data in type_entities.items():
                # Check canonical name
                if canonical_name.lower() in query_lower:
                    entities.append((canonical_name, entity_type))
                    continue

                # Check aliases
                for alias in entity_data.get("aliases", []):
                    if alias.lower() in query_lower:
                        entities.append((canonical_name, entity_type))
                        break

        return entities

    def _extract_entities_from_chunks(
        self, chunks: list[Any]
    ) -> list[tuple[str, str]]:
        """Extract entities mentioned in chunk content.

        Args:
            chunks: Search results or chunked documents.

        Returns:
            List of (entity_name, entity_type) tuples.
        """
        entities = []
        seen = set()

        for chunk in chunks:
            # Get content
            if hasattr(chunk, "content"):
                if hasattr(chunk.content, "text"):
                    content = chunk.content.text
                else:
                    content = str(chunk.content)
            else:
                content = str(chunk)

            content_lower = content.lower()

            # Check canonical entities
            for entity_type, type_entities in self.schema.canonical_entities.items():
                for canonical_name, entity_data in type_entities.items():
                    if canonical_name in seen:
                        continue

                    # Check canonical name
                    if canonical_name.lower() in content_lower:
                        entities.append((canonical_name, entity_type))
                        seen.add(canonical_name)
                        continue

                    # Check aliases
                    for alias in entity_data.get("aliases", []):
                        if alias.lower() in content_lower:
                            entities.append((canonical_name, entity_type))
                            seen.add(canonical_name)
                            break

        return entities

    def _get_related_rules(
        self,
        entities: list[tuple[str, str]],
        character: "ActorEntity | None" = None,
    ) -> list[RelatedRule]:
        """Get related rules for entities via graph traversal.

        Args:
            entities: List of (entity_name, entity_type) tuples.
            character: Optional character for filtering.

        Returns:
            List of RelatedRule objects.
        """
        related: list[RelatedRule] = []
        seen_nodes: set[str] = set()

        for entity_name, entity_type in entities:
            node_id = f"{entity_type}:{entity_name}"

            # Traverse from this entity
            result = self.graph.traverse(
                node_id,
                max_hops=self.max_graph_hops,
                direction="both",
            )

            for node, edge in zip(result.related_nodes, result.edges):
                if node.node_id in seen_nodes:
                    continue
                seen_nodes.add(node.node_id)

                # Check if this is an overriding rule
                is_override = edge.predicate == "overrules"

                # For overriding rules, check character access
                if is_override and character:
                    overrides = self.graph.find_overriding_rules(
                        edge.target_id, character=character
                    )
                    if not any(n.node_id == node.node_id for n, _ in overrides):
                        continue

                # Build explanation
                explanation = self._build_explanation(node, edge, entity_name)

                related.append(
                    RelatedRule(
                        node=node,
                        relationship=edge.predicate,
                        predicate=edge.predicate,
                        chunk_ids=list(node.chunk_ids),
                        explanation=explanation,
                        relevance_score=1.0 if is_override else 0.8,
                    )
                )

        # Sort by relevance (overrides first)
        related.sort(key=lambda r: r.relevance_score, reverse=True)

        return related

    def _build_explanation(
        self, node: GraphNode, edge: GraphEdge, source_entity: str
    ) -> str:
        """Build human-readable explanation for a relationship.

        Args:
            node: The related node.
            edge: The connecting edge.
            source_entity: The source entity name.

        Returns:
            Explanation string.
        """
        predicate = edge.predicate
        condition = edge.metadata.get("condition", "")

        explanations = {
            "overrules": f"**{node.canonical_name}** overrides normal {source_entity} behavior",
            "requires_save": f"**{source_entity}** requires a {node.canonical_name} saving throw",
            "deals_damage": f"**{source_entity}** deals {node.canonical_name} damage",
            "immune_to": f"**{node.canonical_name}** grants immunity to {source_entity}",
            "resistant_to": f"**{node.canonical_name}** grants resistance to {source_entity}",
            "inflicts_condition": f"**{source_entity}** can inflict the {node.canonical_name} condition",
            "grants_feature": f"**{node.canonical_name}** grants the {source_entity} feature",
            "costs_action": f"**{source_entity}** costs a {node.canonical_name}",
            "prerequisite_of": f"**{node.canonical_name}** is a prerequisite for {source_entity}",
        }

        base = explanations.get(
            predicate,
            f"**{node.canonical_name}** is related to {source_entity} via {predicate}",
        )

        if condition:
            base += f" ({condition})"

        return base

    def _build_graph_context(
        self, related_rules: list[RelatedRule]
    ) -> str:
        """Build graph context summary for LLM.

        Args:
            related_rules: List of related rules.

        Returns:
            Formatted context string.
        """
        if not related_rules:
            return ""

        lines = []

        # Group by relationship type
        overrides = [r for r in related_rules if r.predicate == "overrules"]
        mechanics = [r for r in related_rules if r.predicate in (
            "requires_save", "deals_damage", "inflicts_condition"
        )]
        other = [r for r in related_rules if r not in overrides and r not in mechanics]

        if overrides:
            lines.append("### Specific Rules (Override General)")
            for rule in overrides:
                lines.append(f"- {rule.explanation}")
            lines.append("")

        if mechanics:
            lines.append("### Mechanics")
            for rule in mechanics:
                lines.append(f"- {rule.explanation}")
            lines.append("")

        if other:
            lines.append("### Related Information")
            for rule in other:
                lines.append(f"- {rule.explanation}")

        return "\n".join(lines)

    def search(
        self,
        query: str,
        *,
        character: "ActorEntity | None" = None,
        n_results: int | None = None,
        include_graph: bool = True,
    ) -> HybridSearchResult:
        """Perform hybrid graph-vector search.

        Args:
            query: Search query.
            character: Optional character for filtering.
            n_results: Number of vector results.
            include_graph: Whether to include graph traversal.

        Returns:
            HybridSearchResult with combined results.
        """
        logger.info(f"Hybrid search: {query[:50]}...")

        # Step 1: Vector search
        vector_results = self._vector_search(query, n_results)

        result = HybridSearchResult(
            query=query,
            vector_results=vector_results,
            character_filtered=character is not None,
        )

        if not include_graph:
            return result

        # Step 2: Extract entities from query and results
        query_entities = self._extract_entities_from_query(query)
        chunk_entities = self._extract_entities_from_chunks(vector_results)

        all_entities = list(set(query_entities + chunk_entities))

        logger.debug(f"Extracted {len(all_entities)} entities")

        # Step 3: Graph traversal
        related_rules = self._get_related_rules(all_entities, character)

        result.related_rules = related_rules

        # Step 4: Build graph context
        result.graph_context = self._build_graph_context(related_rules)

        # Step 5: Collect all chunk IDs
        chunk_ids: set[str] = set()

        for vr in vector_results:
            if hasattr(vr, "content") and hasattr(vr.content, "content_id"):
                chunk_ids.add(vr.content.content_id)
            elif hasattr(vr, "chunk_id"):
                chunk_ids.add(vr.chunk_id)

        for rule in related_rules:
            chunk_ids.update(rule.chunk_ids)

        result.total_chunks = list(chunk_ids)

        logger.info(
            f"Hybrid search complete: {len(vector_results)} vector results, "
            f"{len(related_rules)} related rules"
        )

        return result

    def search_for_overrides(
        self,
        mechanic: str,
        mechanic_type: str = "Save",
        *,
        character: "ActorEntity | None" = None,
    ) -> list[RelatedRule]:
        """Search specifically for rules that override a mechanic.

        This is useful for checking "specific beats general" scenarios.

        Args:
            mechanic: The mechanic name (e.g., "DEX").
            mechanic_type: The mechanic type.
            character: Optional character for filtering.

        Returns:
            List of overriding rules.
        """
        node_id = f"{mechanic_type}:{mechanic}"

        overrides = self.graph.find_overriding_rules(node_id, character=character)

        related = []
        for node, edge in overrides:
            explanation = self._build_explanation(node, edge, mechanic)
            related.append(
                RelatedRule(
                    node=node,
                    relationship="overrules",
                    predicate="overrules",
                    chunk_ids=list(node.chunk_ids),
                    explanation=explanation,
                    relevance_score=1.0,
                )
            )

        return related

    def get_context_for_spell(
        self,
        spell_name: str,
        *,
        character: "ActorEntity | None" = None,
    ) -> HybridSearchResult:
        """Get complete context for a spell, including related rules.

        Args:
            spell_name: Name of the spell.
            character: Optional target character.

        Returns:
            HybridSearchResult with spell info and related rules.
        """
        return self.search(
            f"{spell_name} spell rules mechanics",
            character=character,
        )

    def get_context_for_action(
        self,
        action_description: str,
        *,
        actor: "ActorEntity | None" = None,
        target: "ActorEntity | None" = None,
    ) -> HybridSearchResult:
        """Get context for a game action.

        Args:
            action_description: Description of the action.
            actor: The acting character.
            target: Optional target character.

        Returns:
            HybridSearchResult with relevant rules.
        """
        # Search with actor context
        result = self.search(
            action_description,
            character=actor,
        )

        # If there's a target, also check for their overriding rules
        if target:
            # Find what saves/mechanics are involved
            for rule in result.related_rules:
                if rule.predicate == "requires_save":
                    # Check if target has override for this save
                    target_overrides = self.search_for_overrides(
                        rule.node.canonical_name,
                        mechanic_type=rule.node.entity_type,
                        character=target,
                    )
                    result.related_rules.extend(target_overrides)

        return result


# =============================================================================
# Factory Function
# =============================================================================


def create_hybrid_retriever(
    vector_store: "ChromaStore | RAGStore",
    graph_path: str | None = None,
    *,
    load_seed_data: bool = True,
) -> HybridRetriever:
    """Create a HybridRetriever with optional graph loading.

    Args:
        vector_store: ChromaStore or RAGStore instance.
        graph_path: Optional path to load graph from.
        load_seed_data: Whether to load seed data if no graph file.

    Returns:
        Configured HybridRetriever.
    """
    from pathlib import Path

    schema = OntologySchema.load_default()
    graph = DnDKnowledgeGraph(schema)

    if graph_path and Path(graph_path).exists():
        graph.load(graph_path)
    elif load_seed_data:
        graph.load_seed_data()

    return HybridRetriever(vector_store, graph, schema)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "RelatedRule",
    "HybridSearchResult",
    "HybridRetriever",
    "create_hybrid_retriever",
]
