"""D&D 5E Knowledge Graph using NetworkX.

This module implements Phase 4 of the knowledge graph pipeline:
Graph Construction & Vector Linking. It provides a NetworkX-based
graph that stores entities and relationships with links to the
ChromaDB vector store.

Key Features:
- Entity nodes with chunk_id links to vector store
- Typed relationship edges with metadata
- Multi-hop traversal for rule discovery
- Character-aware filtering
- JSON serialization for persistence

NEURO-SYMBOLIC PRINCIPLE:
The graph stores the LOGIC of D&D rules (relationships, constraints,
overrides) while the vector store stores the TEXT (descriptions,
flavor). Queries combine both for complete context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.entity_resolver import CanonicalEntity, EntityResolver
from dnd_manager.ingestion.ontology import OntologySchema, SemanticTriple


if TYPE_CHECKING:
    from dnd_manager.models.ecs import ActorEntity

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class GraphNode:
    """A node in the knowledge graph.

    Attributes:
        node_id: Unique identifier (usually "EntityType:CanonicalName").
        canonical_name: The canonical entity name.
        entity_type: The entity type from the ontology.
        chunk_ids: Links to ChromaDB chunks mentioning this entity.
        aliases: Known aliases for this entity.
        attributes: Additional entity attributes.
    """

    node_id: str
    canonical_name: str
    entity_type: str
    chunk_ids: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entity(cls, entity: CanonicalEntity) -> "GraphNode":
        """Create a GraphNode from a CanonicalEntity."""
        node_id = f"{entity.entity_type}:{entity.canonical_name}"
        return cls(
            node_id=node_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            chunk_ids=entity.chunk_ids.copy(),
            aliases=entity.aliases.copy(),
            attributes=entity.attributes.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for NetworkX node data."""
        return {
            "node_id": self.node_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "chunk_ids": list(self.chunk_ids),
            "aliases": list(self.aliases),
            "attributes": self.attributes,
        }


@dataclass
class GraphEdge:
    """An edge in the knowledge graph.

    Attributes:
        source_id: Source node ID.
        target_id: Target node ID.
        predicate: Relationship type from ontology.
        chunk_id: Source chunk where this relationship was found.
        metadata: Additional relationship metadata.
        confidence: Extraction confidence score.
    """

    source_id: str
    target_id: str
    predicate: str
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for NetworkX edge data."""
        return {
            "predicate": self.predicate,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }


@dataclass
class TraversalResult:
    """Result of a graph traversal.

    Attributes:
        start_node: The starting node.
        related_nodes: Nodes found during traversal.
        edges: Edges traversed.
        path: The traversal path (list of node IDs).
        hops: Number of hops from start node.
    """

    start_node: GraphNode
    related_nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    hops: int = 0


# =============================================================================
# Knowledge Graph
# =============================================================================


class DnDKnowledgeGraph:
    """NetworkX-based knowledge graph for D&D 5E rules.

    This class provides:
    - Entity (node) management with chunk_id linking
    - Relationship (edge) management with metadata
    - Multi-hop traversal for rule discovery
    - Character-aware filtering
    - JSON serialization

    Example:
        >>> schema = OntologySchema.load_default()
        >>> graph = DnDKnowledgeGraph(schema)
        >>> 
        >>> # Add entities and relationships
        >>> graph.add_triple(triple)
        >>> 
        >>> # Find related rules
        >>> results = graph.traverse("Spell:Fireball", max_hops=2)
        >>> for node in results.related_nodes:
        ...     print(f"Related: {node.canonical_name} ({node.entity_type})")
    """

    def __init__(
        self,
        schema: OntologySchema | None = None,
        entity_resolver: EntityResolver | None = None,
    ) -> None:
        """Initialize the knowledge graph.

        Args:
            schema: Ontology schema for validation.
            entity_resolver: Entity resolver for canonicalization.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "networkx package required. Install with: pip install networkx"
            ) from exc

        self._graph: Any = nx.MultiDiGraph()  # Allows multiple edges between nodes
        self.schema = schema or OntologySchema.load_default()
        self.resolver = entity_resolver or EntityResolver.from_schema(self.schema)

        # Index: entity_type -> set of node_ids
        self._type_index: dict[str, set[str]] = {}

        # Index: chunk_id -> set of node_ids
        self._chunk_index: dict[str, set[str]] = {}

        logger.info("DnDKnowledgeGraph initialized")

    def _make_node_id(self, entity_type: str, canonical_name: str) -> str:
        """Create a node ID from entity type and name.

        Args:
            entity_type: The entity type.
            canonical_name: The canonical name.

        Returns:
            Node ID in format "EntityType:CanonicalName".
        """
        return f"{entity_type}:{canonical_name}"

    def _parse_node_id(self, node_id: str) -> tuple[str, str]:
        """Parse a node ID into entity type and name.

        Args:
            node_id: The node ID.

        Returns:
            Tuple of (entity_type, canonical_name).
        """
        parts = node_id.split(":", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "Unknown", node_id

    def add_node(
        self,
        entity_type: str,
        canonical_name: str,
        *,
        chunk_ids: set[str] | None = None,
        aliases: set[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add or update a node in the graph.

        Args:
            entity_type: The entity type.
            canonical_name: The canonical name.
            chunk_ids: Optional chunk IDs to link.
            aliases: Optional aliases.
            attributes: Optional attributes.

        Returns:
            The node ID.
        """
        node_id = self._make_node_id(entity_type, canonical_name)

        if self._graph.has_node(node_id):
            # Update existing node
            node_data = self._graph.nodes[node_id]

            if chunk_ids:
                existing_chunks = set(node_data.get("chunk_ids", []))
                existing_chunks.update(chunk_ids)
                node_data["chunk_ids"] = list(existing_chunks)

            if aliases:
                existing_aliases = set(node_data.get("aliases", []))
                existing_aliases.update(aliases)
                node_data["aliases"] = list(existing_aliases)

            if attributes:
                existing_attrs = node_data.get("attributes", {})
                existing_attrs.update(attributes)
                node_data["attributes"] = existing_attrs
        else:
            # Create new node
            node_data = {
                "node_id": node_id,
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "chunk_ids": list(chunk_ids or []),
                "aliases": list(aliases or []),
                "attributes": attributes or {},
            }
            self._graph.add_node(node_id, **node_data)

            # Update type index
            if entity_type not in self._type_index:
                self._type_index[entity_type] = set()
            self._type_index[entity_type].add(node_id)

        # Update chunk index
        for chunk_id in chunk_ids or []:
            if chunk_id not in self._chunk_index:
                self._chunk_index[chunk_id] = set()
            self._chunk_index[chunk_id].add(node_id)

        return node_id

    def add_edge(
        self,
        source_type: str,
        source_name: str,
        predicate: str,
        target_type: str,
        target_name: str,
        *,
        chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> tuple[str, str]:
        """Add an edge (relationship) to the graph.

        Creates source and target nodes if they don't exist.

        Args:
            source_type: Source entity type.
            source_name: Source canonical name.
            predicate: Relationship type.
            target_type: Target entity type.
            target_name: Target canonical name.
            chunk_id: Optional source chunk ID.
            metadata: Optional edge metadata.
            confidence: Extraction confidence.

        Returns:
            Tuple of (source_node_id, target_node_id).
        """
        # Ensure nodes exist
        source_id = self.add_node(
            source_type,
            source_name,
            chunk_ids={chunk_id} if chunk_id else None,
        )
        target_id = self.add_node(
            target_type,
            target_name,
            chunk_ids={chunk_id} if chunk_id else None,
        )

        # Add edge
        edge_data = {
            "predicate": predicate,
            "chunk_id": chunk_id,
            "metadata": metadata or {},
            "confidence": confidence,
        }
        self._graph.add_edge(source_id, target_id, **edge_data)

        return source_id, target_id

    def add_triple(self, triple: SemanticTriple) -> tuple[str, str]:
        """Add a semantic triple to the graph.

        Args:
            triple: The semantic triple to add.

        Returns:
            Tuple of (source_node_id, target_node_id).
        """
        return self.add_edge(
            source_type=triple.subject_type,
            source_name=triple.subject,
            predicate=triple.predicate,
            target_type=triple.object_type,
            target_name=triple.object,
            chunk_id=triple.chunk_id,
            metadata=triple.metadata,
            confidence=triple.confidence,
        )

    def add_triples(self, triples: list[SemanticTriple]) -> int:
        """Add multiple triples to the graph.

        Args:
            triples: List of semantic triples.

        Returns:
            Number of triples added.
        """
        for triple in triples:
            self.add_triple(triple)
        return len(triples)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID.

        Returns:
            GraphNode if found, None otherwise.
        """
        if not self._graph.has_node(node_id):
            return None

        data = self._graph.nodes[node_id]
        return GraphNode(
            node_id=node_id,
            canonical_name=data.get("canonical_name", ""),
            entity_type=data.get("entity_type", ""),
            chunk_ids=set(data.get("chunk_ids", [])),
            aliases=set(data.get("aliases", [])),
            attributes=data.get("attributes", {}),
        )

    def get_node_by_name(
        self, entity_type: str, canonical_name: str
    ) -> GraphNode | None:
        """Get a node by entity type and name.

        Args:
            entity_type: The entity type.
            canonical_name: The canonical name.

        Returns:
            GraphNode if found, None otherwise.
        """
        node_id = self._make_node_id(entity_type, canonical_name)
        return self.get_node(node_id)

    def find_node(self, name: str, entity_type: str | None = None) -> GraphNode | None:
        """Find a node by name, with optional type hint.

        Uses entity resolver for fuzzy matching.

        Args:
            name: Entity name (can be alias).
            entity_type: Optional entity type hint.

        Returns:
            GraphNode if found, None otherwise.
        """
        result = self.resolver.resolve(name, entity_type, auto_register=False)

        if result.match_type == "new":
            return None

        return self.get_node_by_name(result.entity_type, result.canonical_name)

    def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        predicate: str | None = None,
    ) -> list[GraphEdge]:
        """Get edges matching criteria.

        Args:
            source_id: Optional source node ID filter.
            target_id: Optional target node ID filter.
            predicate: Optional predicate filter.

        Returns:
            List of matching GraphEdge objects.
        """
        edges = []

        if source_id:
            if not self._graph.has_node(source_id):
                return []
            edge_iter = self._graph.out_edges(source_id, data=True)
        elif target_id:
            if not self._graph.has_node(target_id):
                return []
            edge_iter = self._graph.in_edges(target_id, data=True)
        else:
            edge_iter = self._graph.edges(data=True)

        for u, v, data in edge_iter:
            if target_id and v != target_id:
                continue
            if source_id and u != source_id:
                continue
            if predicate and data.get("predicate") != predicate:
                continue

            edges.append(
                GraphEdge(
                    source_id=u,
                    target_id=v,
                    predicate=data.get("predicate", ""),
                    chunk_id=data.get("chunk_id"),
                    metadata=data.get("metadata", {}),
                    confidence=data.get("confidence", 1.0),
                )
            )

        return edges

    def traverse(
        self,
        start_node_id: str,
        *,
        max_hops: int = 2,
        predicates: list[str] | None = None,
        entity_types: list[str] | None = None,
        direction: str = "both",  # "out", "in", "both"
    ) -> TraversalResult:
        """Traverse the graph from a starting node.

        Args:
            start_node_id: Starting node ID.
            max_hops: Maximum traversal depth.
            predicates: Optional filter by predicate types.
            entity_types: Optional filter by entity types.
            direction: Traversal direction ("out", "in", "both").

        Returns:
            TraversalResult with found nodes and edges.
        """
        start_node = self.get_node(start_node_id)
        if not start_node:
            return TraversalResult(
                start_node=GraphNode(
                    node_id=start_node_id,
                    canonical_name="",
                    entity_type="",
                ),
            )

        visited: set[str] = {start_node_id}
        related_nodes: list[GraphNode] = []
        traversed_edges: list[GraphEdge] = []
        path: list[str] = [start_node_id]

        # BFS traversal
        current_level = [start_node_id]

        for hop in range(max_hops):
            next_level = []

            for node_id in current_level:
                # Get edges based on direction
                if direction in ("out", "both"):
                    out_edges = self.get_edges(source_id=node_id)
                    for edge in out_edges:
                        if predicates and edge.predicate not in predicates:
                            continue

                        target = self.get_node(edge.target_id)
                        if not target:
                            continue

                        if entity_types and target.entity_type not in entity_types:
                            continue

                        if edge.target_id not in visited:
                            visited.add(edge.target_id)
                            next_level.append(edge.target_id)
                            related_nodes.append(target)
                            traversed_edges.append(edge)
                            path.append(edge.target_id)

                if direction in ("in", "both"):
                    in_edges = self.get_edges(target_id=node_id)
                    for edge in in_edges:
                        if predicates and edge.predicate not in predicates:
                            continue

                        source = self.get_node(edge.source_id)
                        if not source:
                            continue

                        if entity_types and source.entity_type not in entity_types:
                            continue

                        if edge.source_id not in visited:
                            visited.add(edge.source_id)
                            next_level.append(edge.source_id)
                            related_nodes.append(source)
                            traversed_edges.append(edge)
                            path.append(edge.source_id)

            current_level = next_level

        return TraversalResult(
            start_node=start_node,
            related_nodes=related_nodes,
            edges=traversed_edges,
            path=path,
            hops=max_hops,
        )

    def find_overriding_rules(
        self,
        mechanic_node_id: str,
        *,
        character: "ActorEntity | None" = None,
    ) -> list[tuple[GraphNode, GraphEdge]]:
        """Find rules that override a mechanic, optionally filtered by character.

        This implements the "specific beats general" rule lookup.

        Args:
            mechanic_node_id: Node ID of the mechanic/save to check.
            character: Optional character to filter by class/level.

        Returns:
            List of (overriding_node, edge) tuples.
        """
        results: list[tuple[GraphNode, GraphEdge]] = []

        # Find all edges with "overrules" predicate targeting this mechanic
        edges = self.get_edges(target_id=mechanic_node_id, predicate="overrules")

        for edge in edges:
            source_node = self.get_node(edge.source_id)
            if not source_node:
                continue

            # Check if character has access to this feature
            if character:
                # For class features, check if character has the class and level
                if source_node.entity_type == "ClassFeature":
                    level_req = edge.metadata.get(
                        "level_requirement",
                        source_node.attributes.get("level_requirement"),
                    )
                    classes = source_node.attributes.get("classes", [])

                    # Check if character has the required class at required level
                    has_access = False
                    if hasattr(character, "class_features") and character.class_features:
                        for class_level in character.class_features.classes:
                            class_name = class_level.class_name
                            if class_name in classes:
                                if level_req is None or class_level.level >= level_req:
                                    has_access = True
                                    break

                    if not has_access:
                        continue

                # For feats, check if character has the feat
                elif source_node.entity_type == "Feat":
                    # Would need to check character's feats
                    # For now, include all feats
                    pass

            results.append((source_node, edge))

        return results

    def get_nodes_by_type(self, entity_type: str) -> list[GraphNode]:
        """Get all nodes of a specific type.

        Args:
            entity_type: The entity type to filter by.

        Returns:
            List of GraphNode objects.
        """
        node_ids = self._type_index.get(entity_type, set())
        nodes = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                nodes.append(node)
        return nodes

    def get_nodes_for_chunk(self, chunk_id: str) -> list[GraphNode]:
        """Get all nodes linked to a chunk.

        Args:
            chunk_id: The chunk ID.

        Returns:
            List of GraphNode objects.
        """
        node_ids = self._chunk_index.get(chunk_id, set())
        nodes = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                nodes.append(node)
        return nodes

    def get_chunk_ids_for_node(self, node_id: str) -> set[str]:
        """Get all chunk IDs linked to a node.

        Args:
            node_id: The node ID.

        Returns:
            Set of chunk IDs.
        """
        node = self.get_node(node_id)
        return node.chunk_ids if node else set()

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self._graph.number_of_edges()

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary of statistics.
        """
        type_counts = {
            entity_type: len(node_ids)
            for entity_type, node_ids in self._type_index.items()
        }

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "entity_types": len(self._type_index),
            "nodes_by_type": type_counts,
            "chunks_indexed": len(self._chunk_index),
        }

    def save(self, path: Path | str) -> None:
        """Save the graph to a JSON file.

        Args:
            path: Path to save the graph.
        """
        import networkx as nx

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = nx.node_link_data(self._graph)

        # Add indices
        data["_type_index"] = {k: list(v) for k, v in self._type_index.items()}
        data["_chunk_index"] = {k: list(v) for k, v in self._chunk_index.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Knowledge graph saved to {path}",
            nodes=self.node_count,
            edges=self.edge_count,
        )

    def load(self, path: Path | str) -> None:
        """Load the graph from a JSON file.

        Args:
            path: Path to load the graph from.
        """
        import networkx as nx

        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract indices before loading graph
        type_index = data.pop("_type_index", {})
        chunk_index = data.pop("_chunk_index", {})

        # Load graph
        self._graph = nx.node_link_graph(data)

        # Restore indices
        self._type_index = {k: set(v) for k, v in type_index.items()}
        self._chunk_index = {k: set(v) for k, v in chunk_index.items()}

        logger.info(
            f"Knowledge graph loaded from {path}",
            nodes=self.node_count,
            edges=self.edge_count,
        )

    def load_seed_data(self) -> int:
        """Load seed relationships from the ontology schema.

        Returns:
            Number of triples loaded.
        """
        triples = self.schema.get_seed_relationships()
        count = self.add_triples(triples)

        logger.info(f"Loaded {count} seed relationships into graph")
        return count


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "GraphNode",
    "GraphEdge",
    "TraversalResult",
    "DnDKnowledgeGraph",
]
