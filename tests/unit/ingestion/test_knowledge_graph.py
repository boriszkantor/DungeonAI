"""Tests for the Knowledge Graph system.

Tests cover:
1. Ontology schema loading and validation
2. Entity resolution and canonicalization
3. Knowledge graph construction and traversal
4. Hybrid retrieval with graph-vector combination
5. The Fireball/Evasion integration scenario
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dnd_manager.ingestion.ontology import (
    OntologySchema,
    SemanticTriple,
    ValidationResult,
    EntityType,
    Predicate,
)
from dnd_manager.ingestion.entity_resolver import (
    EntityResolver,
    CanonicalEntity,
    ResolutionResult,
)
from dnd_manager.ingestion.knowledge_graph import (
    DnDKnowledgeGraph,
    GraphNode,
    GraphEdge,
    TraversalResult,
)


# =============================================================================
# Ontology Schema Tests
# =============================================================================


class TestOntologySchema:
    """Tests for OntologySchema class."""

    def test_load_default_schema(self):
        """Test loading the default D&D 5E schema."""
        schema = OntologySchema.load_default()
        
        # Should have entity types
        assert len(schema.entity_types) > 0
        assert "Spell" in schema.entity_types
        assert "ClassFeature" in schema.entity_types
        assert "Save" in schema.entity_types
        
        # Should have predicates
        assert len(schema.predicates) > 0
        assert "requires_save" in schema.predicates
        assert "overrules" in schema.predicates
        assert "deals_damage" in schema.predicates

    def test_validate_valid_triple(self):
        """Test validation of a valid triple."""
        schema = OntologySchema.load_default()
        
        triple = SemanticTriple(
            subject="Fireball",
            subject_type="Spell",
            predicate="requires_save",
            object="DEX",
            object_type="Save",
        )
        
        result = schema.validate_triple(triple)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_invalid_predicate(self):
        """Test validation rejects invalid predicate."""
        schema = OntologySchema.load_default()
        
        triple = SemanticTriple(
            subject="Fireball",
            subject_type="Spell",
            predicate="invalid_predicate",
            object="DEX",
            object_type="Save",
        )
        
        result = schema.validate_triple(triple)
        assert not result.is_valid
        assert "Invalid predicate" in result.errors[0]

    def test_validate_invalid_source_type(self):
        """Test validation rejects invalid source type for predicate."""
        schema = OntologySchema.load_default()
        
        # Condition can't be the source of requires_save
        triple = SemanticTriple(
            subject="Blinded",
            subject_type="Condition",
            predicate="requires_save",
            object="DEX",
            object_type="Save",
        )
        
        result = schema.validate_triple(triple)
        assert not result.is_valid
        assert "not valid for predicate" in result.errors[0]

    def test_resolve_alias(self):
        """Test alias resolution."""
        schema = OntologySchema.load_default()
        
        # "Dexterity saving throw" should resolve to "DEX"
        result = schema.resolve_alias("Dexterity saving throw")
        assert result is not None
        assert result[0] == "DEX"
        assert result[1] == "Save"

    def test_get_seed_relationships(self):
        """Test loading seed relationships."""
        schema = OntologySchema.load_default()
        
        triples = schema.get_seed_relationships()
        assert len(triples) > 0
        
        # Should include Evasion overrules DEX
        evasion_overrules = [
            t for t in triples
            if t.subject == "Evasion" and t.predicate == "overrules"
        ]
        assert len(evasion_overrules) > 0


# =============================================================================
# Entity Resolver Tests
# =============================================================================


class TestEntityResolver:
    """Tests for EntityResolver class."""

    def test_create_from_schema(self):
        """Test creating resolver from schema."""
        schema = OntologySchema.load_default()
        resolver = EntityResolver.from_schema(schema)
        
        stats = resolver.get_stats()
        assert stats["total_entities"] > 0
        assert stats["total_aliases"] > 0

    def test_exact_match_resolution(self):
        """Test exact match resolution."""
        schema = OntologySchema.load_default()
        resolver = EntityResolver.from_schema(schema)
        
        result = resolver.resolve("DEX", "Save")
        
        assert result.canonical_name == "DEX"
        assert result.entity_type == "Save"
        assert result.match_type == "exact"
        assert result.confidence == 1.0

    def test_alias_resolution(self):
        """Test alias-based resolution."""
        schema = OntologySchema.load_default()
        resolver = EntityResolver.from_schema(schema)
        
        # "Dex save" is an alias for DEX
        result = resolver.resolve("Dex save", "Save")
        
        assert result.canonical_name == "DEX"
        assert result.match_type == "exact"  # Aliases are considered exact

    def test_fuzzy_match_resolution(self):
        """Test fuzzy matching for similar names."""
        schema = OntologySchema.load_default()
        resolver = EntityResolver.from_schema(schema, fuzzy_threshold=0.7)
        
        # "Fire Ball" should fuzzy match "Fireball"
        result = resolver.resolve("Fire Ball", "Spell")
        
        # Should find Fireball with fuzzy match
        assert result.canonical_name == "Fireball"
        assert result.match_type == "fuzzy"
        assert result.confidence >= 0.7

    def test_resolve_triple(self):
        """Test resolving entities in a triple."""
        schema = OntologySchema.load_default()
        resolver = EntityResolver.from_schema(schema)
        
        triple = SemanticTriple(
            subject="fire ball",  # Wrong case
            subject_type="Spell",
            predicate="requires_save",
            object="Dexterity saving throw",  # Alias
            object_type="Save",
            chunk_id="test-chunk-1",
        )
        
        resolved = resolver.resolve_triple(triple)
        
        assert resolved.subject == "Fireball"
        assert resolved.object == "DEX"
        assert resolved.chunk_id == "test-chunk-1"


# =============================================================================
# Knowledge Graph Tests
# =============================================================================


class TestDnDKnowledgeGraph:
    """Tests for DnDKnowledgeGraph class."""

    def test_create_graph(self):
        """Test creating an empty graph."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self):
        """Test adding a node."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        
        node_id = graph.add_node(
            entity_type="Spell",
            canonical_name="Fireball",
            chunk_ids={"chunk-1"},
        )
        
        assert node_id == "Spell:Fireball"
        assert graph.node_count == 1
        
        node = graph.get_node(node_id)
        assert node is not None
        assert node.canonical_name == "Fireball"
        assert "chunk-1" in node.chunk_ids

    def test_add_triple(self):
        """Test adding a triple creates nodes and edge."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        
        triple = SemanticTriple(
            subject="Fireball",
            subject_type="Spell",
            predicate="requires_save",
            object="DEX",
            object_type="Save",
            chunk_id="chunk-1",
        )
        
        source_id, target_id = graph.add_triple(triple)
        
        assert graph.node_count == 2
        assert graph.edge_count == 1
        
        edges = graph.get_edges(source_id=source_id)
        assert len(edges) == 1
        assert edges[0].predicate == "requires_save"
        assert edges[0].target_id == target_id

    def test_load_seed_data(self):
        """Test loading seed data into the graph."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        
        count = graph.load_seed_data()
        
        assert count > 0
        assert graph.node_count > 0
        assert graph.edge_count > 0

    def test_traverse_graph(self):
        """Test graph traversal from a node."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        # Traverse from Fireball
        result = graph.traverse("Spell:Fireball", max_hops=2)
        
        assert result.start_node.canonical_name == "Fireball"
        assert len(result.related_nodes) > 0
        
        # Should find DEX save
        dex_nodes = [n for n in result.related_nodes if n.canonical_name == "DEX"]
        assert len(dex_nodes) > 0

    def test_find_overriding_rules(self):
        """Test finding rules that override a mechanic."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        # Find what overrides DEX saves
        overrides = graph.find_overriding_rules("Save:DEX")
        
        # Evasion should override DEX saves
        evasion_overrides = [
            (n, e) for n, e in overrides if n.canonical_name == "Evasion"
        ]
        assert len(evasion_overrides) > 0


# =============================================================================
# Integration Test: Fireball + Monk Evasion Scenario
# =============================================================================


class TestFireballEvasionScenario:
    """Integration test for the Fireball/Evasion scenario from the plan.
    
    This tests the core use case: When a Monk is hit by Fireball, the system
    should discover that Evasion (a Monk class feature) overrides the normal
    DEX save behavior.
    """

    def test_fireball_finds_dex_save(self):
        """Test that Fireball is linked to DEX save."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        # Get Fireball's edges
        edges = graph.get_edges(source_id="Spell:Fireball", predicate="requires_save")
        
        assert len(edges) == 1
        assert edges[0].target_id == "Save:DEX"

    def test_dex_save_finds_evasion(self):
        """Test that DEX save is overruled by Evasion."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        # Find overriding rules for DEX saves
        overrides = graph.find_overriding_rules("Save:DEX")
        
        override_names = [n.canonical_name for n, e in overrides]
        assert "Evasion" in override_names

    def test_evasion_linked_to_monk(self):
        """Test that Evasion is granted by Monk class."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        # Traverse from Evasion to find what grants it
        result = graph.traverse(
            "ClassFeature:Evasion",
            max_hops=1,
            direction="in",
            predicates=["grants_feature"],
        )
        
        granting_classes = [n.canonical_name for n in result.related_nodes]
        assert "Monk" in granting_classes

    def test_full_fireball_evasion_chain(self):
        """Test the complete chain: Fireball -> DEX -> Evasion -> Monk."""
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        # Start from Fireball and traverse
        result = graph.traverse("Spell:Fireball", max_hops=3)
        
        # Should find all these nodes in the traversal
        node_names = {n.canonical_name for n in result.related_nodes}
        
        assert "DEX" in node_names  # Fireball requires DEX save
        assert "Fire" in node_names  # Fireball deals Fire damage
        
        # The overriding rules should be discoverable
        overrides = graph.find_overriding_rules("Save:DEX")
        override_names = [n.canonical_name for n, e in overrides]
        
        assert "Evasion" in override_names


# =============================================================================
# Hybrid Retriever Tests (with mocked vector store)
# =============================================================================


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    def test_create_retriever(self):
        """Test creating a hybrid retriever."""
        from dnd_manager.ingestion.hybrid_retriever import HybridRetriever
        
        # Create mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.search = MagicMock(return_value=[])
        
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            knowledge_graph=graph,
            schema=schema,
        )
        
        assert retriever is not None
        assert retriever.graph == graph

    def test_entity_extraction_from_query(self):
        """Test extracting entities from a query."""
        from dnd_manager.ingestion.hybrid_retriever import HybridRetriever
        
        mock_vector_store = MagicMock()
        mock_vector_store.search = MagicMock(return_value=[])
        
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            knowledge_graph=graph,
            schema=schema,
        )
        
        entities = retriever._extract_entities_from_query("The Monk casts Fireball")
        
        entity_names = [name for name, _ in entities]
        assert "Monk" in entity_names
        assert "Fireball" in entity_names

    def test_search_for_overrides(self):
        """Test searching specifically for overriding rules."""
        from dnd_manager.ingestion.hybrid_retriever import HybridRetriever
        
        mock_vector_store = MagicMock()
        mock_vector_store.search = MagicMock(return_value=[])
        
        schema = OntologySchema.load_default()
        graph = DnDKnowledgeGraph(schema)
        graph.load_seed_data()
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            knowledge_graph=graph,
            schema=schema,
        )
        
        # Search for rules that override DEX saves
        related_rules = retriever.search_for_overrides("DEX", "Save")
        
        rule_names = [r.node.canonical_name for r in related_rules]
        assert "Evasion" in rule_names


# =============================================================================
# Triple Extractor Tests (with mocked LLM)
# =============================================================================


class TestTripleExtractor:
    """Tests for TripleExtractor class."""

    def test_parse_response(self):
        """Test parsing LLM response."""
        from dnd_manager.ingestion.triple_extractor import TripleExtractor
        
        schema = OntologySchema.load_default()
        extractor = TripleExtractor(schema)
        
        response = """[
            {"subject": "Fireball", "subject_type": "Spell", "predicate": "requires_save", "object": "DEX", "object_type": "Save", "confidence": 0.95}
        ]"""
        
        triples = extractor._parse_response(response)
        
        assert len(triples) == 1
        assert triples[0]["subject"] == "Fireball"
        assert triples[0]["predicate"] == "requires_save"

    def test_parse_response_with_markdown(self):
        """Test parsing response wrapped in markdown code blocks."""
        from dnd_manager.ingestion.triple_extractor import TripleExtractor
        
        schema = OntologySchema.load_default()
        extractor = TripleExtractor(schema)
        
        response = """```json
[
    {"subject": "Fireball", "subject_type": "Spell", "predicate": "deals_damage", "object": "Fire", "object_type": "DamageType", "confidence": 1.0}
]
```"""
        
        triples = extractor._parse_response(response)
        
        assert len(triples) == 1
        assert triples[0]["object"] == "Fire"

    def test_dict_to_triple(self):
        """Test converting dict to SemanticTriple."""
        from dnd_manager.ingestion.triple_extractor import TripleExtractor
        
        schema = OntologySchema.load_default()
        extractor = TripleExtractor(schema)
        
        data = {
            "subject": "Hold Person",
            "subject_type": "Spell",
            "predicate": "inflicts_condition",
            "object": "Paralyzed",
            "object_type": "Condition",
            "confidence": 0.9,
        }
        
        triple = extractor._dict_to_triple(data, chunk_id="chunk-123")
        
        assert triple is not None
        assert triple.subject == "Hold Person"
        assert triple.predicate == "inflicts_condition"
        assert triple.chunk_id == "chunk-123"
        assert triple.confidence == 0.9
