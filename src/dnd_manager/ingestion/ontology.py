"""D&D 5E Ontology Schema and Triple Validation.

This module provides the formal ontology definition for the knowledge graph.
It defines valid entity types, predicates, and their constraints, ensuring
that all extracted triples conform to the D&D 5E domain model.

NEURO-SYMBOLIC PRINCIPLE:
The ontology is the "contract" between the LLM extraction and the graph database.
By constraining valid relationships, we prevent hallucinated or invalid triples
from corrupting the knowledge base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from dnd_manager.core.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# Entity Types
# =============================================================================


class EntityType(StrEnum):
    """Valid entity types in the D&D 5E ontology."""

    SPELL = "Spell"
    CLASS_FEATURE = "ClassFeature"
    CONDITION = "Condition"
    DAMAGE_TYPE = "DamageType"
    SAVE = "Save"
    RACE = "Race"
    CLASS = "Class"
    SUBCLASS = "Subclass"
    FEAT = "Feat"
    ITEM = "Item"
    ACTION = "Action"
    MONSTER = "Monster"
    MECHANIC = "Mechanic"
    SKILL = "Skill"
    ABILITY_SCORE = "AbilityScore"
    SPELL_SCHOOL = "SpellSchool"
    CREATURE_TYPE = "CreatureType"
    RANGE = "Range"
    SPELL_LEVEL = "SpellLevel"


class Predicate(StrEnum):
    """Valid predicates (edge types) in the D&D 5E ontology."""

    REQUIRES_SAVE = "requires_save"
    DEALS_DAMAGE = "deals_damage"
    INFLICTS_CONDITION = "inflicts_condition"
    OVERRULES = "overrules"
    IMMUNE_TO = "immune_to"
    RESISTANT_TO = "resistant_to"
    VULNERABLE_TO = "vulnerable_to"
    PREREQUISITE_OF = "prerequisite_of"
    GRANTS_FEATURE = "grants_feature"
    COSTS_ACTION = "costs_action"
    GRANTS_EXTRA_ACTION = "grants_extra_action"
    TRIGGERS_REACTION = "triggers_reaction"
    REQUIRES_CONCENTRATION = "requires_concentration"
    HAS_RANGE = "has_range"
    HAS_SPELL_SCHOOL = "has_spell_school"
    HAS_SPELL_LEVEL = "has_spell_level"
    CAN_CAST = "can_cast"
    USES_ABILITY = "uses_ability"
    HAS_CREATURE_TYPE = "has_creature_type"
    ADVANTAGE_ON = "advantage_on"
    DISADVANTAGE_ON = "disadvantage_on"
    PART_OF = "part_of"
    COUNTERS = "counters"
    ENDS_ON = "ends_on"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PredicateConstraint:
    """Defines the valid source and target types for a predicate.

    Attributes:
        predicate: The predicate name.
        valid_sources: Entity types that can be the subject.
        valid_targets: Entity types that can be the object.
        cardinality: Relationship cardinality (many-to-one, many-to-many, etc.).
        metadata_fields: Optional metadata fields for the edge.
    """

    predicate: str
    valid_sources: list[str]
    valid_targets: list[str]
    cardinality: str = "many-to-many"
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class SemanticTriple:
    """A semantic triple representing a relationship in the knowledge graph.

    Attributes:
        subject: The source entity name.
        subject_type: The entity type of the subject.
        predicate: The relationship type.
        object: The target entity name.
        object_type: The entity type of the object.
        metadata: Optional metadata for the relationship.
        chunk_id: The source chunk ID from the vector store.
        confidence: Extraction confidence score (0.0-1.0).
    """

    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticTriple":
        """Create from dictionary."""
        return cls(
            subject=data["subject"],
            subject_type=data["subject_type"],
            predicate=data["predicate"],
            object=data["object"],
            object_type=data["object_type"],
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ValidationResult:
    """Result of triple validation.

    Attributes:
        is_valid: Whether the triple is valid.
        errors: List of validation error messages.
        warnings: List of validation warnings.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Ontology Schema
# =============================================================================


class OntologySchema:
    """Loads and validates against the D&D 5E ontology schema.

    This class provides:
    1. Schema loading from JSON files
    2. Triple validation against predicate constraints
    3. Entity type checking
    4. Canonical entity lookup

    Example:
        >>> schema = OntologySchema.load_default()
        >>> triple = SemanticTriple(
        ...     subject="Fireball",
        ...     subject_type="Spell",
        ...     predicate="requires_save",
        ...     object="DEX",
        ...     object_type="Save",
        ... )
        >>> result = schema.validate_triple(triple)
        >>> assert result.is_valid
    """

    def __init__(
        self,
        entity_types: dict[str, dict[str, Any]],
        predicates: dict[str, PredicateConstraint],
        canonical_entities: dict[str, dict[str, dict[str, Any]]] | None = None,
        seed_relationships: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the ontology schema.

        Args:
            entity_types: Map of entity type name to attributes.
            predicates: Map of predicate name to constraints.
            canonical_entities: Optional map of canonical entity names by type.
            seed_relationships: Optional list of seed relationships.
        """
        self.entity_types = entity_types
        self.predicates = predicates
        self.canonical_entities = canonical_entities or {}
        self.seed_relationships = seed_relationships or []

        # Build reverse alias lookup for fast resolution
        self._alias_to_canonical: dict[str, tuple[str, str]] = {}
        self._build_alias_index()

        logger.info(
            "OntologySchema initialized",
            entity_types=len(entity_types),
            predicates=len(predicates),
            canonical_entities=sum(
                len(v) for v in self.canonical_entities.values()
            ),
        )

    def _build_alias_index(self) -> None:
        """Build reverse lookup from aliases to canonical names."""
        for entity_type, entities in self.canonical_entities.items():
            for canonical_name, entity_data in entities.items():
                # Add canonical name itself
                key = canonical_name.lower()
                self._alias_to_canonical[key] = (canonical_name, entity_type)

                # Add all aliases
                aliases = entity_data.get("aliases", [])
                for alias in aliases:
                    alias_key = alias.lower()
                    if alias_key not in self._alias_to_canonical:
                        self._alias_to_canonical[alias_key] = (
                            canonical_name,
                            entity_type,
                        )

    @classmethod
    def load_from_files(
        cls,
        schema_path: Path | str,
        entities_path: Path | str | None = None,
    ) -> "OntologySchema":
        """Load ontology schema from JSON files.

        Args:
            schema_path: Path to the schema JSON file.
            entities_path: Optional path to the canonical entities JSON file.

        Returns:
            Loaded OntologySchema instance.
        """
        schema_path = Path(schema_path)

        with open(schema_path, encoding="utf-8") as f:
            schema_data = json.load(f)

        # Parse entity types
        entity_types = schema_data.get("entity_types", {})

        # Parse predicates
        predicates: dict[str, PredicateConstraint] = {}
        for name, pred_data in schema_data.get("predicates", {}).items():
            predicates[name] = PredicateConstraint(
                predicate=name,
                valid_sources=pred_data.get("valid_sources", []),
                valid_targets=pred_data.get("valid_targets", []),
                cardinality=pred_data.get("cardinality", "many-to-many"),
                metadata_fields=pred_data.get("metadata", []),
            )

        # Load canonical entities if path provided
        canonical_entities: dict[str, dict[str, dict[str, Any]]] = {}
        seed_relationships: list[dict[str, Any]] = []

        if entities_path:
            entities_path = Path(entities_path)
            if entities_path.exists():
                with open(entities_path, encoding="utf-8") as f:
                    entities_data = json.load(f)

                canonical_entities = entities_data.get("entities", {})

                # Load seed relationships
                relationships = entities_data.get("relationships", {})
                for rel_type, rels in relationships.items():
                    seed_relationships.extend(rels)

        return cls(
            entity_types=entity_types,
            predicates=predicates,
            canonical_entities=canonical_entities,
            seed_relationships=seed_relationships,
        )

    @classmethod
    def load_default(cls) -> "OntologySchema":
        """Load the default D&D 5E ontology schema.

        Returns:
            OntologySchema with default D&D 5E schema.
        """
        # Find the data directory relative to this file
        this_dir = Path(__file__).parent
        data_dir = this_dir.parent.parent.parent / "data" / "ontology"

        schema_path = data_dir / "dnd5e_schema.json"
        entities_path = data_dir / "canonical_entities.json"

        if not schema_path.exists():
            logger.warning(
                f"Default schema not found at {schema_path}, using empty schema"
            )
            return cls(entity_types={}, predicates={})

        return cls.load_from_files(
            schema_path=schema_path,
            entities_path=entities_path if entities_path.exists() else None,
        )

    def is_valid_entity_type(self, entity_type: str) -> bool:
        """Check if an entity type is valid.

        Args:
            entity_type: The entity type to check.

        Returns:
            True if valid, False otherwise.
        """
        return entity_type in self.entity_types

    def is_valid_predicate(self, predicate: str) -> bool:
        """Check if a predicate is valid.

        Args:
            predicate: The predicate to check.

        Returns:
            True if valid, False otherwise.
        """
        return predicate in self.predicates

    def get_predicate_constraint(
        self, predicate: str
    ) -> PredicateConstraint | None:
        """Get the constraint for a predicate.

        Args:
            predicate: The predicate name.

        Returns:
            PredicateConstraint if found, None otherwise.
        """
        return self.predicates.get(predicate)

    def validate_triple(self, triple: SemanticTriple) -> ValidationResult:
        """Validate a semantic triple against the ontology.

        Checks:
        1. Subject type is valid
        2. Object type is valid
        3. Predicate is valid
        4. Subject type is allowed for this predicate
        5. Object type is allowed for this predicate

        Args:
            triple: The triple to validate.

        Returns:
            ValidationResult with validity and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check entity types
        if not self.is_valid_entity_type(triple.subject_type):
            errors.append(f"Invalid subject type: {triple.subject_type}")

        if not self.is_valid_entity_type(triple.object_type):
            errors.append(f"Invalid object type: {triple.object_type}")

        # Check predicate
        if not self.is_valid_predicate(triple.predicate):
            errors.append(f"Invalid predicate: {triple.predicate}")
        else:
            constraint = self.get_predicate_constraint(triple.predicate)
            if constraint:
                # Check source type
                if triple.subject_type not in constraint.valid_sources:
                    errors.append(
                        f"Subject type '{triple.subject_type}' not valid for "
                        f"predicate '{triple.predicate}'. "
                        f"Valid sources: {constraint.valid_sources}"
                    )

                # Check target type
                if triple.object_type not in constraint.valid_targets:
                    errors.append(
                        f"Object type '{triple.object_type}' not valid for "
                        f"predicate '{triple.predicate}'. "
                        f"Valid targets: {constraint.valid_targets}"
                    )

        # Check confidence
        if triple.confidence < 0.5:
            warnings.append(
                f"Low confidence ({triple.confidence:.2f}) for triple: "
                f"{triple.subject} -> {triple.predicate} -> {triple.object}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_triples(
        self, triples: list[SemanticTriple]
    ) -> tuple[list[SemanticTriple], list[tuple[SemanticTriple, ValidationResult]]]:
        """Validate multiple triples, separating valid from invalid.

        Args:
            triples: List of triples to validate.

        Returns:
            Tuple of (valid_triples, invalid_triples_with_results).
        """
        valid: list[SemanticTriple] = []
        invalid: list[tuple[SemanticTriple, ValidationResult]] = []

        for triple in triples:
            result = self.validate_triple(triple)
            if result.is_valid:
                valid.append(triple)
            else:
                invalid.append((triple, result))

        logger.info(
            "Triple validation complete",
            valid=len(valid),
            invalid=len(invalid),
        )

        return valid, invalid

    def resolve_alias(self, name: str) -> tuple[str, str] | None:
        """Resolve an alias to its canonical name and type.

        Args:
            name: The entity name or alias to resolve.

        Returns:
            Tuple of (canonical_name, entity_type) if found, None otherwise.
        """
        return self._alias_to_canonical.get(name.lower())

    def get_canonical_name(
        self, name: str, entity_type: str | None = None
    ) -> str:
        """Get the canonical name for an entity.

        If the name is found as an alias, returns the canonical name.
        Otherwise, returns the original name (title-cased).

        Args:
            name: The entity name to canonicalize.
            entity_type: Optional entity type hint.

        Returns:
            The canonical name.
        """
        resolved = self.resolve_alias(name)
        if resolved:
            return resolved[0]

        # Not found - normalize to title case
        return name.title().strip()

    def get_canonical_entity(
        self, entity_type: str, name: str
    ) -> dict[str, Any] | None:
        """Get canonical entity data.

        Args:
            entity_type: The entity type.
            name: The entity name or alias.

        Returns:
            Entity data dict if found, None otherwise.
        """
        entities = self.canonical_entities.get(entity_type, {})

        # Try exact match first
        if name in entities:
            return entities[name]

        # Try alias resolution
        resolved = self.resolve_alias(name)
        if resolved and resolved[1] == entity_type:
            return entities.get(resolved[0])

        return None

    def get_entity_types_list(self) -> list[str]:
        """Get list of all valid entity types.

        Returns:
            List of entity type names.
        """
        return list(self.entity_types.keys())

    def get_predicates_list(self) -> list[str]:
        """Get list of all valid predicates.

        Returns:
            List of predicate names.
        """
        return list(self.predicates.keys())

    def get_seed_relationships(self) -> list[SemanticTriple]:
        """Get seed relationships as SemanticTriple objects.

        Returns:
            List of SemanticTriple objects from seed data.
        """
        triples: list[SemanticTriple] = []

        for rel in self.seed_relationships:
            # Resolve entity types
            subject_resolved = self.resolve_alias(rel["subject"])
            object_resolved = self.resolve_alias(rel["object"])

            subject_type = subject_resolved[1] if subject_resolved else "Unknown"
            object_type = object_resolved[1] if object_resolved else "Unknown"

            triple = SemanticTriple(
                subject=rel["subject"],
                subject_type=subject_type,
                predicate=rel["predicate"],
                object=rel["object"],
                object_type=object_type,
                metadata=rel.get("metadata", {}),
                confidence=1.0,  # Seed data is always high confidence
            )
            triples.append(triple)

        return triples

    def to_prompt_format(self) -> str:
        """Format the ontology for LLM prompts.

        Returns:
            String representation suitable for LLM prompts.
        """
        lines = ["## ONTOLOGY CONSTRAINTS", ""]

        # Entity types
        lines.append("### Valid Entity Types:")
        for entity_type in sorted(self.entity_types.keys()):
            examples = self.entity_types[entity_type].get("examples", [])
            example_str = f" (e.g., {', '.join(examples[:3])})" if examples else ""
            lines.append(f"- {entity_type}{example_str}")
        lines.append("")

        # Predicates
        lines.append("### Valid Predicates:")
        for pred_name, constraint in sorted(self.predicates.items()):
            sources = ", ".join(constraint.valid_sources[:3])
            targets = ", ".join(constraint.valid_targets[:3])
            lines.append(f"- {pred_name}: [{sources}] -> [{targets}]")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "EntityType",
    "Predicate",
    "PredicateConstraint",
    "SemanticTriple",
    "ValidationResult",
    "OntologySchema",
]
