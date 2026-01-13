"""LLM-Driven Semantic Triple Extraction.

This module implements Phase 2 of the knowledge graph pipeline:
Information Extraction (IE) using LLMs to decompose text chunks
into semantic triples conforming to the D&D 5E ontology.

NEURO-SYMBOLIC PRINCIPLE:
The LLM acts as a reasoning engine, not a free-form generator.
It is constrained by the ontology schema to output only valid
relationships. The Python engine validates all outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dnd_manager.core.config import get_settings
from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.ontology import OntologySchema, SemanticTriple


if TYPE_CHECKING:
    from dnd_manager.ingestion.universal_loader import ChunkedDocument

logger = get_logger(__name__)


# =============================================================================
# Extraction Prompts
# =============================================================================


SYSTEM_PROMPT = """You are a D&D 5E rules parser specializing in extracting structured relationships from rulebook text.

Your task is to decompose text into SEMANTIC TRIPLES: (Subject, Predicate, Object) relationships.

## CRITICAL RULES:
1. ONLY extract relationships that are EXPLICITLY stated in the text
2. DO NOT infer or hallucinate relationships
3. Use ONLY the entity types and predicates from the ontology below
4. If no valid relationships exist in the text, return an empty array
5. Use canonical names (e.g., "DEX" not "Dexterity saving throw")

{ontology_constraints}

## OUTPUT FORMAT (JSON array only, no markdown):
[
  {{"subject": "Fireball", "subject_type": "Spell", "predicate": "requires_save", "object": "DEX", "object_type": "Save", "confidence": 0.95}},
  {{"subject": "Fireball", "subject_type": "Spell", "predicate": "deals_damage", "object": "Fire", "object_type": "DamageType", "confidence": 0.95}}
]

## CONFIDENCE SCORING:
- 1.0: Explicitly stated word-for-word
- 0.9: Clearly implied by context
- 0.8: Reasonable inference from rules text
- 0.7 or below: Uncertain, may be incorrect

## EXAMPLES:

Input: "Fireball. 3rd-level evocation. A bright streak flashes from your pointing finger to a point you choose within range and then blossoms with a low roar into an explosion of flame. Each creature in a 20-foot-radius sphere centered on that point must make a Dexterity saving throw. A target takes 8d6 fire damage on a failed save, or half as much damage on a successful one."

Output:
[
  {{"subject": "Fireball", "subject_type": "Spell", "predicate": "requires_save", "object": "DEX", "object_type": "Save", "confidence": 1.0}},
  {{"subject": "Fireball", "subject_type": "Spell", "predicate": "deals_damage", "object": "Fire", "object_type": "DamageType", "confidence": 1.0}},
  {{"subject": "Fireball", "subject_type": "Spell", "predicate": "has_spell_school", "object": "Evocation", "object_type": "SpellSchool", "confidence": 1.0}},
  {{"subject": "Fireball", "subject_type": "Spell", "predicate": "has_spell_level", "object": "3", "object_type": "SpellLevel", "confidence": 1.0}}
]

Input: "Evasion. Beginning at 7th level, you can nimbly dodge out of the way of certain area effects, such as a red dragon's fiery breath or an ice storm spell. When you are subjected to an effect that allows you to make a Dexterity saving throw to take only half damage, you instead take no damage if you succeed on the saving throw, and only half damage if you fail."

Output:
[
  {{"subject": "Monk", "subject_type": "Class", "predicate": "grants_feature", "object": "Evasion", "object_type": "ClassFeature", "confidence": 0.9, "metadata": {{"level_requirement": 7}}}},
  {{"subject": "Evasion", "subject_type": "ClassFeature", "predicate": "overrules", "object": "DEX", "object_type": "Save", "confidence": 1.0, "metadata": {{"condition": "On successful save, take no damage instead of half"}}}}
]
"""


USER_PROMPT_TEMPLATE = """Extract semantic triples from the following D&D 5E text chunk.

Source: {source}
Title: {title}
Category: {category}

TEXT:
{content}

Return ONLY a JSON array of triples. If no valid relationships found, return [].
"""


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ExtractionResult:
    """Result of triple extraction from a chunk.

    Attributes:
        chunk_id: The source chunk ID.
        triples: Extracted semantic triples.
        raw_response: Raw LLM response for debugging.
        error: Any error that occurred during extraction.
        token_usage: Token counts for cost tracking.
    """

    chunk_id: str
    triples: list[SemanticTriple] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractionStats:
    """Statistics for batch extraction.

    Attributes:
        total_chunks: Number of chunks processed.
        successful_chunks: Number of successfully processed chunks.
        failed_chunks: Number of failed chunks.
        total_triples: Total triples extracted.
        valid_triples: Triples that passed validation.
        invalid_triples: Triples that failed validation.
        total_tokens: Total tokens used.
        estimated_cost: Estimated API cost in USD.
    """

    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    total_triples: int = 0
    valid_triples: int = 0
    invalid_triples: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


# =============================================================================
# Triple Extractor
# =============================================================================


class TripleExtractor:
    """Extracts semantic triples from text chunks using LLM.

    This class implements the Information Extraction phase of the
    knowledge graph pipeline. It uses a constrained LLM prompt to
    decompose text into valid ontology triples.

    Example:
        >>> schema = OntologySchema.load_default()
        >>> extractor = TripleExtractor(schema)
        >>> result = extractor.extract_from_chunk(chunk)
        >>> for triple in result.triples:
        ...     print(f"{triple.subject} -> {triple.predicate} -> {triple.object}")
    """

    def __init__(
        self,
        schema: OntologySchema,
        *,
        model: str = "google/gemini-2.0-flash-001",
        use_openrouter: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        validate_on_extract: bool = True,
    ) -> None:
        """Initialize the triple extractor.

        Args:
            schema: The ontology schema for validation.
            model: LLM model to use for extraction.
            use_openrouter: Whether to use OpenRouter API.
            temperature: Sampling temperature (low for deterministic output).
            max_tokens: Maximum response tokens.
            validate_on_extract: Whether to validate triples immediately.
        """
        self.schema = schema
        self.model = model
        self.use_openrouter = use_openrouter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.validate_on_extract = validate_on_extract
        self._client: Any = None

        # Build system prompt with ontology constraints
        self._system_prompt = SYSTEM_PROMPT.format(
            ontology_constraints=schema.to_prompt_format()
        )

        logger.info(
            "TripleExtractor initialized",
            model=model,
            validate_on_extract=validate_on_extract,
        )

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                settings = get_settings()

                if self.use_openrouter:
                    api_key = settings.ai.openrouter_api_key
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
                logger.error("openai package not installed")
                raise ImportError(
                    "openai package required for triple extraction"
                ) from exc

        return self._client

    def _parse_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into triple dictionaries.

        Args:
            response: Raw LLM response string.

        Returns:
            List of triple dictionaries.
        """
        response = response.strip()

        # Strip markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        if not response:
            return []

        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "triples" in data:
                return data["triples"]
            else:
                logger.warning(f"Unexpected response format: {type(data)}")
                return []
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON response: {exc}")
            return []

    def _dict_to_triple(
        self, data: dict[str, Any], chunk_id: str | None = None
    ) -> SemanticTriple | None:
        """Convert dictionary to SemanticTriple.

        Args:
            data: Dictionary with triple data.
            chunk_id: Source chunk ID.

        Returns:
            SemanticTriple if valid, None otherwise.
        """
        required_fields = [
            "subject",
            "subject_type",
            "predicate",
            "object",
            "object_type",
        ]

        for field_name in required_fields:
            if field_name not in data:
                logger.warning(f"Missing required field: {field_name}")
                return None

        return SemanticTriple(
            subject=data["subject"],
            subject_type=data["subject_type"],
            predicate=data["predicate"],
            object=data["object"],
            object_type=data["object_type"],
            metadata=data.get("metadata", {}),
            chunk_id=chunk_id,
            confidence=data.get("confidence", 0.8),
        )

    def extract_from_text(
        self,
        text: str,
        *,
        source: str = "unknown",
        title: str = "",
        category: str = "",
        chunk_id: str | None = None,
    ) -> ExtractionResult:
        """Extract triples from raw text.

        Args:
            text: The text to extract from.
            source: Source document name.
            title: Section title.
            category: Content category.
            chunk_id: Optional chunk ID for linking.

        Returns:
            ExtractionResult with extracted triples.
        """
        from openai import APIConnectionError, APIStatusError, RateLimitError

        result = ExtractionResult(chunk_id=chunk_id or "")

        user_prompt = USER_PROMPT_TEMPLATE.format(
            source=source,
            title=title,
            category=category,
            content=text[:3000],  # Limit content length
        )

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result.raw_response = response.choices[0].message.content or ""

            # Track token usage
            if response.usage:
                result.token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Parse response
            triple_dicts = self._parse_response(result.raw_response)

            # Convert to SemanticTriple objects
            for triple_dict in triple_dicts:
                triple = self._dict_to_triple(triple_dict, chunk_id)
                if triple:
                    # Validate if enabled
                    if self.validate_on_extract:
                        validation = self.schema.validate_triple(triple)
                        if validation.is_valid:
                            result.triples.append(triple)
                        else:
                            logger.debug(
                                f"Invalid triple rejected: {validation.errors}"
                            )
                    else:
                        result.triples.append(triple)

            logger.info(
                f"Extracted {len(result.triples)} triples from chunk",
                chunk_id=chunk_id,
                source=source,
            )

        except RateLimitError as exc:
            result.error = f"Rate limited: {exc}"
            logger.warning(result.error)
        except APIConnectionError as exc:
            result.error = f"Connection error: {exc}"
            logger.warning(result.error)
        except APIStatusError as exc:
            result.error = f"API error: {exc}"
            logger.warning(result.error)
        except Exception as exc:
            result.error = f"Extraction failed: {exc}"
            logger.exception("Triple extraction failed")

        return result

    def extract_from_chunk(
        self, chunk: "ChunkedDocument"
    ) -> ExtractionResult:
        """Extract triples from a ChunkedDocument.

        Args:
            chunk: The document chunk to process.

        Returns:
            ExtractionResult with extracted triples.
        """
        return self.extract_from_text(
            text=chunk.content,
            source=chunk.source,
            title=chunk.metadata.get("title", ""),
            category=chunk.metadata.get("category", ""),
            chunk_id=chunk.chunk_id,
        )

    def extract_batch(
        self,
        chunks: list["ChunkedDocument"],
        *,
        progress_callback: Any = None,
    ) -> tuple[list[SemanticTriple], ExtractionStats]:
        """Extract triples from multiple chunks.

        Args:
            chunks: List of document chunks to process.
            progress_callback: Optional callback for progress updates.
                              Called with (current, total, chunk_id).

        Returns:
            Tuple of (all_valid_triples, extraction_stats).
        """
        stats = ExtractionStats(total_chunks=len(chunks))
        all_triples: list[SemanticTriple] = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks), chunk.chunk_id)

            result = self.extract_from_chunk(chunk)

            if result.error:
                stats.failed_chunks += 1
            else:
                stats.successful_chunks += 1
                stats.total_triples += len(result.triples)
                all_triples.extend(result.triples)

            # Track tokens
            stats.total_tokens += result.token_usage.get("total_tokens", 0)

        # Validate all triples
        valid, invalid = self.schema.validate_triples(all_triples)
        stats.valid_triples = len(valid)
        stats.invalid_triples = len(invalid)

        # Estimate cost (rough approximation for Gemini Flash)
        # ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens
        stats.estimated_cost = stats.total_tokens * 0.0001 / 1000  # Conservative

        logger.info(
            "Batch extraction complete",
            total_chunks=stats.total_chunks,
            successful=stats.successful_chunks,
            failed=stats.failed_chunks,
            valid_triples=stats.valid_triples,
            invalid_triples=stats.invalid_triples,
            estimated_cost=f"${stats.estimated_cost:.4f}",
        )

        return valid, stats


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_triples_from_chunks(
    chunks: list["ChunkedDocument"],
    schema: OntologySchema | None = None,
    *,
    model: str = "google/gemini-2.0-flash-001",
    progress_callback: Any = None,
) -> tuple[list[SemanticTriple], ExtractionStats]:
    """Convenience function to extract triples from chunks.

    Args:
        chunks: Document chunks to process.
        schema: Ontology schema (loads default if None).
        model: LLM model to use.
        progress_callback: Optional progress callback.

    Returns:
        Tuple of (valid_triples, stats).
    """
    if schema is None:
        schema = OntologySchema.load_default()

    extractor = TripleExtractor(schema, model=model)
    return extractor.extract_batch(chunks, progress_callback=progress_callback)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "ExtractionResult",
    "ExtractionStats",
    "TripleExtractor",
    "extract_triples_from_chunks",
]
