"""Rule book ingestion and chunking for RAG.

This module provides functionality to ingest D&D rule books, expansions,
and source materials, chunking them into semantically meaningful units
suitable for retrieval-augmented generation.

Chunking Strategy:
- Rule books are chunked by logical units (spells, feats, rules sections)
- Smaller chunk sizes for precise retrieval
- Metadata tagging for filtering (source, type, category)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from dnd_manager.core.exceptions import IngestionError, PDFParseError
from dnd_manager.core.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class DocumentType(StrEnum):
    """Types of D&D reference documents."""

    RULE = "rule"
    SPELL = "spell"
    ITEM = "item"
    MONSTER = "monster"
    FEAT = "feat"
    CLASS_FEATURE = "class_feature"
    RACE_TRAIT = "race_trait"
    CONDITION = "condition"
    BACKGROUND = "background"
    GENERAL = "general"


@dataclass
class Document:
    """A chunked document for the RAG knowledge base.

    Attributes:
        content: The text content of the chunk.
        doc_id: Unique identifier for this document.
        source: Source book or document name.
        doc_type: Type of content (spell, rule, item, etc.).
        title: Title or heading of the section.
        category: Additional categorization (e.g., spell school).
        page_number: Original page number if available.
        metadata: Additional metadata for filtering.
    """

    content: str
    doc_id: str
    source: str
    doc_type: DocumentType
    title: str = ""
    category: str = ""
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate doc_id if not provided."""
        if not self.doc_id:
            # Generate hash-based ID from content
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
            self.doc_id = f"{self.source}_{self.doc_type}_{content_hash}"


# =============================================================================
# Text Splitter
# =============================================================================


class SemanticTextSplitter:
    """Split text into semantically meaningful chunks.

    This splitter is optimized for D&D rule content, recognizing
    common patterns like spell blocks, feat descriptions, and rules.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the text splitter.

        Args:
            chunk_size: Target maximum chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            separators: List of separators to split on (in priority order).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n\n",  # Major section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            ", ",      # Clauses
            " ",       # Words
        ]

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of chunks.
        """
        if not text.strip():
            return []

        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # No more separators, force split at chunk_size
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on current separator
        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            test_chunk = current_chunk + separator + split if current_chunk else split

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single split is too large, recursively split it
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


# =============================================================================
# Rule Book Parser
# =============================================================================


class RuleBookParser:
    """Parse and chunk D&D rule books for RAG indexing.

    This parser converts PDF rule books to markdown, then chunks
    them into semantically meaningful units with appropriate metadata.
    """

    # Patterns for detecting content types
    SPELL_PATTERN = re.compile(
        r"^(?P<name>[A-Z][A-Za-z\s\'-]+)\n"
        r"(?P<level>\d+(?:st|nd|rd|th)-level\s+)?(?P<school>\w+)(?:\s+\(ritual\))?\n",
        re.MULTILINE
    )

    FEAT_PATTERN = re.compile(
        r"^(?P<name>[A-Z][A-Za-z\s\'-]+)\n"
        r"(?:Prerequisite:[^\n]+\n)?",
        re.MULTILINE
    )

    ITEM_PATTERN = re.compile(
        r"^(?P<name>[A-Z][A-Za-z\s\'-]+)\n"
        r"(?P<type>Weapon|Armor|Wondrous item|Ring|Rod|Staff|Wand|Potion|Scroll)",
        re.MULTILINE
    )

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialize the parser.

        Args:
            chunk_size: Target chunk size for splitting.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SemanticTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        logger.info(
            "RuleBookParser initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _extract_markdown_from_pdf(self, pdf_path: Path) -> str:
        """Extract markdown text from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Markdown-formatted text.

        Raises:
            PDFParseError: If extraction fails.
        """
        try:
            import pymupdf4llm

            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            logger.info(f"Extracted {len(md_text)} characters from PDF")
            return md_text

        except ImportError as exc:
            raise PDFParseError(
                "pymupdf4llm not installed. Install with: pip install pymupdf4llm",
                source_file=str(pdf_path),
            ) from exc
        except Exception as exc:
            raise PDFParseError(
                f"Failed to extract text from PDF: {exc}",
                source_file=str(pdf_path),
            ) from exc

    def _detect_document_type(self, content: str, title: str) -> DocumentType:
        """Detect the type of content based on patterns.

        Args:
            content: The text content.
            title: The section title.

        Returns:
            Detected DocumentType.
        """
        content_lower = content.lower()
        title_lower = title.lower()

        # Check for spells
        if self.SPELL_PATTERN.search(content):
            return DocumentType.SPELL
        if "casting time" in content_lower and "range" in content_lower:
            return DocumentType.SPELL

        # Check for items
        if self.ITEM_PATTERN.search(content):
            return DocumentType.ITEM
        if "attunement" in content_lower or "magic item" in title_lower:
            return DocumentType.ITEM

        # Check for feats
        if "feat" in title_lower or "prerequisite:" in content_lower:
            return DocumentType.FEAT

        # Check for monsters
        if "challenge" in content_lower and "hit points" in content_lower:
            return DocumentType.MONSTER

        # Check for conditions
        if any(cond in title_lower for cond in [
            "blinded", "charmed", "deafened", "frightened", "grappled",
            "incapacitated", "invisible", "paralyzed", "petrified",
            "poisoned", "prone", "restrained", "stunned", "unconscious"
        ]):
            return DocumentType.CONDITION

        # Check for class features
        if "class feature" in title_lower or "level feature" in content_lower:
            return DocumentType.CLASS_FEATURE

        # Check for racial traits
        if "racial trait" in title_lower or "ability score increase" in content_lower:
            return DocumentType.RACE_TRAIT

        # Check for backgrounds
        if "background" in title_lower and "proficiencies" in content_lower:
            return DocumentType.BACKGROUND

        # Default to rule/general
        return DocumentType.RULE

    def _extract_title(self, content: str) -> str:
        """Extract a title from content.

        Args:
            content: The text content.

        Returns:
            Extracted or generated title.
        """
        lines = content.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            # Remove markdown headers
            first_line = re.sub(r"^#+\s*", "", first_line)
            # Limit length
            if len(first_line) > 100:
                first_line = first_line[:100] + "..."
            return first_line
        return "Untitled Section"

    def _split_by_headers(self, markdown: str) -> list[tuple[str, str]]:
        """Split markdown by headers, preserving hierarchy.

        Args:
            markdown: Markdown text.

        Returns:
            List of (heading, content) tuples.
        """
        # Pattern to match markdown headers
        header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

        sections = []
        last_end = 0
        current_heading = ""

        for match in header_pattern.finditer(markdown):
            # Get content before this header
            if last_end < match.start():
                content = markdown[last_end:match.start()].strip()
                if content:
                    sections.append((current_heading, content))

            current_heading = match.group(2).strip()
            last_end = match.end()

        # Get remaining content
        if last_end < len(markdown):
            content = markdown[last_end:].strip()
            if content:
                sections.append((current_heading, content))

        return sections

    def parse(
        self,
        pdf_path: Path | str,
        *,
        source_name: str | None = None,
    ) -> list[Document]:
        """Parse a rule book PDF into Document objects.

        Args:
            pdf_path: Path to the PDF file.
            source_name: Name of the source (defaults to filename).

        Returns:
            List of Document objects.

        Raises:
            PDFParseError: If PDF processing fails.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise PDFParseError(
                f"PDF file not found: {pdf_path}",
                source_file=str(pdf_path),
            )

        source = source_name or pdf_path.stem

        logger.info(f"Parsing rule book: {source}")

        # Extract markdown
        markdown = self._extract_markdown_from_pdf(pdf_path)

        # Split by headers first
        header_sections = self._split_by_headers(markdown)

        documents: list[Document] = []

        for heading, content in header_sections:
            # Further chunk if content is too large
            if len(content) > self.chunk_size:
                chunks = self.splitter.split_text(content)
            else:
                chunks = [content] if content.strip() else []

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                # Detect document type
                doc_type = self._detect_document_type(chunk, heading)

                # Create title
                if heading:
                    title = heading
                    if len(chunks) > 1:
                        title = f"{heading} (Part {i + 1})"
                else:
                    title = self._extract_title(chunk)

                doc = Document(
                    content=chunk,
                    doc_id="",  # Will be auto-generated
                    source=source,
                    doc_type=doc_type,
                    title=title,
                    metadata={
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "has_heading": bool(heading),
                    },
                )
                documents.append(doc)

        logger.info(
            f"Parsed {len(documents)} documents from {source}",
            source=source,
            document_count=len(documents),
        )

        return documents


# =============================================================================
# Convenience Functions
# =============================================================================


def ingest_rule_book(
    pdf_path: str | Path,
    *,
    source_name: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Ingest a D&D rule book PDF into chunked documents.

    This is the main entry point for rule book ingestion.

    Args:
        pdf_path: Path to the PDF file.
        source_name: Name of the source book.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Document objects ready for indexing.

    Raises:
        PDFParseError: If PDF processing fails.

    Example:
        >>> documents = ingest_rule_book(
        ...     "players_handbook.pdf",
        ...     source_name="Player's Handbook",
        ... )
        >>> print(f"Ingested {len(documents)} document chunks")
    """
    parser = RuleBookParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser.parse(pdf_path, source_name=source_name)


def ingest_spell_list(
    pdf_path: str | Path,
    *,
    source_name: str | None = None,
) -> list[Document]:
    """Ingest a spell list or spell compendium.

    Uses smaller chunks optimized for individual spells.

    Args:
        pdf_path: Path to the PDF file.
        source_name: Name of the source.

    Returns:
        List of spell Document objects.
    """
    parser = RuleBookParser(chunk_size=600, chunk_overlap=50)
    documents = parser.parse(pdf_path, source_name=source_name)

    # Filter to spells only
    return [doc for doc in documents if doc.doc_type == DocumentType.SPELL]


__all__ = [
    "DocumentType",
    "Document",
    "SemanticTextSplitter",
    "RuleBookParser",
    "ingest_rule_book",
    "ingest_spell_list",
]
