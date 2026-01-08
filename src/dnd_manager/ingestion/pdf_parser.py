"""PDF document parsing for the RAG pipeline.

This module provides functionality for parsing PDF documents into
text chunks suitable for embedding and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dnd_manager.core.exceptions import PDFParseError
from dnd_manager.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text extracted from a document.

    Attributes:
        content: The text content of the chunk.
        source_file: Path to the source file.
        page_number: Page number where the chunk originated.
        chunk_index: Index of this chunk within the document.
        metadata: Additional metadata about the chunk.
    """

    content: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


class PDFParser:
    """Parse PDF documents into text chunks.

    This class handles PDF parsing using PyMuPDF4LLM for optimal
    text extraction, including table and structure preservation.

    Attributes:
        chunk_size: Target size for text chunks.
        chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the PDF parser.

        Args:
            chunk_size: Target size for text chunks in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Raises:
            PDFParseError: If chunk_overlap >= chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise PDFParseError(
                "chunk_overlap must be less than chunk_size",
                details={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "PDFParser initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def parse(self, file_path: Path | str) -> list[TextChunk]:
        """Parse a PDF file into text chunks.

        Args:
            file_path: Path to the PDF file to parse.

        Returns:
            List of TextChunk objects extracted from the PDF.

        Raises:
            PDFParseError: If the file cannot be parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise PDFParseError(
                f"PDF file not found: {file_path}",
                source_file=str(file_path),
            )

        if not file_path.suffix.lower() == ".pdf":
            raise PDFParseError(
                f"File is not a PDF: {file_path}",
                source_file=str(file_path),
            )

        logger.info("Parsing PDF", file_path=str(file_path))

        try:
            # Lazy import to avoid loading PyMuPDF until needed
            import pymupdf4llm

            # Extract markdown-formatted text from PDF
            md_text = pymupdf4llm.to_markdown(str(file_path))

            # Split into chunks
            chunks = list(self._create_chunks(md_text, str(file_path)))

            logger.info(
                "PDF parsed successfully",
                file_path=str(file_path),
                chunk_count=len(chunks),
            )

            return chunks

        except ImportError as exc:
            raise PDFParseError(
                "pymupdf4llm is not installed. Install with: pip install pymupdf4llm",
                source_file=str(file_path),
            ) from exc
        except Exception as exc:
            raise PDFParseError(
                f"Failed to parse PDF: {exc}",
                source_file=str(file_path),
                details={"error_type": type(exc).__name__},
            ) from exc

    def _create_chunks(
        self,
        text: str,
        source_file: str,
    ) -> Iterator[TextChunk]:
        """Split text into overlapping chunks.

        Args:
            text: The full text to split.
            source_file: Path to the source file.

        Yields:
            TextChunk objects.
        """
        if not text.strip():
            return

        # Split on paragraph boundaries when possible
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_index = 0
        page_number = 1  # PyMuPDF4LLM doesn't preserve page numbers in markdown

        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                if current_chunk:
                    yield TextChunk(
                        content=current_chunk.strip(),
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1

                    # Keep overlap from end of previous chunk
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Yield final chunk
        if current_chunk.strip():
            yield TextChunk(
                content=current_chunk.strip(),
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
            )


def parse_pdf(
    file_path: Path | str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Convenience function to parse a PDF file.

    Args:
        file_path: Path to the PDF file.
        chunk_size: Target size for text chunks.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of TextChunk objects.

    Raises:
        PDFParseError: If parsing fails.
    """
    parser = PDFParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser.parse(file_path)


__all__ = [
    "TextChunk",
    "PDFParser",
    "parse_pdf",
]
