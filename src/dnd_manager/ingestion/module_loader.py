"""Adventure module ingestion for scene extraction.

This module provides functionality to ingest D&D adventure modules,
extracting scenes and narrative content while preserving the story flow.

Chunking Strategy:
- Split on markdown headers to preserve narrative structure
- Larger chunks than rule books for context continuity
- Scene detection based on common adventure module patterns
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from dnd_manager.core.exceptions import IngestionError, PDFParseError
from dnd_manager.core.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class SceneCategory(StrEnum):
    """Categories of adventure scenes."""

    COMBAT = "combat"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    PUZZLE = "puzzle"
    TRAP = "trap"
    NARRATIVE = "narrative"
    INTRODUCTION = "introduction"
    CONCLUSION = "conclusion"
    LOCATION = "location"
    NPC = "npc"
    TREASURE = "treasure"


@dataclass
class ModuleScene:
    """A scene extracted from an adventure module.

    Represents a discrete scene or location from an adventure module,
    preserving narrative content and structure.

    Attributes:
        uid: Unique identifier for this scene.
        title: Scene title (from header or generated).
        content: Full scene content/description.
        read_aloud: Boxed/read-aloud text if detected.
        source_module: Name of the source adventure module.
        chapter: Chapter or section name.
        category: Type of scene (combat, social, etc.).
        location: Location name if detected.
        npcs_mentioned: NPCs mentioned in the scene.
        monsters_mentioned: Monsters mentioned in the scene.
        items_mentioned: Items or treasure mentioned.
        page_reference: Original page reference if available.
        metadata: Additional metadata.
    """

    title: str
    content: str
    source_module: str
    uid: UUID = field(default_factory=uuid4)
    read_aloud: str = ""
    chapter: str = ""
    category: SceneCategory = SceneCategory.NARRATIVE
    location: str = ""
    npcs_mentioned: list[str] = field(default_factory=list)
    monsters_mentioned: list[str] = field(default_factory=list)
    items_mentioned: list[str] = field(default_factory=list)
    page_reference: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scene_id(self) -> str:
        """Generate a readable scene ID."""
        title_slug = re.sub(r"[^a-z0-9]+", "_", self.title.lower())[:30]
        return f"{self.source_module}_{title_slug}_{str(self.uid)[:8]}"


# =============================================================================
# Header-Based Text Splitter
# =============================================================================


class MarkdownHeaderSplitter:
    """Split markdown text by headers while preserving hierarchy.

    This splitter recognizes markdown headers (#, ##, ###, ####) and
    splits content at appropriate boundaries for adventure modules.
    """

    # Header pattern with levels
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        *,
        split_on_levels: list[int] | None = None,
        max_chunk_size: int = 2000,
        include_header_in_content: bool = True,
    ) -> None:
        """Initialize the header splitter.

        Args:
            split_on_levels: Header levels to split on (default [1, 2, 3]).
            max_chunk_size: Maximum chunk size before forcing a split.
            include_header_in_content: Include the header in chunk content.
        """
        self.split_on_levels = split_on_levels or [1, 2, 3]
        self.max_chunk_size = max_chunk_size
        self.include_header_in_content = include_header_in_content

    def split(self, markdown: str) -> list[dict[str, Any]]:
        """Split markdown by headers.

        Args:
            markdown: Markdown text to split.

        Returns:
            List of dicts with 'level', 'header', 'content', 'parent_headers'.
        """
        sections: list[dict[str, Any]] = []
        header_stack: list[tuple[int, str]] = []  # (level, header_text)

        # Find all headers
        header_matches = list(self.HEADER_PATTERN.finditer(markdown))

        for i, match in enumerate(header_matches):
            level = len(match.group(1))
            header = match.group(2).strip()

            # Determine content range
            start = match.end()
            if i + 1 < len(header_matches):
                end = header_matches[i + 1].start()
            else:
                end = len(markdown)

            content = markdown[start:end].strip()

            # Update header stack for hierarchy
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            parent_headers = [h for _, h in header_stack]
            header_stack.append((level, header))

            # Only create sections for specified levels
            if level in self.split_on_levels:
                if self.include_header_in_content:
                    full_content = f"{'#' * level} {header}\n\n{content}"
                else:
                    full_content = content

                sections.append({
                    "level": level,
                    "header": header,
                    "content": full_content,
                    "parent_headers": parent_headers.copy(),
                })

        # Handle content before first header
        if header_matches:
            pre_content = markdown[:header_matches[0].start()].strip()
            if pre_content:
                sections.insert(0, {
                    "level": 0,
                    "header": "Introduction",
                    "content": pre_content,
                    "parent_headers": [],
                })

        return sections


# =============================================================================
# Adventure Module Parser
# =============================================================================


class AdventureModuleParser:
    """Parse adventure modules into scene objects.

    This parser extracts scenes from adventure module PDFs,
    detecting read-aloud text, NPCs, monsters, and categorizing content.
    """

    # Pattern for read-aloud/boxed text (common conventions)
    READ_ALOUD_PATTERNS = [
        re.compile(r">\s*(.+?)(?=\n[^>]|\Z)", re.DOTALL),  # Blockquotes
        re.compile(r"\*\*Read Aloud:\*\*\s*(.+?)(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE),
        re.compile(r"_([^_]+)_", re.DOTALL),  # Italics (sometimes used for read-aloud)
    ]

    # Common monster names (subset for detection)
    COMMON_MONSTERS = {
        "goblin", "orc", "skeleton", "zombie", "dragon", "giant", "troll",
        "ogre", "wolf", "spider", "rat", "bat", "snake", "owlbear",
        "beholder", "mind flayer", "lich", "vampire", "werewolf", "demon",
        "devil", "elemental", "golem", "mimic", "doppelganger", "bandit",
        "cultist", "guard", "knight", "mage", "priest", "assassin",
    }

    # Patterns for detecting scene types
    COMBAT_INDICATORS = [
        "attack", "initiative", "combat", "fight", "battle", "ambush",
        "hostile", "enemy", "enemies", "creature attacks",
    ]
    SOCIAL_INDICATORS = [
        "negotiate", "persuade", "convince", "talk to", "speak with",
        "roleplay", "conversation", "dialogue", "npc",
    ]
    PUZZLE_INDICATORS = [
        "puzzle", "riddle", "mechanism", "solve", "combination", "code",
        "unlock", "decipher",
    ]
    TRAP_INDICATORS = [
        "trap", "trigger", "dc", "saving throw", "damage on", "dexterity save",
    ]

    def __init__(
        self,
        *,
        max_scene_size: int = 3000,
        detect_read_aloud: bool = True,
    ) -> None:
        """Initialize the parser.

        Args:
            max_scene_size: Maximum scene content size.
            detect_read_aloud: Whether to detect and extract read-aloud text.
        """
        self.max_scene_size = max_scene_size
        self.detect_read_aloud = detect_read_aloud
        self.header_splitter = MarkdownHeaderSplitter(
            split_on_levels=[1, 2, 3],
            max_chunk_size=max_scene_size,
        )

        logger.info(
            "AdventureModuleParser initialized",
            max_scene_size=max_scene_size,
        )

    def _extract_markdown_from_pdf(self, pdf_path: Path) -> str:
        """Extract markdown from PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Markdown text.
        """
        try:
            import pymupdf4llm

            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            logger.info(f"Extracted {len(md_text)} characters from module PDF")
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

    def _extract_read_aloud(self, content: str) -> tuple[str, str]:
        """Extract read-aloud text from content.

        Args:
            content: Scene content.

        Returns:
            Tuple of (read_aloud_text, remaining_content).
        """
        if not self.detect_read_aloud:
            return "", content

        read_aloud_parts = []

        for pattern in self.READ_ALOUD_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                read_aloud_parts.extend(matches)

        if read_aloud_parts:
            # Take the first substantial read-aloud section
            read_aloud = max(read_aloud_parts, key=len)
            return read_aloud.strip(), content

        return "", content

    def _detect_monsters(self, content: str) -> list[str]:
        """Detect monster names in content.

        Args:
            content: Scene content.

        Returns:
            List of detected monster names.
        """
        content_lower = content.lower()
        found = []

        for monster in self.COMMON_MONSTERS:
            # Check for monster name (case-insensitive, word boundary)
            pattern = rf"\b{re.escape(monster)}s?\b"
            if re.search(pattern, content_lower):
                found.append(monster.title())

        return found

    def _detect_npcs(self, content: str) -> list[str]:
        """Detect NPC names in content.

        Args:
            content: Scene content.

        Returns:
            List of potential NPC names.
        """
        # Pattern for capitalized names (simple heuristic)
        name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")

        # Find potential names
        matches = name_pattern.findall(content)

        # Filter out common words and very short names
        common_words = {
            "The", "This", "That", "These", "Those", "When", "Where", "What",
            "Who", "How", "Then", "They", "Their", "There", "Here", "Chapter",
            "Part", "Section", "Area", "Room", "Level", "Dungeon", "Tower",
        }

        npcs = []
        for name in matches:
            if name not in common_words and len(name) > 3:
                npcs.append(name)

        # Deduplicate while preserving order
        seen = set()
        unique_npcs = []
        for npc in npcs:
            if npc not in seen:
                seen.add(npc)
                unique_npcs.append(npc)

        return unique_npcs[:10]  # Limit to top 10

    def _detect_items(self, content: str) -> list[str]:
        """Detect magic items or treasure in content.

        Args:
            content: Scene content.

        Returns:
            List of detected items.
        """
        items = []

        # Pattern for items (italicized or specific patterns)
        item_patterns = [
            re.compile(r"\*([^*]+(?:sword|staff|ring|amulet|potion|scroll|wand|rod|armor|shield|cloak|boots|helm|gauntlets)[^*]*)\*", re.IGNORECASE),
            re.compile(r"\+\d\s+(\w+(?:\s+\w+)?)", re.IGNORECASE),  # +1 sword, etc.
        ]

        for pattern in item_patterns:
            matches = pattern.findall(content)
            items.extend(matches)

        # Also check for treasure mentions
        treasure_pattern = re.compile(r"(\d+)\s*(gp|sp|cp|pp|gold|silver|copper|platinum)", re.IGNORECASE)
        treasure_matches = treasure_pattern.findall(content)
        for amount, currency in treasure_matches:
            items.append(f"{amount} {currency}")

        return items[:10]  # Limit

    def _categorize_scene(self, content: str, title: str) -> SceneCategory:
        """Categorize a scene based on content analysis.

        Args:
            content: Scene content.
            title: Scene title.

        Returns:
            Scene category.
        """
        content_lower = content.lower()
        title_lower = title.lower()

        # Check for introduction/conclusion
        if any(word in title_lower for word in ["introduction", "beginning", "prologue"]):
            return SceneCategory.INTRODUCTION
        if any(word in title_lower for word in ["conclusion", "ending", "epilogue"]):
            return SceneCategory.CONCLUSION

        # Check for specific area/location
        if any(word in title_lower for word in ["room", "area", "chamber", "hall", "cave"]):
            return SceneCategory.LOCATION

        # Check for NPC focus
        if "npc" in title_lower or "character" in title_lower:
            return SceneCategory.NPC

        # Check content indicators
        combat_score = sum(1 for ind in self.COMBAT_INDICATORS if ind in content_lower)
        social_score = sum(1 for ind in self.SOCIAL_INDICATORS if ind in content_lower)
        puzzle_score = sum(1 for ind in self.PUZZLE_INDICATORS if ind in content_lower)
        trap_score = sum(1 for ind in self.TRAP_INDICATORS if ind in content_lower)

        scores = {
            SceneCategory.COMBAT: combat_score,
            SceneCategory.SOCIAL: social_score,
            SceneCategory.PUZZLE: puzzle_score,
            SceneCategory.TRAP: trap_score,
        }

        max_category = max(scores, key=lambda k: scores[k])
        if scores[max_category] >= 2:
            return max_category

        # Check for treasure
        if "treasure" in content_lower or "gold" in content_lower:
            return SceneCategory.TREASURE

        # Default to exploration or narrative
        if any(word in content_lower for word in ["explore", "search", "investigate"]):
            return SceneCategory.EXPLORATION

        return SceneCategory.NARRATIVE

    def parse(
        self,
        pdf_path: Path | str,
        *,
        module_name: str | None = None,
    ) -> list[ModuleScene]:
        """Parse an adventure module PDF into scenes.

        Args:
            pdf_path: Path to the PDF file.
            module_name: Name of the adventure module.

        Returns:
            List of ModuleScene objects.

        Raises:
            PDFParseError: If PDF processing fails.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise PDFParseError(
                f"PDF file not found: {pdf_path}",
                source_file=str(pdf_path),
            )

        source = module_name or pdf_path.stem

        logger.info(f"Parsing adventure module: {source}")

        # Extract markdown
        markdown = self._extract_markdown_from_pdf(pdf_path)

        # Split by headers
        sections = self.header_splitter.split(markdown)

        scenes: list[ModuleScene] = []

        for section in sections:
            header = section["header"]
            content = section["content"]
            parent_headers = section["parent_headers"]

            if not content.strip():
                continue

            # Skip very small sections
            if len(content) < 50:
                continue

            # Extract read-aloud text
            read_aloud, _ = self._extract_read_aloud(content)

            # Detect entities
            monsters = self._detect_monsters(content)
            npcs = self._detect_npcs(content)
            items = self._detect_items(content)

            # Categorize scene
            category = self._categorize_scene(content, header)

            # Determine chapter from parent headers
            chapter = parent_headers[0] if parent_headers else ""

            scene = ModuleScene(
                title=header,
                content=content,
                source_module=source,
                read_aloud=read_aloud,
                chapter=chapter,
                category=category,
                npcs_mentioned=npcs,
                monsters_mentioned=monsters,
                items_mentioned=items,
                metadata={
                    "level": section["level"],
                    "parent_headers": parent_headers,
                },
            )
            scenes.append(scene)

        logger.info(
            f"Parsed {len(scenes)} scenes from {source}",
            source=source,
            scene_count=len(scenes),
        )

        return scenes


# =============================================================================
# Convenience Functions
# =============================================================================


def ingest_adventure_module(
    pdf_path: str | Path,
    *,
    module_name: str | None = None,
    max_scene_size: int = 3000,
) -> list[ModuleScene]:
    """Ingest a D&D adventure module PDF into scenes.

    This is the main entry point for adventure module ingestion.

    Args:
        pdf_path: Path to the PDF file.
        module_name: Name of the adventure module.
        max_scene_size: Maximum scene content size.

    Returns:
        List of ModuleScene objects.

    Raises:
        PDFParseError: If PDF processing fails.

    Example:
        >>> scenes = ingest_adventure_module(
        ...     "lost_mine_of_phandelver.pdf",
        ...     module_name="Lost Mine of Phandelver",
        ... )
        >>> for scene in scenes:
        ...     print(f"{scene.title}: {scene.category}")
    """
    parser = AdventureModuleParser(max_scene_size=max_scene_size)
    return parser.parse(pdf_path, module_name=module_name)


def extract_combat_encounters(scenes: list[ModuleScene]) -> list[ModuleScene]:
    """Filter scenes to only combat encounters.

    Args:
        scenes: List of all scenes.

    Returns:
        List of combat scenes only.
    """
    return [s for s in scenes if s.category == SceneCategory.COMBAT]


def extract_npcs(scenes: list[ModuleScene]) -> dict[str, list[str]]:
    """Extract all NPCs mentioned across scenes.

    Args:
        scenes: List of scenes.

    Returns:
        Dict mapping NPC names to scenes they appear in.
    """
    npc_scenes: dict[str, list[str]] = {}

    for scene in scenes:
        for npc in scene.npcs_mentioned:
            if npc not in npc_scenes:
                npc_scenes[npc] = []
            npc_scenes[npc].append(scene.title)

    return npc_scenes


__all__ = [
    "SceneCategory",
    "ModuleScene",
    "MarkdownHeaderSplitter",
    "AdventureModuleParser",
    "ingest_adventure_module",
    "extract_combat_encounters",
    "extract_npcs",
]
