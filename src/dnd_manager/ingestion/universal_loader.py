"""Universal document loader with vision pipeline.

This module provides a unified ingestion system for all D&D content:
- Character sheets (via vision AI - NO OCR)
- Rulebooks (via markdown extraction + RAG)
- Adventure modules (via header-based chunking)

NEURO-SYMBOLIC PRINCIPLE:
Vision AI extracts structured data into Pydantic models.
The Python engine validates and owns the resulting state.
LLMs cannot invent stats - they must extract from the document.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from dnd_manager.core.config import get_settings
from dnd_manager.core.exceptions import IngestionError, PDFParseError
from dnd_manager.core.logging import get_logger
from dnd_manager.models.ecs import (
    ActorEntity,
    ActorType,
    ClassFeatureComponent,
    ClassLevel,
    DefenseComponent,
    HealthComponent,
    InventoryComponent,
    ItemStack,
    ItemType,
    JournalComponent,
    MovementComponent,
    SpellbookComponent,
    StatsComponent,
)


logger = get_logger(__name__)


# =============================================================================
# Document Types
# =============================================================================


class DocumentType(StrEnum):
    """Types of documents that can be ingested."""

    CHARACTER_SHEET = "character_sheet"
    SOURCEBOOK = "sourcebook"  # Official WotC books (PHB, XGE, MM, etc.)
    ADVENTURE = "adventure"     # Campaign modules (Strahd, Dragon Heist)
    THIRD_PARTY = "third_party" # DMs Guild, homebrew, non-WotC
    UNKNOWN = "unknown"


@dataclass
class ChunkedDocument:
    """A document chunk for RAG storage."""

    chunk_id: str
    content: str
    source: str
    doc_type: DocumentType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate."""
        return len(self.content) // 4


# =============================================================================
# OpenRouter Client
# =============================================================================


class OpenRouterClient:
    """OpenRouter API client using openai SDK.
    
    Configured for vision models (Gemini) and reasoning models (Claude).
    """

    def __init__(
        self,
        api_key: str | None = None,
        vision_model: str = "google/gemini-2.0-flash-001",
        reasoning_model: str = "google/gemini-3-pro-preview",
    ) -> None:
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, reads from settings.
            vision_model: Model for vision/extraction tasks.
            reasoning_model: Model for reasoning/DM tasks.
        """
        self.vision_model = vision_model
        self.reasoning_model = reasoning_model
        self._client: Any = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Get or create the OpenAI client configured for OpenRouter."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise IngestionError(
                    "openai package not installed",
                    details={"package": "openai"},
                ) from exc

            api_key = self._api_key
            if api_key is None:
                settings = get_settings()
                key = settings.ai.openai_api_key
                if key:
                    api_key = key.get_secret_value()

            if not api_key:
                raise IngestionError(
                    "OpenRouter API key not configured",
                    details={"env_var": "DND_MANAGER_OPENAI_API_KEY"},
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/dungeon-ai/campaign-manager",
                    "X-Title": "DungeonAI Campaign Manager",
                },
            )

        return self._client

    def vision_extract(
        self,
        images_base64: list[str],
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Send images to vision model for extraction.
        
        Args:
            images_base64: List of base64-encoded images.
            system_prompt: System instructions.
            user_prompt: User request.
            json_schema: Optional JSON schema for structured output.
            temperature: Sampling temperature.
            
        Returns:
            Model response text.
        """
        from openai import APIConnectionError, APIStatusError, RateLimitError
        from tenacity import retry, stop_after_attempt, wait_exponential

        client = self._get_client()

        # Build content with images
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        for img_b64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",
                },
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        )
        def _call() -> str:
            try:
                response = client.chat.completions.create(
                    model=self.vision_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4096,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as exc:
                logger.warning("Rate limited, retrying...")
                raise
            except APIConnectionError as exc:
                raise IngestionError(
                    f"Failed to connect to OpenRouter: {exc}",
                    details={"model": self.vision_model},
                ) from exc
            except APIStatusError as exc:
                raise IngestionError(
                    f"OpenRouter API error: {exc}",
                    details={"status_code": exc.status_code},
                ) from exc

        return _call()

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request.
        
        Args:
            messages: Chat messages.
            model: Model to use (defaults to reasoning_model).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            
        Returns:
            Model response text.
        """
        client = self._get_client()
        model = model or self.reasoning_model

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""


# =============================================================================
# PDF to Image Conversion (Vision Pipeline)
# =============================================================================


def pdf_to_images(
    pdf_source: str | bytes | Path,
    dpi: int = 200,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[bytes]:
    """Convert PDF pages to PNG images.
    
    Uses PyMuPDF (fitz) for high-quality rendering.
    NO OCR - we send images directly to vision AI.
    
    Args:
        pdf_source: Path to PDF or raw bytes.
        dpi: Resolution for rendering.
        first_page: First page to convert (1-indexed, default 1).
        last_page: Last page to convert (1-indexed, default all).
        
    Returns:
        List of PNG image bytes.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise IngestionError(
            "PyMuPDF not installed. Install with: pip install pymupdf",
            details={"package": "pymupdf"},
        ) from exc

    try:
        # Open PDF
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            path = Path(pdf_source)
            if not path.exists():
                raise PDFParseError(f"PDF file not found: {path}")
            doc = fitz.open(str(path))

        # Calculate page range (convert 1-indexed to 0-indexed)
        start_page = (first_page - 1) if first_page else 0
        end_page = last_page if last_page else len(doc)
        
        # Validate range
        start_page = max(0, min(start_page, len(doc) - 1))
        end_page = max(start_page + 1, min(end_page, len(doc)))

        # Calculate zoom factor for desired DPI (default PDF is 72 DPI)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Convert pages to images
        images: list[bytes] = []
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))

        doc.close()

        logger.info(f"Converted PDF to {len(images)} images at {dpi} DPI")
        return images

    except Exception as exc:
        error_msg = str(exc).lower()
        if "password" in error_msg:
            raise PDFParseError("PDF is password-protected") from exc
        elif "corrupt" in error_msg or "invalid" in error_msg:
            raise PDFParseError(f"Invalid or corrupted PDF: {exc}") from exc
        else:
            raise PDFParseError(f"PDF conversion failed: {exc}") from exc


def images_to_base64(images: list[bytes]) -> list[str]:
    """Convert image bytes to base64 strings."""
    return [base64.b64encode(img).decode("utf-8") for img in images]


# =============================================================================
# Character Sheet Extraction (Vision Pipeline)
# =============================================================================


# System prompt for character extraction
CHARACTER_EXTRACTION_PROMPT = """You are an expert D&D 5E character sheet parser. Extract ALL data from this character sheet image(s) into the exact JSON structure provided.

CRITICAL RULES:
1. Extract ONLY what you can see. Do not invent or assume any values.
2. If a field is not visible or unclear, use null.
3. Ability scores MUST be integers between 1 and 30.
4. HP values must be positive integers.
5. Level must be between 1 and 20.
6. Include ALL visible spells, items, and features.

OUTPUT FORMAT (JSON only, no markdown):
{
  "name": "Character Name",
  "race": "Race",
  "classes": [{"class_name": "Fighter", "level": 5, "subclass": "Champion"}],
  "stats": {
    "strength": 16,
    "dexterity": 14,
    "constitution": 15,
    "intelligence": 10,
    "wisdom": 12,
    "charisma": 8
  },
  "hp_current": 45,
  "hp_max": 45,
  "armor_class": 18,
  "proficiency_bonus": 3,
  "speed": 30,
  "saving_throw_proficiencies": ["strength", "constitution"],
  "skill_proficiencies": {
    "athletics": 1,
    "perception": 1,
    "intimidation": 2
  },
  "inventory": [
    {"name": "Longsword", "quantity": 1},
    {"name": "Shield", "quantity": 1}
  ],
  "spells_known": ["Shield", "Magic Missile"],
  "cantrips": ["Fire Bolt"],
  "spell_slots": {"1": 4, "2": 3},
  "features": ["Second Wind", "Action Surge"],
  "personality_traits": ["I am fearless"],
  "ideals": ["Honor"],
  "bonds": ["My sword belonged to my father"],
  "flaws": ["I am reckless"],
  "background": "Soldier",
  "alignment": "Lawful Good"
}"""


class CharacterExtractor:
    """Extract character data from PDF character sheets using vision AI."""

    def __init__(
        self, 
        client: OpenRouterClient | None = None,
        chroma_store: "ChromaStore | None" = None,
    ) -> None:
        """Initialize extractor with OpenRouter client."""
        self.client = client or OpenRouterClient()
        self.chroma_store = chroma_store  # Optional, for RAG item lookup

    def extract(
        self,
        pdf_source: str | bytes | Path,
        max_pages: int = 2,
    ) -> ActorEntity:
        """Extract character data from a PDF character sheet.
        
        Uses vision AI - NO OCR. The PDF is converted to images
        and sent to Gemini for extraction.
        
        Args:
            pdf_source: Path to PDF or raw bytes.
            max_pages: Maximum pages to process (first N).
            
        Returns:
            ActorEntity populated with extracted data.
        """
        logger.info("Starting character extraction via vision pipeline")

        # Step 1: Convert PDF to images
        images = pdf_to_images(pdf_source, dpi=200, last_page=max_pages)
        images_b64 = images_to_base64(images)

        logger.info(f"Sending {len(images_b64)} page(s) to vision model")

        # Step 2: Send to vision model
        response = self.client.vision_extract(
            images_base64=images_b64,
            system_prompt=CHARACTER_EXTRACTION_PROMPT,
            user_prompt="Extract the complete character data from this D&D 5E character sheet. Output valid JSON only.",
            temperature=0.1,
        )

        # Step 3: Parse JSON response
        data = self._parse_json_response(response)

        # Step 4: Convert to ActorEntity
        return self._build_actor_entity(data)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from model response."""
        # Strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```json and ```)
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

        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"Failed to parse character data JSON: {exc}",
                details={"response_preview": response[:500]},
            ) from exc

    def _build_actor_entity(self, data: dict[str, Any]) -> ActorEntity:
        """Build ActorEntity from extracted data."""
        try:
            # Stats
            stats_data = data.get("stats", {})
            stats = StatsComponent(
                strength=stats_data.get("strength", 10),
                dexterity=stats_data.get("dexterity", 10),
                constitution=stats_data.get("constitution", 10),
                intelligence=stats_data.get("intelligence", 10),
                wisdom=stats_data.get("wisdom", 10),
                charisma=stats_data.get("charisma", 10),
                proficiency_bonus=data.get("proficiency_bonus", 2),
                save_proficiencies=[
                    p for p in data.get("saving_throw_proficiencies", [])
                ],
                skill_proficiencies=data.get("skill_proficiencies", {}),
            )

            # Health
            hp_max = data.get("hp_max", 10)
            hp_current = data.get("hp_current", hp_max)
            health = HealthComponent(
                hp_current=hp_current,
                hp_max=hp_max,
            )

            # Defense - will be updated after inventory is created
            ac_from_sheet = data.get("armor_class", 10)

            # Classes
            classes_data = data.get("classes", [])
            if not classes_data:
                # Fallback for simple extraction
                classes_data = [{"class_name": "Fighter", "level": 1}]

            class_features = ClassFeatureComponent(
                classes=[
                    ClassLevel(
                        class_name=c.get("class_name", "Fighter"),
                        level=c.get("level", 1),
                        subclass=c.get("subclass"),
                    )
                    for c in classes_data
                ],
                features=data.get("features", []),
            )

            # Inventory - resolve items to get proper stats
            from dnd_manager.models.equipment import resolve_item_name, resolve_item_from_rag
            
            inventory_items = []
            equipped_weapon = None
            equipped_armor = None
            
            for item_data in data.get("inventory", []):
                item_name = item_data.get("name", "Unknown")
                quantity = item_data.get("quantity", 1)
                
                # Try to resolve the item
                item = resolve_item_name(item_name)
                
                # If unknown, try RAG lookup
                if item and item.item_type == ItemType.GEAR and "Unknown" in item.description:
                    if self.chroma_store:
                        rag_item = resolve_item_from_rag(item_name, self.chroma_store)
                        if rag_item:
                            item = rag_item
                
                if item:
                    item.quantity = quantity
                    
                    # Auto-equip first weapon and armor found
                    if item.item_type == ItemType.WEAPON and equipped_weapon is None:
                        item.equipped = True
                        equipped_weapon = item.name
                    elif item.item_type in (ItemType.ARMOR, ItemType.SHIELD) and equipped_armor is None:
                        item.equipped = True
                        equipped_armor = item.name
                    
                    inventory_items.append(item)
            
            inventory = InventoryComponent(items=inventory_items)
            
            # Calculate defense from equipped armor
            ac_base = 10
            max_dex_bonus = None
            uses_dex = True
            shield_bonus = 0
            
            for item in inventory_items:
                if not item.equipped:
                    continue
                if item.item_type == ItemType.ARMOR and item.ac_base is not None:
                    ac_base = item.ac_base
                    max_dex_bonus = item.max_dex_bonus
                    uses_dex = max_dex_bonus is None or max_dex_bonus > 0
                elif item.item_type == ItemType.SHIELD and item.ac_bonus is not None:
                    shield_bonus = item.ac_bonus
            
            # If we found armor, use calculated AC; otherwise use sheet AC
            if any(i.equipped and i.item_type == ItemType.ARMOR for i in inventory_items):
                defense = DefenseComponent(
                    ac_base=ac_base + shield_bonus,
                    uses_dex=uses_dex,
                    max_dex_bonus=max_dex_bonus,
                )
            else:
                # Use AC from character sheet (may include unarmored defense, etc.)
                defense = DefenseComponent(ac_base=ac_from_sheet, uses_dex=False)

            # Spellbook (if spells found)
            spellbook = None
            if data.get("spells_known") or data.get("cantrips"):
                # Determine spellcasting ability from class
                primary_class = classes_data[0].get("class_name", "").lower() if classes_data else ""
                spell_ability = {
                    "wizard": "intelligence",
                    "sorcerer": "charisma",
                    "warlock": "charisma",
                    "bard": "charisma",
                    "cleric": "wisdom",
                    "druid": "wisdom",
                    "paladin": "charisma",
                    "ranger": "wisdom",
                }.get(primary_class, "intelligence")

                spell_slots = {
                    int(k): (v, v) for k, v in data.get("spell_slots", {}).items()
                }

                spellbook = SpellbookComponent(
                    spellcasting_ability=spell_ability,
                    spell_save_dc=data.get("spell_save_dc", 8 + stats.proficiency_bonus),
                    spell_attack_bonus=data.get("spell_attack_bonus", stats.proficiency_bonus),
                    spells_known=data.get("spells_known", []),
                    cantrips=data.get("cantrips", []),
                    spell_slots=spell_slots,
                )

            # Journal/Personality
            journal = JournalComponent(
                personality_traits=data.get("personality_traits", []),
                ideals=data.get("ideals", []),
                bonds=data.get("bonds", []),
                flaws=data.get("flaws", []),
            )

            # Movement
            speed = data.get("speed", 30)
            movement = MovementComponent(
                speed_walk=speed,
                movement_remaining=speed,
            )

            # Build entity
            entity = ActorEntity(
                name=data.get("name", "Unknown Hero"),
                type=ActorType.PLAYER,
                race=data.get("race", "Unknown"),
                alignment=data.get("alignment", "Neutral"),
                stats=stats,
                health=health,
                defense=defense,
                inventory=inventory,
                spellbook=spellbook,
                journal=journal,
                movement=movement,
                class_features=class_features,
            )

            logger.info(f"Successfully extracted character: {entity.name}")
            return entity

        except Exception as exc:
            raise IngestionError(
                f"Failed to build character entity: {exc}",
                details={"data_keys": list(data.keys())},
            ) from exc


# =============================================================================
# Rulebook/Module Ingestion (Markdown + RAG)
# =============================================================================


def pdf_to_markdown(pdf_path: str | Path) -> str:
    """Convert PDF to Markdown using pymupdf4llm.
    
    Preserves headers, tables, and formatting for RAG.
    
    Args:
        pdf_path: Path to PDF file.
        
    Returns:
        Markdown text.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise IngestionError(
            "pymupdf4llm not installed. Install with: pip install pymupdf4llm",
            details={"package": "pymupdf4llm"},
        ) from exc

    path = Path(pdf_path)
    if not path.exists():
        raise PDFParseError(f"PDF file not found: {path}")

    try:
        markdown = pymupdf4llm.to_markdown(str(path))
        logger.info(f"Converted PDF to markdown: {len(markdown)} characters")
        return markdown
    except Exception as exc:
        raise PDFParseError(f"PDF to markdown conversion failed: {exc}") from exc


def chunk_by_headers(
    markdown: str,
    source: str,
    doc_type: DocumentType,
    max_chunk_size: int = 1000,
) -> list[ChunkedDocument]:
    """Chunk markdown by headers for RAG storage.
    
    Uses LangChain's MarkdownHeaderTextSplitter for semantic chunking.
    
    Args:
        markdown: Markdown text to chunk.
        source: Source document name.
        doc_type: Type of document.
        max_chunk_size: Maximum chunk size in characters.
        
    Returns:
        List of ChunkedDocument objects.
    """
    try:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
    except ImportError as exc:
        raise IngestionError(
            "langchain-text-splitters not installed",
            details={"package": "langchain-text-splitters"},
        ) from exc

    # Global chunk counter for unique IDs
    global_chunk_idx = 0
    
    # Split by headers first
    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,
    )

    header_splits = header_splitter.split_text(markdown)

    # Further split large chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[ChunkedDocument] = []

    for doc in header_splits:
        content = doc.page_content
        metadata = doc.metadata

        # Build title from headers
        title_parts = []
        for h in ["h1", "h2", "h3", "h4"]:
            if h in metadata:
                title_parts.append(metadata[h])
        title = " > ".join(title_parts) if title_parts else "Untitled"

        # Split if too large
        if len(content) > max_chunk_size:
            sub_splits = text_splitter.split_text(content)
            for i, sub in enumerate(sub_splits):
                # Use content hash for truly unique IDs
                chunk_id = hashlib.sha256(f"{source}:{global_chunk_idx}:{sub[:100]}".encode()).hexdigest()[:16]
                global_chunk_idx += 1
                chunks.append(ChunkedDocument(
                    chunk_id=chunk_id,
                    content=sub,
                    source=source,
                    doc_type=doc_type,
                    metadata={
                        "title": title,
                        "chunk_index": i,
                        **metadata,
                    },
                ))
        else:
            # Use content hash for truly unique IDs
            chunk_id = hashlib.sha256(f"{source}:{global_chunk_idx}:{content[:100]}".encode()).hexdigest()[:16]
            global_chunk_idx += 1
            chunks.append(ChunkedDocument(
                chunk_id=chunk_id,
                content=content,
                source=source,
                doc_type=doc_type,
                metadata={
                    "title": title,
                    **metadata,
                },
            ))

    logger.info(f"Created {len(chunks)} chunks from {source}")
    return chunks


# =============================================================================
# ChromaDB Vector Store
# =============================================================================


def get_openai_embedding_function(api_key: str | None = None):
    """Get ChromaDB's OpenAI embedding function configured for OpenRouter.
    
    Uses chromadb's built-in OpenAIEmbeddingFunction which properly
    implements the required interface.
    """
    try:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    except ImportError:
        logger.warning("Could not import OpenAIEmbeddingFunction, using default embeddings")
        return None
    
    if api_key is None:
        settings = get_settings()
        key = settings.ai.openai_api_key
        if key:
            api_key = key.get_secret_value()
    
    if not api_key:
        logger.warning("No API key for embeddings, using default")
        return None
    
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        model_name="openai/text-embedding-3-small",
    )


class ChromaStore:
    """ChromaDB vector store for RAG retrieval.
    
    Provides local persistence and semantic search over
    ingested documents (rulebooks, modules, etc.).
    """

    def __init__(
        self,
        collection_name: str = "dungeon_ai",
        persist_directory: str | Path | None = None,
        use_openrouter_embeddings: bool = True,
    ) -> None:
        """Initialize ChromaDB store.
        
        Args:
            collection_name: Name of the collection.
            persist_directory: Path for persistent storage.
            use_openrouter_embeddings: Use OpenRouter for embeddings (recommended).
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_openrouter_embeddings = use_openrouter_embeddings
        self._client: Any = None
        self._collection: Any = None
        self._embedding_fn: Any = None

    def _get_collection(self) -> Any:
        """Get or create the ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as exc:
            raise IngestionError(
                "chromadb not installed. Install with: pip install chromadb",
                details={"package": "chromadb"},
            ) from exc

        if self.persist_directory:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
            )
        else:
            self._client = chromadb.Client()

        # Use OpenRouter embeddings to avoid ONNX issues
        embedding_function = None
        if self.use_openrouter_embeddings:
            self._embedding_fn = get_openai_embedding_function()
            embedding_function = self._embedding_fn

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )

        logger.info(f"ChromaDB collection '{self.collection_name}' ready")
        return self._collection

    def add_documents(self, documents: list[ChunkedDocument]) -> int:
        """Add documents to the vector store.
        
        Args:
            documents: Documents to add.
            
        Returns:
            Number of documents added.
        """
        collection = self._get_collection()

        ids = [doc.chunk_id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [
            {
                "source": doc.source,
                "doc_type": doc.doc_type.value,
                **doc.metadata,
            }
            for doc in documents
        ]

        # Add in batches to avoid memory issues
        batch_size = 100
        added = 0

        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_contents = contents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            collection.upsert(
                ids=batch_ids,
                documents=batch_contents,
                metadatas=batch_metadatas,
            )
            added += len(batch_ids)

        logger.info(f"Added {added} documents to ChromaDB")
        return added

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_type: DocumentType | None = None,
        source_filter: str | None = None,
    ) -> list[ChunkedDocument]:
        """Search for relevant documents.
        
        Args:
            query: Search query.
            n_results: Maximum results to return.
            doc_type: Filter by document type.
            source_filter: Filter by source name.
            
        Returns:
            List of matching ChunkedDocument objects.
        """
        collection = self._get_collection()

        # Build where filter
        where_filter = {}
        if doc_type:
            where_filter["doc_type"] = doc_type.value
        if source_filter:
            where_filter["source"] = source_filter

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None,
        )

        # Convert to ChunkedDocument objects
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, content in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                chunk_id = results["ids"][0][i] if results["ids"] else str(uuid4())

                # Map old doc_type values to new ones
                raw_doc_type = metadata.get("doc_type", "unknown")
                doc_type_map = {
                    "rulebook": "sourcebook",
                    "adventure_module": "adventure",
                    "spell_list": "sourcebook",
                    "monster_manual": "sourcebook",
                }
                mapped_type = doc_type_map.get(raw_doc_type, raw_doc_type)
                
                try:
                    doc_type_enum = DocumentType(mapped_type)
                except ValueError:
                    doc_type_enum = DocumentType.UNKNOWN
                
                documents.append(ChunkedDocument(
                    chunk_id=chunk_id,
                    content=content,
                    source=metadata.get("source", "unknown"),
                    doc_type=doc_type_enum,
                    metadata=metadata,
                ))

        logger.info(f"Found {len(documents)} results for query: {query[:50]}...")
        return documents

    def count(self) -> int:
        """Get total document count."""
        collection = self._get_collection()
        return collection.count()


# =============================================================================
# Universal Ingestor
# =============================================================================


class UniversalIngestor:
    """Universal document ingestion system.
    
    Handles all document types through a unified interface:
    - Character sheets → Vision pipeline → ActorEntity
    - Rulebooks → Markdown → ChromaDB
    - Adventure modules → Markdown → ChromaDB
    """

    def __init__(
        self,
        chroma_store: ChromaStore | None = None,
        openrouter_client: OpenRouterClient | None = None,
    ) -> None:
        """Initialize the universal ingestor.
        
        Args:
            chroma_store: ChromaDB store for RAG documents.
            openrouter_client: OpenRouter client for vision extraction.
        """
        self.chroma_store = chroma_store or ChromaStore()
        self.openrouter_client = openrouter_client or OpenRouterClient()
        self.character_extractor = CharacterExtractor(
            client=self.openrouter_client,
            chroma_store=self.chroma_store,
        )

    def detect_document_type(self, pdf_path: str | Path) -> DocumentType:
        """Detect the type of document based on content.
        
        Uses first page vision to classify the document.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            Detected document type.
        """
        # Quick heuristic based on filename
        filename = Path(pdf_path).stem.lower()

        if any(kw in filename for kw in ["character", "sheet", "pc"]):
            return DocumentType.CHARACTER_SHEET
        if any(kw in filename for kw in ["phb", "dmg", "xge", "tce", "rule", "spell", "monster", "mm"]):
            return DocumentType.SOURCEBOOK
        if any(kw in filename for kw in ["module", "adventure", "curse", "descent", "tyranny", "phandelver"]):
            return DocumentType.ADVENTURE

        return DocumentType.UNKNOWN

    def ingest_character_sheet(
        self,
        pdf_source: str | bytes | Path,
    ) -> ActorEntity:
        """Ingest a character sheet PDF via vision pipeline.
        
        Args:
            pdf_source: Path to PDF or raw bytes.
            
        Returns:
            Extracted ActorEntity.
        """
        return self.character_extractor.extract(pdf_source)

    def ingest_rulebook(
        self,
        pdf_path: str | Path,
        source_name: str | None = None,
    ) -> int:
        """Ingest a rulebook PDF into ChromaDB.
        
        Args:
            pdf_path: Path to PDF file.
            source_name: Name for the source (defaults to filename).
            
        Returns:
            Number of chunks added.
        """
        path = Path(pdf_path)
        source = source_name or path.stem

        # Convert to markdown
        markdown = pdf_to_markdown(path)

        # Chunk by headers
        chunks = chunk_by_headers(
            markdown,
            source=source,
            doc_type=DocumentType.SOURCEBOOK,
        )

        # Add to ChromaDB
        return self.chroma_store.add_documents(chunks)

    def ingest_adventure_module(
        self,
        pdf_path: str | Path,
        source_name: str | None = None,
    ) -> int:
        """Ingest an adventure module PDF into ChromaDB.
        
        Args:
            pdf_path: Path to PDF file.
            source_name: Name for the source (defaults to filename).
            
        Returns:
            Number of chunks added.
        """
        path = Path(pdf_path)
        source = source_name or path.stem

        # Convert to markdown
        markdown = pdf_to_markdown(path)

        # Chunk by headers (larger chunks for narrative)
        chunks = chunk_by_headers(
            markdown,
            source=source,
            doc_type=DocumentType.ADVENTURE,
            max_chunk_size=1500,  # Larger for narrative content
        )

        # Add to ChromaDB
        return self.chroma_store.add_documents(chunks)

    def ingest(
        self,
        pdf_source: str | bytes | Path,
        doc_type: DocumentType | None = None,
    ) -> ActorEntity | int:
        """Ingest a document automatically.
        
        Detects type if not provided and routes to appropriate handler.
        
        Args:
            pdf_source: Path to PDF or raw bytes.
            doc_type: Document type (auto-detected if None).
            
        Returns:
            ActorEntity for character sheets, chunk count otherwise.
        """
        # Detect type if not provided
        if doc_type is None:
            if isinstance(pdf_source, bytes):
                doc_type = DocumentType.CHARACTER_SHEET  # Assume bytes = uploaded sheet
            else:
                doc_type = self.detect_document_type(pdf_source)

        logger.info(f"Ingesting document as {doc_type.value}")

        if doc_type == DocumentType.CHARACTER_SHEET:
            return self.ingest_character_sheet(pdf_source)
        elif doc_type == DocumentType.SOURCEBOOK:
            return self.ingest_rulebook(pdf_source)
        elif doc_type == DocumentType.ADVENTURE:
            return self.ingest_adventure_module(pdf_source)
        elif doc_type == DocumentType.THIRD_PARTY:
            return self.ingest_rulebook(pdf_source)  # Treat like sourcebook
        else:
            # Unknown - try as sourcebook
            return self.ingest_rulebook(pdf_source)

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_type: DocumentType | None = None,
    ) -> list[ChunkedDocument]:
        """Search ingested documents.
        
        Args:
            query: Search query.
            n_results: Maximum results.
            doc_type: Filter by type.
            
        Returns:
            Matching documents.
        """
        return self.chroma_store.search(query, n_results, doc_type)


__all__ = [
    # Types
    "DocumentType",
    "ChunkedDocument",
    # Client
    "OpenRouterClient",
    # PDF Processing
    "pdf_to_images",
    "images_to_base64",
    "pdf_to_markdown",
    "chunk_by_headers",
    # Character Extraction
    "CharacterExtractor",
    "CHARACTER_EXTRACTION_PROMPT",
    # Vector Store
    "ChromaStore",
    # Universal Ingestor
    "UniversalIngestor",
]
