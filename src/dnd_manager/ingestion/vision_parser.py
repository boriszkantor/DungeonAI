"""Vision-based character sheet extraction using OpenRouter.

This module provides functionality to extract D&D 5E character data from
scanned character sheets or PDF images using vision-capable LLMs via OpenRouter.

The extraction pipeline:
1. Convert PDF page to base64 image
2. Send to vision model with structured extraction prompt
3. Parse JSON response into PlayerCharacter model
4. Retry with error correction if validation fails
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dnd_manager.core.config import get_settings
from dnd_manager.core.exceptions import IngestionError, PDFParseError
from dnd_manager.core.logging import get_logger
from dnd_manager.models import (
    AutonomyLevel,
    ClassLevel,
    HealthComponent,
    PersonaComponent,
    PlayerCharacter,
    SkillsComponent,
    StatsComponent,
)


if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

logger = get_logger(__name__)

# =============================================================================
# System Prompts
# =============================================================================

CHARACTER_EXTRACTION_SYSTEM_PROMPT = """You are a D&D 5E character sheet data extractor. Your task is to analyze the provided character sheet image and extract all character information into a structured JSON format.

IMPORTANT: You must output ONLY valid JSON with no additional text, markdown formatting, or explanation.
IMPORTANT: Extract ALL page 2 content including backstory, allies, appearance details, etc.

Extract the following information into this exact JSON structure:

{
  "name": "Character name (string)",
  "race": "Character race (string)",
  "background": "Character background (string, e.g., 'Soldier', 'Noble')",
  "alignment": "Alignment as lowercase with underscores (e.g., 'lawful_good', 'chaotic_neutral', 'true_neutral')",
  "experience_points": 0,
  "classes": [
    {
      "class_name": "Class name (string)",
      "subclass": "Subclass name or null",
      "level": 1,
      "hit_die": 8
    }
  ],
  "stats": {
    "strength": 10,
    "dexterity": 10,
    "constitution": 10,
    "intelligence": 10,
    "wisdom": 10,
    "charisma": 10
  },
  "health": {
    "current_hp": 10,
    "max_hp": 10,
    "temp_hp": 0
  },
  "armor_class": 10,
  "speed": 30,
  "proficient_skills": ["List of proficient skill names in lowercase_with_underscores"],
  "expert_skills": ["List of expertise skill names"],
  "saving_throw_proficiencies": ["List of ability names: strength, dexterity, etc."],
  "languages": ["List of known languages"],
  "tool_proficiencies": ["List of tool proficiencies"],
  "weapon_proficiencies": ["List of weapon proficiencies"],
  "armor_proficiencies": ["List of armor proficiencies"],
  "features": ["List of class features, racial traits, and feats"],
  "known_spells": ["List of spell names"],
  "prepared_spells": ["List of prepared spell names"],
  "inventory": [
    {
      "name": "Item name",
      "quantity": 1,
      "equipped": false
    }
  ],
  "personality_traits": ["Personality trait strings"],
  "ideals": ["Ideal strings"],
  "bonds": ["Bond strings"],
  "flaws": ["Flaw strings"],
  "backstory": "Full character backstory text from page 2",
  "allies_and_organizations": "Allies, organizations, and faction affiliations from page 2",
  "appearance": "Physical appearance description from page 2",
  "age": "Character age",
  "height": "Character height",
  "weight": "Character weight",
  "eyes": "Eye color",
  "hair": "Hair color/style",
  "skin": "Skin tone",
  "treasure": "Additional treasure from page 2",
  "additional_features_traits": "Additional features and traits from page 2",
  "currency": {
    "copper": 0,
    "silver": 0,
    "electrum": 0,
    "gold": 0,
    "platinum": 0
  }
}

RULES:
1. Use snake_case for all keys and enum values
2. Ability scores must be integers between 1-30
3. Level must be between 1-20
4. Hit die must be 6, 8, 10, or 12
5. If a value is not visible or unclear, use empty string or null
6. For alignment, use exactly one of: lawful_good, neutral_good, chaotic_good, lawful_neutral, true_neutral, chaotic_neutral, lawful_evil, neutral_evil, chaotic_evil
7. For skills, use: athletics, acrobatics, sleight_of_hand, stealth, arcana, history, investigation, nature, religion, animal_handling, insight, medicine, perception, survival, deception, intimidation, performance, persuasion
8. Extract ALL text from page 2 fields like backstory, allies, and appearance - do not truncate

Output ONLY the JSON object, no other text."""

CORRECTION_PROMPT_TEMPLATE = """The previous JSON extraction had validation errors. Please fix the following issues and return corrected JSON:

VALIDATION ERRORS:
{errors}

PREVIOUS JSON:
{previous_json}

Return ONLY the corrected JSON with no additional text."""


# =============================================================================
# PDF to Image Conversion
# =============================================================================


def pdf_page_to_base64_image(
    pdf_bytes: bytes,
    page_number: int = 0,
    dpi: int = 150,
) -> str:
    """Convert a PDF page to a base64-encoded PNG image.

    Args:
        pdf_bytes: Raw PDF file bytes.
        page_number: Page number to convert (0-indexed).
        dpi: Resolution for rendering.

    Returns:
        Base64-encoded PNG image string.

    Raises:
        PDFParseError: If PDF cannot be processed.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if page_number >= len(doc):
            raise PDFParseError(
                f"Page {page_number} not found in PDF with {len(doc)} pages",
                page_number=page_number,
            )

        page = doc[page_number]
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale for DPI
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        png_bytes = pix.tobytes("png")

        # Encode to base64
        base64_image = base64.b64encode(png_bytes).decode("utf-8")

        doc.close()

        logger.debug(
            "PDF page converted to image",
            page_number=page_number,
            image_size=len(png_bytes),
        )

        return base64_image

    except ImportError as exc:
        raise PDFParseError(
            "PyMuPDF (fitz) not installed. Install with: pip install pymupdf"
        ) from exc
    except Exception as exc:
        raise PDFParseError(
            f"Failed to convert PDF page to image: {exc}",
            page_number=page_number,
        ) from exc


# =============================================================================
# OpenRouter Client
# =============================================================================


def get_openrouter_client() -> "OpenAI":
    """Get an OpenAI client configured for OpenRouter.

    Returns:
        Configured OpenAI client.

    Raises:
        IngestionError: If API key is not configured.
    """
    try:
        from openai import OpenAI

        settings = get_settings()

        # Try OpenAI key first (works with OpenRouter)
        api_key = None
        if settings.ai.openai_api_key:
            api_key = settings.ai.openai_api_key.get_secret_value()

        if not api_key:
            raise IngestionError(
                "OpenRouter API key not configured. Set DND_MANAGER_OPENAI_API_KEY"
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/dnd-campaign-manager",
                "X-Title": "D&D Campaign Manager",
            },
        )

        return client

    except ImportError as exc:
        raise IngestionError(
            "openai package not installed. Install with: pip install openai"
        ) from exc


# =============================================================================
# Vision Extraction
# =============================================================================


class CharacterSheetExtractor:
    """Extract character data from character sheet images using vision LLMs.

    This class handles the full extraction pipeline including:
    - PDF to image conversion
    - Vision model API calls via OpenRouter
    - JSON parsing and validation
    - Retry with error correction

    Attributes:
        client: OpenAI client configured for OpenRouter.
        model: Vision model to use for extraction.
        max_retries: Maximum correction attempts for validation errors.
    """

    # Supported vision models on OpenRouter
    SUPPORTED_MODELS = [
        "google/gemini-2.0-flash-001",
        "google/gemini-pro-vision",
        "openai/gpt-4o",
        "openai/gpt-4-vision-preview",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
    ]

    def __init__(
        self,
        *,
        model: str = "google/gemini-2.0-flash-001",
        max_retries: int = 3,
    ) -> None:
        """Initialize the character sheet extractor.

        Args:
            model: Vision model to use (OpenRouter model ID).
            max_retries: Maximum attempts to correct validation errors.
        """
        self.client = get_openrouter_client()
        self.model = model
        self.max_retries = max_retries

        logger.info(
            "CharacterSheetExtractor initialized",
            model=model,
            max_retries=max_retries,
        )

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_vision_api(
        self,
        base64_image: str,
        system_prompt: str,
        user_message: str | None = None,
    ) -> str:
        """Call the vision API with retry logic.

        Args:
            base64_image: Base64-encoded image.
            system_prompt: System prompt for extraction.
            user_message: Optional user message (for corrections).

        Returns:
            Raw response text from the model.

        Raises:
            IngestionError: If API call fails.
        """
        try:
            from openai import APIConnectionError, APIStatusError, RateLimitError

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
            ]

            # Build content with image
            content: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ]

            if user_message:
                content.append({"type": "text", "text": user_message})
            else:
                content.append({
                    "type": "text",
                    "text": "Extract all character data from this D&D 5E character sheet image.",
                })

            messages.append({"role": "user", "content": content})

            # Make API call
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,  # Low temperature for structured extraction
            )

            result = response.choices[0].message.content or ""

            logger.debug(
                "Vision API response received",
                model=self.model,
                response_length=len(result),
            )

            return result

        except RateLimitError as exc:
            raise IngestionError(
                f"OpenRouter rate limit exceeded: {exc}",
                details={"model": self.model, "error_type": "rate_limit"},
            ) from exc
        except APIConnectionError as exc:
            raise IngestionError(
                f"Failed to connect to OpenRouter: {exc}",
                details={"model": self.model, "error_type": "connection"},
            ) from exc
        except APIStatusError as exc:
            raise IngestionError(
                f"OpenRouter API error: {exc}",
                details={"model": self.model, "status_code": exc.status_code},
            ) from exc
        except Exception as exc:
            raise IngestionError(
                f"Vision API call failed: {exc}",
                details={"model": self.model},
            ) from exc

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from model response, handling markdown code blocks.

        Args:
            response: Raw response text.

        Returns:
            Parsed JSON dictionary.

        Raises:
            IngestionError: If JSON parsing fails.
        """
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            # Remove opening fence
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"Failed to parse JSON from model response: {exc}",
                details={"response_preview": text[:500]},
            ) from exc

    def _map_to_player_character(
        self,
        data: dict[str, Any],
    ) -> PlayerCharacter:
        """Map extracted JSON data to PlayerCharacter model.

        Args:
            data: Extracted character data dictionary.

        Returns:
            Validated PlayerCharacter instance.

        Raises:
            IngestionError: If validation fails.
        """
        from pydantic import ValidationError

        from dnd_manager.models.enums import Ability, Alignment, Skill

        try:
            # Build stats component
            stats_data = data.get("stats", {})
            stats = StatsComponent(
                strength=stats_data.get("strength", 10),
                dexterity=stats_data.get("dexterity", 10),
                constitution=stats_data.get("constitution", 10),
                intelligence=stats_data.get("intelligence", 10),
                wisdom=stats_data.get("wisdom", 10),
                charisma=stats_data.get("charisma", 10),
            )

            # Build health component
            health_data = data.get("health", {})
            max_hp = health_data.get("max_hp", 10)
            health = HealthComponent(
                current_hp=health_data.get("current_hp", max_hp),
                max_hp=max_hp,
                temp_hp=health_data.get("temp_hp", 0),
            )

            # Build persona component (player-controlled by default)
            persona = PersonaComponent(
                name=data.get("name", "Unknown"),
                autonomy=AutonomyLevel.NONE,
                biography=data.get("biography", ""),
                personality_traits=data.get("personality_traits", []),
                ideals=data.get("ideals", []),
                bonds=data.get("bonds", []),
                flaws=data.get("flaws", []),
            )

            # Build classes
            classes_data = data.get("classes", [{"class_name": "Fighter", "level": 1}])
            classes = []
            for cls in classes_data:
                classes.append(ClassLevel(
                    class_name=cls.get("class_name", "Fighter"),
                    subclass=cls.get("subclass"),
                    level=cls.get("level", 1),
                    hit_die=cls.get("hit_die", 8),
                ))

            # Build skills component
            proficient_skills = set()
            expert_skills = set()
            for skill_name in data.get("proficient_skills", []):
                try:
                    proficient_skills.add(Skill(skill_name))
                except ValueError:
                    logger.warning(f"Unknown skill: {skill_name}")

            for skill_name in data.get("expert_skills", []):
                try:
                    skill = Skill(skill_name)
                    proficient_skills.add(skill)  # Expertise requires proficiency
                    expert_skills.add(skill)
                except ValueError:
                    logger.warning(f"Unknown expertise skill: {skill_name}")

            skills = SkillsComponent(
                proficiencies=proficient_skills,
                expertise=expert_skills,
            )

            # Map alignment
            alignment_str = data.get("alignment", "true_neutral")
            try:
                alignment = Alignment(alignment_str)
            except ValueError:
                alignment = Alignment.TRUE_NEUTRAL

            # Build inventory
            from dnd_manager.models.entities import InventoryItem
            inventory = []
            for item_data in data.get("inventory", []):
                inventory.append(InventoryItem(
                    name=item_data.get("name", "Unknown Item"),
                    quantity=item_data.get("quantity", 1),
                    equipped=item_data.get("equipped", False),
                ))

            # Create PlayerCharacter
            character = PlayerCharacter(
                name=data.get("name", "Unknown Character"),
                race=data.get("race", "Human"),
                background=data.get("background", ""),
                alignment=alignment,
                experience_points=data.get("experience_points", 0),
                stats=stats,
                health=health,
                persona=persona,
                classes=classes,
                skills=skills,
                inventory=inventory,
                languages=data.get("languages", ["Common"]),
                tool_proficiencies=data.get("tool_proficiencies", []),
                weapon_proficiencies=data.get("weapon_proficiencies", []),
                armor_proficiencies=data.get("armor_proficiencies", []),
                features=data.get("features", []),
                known_spells=data.get("known_spells", []),
                prepared_spells=data.get("prepared_spells", []),
            )

            logger.info(
                "Character extracted successfully",
                name=character.name,
                level=character.total_level,
                race=character.race,
            )

            return character

        except ValidationError as exc:
            # Re-raise with details for correction
            raise IngestionError(
                "Character data validation failed",
                details={
                    "validation_errors": str(exc),
                    "extracted_data": data,
                },
            ) from exc

    def extract_from_image(
        self,
        base64_image: str,
    ) -> PlayerCharacter:
        """Extract character data from a base64-encoded image.

        Args:
            base64_image: Base64-encoded character sheet image.

        Returns:
            Extracted and validated PlayerCharacter.

        Raises:
            IngestionError: If extraction or validation fails.
        """
        last_json: dict[str, Any] = {}
        last_error: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    # Initial extraction
                    response = self._call_vision_api(
                        base64_image,
                        CHARACTER_EXTRACTION_SYSTEM_PROMPT,
                    )
                else:
                    # Correction attempt
                    logger.info(
                        f"Attempting correction (attempt {attempt}/{self.max_retries})"
                    )
                    correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(
                        errors=last_error,
                        previous_json=json.dumps(last_json, indent=2),
                    )
                    response = self._call_vision_api(
                        base64_image,
                        CHARACTER_EXTRACTION_SYSTEM_PROMPT,
                        user_message=correction_prompt,
                    )

                # Parse JSON
                data = self._parse_json_response(response)
                last_json = data

                # Map to PlayerCharacter
                return self._map_to_player_character(data)

            except IngestionError as exc:
                if "validation" in str(exc).lower() and attempt < self.max_retries:
                    last_error = str(exc.details.get("validation_errors", str(exc)))
                    logger.warning(
                        "Validation failed, will retry",
                        attempt=attempt,
                        error=last_error[:200],
                    )
                    continue
                raise

        raise IngestionError(
            f"Failed to extract valid character after {self.max_retries} attempts",
            details={"last_json": last_json, "last_error": last_error},
        )

    def extract_from_pdf(
        self,
        pdf_bytes: bytes,
        page_number: int = 0,
    ) -> PlayerCharacter:
        """Extract character data from a PDF file.

        Args:
            pdf_bytes: Raw PDF file bytes.
            page_number: Page to extract from (0-indexed).

        Returns:
            Extracted and validated PlayerCharacter.

        Raises:
            PDFParseError: If PDF conversion fails.
            IngestionError: If extraction fails.
        """
        logger.info("Extracting character from PDF", page_number=page_number)

        # Convert PDF page to image
        base64_image = pdf_page_to_base64_image(pdf_bytes, page_number)

        # Extract character data
        return self.extract_from_image(base64_image)


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_character_from_pdf(
    pdf_bytes: bytes,
    *,
    model: str = "google/gemini-2.0-flash-001",
    page_number: int = 0,
) -> PlayerCharacter:
    """Extract a D&D 5E character from a PDF character sheet.

    This is the main entry point for character sheet extraction.

    Args:
        pdf_bytes: Raw PDF file bytes.
        model: Vision model to use for extraction.
        page_number: Page number to extract from (0-indexed).

    Returns:
        Extracted and validated PlayerCharacter instance.

    Raises:
        PDFParseError: If PDF cannot be processed.
        IngestionError: If extraction fails.

    Example:
        >>> with open("character_sheet.pdf", "rb") as f:
        ...     pdf_bytes = f.read()
        >>> character = extract_character_from_pdf(pdf_bytes)
        >>> print(f"{character.name} - Level {character.total_level}")
    """
    extractor = CharacterSheetExtractor(model=model)
    return extractor.extract_from_pdf(pdf_bytes, page_number=page_number)


def extract_character_from_image_file(
    image_path: Path | str,
    *,
    model: str = "google/gemini-2.0-flash-001",
) -> PlayerCharacter:
    """Extract a D&D 5E character from an image file.

    Args:
        image_path: Path to the image file.
        model: Vision model to use for extraction.

    Returns:
        Extracted and validated PlayerCharacter instance.

    Raises:
        IngestionError: If extraction fails.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise IngestionError(f"Image file not found: {image_path}")

    # Read and encode image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    extractor = CharacterSheetExtractor(model=model)
    return extractor.extract_from_image(base64_image)


__all__ = [
    "CharacterSheetExtractor",
    "extract_character_from_pdf",
    "extract_character_from_image_file",
    "pdf_page_to_base64_image",
    "CHARACTER_EXTRACTION_SYSTEM_PROMPT",
]
