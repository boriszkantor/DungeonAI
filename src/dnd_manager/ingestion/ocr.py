"""OCR processing for image-based content.

This module provides OCR functionality for extracting text from
images, scanned documents, and character sheets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dnd_manager.core.exceptions import OCRError
from dnd_manager.core.logging import get_logger


logger = get_logger(__name__)

SupportedImageFormat = Literal["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"]

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"]
)


@dataclass(frozen=True)
class OCRResult:
    """Result of OCR processing.

    Attributes:
        text: Extracted text content.
        confidence: Overall confidence score (0.0-1.0).
        source_file: Path to the source image.
        language: Detected or specified language.
    """

    text: str
    confidence: float
    source_file: str
    language: str = "eng"


class OCRProcessor:
    """Process images with Optical Character Recognition.

    This class provides OCR functionality using various backends,
    with support for D&D-specific content like character sheets
    and rulebook scans.

    Attributes:
        language: OCR language code.
        backend: OCR backend to use ('tesseract' or 'easyocr').
    """

    def __init__(
        self,
        *,
        language: str = "eng",
        backend: Literal["tesseract", "easyocr"] = "tesseract",
    ) -> None:
        """Initialize the OCR processor.

        Args:
            language: OCR language code (e.g., 'eng', 'deu').
            backend: OCR backend to use.
        """
        self.language = language
        self.backend = backend
        logger.info(
            "OCRProcessor initialized",
            language=language,
            backend=backend,
        )

    def process(self, image_path: Path | str) -> OCRResult:
        """Process an image with OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult containing extracted text and metadata.

        Raises:
            OCRError: If OCR processing fails.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise OCRError(
                f"Image file not found: {image_path}",
                source_file=str(image_path),
            )

        suffix = image_path.suffix.lower().lstrip(".")
        if suffix not in SUPPORTED_FORMATS:
            raise OCRError(
                f"Unsupported image format: {suffix}",
                source_file=str(image_path),
                details={"supported_formats": list(SUPPORTED_FORMATS)},
            )

        logger.info("Processing image with OCR", image_path=str(image_path))

        if self.backend == "tesseract":
            return self._process_tesseract(image_path)
        elif self.backend == "easyocr":
            return self._process_easyocr(image_path)
        else:
            raise OCRError(
                f"Unknown OCR backend: {self.backend}",
                source_file=str(image_path),
            )

    def _process_tesseract(self, image_path: Path) -> OCRResult:
        """Process image using Tesseract OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult from Tesseract processing.

        Raises:
            OCRError: If Tesseract is not available or processing fails.
        """
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.language)

            # Get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [c for c in data["conf"] if c > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.info(
                "Tesseract OCR completed",
                image_path=str(image_path),
                confidence=avg_confidence / 100,
            )

            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100,
                source_file=str(image_path),
                language=self.language,
            )

        except ImportError as exc:
            raise OCRError(
                "pytesseract or PIL not installed. Install with: "
                "pip install pytesseract pillow",
                source_file=str(image_path),
            ) from exc
        except Exception as exc:
            raise OCRError(
                f"Tesseract OCR failed: {exc}",
                source_file=str(image_path),
                details={"error_type": type(exc).__name__},
            ) from exc

    def _process_easyocr(self, image_path: Path) -> OCRResult:
        """Process image using EasyOCR.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult from EasyOCR processing.

        Raises:
            OCRError: If EasyOCR is not available or processing fails.
        """
        try:
            import easyocr

            reader = easyocr.Reader([self.language])
            results = reader.readtext(str(image_path))

            # Combine text and calculate average confidence
            texts = []
            confidences = []
            for _, text, conf in results:
                texts.append(text)
                confidences.append(conf)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.info(
                "EasyOCR completed",
                image_path=str(image_path),
                confidence=avg_confidence,
            )

            return OCRResult(
                text=combined_text.strip(),
                confidence=avg_confidence,
                source_file=str(image_path),
                language=self.language,
            )

        except ImportError as exc:
            raise OCRError(
                "easyocr not installed. Install with: pip install easyocr",
                source_file=str(image_path),
            ) from exc
        except Exception as exc:
            raise OCRError(
                f"EasyOCR failed: {exc}",
                source_file=str(image_path),
                details={"error_type": type(exc).__name__},
            ) from exc


__all__ = [
    "OCRResult",
    "OCRProcessor",
    "SUPPORTED_FORMATS",
]
