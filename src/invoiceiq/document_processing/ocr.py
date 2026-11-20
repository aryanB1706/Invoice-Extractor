"""Tesseract OCR helpers."""

from pathlib import Path

import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

from invoiceiq.config import Settings, get_settings


class OCRError(RuntimeError):
    """Raised when an image cannot be read or Tesseract OCR fails."""


def image_to_text(image: str | Path | Image.Image) -> str:
    """Run Tesseract over an image and return the recognized text."""
    settings = get_settings()
    _configure_tesseract(settings)

    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as opened:
                return _run_ocr(opened, settings.ocr_language)
        except UnidentifiedImageError as exc:
            raise OCRError(f"Unreadable image '{image}': {exc}") from exc

    return _run_ocr(image, settings.ocr_language)


def _configure_tesseract(settings: Settings) -> None:
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def _run_ocr(image: Image.Image, language: str) -> str:
    try:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        return pytesseract.image_to_string(prepared, lang=language)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError(
            "Tesseract OCR is not installed or TESSERACT_CMD is incorrect."
        ) from exc
    except (pytesseract.TesseractError, OSError, ValueError) as exc:
        raise OCRError(f"OCR failed: {exc}") from exc
