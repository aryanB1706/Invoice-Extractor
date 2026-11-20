"""Tesseract OCR helpers."""

from pathlib import Path

import pytesseract
from PIL import Image

from invoiceiq.config import get_settings


def image_to_text(image: str | Path | Image.Image) -> str:
    """Run Tesseract over an image and return the recognized text."""
    settings = get_settings()

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return pytesseract.image_to_string(opened, lang=settings.ocr_language)

    return pytesseract.image_to_string(image, lang=settings.ocr_language)
