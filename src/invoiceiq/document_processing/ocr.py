"""Tesseract OCR helpers."""

import re
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

from invoiceiq.config import Settings, get_settings

_ORIENTATION_ANGLES = (90, 180, 270)
_WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z&.'/-]{2,}$")
_INVOICE_KEYWORDS = frozenset(
    {
        "invoice",
        "tax",
        "total",
        "subtotal",
        "date",
        "amount",
        "description",
        "quantity",
        "qty",
        "vendor",
        "bill",
        "number",
        "discount",
        "rate",
        "unit",
        "price",
        "gst",
        "vat",
    }
)


class OCRError(RuntimeError):
    """Raised when an image cannot be read or Tesseract OCR fails."""


def image_to_text(image: str | Path | Image.Image) -> str:
    """Run Tesseract over an image and return the recognized text."""
    settings = get_settings()
    _configure_tesseract(settings)

    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as opened:
                return _run_ocr(opened, settings)
        except UnidentifiedImageError as exc:
            raise OCRError(f"Unreadable image '{image}': {exc}") from exc

    return _run_ocr(image, settings)


def _configure_tesseract(settings: Settings) -> None:
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def _run_ocr(image: Image.Image, settings: Settings) -> str:
    try:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        if settings.ocr_auto_rotate:
            return _ocr_best_orientation(prepared, settings.ocr_language)
        return _ocr_pass(prepared, settings.ocr_language)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError(
            "Tesseract OCR is not installed or TESSERACT_CMD is incorrect."
        ) from exc
    except (pytesseract.TesseractError, OSError, ValueError) as exc:
        raise OCRError(f"OCR failed: {exc}") from exc


def _ocr_best_orientation(image: Image.Image, language: str) -> str:
    """Return OCR text, correcting rotated or upside-down scans when needed.

    Clean documents are recognized in a single pass. Only text without clear
    invoice keywords is re-tested at 90, 180, and 270 degrees, which keeps the
    common path fast while recovering rotated scans.
    """
    base_text = _ocr_pass(image, language)
    base_score, base_keywords = _rotation_metrics(base_text)
    if base_keywords >= 2:
        return base_text

    best_text = base_text
    best_score = base_score
    for angle in _ORIENTATION_ANGLES:
        candidate = _ocr_pass(image.rotate(angle, expand=True), language)
        candidate_score, _ = _rotation_metrics(candidate)
        if candidate_score > best_score:
            best_text = candidate
            best_score = candidate_score

    if best_score >= base_score * 1.5 and best_score - base_score >= 10:
        return best_text
    return base_text


def _ocr_pass(image: Image.Image, language: str) -> str:
    return pytesseract.image_to_string(image, lang=language)


def _rotation_metrics(text: str) -> tuple[int, int]:
    """Score OCR text by readable words and invoice keywords."""
    tokens = [token.strip(".,:;()[]{}<>\"'") for token in text.split()]
    good_words = sum(1 for token in tokens if _WORD_PATTERN.match(token))
    keyword_hits = sum(1 for token in tokens if token.lower() in _INVOICE_KEYWORDS)
    return good_words + 3 * keyword_hits, keyword_hits
