"""Load text from PDFs and images.

Digital PDFs are read directly with PyMuPDF. Pages with little or no text are
rendered to images and passed through Tesseract OCR. Images are always handled
by OCR. The resulting page text is handed to the extraction layer.
"""

from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image, UnidentifiedImageError

from invoiceiq.config import get_settings
from invoiceiq.document_processing.ocr import OCRError, image_to_text
from invoiceiq.schemas import PageText

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTENSION = ".pdf"


class DocumentProcessingError(RuntimeError):
    """Raised when a document cannot be opened or processed."""


def load_document(path: str | Path) -> list[PageText]:
    """Return extracted text for every page of a supported document."""
    document_path = Path(path)

    if not document_path.is_file():
        raise FileNotFoundError(f"Document not found: {document_path}")
    if document_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{document_path.suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if document_path.suffix.lower() == PDF_EXTENSION:
        return _load_pdf(document_path)
    return [_load_image(document_path)]


def load_document_bytes(content: bytes, extension: str) -> list[PageText]:
    """Load text from in-memory PDF or image content."""
    normalized_extension = extension.lower()
    if normalized_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{extension}'.")
    if not content:
        raise DocumentProcessingError("Document content is empty.")

    if normalized_extension == PDF_EXTENSION:
        try:
            document = pymupdf.open(stream=content, filetype="pdf")
        except (
            pymupdf.FileDataError,
            pymupdf.EmptyFileError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise DocumentProcessingError(f"Could not open PDF: {exc}") from exc
        return _process_pdf_document(document)

    try:
        with Image.open(BytesIO(content)) as image:
            text = _clean_text(image_to_text(image))
    except (
        Image.DecompressionBombError,
        OCRError,
        OSError,
        RuntimeError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        return [
            PageText(
                page_number=1,
                text="",
                used_ocr=True,
                error=f"Could not read image: {exc}",
            )
        ]
    return [
        PageText(
            page_number=1,
            text=text,
            used_ocr=True,
            error=None if text else "No readable text found in image.",
        )
    ]


def _load_pdf(path: Path) -> list[PageText]:
    try:
        document = pymupdf.open(path)
    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise DocumentProcessingError(f"Could not open PDF '{path.name}': {exc}") from exc

    return _process_pdf_document(document, path.name)


def _process_pdf_document(
    document: pymupdf.Document, filename: str = "document.pdf"
) -> list[PageText]:
    with document:
        if document.needs_pass:
            raise DocumentProcessingError(f"PDF '{filename}' is password protected.")

        return [
            _process_pdf_page(page, page_number)
            for page_number, page in enumerate(document, start=1)
        ]


def _process_pdf_page(page: pymupdf.Page, page_number: int) -> PageText:
    text = _clean_text(page.get_text("text"))
    used_ocr = False
    error: str | None = None

    if _needs_ocr(text):
        used_ocr = True
        try:
            ocr_text = _clean_text(image_to_text(_render_page(page)))
        except (OCRError, RuntimeError, ValueError) as exc:
            error = f"OCR failed for page {page_number}: {exc}"
        else:
            if _text_size(ocr_text) >= _text_size(text):
                text = ocr_text
            if not text:
                error = f"No readable text found on page {page_number}."

    return PageText(
        page_number=page_number,
        text=text,
        used_ocr=used_ocr,
        error=error,
    )


def _load_image(path: Path) -> PageText:
    try:
        text = _clean_text(image_to_text(path))
    except OCRError as exc:
        return PageText(page_number=1, text="", used_ocr=True, error=str(exc))

    return PageText(
        page_number=1,
        text=text,
        used_ocr=True,
        error=None if text else "No readable text found in image.",
    )


def _render_page(page: pymupdf.Page) -> Image.Image:
    settings = get_settings()
    pixmap = page.get_pixmap(dpi=settings.ocr_render_dpi, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _needs_ocr(text: str) -> bool:
    return _text_size(text) < get_settings().ocr_min_text_chars


def _text_size(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def _clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)
