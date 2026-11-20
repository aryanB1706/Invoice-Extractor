"""Load text from PDFs and images.

PDF pages are read with PyMuPDF. Image handling is delegated to the OCR
module so the rest of the pipeline works with a common ``PageText`` shape.
"""

from pathlib import Path

import pymupdf

from invoiceiq.schemas import PageText

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_document(path: str | Path) -> list[PageText]:
    """Return extracted text for every page of a supported document."""
    document_path = Path(path)

    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    if document_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{document_path.suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if document_path.suffix.lower() == ".pdf":
        return _load_pdf(document_path)
    return [_load_image(document_path)]


def _load_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append(PageText(page_number=page_number, text=page.get_text("text")))
    return pages


def _load_image(path: Path) -> PageText:
    from invoiceiq.document_processing.ocr import image_to_text

    return PageText(page_number=1, text=image_to_text(path), used_ocr=True)
