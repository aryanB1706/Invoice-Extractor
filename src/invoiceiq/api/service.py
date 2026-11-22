"""Document-to-invoice orchestration for the HTTP API."""

from collections.abc import Callable
from pathlib import Path

from invoiceiq.config import get_settings
from invoiceiq.document_processing.loader import load_document, load_document_bytes
from invoiceiq.extraction import InvoiceExtractor
from invoiceiq.schemas import DocumentQuality, ExtractionResponse, PageText
from invoiceiq.validation import validate_invoice


class UnreadableDocumentError(ValueError):
    """Raised when a supported document contains no readable text."""


def process_document(
    path: str | Path,
    *,
    on_pages_loaded: Callable[[DocumentQuality], None] | None = None,
) -> ExtractionResponse:
    """Extract and validate invoice data from a document on disk."""
    pages = load_document(path)
    if on_pages_loaded is not None:
        on_pages_loaded(
            DocumentQuality(
                page_count=len(pages),
                ocr_pages=sum(page.used_ocr for page in pages),
                text_characters=sum(
                    not character.isspace()
                    for page in pages
                    for character in page.text
                ),
                page_errors=[page.error for page in pages if page.error],
            )
        )
    return _process_pages(pages)


def process_document_bytes(content: bytes, extension: str) -> ExtractionResponse:
    """Extract and validate invoice data from in-memory uploaded content."""
    return _process_pages(load_document_bytes(content, extension))


def _process_pages(pages: list[PageText]) -> ExtractionResponse:
    text = "\n\n".join(page.text for page in pages if page.text.strip())
    if not text:
        page_errors = [page.error for page in pages if page.error]
        detail = "; ".join(page_errors) if page_errors else "No readable text was found."
        raise UnreadableDocumentError(detail)

    extraction = InvoiceExtractor().extract(text)
    report = validate_invoice(
        extraction.invoice,
        tolerance=get_settings().invoice_amount_tolerance,
    )
    warnings = [page.error for page in pages if page.error]
    ocr_min_chars = get_settings().ocr_min_text_chars
    sparse_ocr_pages = [
        page.page_number
        for page in pages
        if page.used_ocr
        and sum(not character.isspace() for character in page.text) < ocr_min_chars
    ]
    if sparse_ocr_pages:
        page_numbers = ", ".join(str(number) for number in sparse_ocr_pages)
        warnings.append(
            f"OCR returned fewer than {ocr_min_chars} non-whitespace characters "
            f"on page(s) {page_numbers}."
        )

    return ExtractionResponse(
        invoice=extraction.invoice,
        validation=report,
        needs_review=(extraction.invoice.is_invoice and not report.is_valid) or bool(warnings),
        warnings=warnings,
        model=extraction.model,
        usage=extraction.usage,
    )


__all__ = ["UnreadableDocumentError", "process_document", "process_document_bytes"]
