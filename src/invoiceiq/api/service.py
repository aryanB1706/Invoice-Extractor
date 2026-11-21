"""Document-to-invoice orchestration for the HTTP API."""

from pathlib import Path

from invoiceiq.config import get_settings
from invoiceiq.document_processing.loader import load_document, load_document_bytes
from invoiceiq.extraction import InvoiceExtractor
from invoiceiq.schemas import ExtractionResponse, PageText
from invoiceiq.validation import validate_invoice


class UnreadableDocumentError(ValueError):
    """Raised when a supported document contains no readable text."""


def process_document(path: str | Path) -> ExtractionResponse:
    """Extract and validate invoice data from a document on disk."""
    return _process_pages(load_document(path))


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

    return ExtractionResponse(
        invoice=extraction.invoice,
        validation=report,
        needs_review=not report.is_valid or bool(warnings),
        warnings=warnings,
        model=extraction.model,
        usage=extraction.usage,
    )


__all__ = ["UnreadableDocumentError", "process_document", "process_document_bytes"]
