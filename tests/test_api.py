"""API smoke tests."""

from datetime import date

from fastapi.testclient import TestClient

from invoiceiq.api import routes, service
from invoiceiq.api.app import app
from invoiceiq.config import Settings
from invoiceiq.schemas import (
    ExtractionResponse,
    ExtractionResult,
    Invoice,
    PageText,
    ValidationReport,
)

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def _valid_response() -> ExtractionResponse:
    return ExtractionResponse(
        invoice=Invoice(
            is_invoice=True,
            vendor="Acme",
            invoice_number="INV-1",
            invoice_date=date(2026, 1, 1),
            currency="USD",
            line_items=[{"description": "Widget", "quantity": 1, "amount": 100.0}],
            tax_total=18.0,
            total=118.0,
        ),
        validation=ValidationReport(is_valid=True),
        needs_review=False,
    )


def test_extract_endpoint_returns_validated_invoice(monkeypatch) -> None:
    monkeypatch.setattr(routes, "process_document", lambda path: _valid_response())

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["invoice"]["invoice_number"] == "INV-1"
    assert response.json()["validation"]["is_valid"] is True
    assert response.json()["needs_review"] is False


def test_versioned_extract_route_remains_available(monkeypatch) -> None:
    monkeypatch.setattr(routes, "process_document", lambda path: _valid_response())

    response = client.post(
        "/api/v1/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200


def test_extract_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/extract",
        files={"file": ("invoice.tiff", b"data", "image/tiff")},
    )

    assert response.status_code == 415


def test_extract_rejects_invalid_pdf() -> None:
    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 422


def test_extract_rejects_empty_file() -> None:
    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_pipeline_marks_inconsistent_totals_for_review(monkeypatch) -> None:
    invoice = _valid_response().invoice.model_copy(update={"total": 120.0})

    class FakeExtractor:
        def extract(self, text: str) -> ExtractionResult:
            assert "invoice text" in text
            return ExtractionResult(invoice=invoice)

    monkeypatch.setattr(
        service,
        "load_document",
        lambda path: [PageText(page_number=1, text="invoice text")],
    )
    monkeypatch.setattr(service, "InvoiceExtractor", FakeExtractor)
    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: Settings(_env_file=None, invoice_amount_tolerance=0.01),
    )

    response = service.process_document("invoice.pdf")

    assert response.validation.is_valid is False
    assert response.needs_review is True
    assert any(issue.field == "total" for issue in response.validation.issues)
