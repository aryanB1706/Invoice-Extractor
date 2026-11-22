"""End-to-end pipeline tests with OCR and Groq boundaries mocked."""

from datetime import date
from io import BytesIO
from typing import Any

import groq
import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invoiceiq.api import service
from invoiceiq.api.app import app
from invoiceiq.config import Settings
from invoiceiq.document_processing import loader
from invoiceiq.document_processing.ocr import OCRError
from invoiceiq.extraction import InvoiceExtractor, LLMConfigurationError
from invoiceiq.schemas import Invoice, LineItem

client = TestClient(app)
INVOICE_TEXT = (
    "Invoice INV-2041\nVendor: Acme Supplies\nDate: 2026-03-14\n"
    "A4 paper, quantity 10, line amount 3200.00\nTax: 576.00\nTotal: 3776.00"
)


class FakeStructuredLLM:
    """Return a predetermined structured response without calling Groq."""

    def __init__(self, parsed: Invoice | None = None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.messages: list[Any] | None = None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "FakeStructuredLLM":
        return self

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return {"raw": None, "parsed": self.parsed, "parsing_error": None}


def _invoice(**overrides: Any) -> Invoice:
    values: dict[str, Any] = {
        "is_invoice": True,
        "vendor": "Acme Supplies",
        "invoice_number": "INV-2041",
        "invoice_date": date(2026, 3, 14),
        "currency": "INR",
        "line_items": [
            LineItem(description="A4 paper", quantity=10, amount=3200.0),
        ],
        "tax_total": 576.0,
        "total": 3776.0,
    }
    values.update(overrides)
    return Invoice(**values)


def _non_invoice() -> Invoice:
    return Invoice(
        is_invoice=False,
        vendor=None,
        invoice_number=None,
        invoice_date=None,
        currency=None,
        line_items=None,
        tax_total=None,
        total=None,
    )


def _install_extractor(
    monkeypatch: pytest.MonkeyPatch,
    parsed: Invoice | None = None,
    error: Exception | None = None,
) -> FakeStructuredLLM:
    llm = FakeStructuredLLM(parsed=parsed, error=error)
    settings = Settings(_env_file=None, groq_model="test-model")
    monkeypatch.setattr(
        service,
        "InvoiceExtractor",
        lambda: InvoiceExtractor(llm=llm, settings=settings),
    )
    return llm


def _pdf_bytes(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    output = BytesIO()
    document.save(output)
    document.close()
    return output.getvalue()


def _scanned_pdf_bytes() -> bytes:
    image = BytesIO()
    Image.new("RGB", (400, 200), "white").save(image, format="PNG")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image.getvalue())
    output = BytesIO()
    document.save(output)
    document.close()
    return output.getvalue()


def _image_bytes(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (300, 150), "white").save(output, format=image_format)
    return output.getvalue()


def test_digital_pdf_runs_through_extraction_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install_extractor(monkeypatch, _invoice())

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", _pdf_bytes([INVOICE_TEXT]), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["invoice"]["invoice_number"] == "INV-2041"
    assert body["validation"]["is_valid"] is True
    assert body["needs_review"] is False
    assert llm.messages is not None
    assert "A4 paper" in str(llm.messages[1].content)


def test_scanned_pdf_uses_ocr_then_extracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install_extractor(monkeypatch, _invoice())
    ocr_calls: list[object] = []

    def fake_ocr(image: object) -> str:
        ocr_calls.append(image)
        return INVOICE_TEXT

    monkeypatch.setattr(loader, "image_to_text", fake_ocr)
    response = client.post(
        "/extract",
        files={"file": ("scanned.pdf", _scanned_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["invoice"]["is_invoice"] is True
    assert len(ocr_calls) == 1
    assert llm.messages is not None


@pytest.mark.parametrize(("extension", "image_format", "content_type"), [
    ("jpg", "JPEG", "image/jpeg"),
    ("png", "PNG", "image/png"),
])
def test_invoice_images_run_through_ocr_and_extraction(
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
    image_format: str,
    content_type: str,
) -> None:
    _install_extractor(monkeypatch, _invoice())
    monkeypatch.setattr(loader, "image_to_text", lambda image: INVOICE_TEXT)

    response = client.post(
        "/extract",
        files={
            "file": (
                f"invoice.{extension}",
                _image_bytes(image_format),
                content_type,
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["invoice"]["total"] == 3776.0
    assert response.json()["validation"]["is_valid"] is True


def test_multi_page_invoice_combines_page_text_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install_extractor(monkeypatch, _invoice())
    first_page = "Invoice INV-2041 vendor Acme Supplies " + "billing details " * 5
    second_page = "Line items and tax details " + "A4 paper 3200 tax 576 total 3776 " * 3

    response = client.post(
        "/extract",
        files={"file": ("multi.pdf", _pdf_bytes([first_page, second_page]), "application/pdf")},
    )

    assert response.status_code == 200
    assert llm.messages is not None
    prompt = str(llm.messages[1].content)
    assert "billing details" in prompt
    assert "tax 576" in prompt


@pytest.mark.parametrize(
    "document_text",
    ["Bank statement account activity", "Purchase order PO-1"],
)
def test_non_invoice_documents_return_null_invoice_fields(
    monkeypatch: pytest.MonkeyPatch, document_text: str
) -> None:
    _install_extractor(monkeypatch, _non_invoice())

    response = client.post(
        "/extract",
        files={"file": ("document.pdf", _pdf_bytes([document_text * 5]), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["invoice"]["is_invoice"] is False
    assert all(value is None for key, value in body["invoice"].items() if key != "is_invoice")
    assert body["needs_review"] is False


def test_missing_invoice_number_is_returned_for_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_extractor(monkeypatch, _invoice(invoice_number=None))

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", _pdf_bytes([INVOICE_TEXT]), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["invoice"]["invoice_number"] is None
    assert body["needs_review"] is True
    assert any(issue["field"] == "invoice_number" for issue in body["validation"]["issues"])


def test_unreadable_scanned_pdf_returns_unprocessable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install_extractor(monkeypatch, _invoice())
    monkeypatch.setattr(loader, "image_to_text", lambda image: "")

    response = client.post(
        "/extract",
        files={"file": ("blank-scan.pdf", _scanned_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 422
    assert "No readable text" in response.json()["detail"]
    assert llm.messages is None


def test_corrupt_pdf_returns_unprocessable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install_extractor(monkeypatch, _invoice())

    response = client.post(
        "/extract",
        files={"file": ("corrupt.pdf", b"%PDF-1.4 not a valid document", "application/pdf")},
    )

    assert response.status_code == 422
    assert llm.messages is None


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (
            groq.APITimeoutError(
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            ),
            504,
        ),
        (
            groq.APIConnectionError(
                request=httpx.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions"
                )
            ),
            502,
        ),
    ],
)
def test_groq_failures_return_gateway_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    _install_extractor(monkeypatch, error=error)

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", _pdf_bytes([INVOICE_TEXT]), "application/pdf")},
    )

    assert response.status_code == status_code


def test_ocr_warning_marks_response_for_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_extractor(monkeypatch, _invoice())

    def failing_ocr(image: object) -> str:
        raise OCRError("Tesseract is unavailable")

    monkeypatch.setattr(loader, "image_to_text", failing_ocr)

    response = client.post(
        "/extract",
        files={"file": ("mixed.pdf", _pdf_bytes([INVOICE_TEXT, ""]), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["needs_review"] is True
    assert body["warnings"]


def test_unreadable_image_returns_unprocessable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_extractor(monkeypatch, _invoice())
    monkeypatch.setattr(loader, "image_to_text", lambda image: "")

    response = client.post(
        "/extract",
        files={"file": ("blank.png", _image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 422
    assert "No readable text" in response.json()["detail"]


def test_rate_limit_returns_bad_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.RateLimitError(
        "rate limit exceeded",
        response=httpx.Response(429, request=request),
        body=None,
    )
    _install_extractor(monkeypatch, error=error)

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", _pdf_bytes([INVOICE_TEXT]), "application/pdf")},
    )

    assert response.status_code == 502


def test_authentication_error_returns_bad_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.AuthenticationError(
        "invalid api key",
        response=httpx.Response(401, request=request),
        body=None,
    )
    _install_extractor(monkeypatch, error=error)

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", _pdf_bytes([INVOICE_TEXT]), "application/pdf")},
    )

    assert response.status_code == 502


def test_missing_groq_configuration_returns_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_configuration_error() -> None:
        raise LLMConfigurationError("Groq is not configured.")

    monkeypatch.setattr(service, "InvoiceExtractor", raise_configuration_error)

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", _pdf_bytes([INVOICE_TEXT]), "application/pdf")},
    )

    assert response.status_code == 503
