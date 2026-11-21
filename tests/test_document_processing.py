"""Document loader and OCR tests."""

from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from invoiceiq.config import Settings
from invoiceiq.document_processing import loader, ocr
from invoiceiq.document_processing.loader import (
    DocumentProcessingError,
    load_document,
    load_document_bytes,
)
from invoiceiq.document_processing.ocr import OCRError, image_to_text

DIGITAL_TEXT = (
    "Invoice INV-001\n"
    "Vendor: Acme Corporation\n"
    "Billing address: 123 Main Street\n"
    "Total: 100.00 USD\n"
    "Thank you for your business."
)


def _write_pdf(path: Path, page_texts: list[str]) -> Path:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path


def test_digital_pdf_uses_embedded_text_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pdf(tmp_path / "digital.pdf", [DIGITAL_TEXT])

    def fail_ocr(*args: object, **kwargs: object) -> str:
        raise AssertionError("OCR must not run for a digital PDF page")

    monkeypatch.setattr(loader, "image_to_text", fail_ocr)

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "INV-001" in pages[0].text
    assert pages[0].used_ocr is False
    assert pages[0].error is None


def test_multi_page_pdf_uses_ocr_only_for_empty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pdf(tmp_path / "mixed.pdf", [DIGITAL_TEXT, ""])
    calls: list[object] = []

    def fake_ocr(image: object) -> str:
        calls.append(image)
        return "Scanned invoice page with enough readable content."

    monkeypatch.setattr(loader, "image_to_text", fake_ocr)

    pages = load_document(path)

    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].used_ocr is False
    assert pages[1].used_ocr is True
    assert "Scanned invoice" in pages[1].text
    assert len(calls) == 1
    assert isinstance(calls[0], Image.Image)


def test_low_text_pdf_page_is_ocr_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pdf(tmp_path / "low-text.pdf", ["Hi"])

    monkeypatch.setattr(
        loader,
        "image_to_text",
        lambda image: "Recovered invoice text from the scanned page.",
    )

    pages = load_document(path)

    assert pages[0].used_ocr is True
    assert "Recovered invoice" in pages[0].text
    assert pages[0].error is None


def test_png_image_is_processed_with_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "invoice.png"
    Image.new("RGB", (240, 120), "white").save(path)

    monkeypatch.setattr(
        loader,
        "image_to_text",
        lambda image: "Image invoice INV-2024 total 42.00",
    )

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].used_ocr is True
    assert "INV-2024" in pages[0].text
    assert pages[0].error is None


def test_unreadable_image_returns_error_page(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not an image")

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].text == ""
    assert pages[0].used_ocr is True
    assert pages[0].error is not None
    assert "Unreadable image" in pages[0].error


def test_blank_page_reports_no_text_when_ocr_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pdf(tmp_path / "blank.pdf", [""])

    monkeypatch.setattr(loader, "image_to_text", lambda image: "")

    pages = load_document(path)

    assert pages[0].text == ""
    assert pages[0].used_ocr is True
    assert pages[0].error == "No readable text found on page 1."


def test_ocr_failure_is_reported_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pdf(tmp_path / "ocr-failure.pdf", [""])

    def failing_ocr(image: object) -> str:
        raise OCRError("Tesseract is unavailable")

    monkeypatch.setattr(loader, "image_to_text", failing_ocr)

    pages = load_document(path)

    assert pages[0].text == ""
    assert pages[0].used_ocr is True
    assert pages[0].error is not None
    assert "Tesseract is unavailable" in pages[0].error


def test_corrupt_pdf_raises_processing_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4 this file is broken")

    with pytest.raises(DocumentProcessingError):
        load_document(path)


def test_missing_document_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.pdf")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "invoice.txt"
    path.write_text("not a document", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(path)


def test_image_to_text_uses_pytesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 50), "white").save(path)

    monkeypatch.setattr(
        "invoiceiq.document_processing.ocr.pytesseract.image_to_string",
        lambda image, lang: f"recognized with {lang}",
    )

    assert image_to_text(path) == "recognized with eng"


def test_auto_rotation_recovers_upside_down_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_ocr(image: Image.Image, lang: str | None = None) -> str:
        calls.append(image.size)
        if len(calls) == 3:
            return "Lakeside Printing Co INVOICE Total 100.00 Tax 18.00"
        return "ZE60T-d7 | 03 Sunuud apisere7  \ufffdssauisng sno{ 40} Nok xUeUL"

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", fake_ocr)

    text = image_to_text(Image.new("RGB", (200, 100), "white"))

    assert "Lakeside Printing Co INVOICE" in text
    assert len(calls) == 4


def test_auto_rotation_skips_readable_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_ocr(image: Image.Image, lang: str | None = None) -> str:
        calls.append(1)
        return "Invoice INV-1 Total 118.00 Tax 18.00 Date 2026-01-01"

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", fake_ocr)

    text = image_to_text(Image.new("RGB", (200, 100), "white"))

    assert "Invoice INV-1" in text
    assert len(calls) == 1


def test_auto_rotation_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_ocr(image: Image.Image, lang: str | None = None) -> str:
        calls.append(1)
        return "ZE60T-d7 | 03 Sunuud"

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", fake_ocr)
    monkeypatch.setattr(
        ocr,
        "get_settings",
        lambda: Settings(_env_file=None, ocr_auto_rotate=False),
    )

    text = image_to_text(Image.new("RGB", (200, 100), "white"))

    assert text == "ZE60T-d7 | 03 Sunuud"
    assert len(calls) == 1


def test_rotation_metrics_reward_readable_text() -> None:
    readable_score, readable_keywords = ocr._rotation_metrics(
        "Invoice total tax amount 100.00"
    )
    garbled_score, garbled_keywords = ocr._rotation_metrics("ZE60T-d7 | 03 Sunuud")

    assert readable_score > garbled_score
    assert readable_keywords > garbled_keywords


def test_load_document_bytes_reads_pdf() -> None:
    pages = load_document_bytes(_pdf_bytes([DIGITAL_TEXT]), ".pdf")

    assert len(pages) == 1
    assert "INV-001" in pages[0].text
    assert pages[0].used_ocr is False


def test_load_document_bytes_reads_image(monkeypatch: pytest.MonkeyPatch) -> None:
    output = BytesIO()
    Image.new("RGB", (100, 50), "white").save(output, format="PNG")
    monkeypatch.setattr(loader, "image_to_text", lambda image: "Image invoice text")

    pages = load_document_bytes(output.getvalue(), ".png")

    assert pages[0].text == "Image invoice text"
    assert pages[0].used_ocr is True
    assert pages[0].error is None


def test_load_document_bytes_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError):
        load_document_bytes(b"data", ".txt")


def test_load_document_bytes_rejects_empty_content() -> None:
    with pytest.raises(DocumentProcessingError):
        load_document_bytes(b"", ".pdf")


def test_load_document_bytes_rejects_corrupt_pdf() -> None:
    with pytest.raises(DocumentProcessingError):
        load_document_bytes(b"%PDF-1.4 this file is broken", ".pdf")


def test_load_document_bytes_reports_unreadable_image() -> None:
    pages = load_document_bytes(b"not an image", ".png")

    assert pages[0].text == ""
    assert pages[0].used_ocr is True
    assert pages[0].error is not None


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
