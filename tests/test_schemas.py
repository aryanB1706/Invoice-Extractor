"""Schema and validation tests."""

from datetime import date

from invoiceiq.schemas import Invoice, LineItem
from invoiceiq.validation import validate_invoice


def test_invoice_schema_matches_required_json_keys() -> None:
    invoice = Invoice(
        is_invoice=True,
        vendor="Acme",
        invoice_number="INV-1",
        invoice_date=date(2026, 3, 14),
        currency="INR",
        line_items=[LineItem(description="Widget", quantity=2, amount=100.0)],
        tax_total=18.0,
        total=118.0,
    )

    payload = invoice.model_dump(mode="json")

    assert set(payload) == {
        "is_invoice",
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    }
    assert set(payload["line_items"][0]) == {"description", "quantity", "amount"}
    assert payload["invoice_date"] == "2026-03-14"


def test_valid_invoice_has_no_issues() -> None:
    invoice = Invoice(
        is_invoice=True,
        vendor="Acme",
        invoice_number="INV-1",
        invoice_date=date(2026, 1, 1),
        line_items=[LineItem(description="Widget", quantity=1, amount=100.0)],
        tax_total=18.0,
        total=118.0,
    )

    report = validate_invoice(invoice)
    assert report.is_valid is True
    assert report.issues == []


def test_incorrect_total_is_reported() -> None:
    invoice = Invoice(
        is_invoice=True,
        vendor="Acme",
        invoice_number="INV-1",
        invoice_date=date(2026, 1, 1),
        line_items=[LineItem(description="Widget", quantity=1, amount=100.0)],
        tax_total=18.0,
        total=100.0,
    )

    report = validate_invoice(invoice)
    assert report.is_valid is False
    assert report.issues[0].field == "total"


def test_non_invoice_is_not_valid_invoice_data() -> None:
    report = validate_invoice(Invoice(is_invoice=False))

    assert report.is_valid is False
    assert report.issues[0].field == "is_invoice"
