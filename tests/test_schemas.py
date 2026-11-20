"""Schema and validation tests."""

from datetime import date
from decimal import Decimal

from invoiceiq.schemas import Invoice, LineItem
from invoiceiq.validation import validate_invoice


def test_invoice_schema_accepts_line_items() -> None:
    invoice = Invoice(
        vendor_name="Acme",
        invoice_number="INV-1",
        total=Decimal("10.00"),
        line_items=[LineItem(description="Widget", amount=Decimal("10.00"))],
    )

    assert invoice.line_items[0].description == "Widget"
    assert invoice.total == Decimal("10.00")


def test_valid_invoice_has_no_issues() -> None:
    invoice = Invoice(
        vendor_name="Acme",
        invoice_number="INV-1",
        invoice_date=date(2026, 1, 1),
        subtotal=Decimal("100.00"),
        tax=Decimal("18.00"),
        total=Decimal("118.00"),
    )

    report = validate_invoice(invoice)
    assert report.is_valid is True
    assert report.issues == []


def test_incorrect_total_is_reported() -> None:
    invoice = Invoice(
        vendor_name="Acme",
        invoice_number="INV-1",
        subtotal=Decimal("100.00"),
        tax=Decimal("18.00"),
        total=Decimal("100.00"),
    )

    report = validate_invoice(invoice)
    assert report.is_valid is False
    assert report.issues[0].field == "total"
