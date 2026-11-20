"""Mathematical invoice validation tests, separate from schema validation."""

from datetime import date

from invoiceiq.schemas import Invoice, LineItem
from invoiceiq.validation import validate_invoice


def _invoice(**overrides: object) -> Invoice:
    values = {
        "is_invoice": True,
        "vendor": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": date(2026, 1, 1),
        "currency": "INR",
        "line_items": [LineItem(description="Widget", quantity=1, amount=100.0)],
        "tax_total": 18.0,
        "total": 118.0,
    }
    values.update(overrides)
    return Invoice(**values)


def test_valid_invoice_has_no_issues() -> None:
    report = validate_invoice(_invoice())

    assert report.is_valid is True
    assert report.issues == []


def test_incorrect_total_is_reported() -> None:
    report = validate_invoice(_invoice(total=100.0))

    assert report.is_valid is False
    assert report.issues[0].field == "total"


def test_missing_total_is_reported() -> None:
    report = validate_invoice(_invoice(total=None))

    assert report.is_valid is False
    assert any(issue.field == "total" for issue in report.issues)


def test_non_invoice_is_not_valid_invoice_data() -> None:
    report = validate_invoice(
        Invoice(
            is_invoice=False,
            vendor=None,
            invoice_number=None,
            invoice_date=None,
            currency=None,
            line_items=None,
            tax_total=None,
            total=None,
        )
    )

    assert report.is_valid is False
    assert report.issues[0].field == "is_invoice"


def test_calculation_is_skipped_when_amounts_are_missing() -> None:
    invoice = _invoice(
        line_items=[LineItem(description="Widget", quantity=None, amount=None)],
        tax_total=None,
        total=118.0,
    )

    report = validate_invoice(invoice)

    assert any(issue.field == "line_items" for issue in report.issues)
    assert all(issue.field != "total" for issue in report.issues)


def test_missing_tax_is_not_assumed_to_be_zero() -> None:
    report = validate_invoice(_invoice(tax_total=None))

    assert any(issue.field == "tax_total" for issue in report.issues)
    assert all(issue.field != "total" for issue in report.issues)


def test_validation_tolerance_is_configurable() -> None:
    invoice = _invoice(total=118.05)

    assert validate_invoice(invoice, tolerance=0.1).is_valid is True
    assert validate_invoice(invoice, tolerance=0.01).is_valid is False
