"""Mathematical invoice validation tests, separate from schema validation."""

from datetime import date

import pytest

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


def test_missing_vendor_is_reported() -> None:
    report = validate_invoice(_invoice(vendor=None))

    assert report.is_valid is False
    assert any(issue.field == "vendor" for issue in report.issues)


def test_missing_invoice_number_is_reported() -> None:
    report = validate_invoice(_invoice(invoice_number=None))

    assert report.is_valid is False
    assert any(issue.field == "invoice_number" for issue in report.issues)


def test_missing_invoice_date_is_reported() -> None:
    report = validate_invoice(_invoice(invoice_date=None))

    assert report.is_valid is False
    assert any(issue.field == "invoice_date" for issue in report.issues)


def test_missing_currency_is_reported() -> None:
    report = validate_invoice(_invoice(currency=None))

    assert report.is_valid is False
    assert any(issue.field == "currency" for issue in report.issues)


def test_currency_must_be_a_three_letter_code() -> None:
    report = validate_invoice(_invoice(currency="US$"))

    assert report.is_valid is False
    assert any("three-letter" in issue.message for issue in report.issues)


def test_whitespace_only_required_identifiers_are_missing() -> None:
    report = validate_invoice(_invoice(vendor="  ", invoice_number="\t"))

    assert report.is_valid is False
    assert any(issue.field == "vendor" for issue in report.issues)
    assert any(issue.field == "invoice_number" for issue in report.issues)


def test_negative_total_is_reported() -> None:
    report = validate_invoice(_invoice(total=-1.0))

    assert report.is_valid is False
    assert any(issue.field == "total" for issue in report.issues)


def test_missing_line_items_is_reported() -> None:
    report = validate_invoice(_invoice(line_items=None))

    assert report.is_valid is False
    assert any(issue.field == "line_items" for issue in report.issues)


def test_line_item_without_amount_is_reported() -> None:
    report = validate_invoice(
        _invoice(line_items=[LineItem(description="Widget", quantity=1, amount=None)])
    )

    assert report.is_valid is False
    assert any(issue.field == "line_items" for issue in report.issues)


def test_missing_line_item_quantity_is_reported() -> None:
    report = validate_invoice(
        _invoice(
            line_items=[LineItem(description="Widget", quantity=None, amount=100.0)]
        )
    )

    assert report.is_valid is False
    assert any("quantity" in issue.message for issue in report.issues)


def test_blank_line_item_description_is_reported() -> None:
    report = validate_invoice(
        _invoice(
            line_items=[LineItem(description="  ", quantity=1, amount=100.0)]
        )
    )

    assert report.is_valid is False
    assert any("description" in issue.message for issue in report.issues)


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_invoice(_invoice(), tolerance=-0.01)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError):
        validate_invoice(_invoice(), tolerance=tolerance)
