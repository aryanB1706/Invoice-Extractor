"""Unit tests for transparent evaluation matching rules."""

from datetime import date

from invoiceiq.evaluation.metrics import (
    compare_invoices,
    compare_line_items,
    dates_match,
    evaluate_invoice,
    failed_field_comparisons,
    normalize_text,
    numbers_match,
    texts_match,
)
from invoiceiq.schemas import Invoice, LineItem


def _invoice(**overrides: object) -> Invoice:
    values: dict[str, object] = {
        "is_invoice": True,
        "vendor": "Acme Supplies Pvt Ltd",
        "invoice_number": "INV-2041",
        "invoice_date": date(2026, 3, 14),
        "currency": "INR",
        "line_items": [
            LineItem(description="A4 paper, 500 sheets", quantity=10, amount=3200.0)
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


def test_numbers_match_treats_1200_and_1200_00_as_equal() -> None:
    assert numbers_match(1200, 1200.00, tolerance=0.01) is True


def test_numbers_match_respects_tolerance() -> None:
    assert numbers_match(1200.0, 1200.02, tolerance=0.01) is False
    assert numbers_match(1200.0, 1200.02, tolerance=0.05) is True


def test_numbers_match_null_handling() -> None:
    assert numbers_match(None, None, tolerance=0.01) is True
    assert numbers_match(None, 0.0, tolerance=0.01) is False
    assert numbers_match(0.0, None, tolerance=0.01) is False


def test_text_match_handles_case_and_whitespace() -> None:
    assert texts_match("  Acme   Supplies  ", "acme supplies") is True
    assert normalize_text("ACME\nSupplies") == "acme supplies"


def test_text_match_preserves_meaningful_differences() -> None:
    assert texts_match("INV-2041", "INV 2041") is False
    assert texts_match("Acme Pvt Ltd", "Acme Pvt. Ltd.") is False


def test_dates_match_iso_format() -> None:
    assert dates_match("2026-03-14", date(2026, 3, 14)) is True
    assert dates_match("2026-03-14", "2026-03-15") is False
    assert dates_match(None, None) is True
    assert dates_match(None, "2026-03-14") is False


def test_compare_invoices_all_fields_match() -> None:
    comparisons, line_metrics = compare_invoices(
        _invoice(), _invoice(), tolerance=0.01
    )

    assert all(comparison.match for comparison in comparisons.values())
    assert line_metrics is not None
    assert line_metrics.matched_items == 1
    assert line_metrics.count_match is True


def test_compare_invoices_detects_genuine_mismatches() -> None:
    predicted = _invoice(total=3777.0, vendor="Different Vendor")

    comparisons, _ = compare_invoices(_invoice(), predicted, tolerance=0.01)

    assert comparisons["total"].match is False
    assert comparisons["vendor"].match is False
    assert comparisons["currency"].match is True
    assert comparisons["is_invoice"].match is True


def test_compare_invoices_accepts_case_and_spacing_differences() -> None:
    predicted = _invoice(
        vendor="  acme   supplies pvt ltd ",
        invoice_number="inv-2041",
        currency="inr",
    )

    comparisons, _ = compare_invoices(_invoice(), predicted, tolerance=0.01)

    assert comparisons["vendor"].match is True
    assert comparisons["invoice_number"].match is True
    assert comparisons["currency"].match is True


def test_compare_line_items_detects_quantity_and_amount_mismatch() -> None:
    expected = [LineItem(description="Widget", quantity=2, amount=100.0)]
    predicted = [LineItem(description="Widget", quantity=3, amount=101.0)]

    field_match, metrics = compare_line_items(expected, predicted, tolerance=0.01)

    assert field_match is False
    assert metrics is not None
    assert metrics.matched_items == 0
    assert metrics.quantity_matches == 0
    assert metrics.amount_matches == 0
    assert metrics.details


def test_compare_line_items_does_not_ignore_extra_items() -> None:
    expected = [LineItem(description="Widget", quantity=1, amount=10.0)]
    predicted = [
        LineItem(description="Widget", quantity=1, amount=10.0),
        LineItem(description="Extra", quantity=1, amount=5.0),
    ]

    field_match, metrics = compare_line_items(expected, predicted, tolerance=0.01)

    assert field_match is False
    assert metrics is not None
    assert metrics.count_match is False
    assert metrics.matched_items == 1
    assert metrics.aligned_items == 2
    assert metrics.item_match_rate == 0.5


def test_compare_line_items_handles_missing_side() -> None:
    field_match, metrics = compare_line_items(
        [LineItem(description="Widget", quantity=1, amount=10.0)], None, tolerance=0.01
    )

    assert field_match is False
    assert metrics is not None
    assert metrics.count_match is False
    assert metrics.expected_count == 1
    assert metrics.predicted_count == 0


def test_non_invoice_null_fields_match() -> None:
    comparisons, line_metrics = compare_invoices(
        _non_invoice(), _non_invoice(), tolerance=0.01
    )

    assert all(comparison.match for comparison in comparisons.values())
    assert line_metrics is None


def test_evaluate_invoice_returns_per_field_booleans() -> None:
    result = evaluate_invoice(_invoice(), _invoice())

    assert set(result) == {
        "is_invoice",
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    }
    assert all(result.values())


def test_failed_field_comparisons_are_all_false() -> None:
    comparisons = failed_field_comparisons(_invoice().model_dump(mode="json"))

    assert all(comparison.match is False for comparison in comparisons.values())
    assert comparisons["is_invoice"].expected is True
    assert comparisons["is_invoice"].predicted is None
