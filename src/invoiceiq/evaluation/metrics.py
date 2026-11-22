"""Transparent comparison rules for invoice evaluation.

The rules are intentionally strict enough to surface real extraction errors:

* Text fields are compared after Unicode normalization, whitespace collapsing,
  and case folding. Punctuation and digits are preserved.
* Dates are compared as ISO ``date`` objects.
* Numbers are compared with an absolute tolerance (default ``0.01``).
* Line items are compared in printed order. A document matches on
  ``line_items`` only when the item count matches and every item matches on
  description, quantity, and amount.
"""

import math
import unicodedata
from datetime import date, datetime

from invoiceiq.evaluation.models import FieldComparison, LineItemMetrics
from invoiceiq.schemas import Invoice, LineItem


def normalize_text(value: str | None) -> str | None:
    """Normalize text for comparison without removing meaningful characters."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def normalize_date(value: date | datetime | str | None) -> date | None:
    """Normalize a date or ISO date string to a ``date`` object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def texts_match(expected: str | None, predicted: str | None) -> bool:
    """Compare text fields after normalization."""
    return normalize_text(expected) == normalize_text(predicted)


def dates_match(
    expected: date | datetime | str | None,
    predicted: date | datetime | str | None,
) -> bool:
    """Compare dates after ISO normalization."""
    expected_date = normalize_date(expected)
    predicted_date = normalize_date(predicted)
    if expected_date is None or predicted_date is None:
        return expected_date is None and predicted_date is None
    return expected_date == predicted_date


def numbers_match(
    expected: float | None, predicted: float | None, tolerance: float
) -> bool:
    """Compare numbers with an absolute tolerance."""
    if expected is None or predicted is None:
        return expected is None and predicted is None
    return math.isclose(
        float(expected), float(predicted), rel_tol=0.0, abs_tol=tolerance
    )


def compare_line_items(
    expected: list[LineItem] | None,
    predicted: list[LineItem] | None,
    *,
    tolerance: float,
) -> tuple[bool, LineItemMetrics | None]:
    """Compare line items in printed order and return item-level metrics."""
    if expected is None and predicted is None:
        return True, None

    if expected is None or predicted is None:
        expected_count = 0 if expected is None else len(expected)
        predicted_count = 0 if predicted is None else len(predicted)
        aligned = max(expected_count, predicted_count)
        return False, LineItemMetrics(
            expected_count=expected_count,
            predicted_count=predicted_count,
            count_match=False,
            matched_items=0,
            aligned_items=aligned,
            item_match_rate=0.0 if aligned else 1.0,
            description_matches=0,
            quantity_matches=0,
            amount_matches=0,
            details=["One side has line items and the other does not."],
        )

    expected_count = len(expected)
    predicted_count = len(predicted)
    aligned = max(expected_count, predicted_count)
    description_matches = 0
    quantity_matches = 0
    amount_matches = 0
    matched_items = 0
    details: list[str] = []

    for index in range(aligned):
        expected_item = expected[index] if index < expected_count else None
        predicted_item = predicted[index] if index < predicted_count else None

        if expected_item is None or predicted_item is None:
            side = "predicted" if predicted_item is None else "expected"
            details.append(f"item {index + 1}: missing on {side} side")
            continue

        description_ok = texts_match(
            expected_item.description, predicted_item.description
        )
        quantity_ok = numbers_match(
            expected_item.quantity, predicted_item.quantity, tolerance
        )
        amount_ok = numbers_match(expected_item.amount, predicted_item.amount, tolerance)

        description_matches += int(description_ok)
        quantity_matches += int(quantity_ok)
        amount_matches += int(amount_ok)

        if description_ok and quantity_ok and amount_ok:
            matched_items += 1
        else:
            problems = []
            if not description_ok:
                problems.append(
                    f"description {expected_item.description!r} != "
                    f"{predicted_item.description!r}"
                )
            if not quantity_ok:
                problems.append(
                    f"quantity {expected_item.quantity!r} != "
                    f"{predicted_item.quantity!r}"
                )
            if not amount_ok:
                problems.append(
                    f"amount {expected_item.amount!r} != {predicted_item.amount!r}"
                )
            details.append(f"item {index + 1}: " + "; ".join(problems))

    count_match = expected_count == predicted_count
    field_match = (
        count_match and matched_items == expected_count and aligned == expected_count
    )
    return field_match, LineItemMetrics(
        expected_count=expected_count,
        predicted_count=predicted_count,
        count_match=count_match,
        matched_items=matched_items,
        aligned_items=aligned,
        item_match_rate=matched_items / aligned if aligned else 1.0,
        description_matches=description_matches,
        quantity_matches=quantity_matches,
        amount_matches=amount_matches,
        details=details,
    )


def compare_invoices(
    expected: Invoice, predicted: Invoice, *, tolerance: float
) -> tuple[dict[str, FieldComparison], LineItemMetrics | None]:
    """Compare two invoices field by field."""
    expected_data = expected.model_dump(mode="json")
    predicted_data = predicted.model_dump(mode="json")
    line_items_match, line_item_metrics = compare_line_items(
        expected.line_items, predicted.line_items, tolerance=tolerance
    )

    matches = {
        "is_invoice": expected.is_invoice == predicted.is_invoice,
        "vendor": texts_match(expected.vendor, predicted.vendor),
        "invoice_number": texts_match(
            expected.invoice_number, predicted.invoice_number
        ),
        "invoice_date": dates_match(expected.invoice_date, predicted.invoice_date),
        "currency": texts_match(expected.currency, predicted.currency),
        "line_items": line_items_match,
        "tax_total": numbers_match(expected.tax_total, predicted.tax_total, tolerance),
        "total": numbers_match(expected.total, predicted.total, tolerance),
    }

    comparisons = {
        field: FieldComparison(
            expected=expected_data.get(field),
            predicted=predicted_data.get(field),
            match=match,
        )
        for field, match in matches.items()
    }
    return comparisons, line_item_metrics


def evaluate_invoice(
    expected: Invoice, predicted: Invoice, *, tolerance: float = 0.01
) -> dict[str, bool]:
    """Return a per-field match report for a single invoice."""
    comparisons, _ = compare_invoices(expected, predicted, tolerance=tolerance)
    return {field: comparison.match for field, comparison in comparisons.items()}


def failed_field_comparisons(
    expected_data: dict[str, object] | None,
) -> dict[str, FieldComparison]:
    """Return all-false comparisons for a document whose extraction failed."""
    return {
        field: FieldComparison(
            expected=(expected_data or {}).get(field),
            predicted=None,
            match=False,
        )
        for field in (
            "is_invoice",
            "vendor",
            "invoice_number",
            "invoice_date",
            "currency",
            "line_items",
            "tax_total",
            "total",
        )
    }
