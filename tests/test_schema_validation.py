"""Strict Pydantic schema validation tests."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from invoiceiq.schemas import Invoice

REQUIRED_KEYS = (
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "is_invoice": True,
        "vendor": "Acme Supplies Pvt Ltd",
        "invoice_number": "INV-2041",
        "invoice_date": "2026-03-14",
        "currency": "INR",
        "line_items": [
            {
                "description": "A4 paper, 500 sheets",
                "quantity": 10,
                "amount": 3200.0,
            }
        ],
        "tax_total": 576.0,
        "total": 3776.0,
    }
    payload.update(overrides)
    return payload


def _non_invoice_payload() -> dict[str, Any]:
    return {
        "is_invoice": False,
        "vendor": None,
        "invoice_number": None,
        "invoice_date": None,
        "currency": None,
        "line_items": None,
        "tax_total": None,
        "total": None,
    }


def test_valid_payload_parses_with_exact_keys() -> None:
    invoice = Invoice.model_validate(_payload())
    result = invoice.model_dump(mode="json")

    assert set(result) == set(REQUIRED_KEYS)
    assert result["invoice_date"] == "2026-03-14"
    assert set(result["line_items"][0]) == {"description", "quantity", "amount"}


@pytest.mark.parametrize("missing_key", REQUIRED_KEYS)
def test_missing_required_key_is_rejected(missing_key: str) -> None:
    payload = _payload()
    payload.pop(missing_key)

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


@pytest.mark.parametrize("missing_key", ["description", "quantity", "amount"])
def test_missing_line_item_key_is_rejected(missing_key: str) -> None:
    payload = _payload()
    payload["line_items"][0].pop(missing_key)

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_extra_keys_are_rejected() -> None:
    payload = _payload(extra_field="unexpected")

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_extra_line_item_keys_are_rejected() -> None:
    payload = _payload()
    payload["line_items"][0]["unit_price"] = 320.0

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


@pytest.mark.parametrize(
    "invoice_date",
    ["2026-3-4", "03/14/2026", "2026-03-14T00:00:00", "14 March 2026"],
)
def test_invalid_date_format_is_rejected(invoice_date: str) -> None:
    with pytest.raises(ValidationError):
        Invoice.model_validate(_payload(invoice_date=invoice_date))


def test_date_must_be_string_or_date() -> None:
    with pytest.raises(ValidationError):
        Invoice.model_validate(_payload(invoice_date=20260314))


@pytest.mark.parametrize("field_name", ["tax_total", "total"])
def test_invoice_numbers_must_be_numeric(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Invoice.model_validate(_payload(**{field_name: "576.00"}))


@pytest.mark.parametrize("field_name", ["quantity", "amount"])
def test_line_item_numbers_must_be_numeric(field_name: str) -> None:
    payload = _payload()
    payload["line_items"][0][field_name] = "10"

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_not_accepted_as_numbers(value: bool) -> None:
    payload = _payload()
    payload["line_items"][0]["quantity"] = value

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        Invoice.model_validate(_payload(total=value))


def test_non_invoice_with_all_null_fields_is_valid() -> None:
    invoice = Invoice.model_validate(_non_invoice_payload())

    assert invoice.model_dump(mode="json") == _non_invoice_payload()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("vendor", "Acme"),
        ("invoice_number", "INV-1"),
        ("invoice_date", "2026-03-14"),
        ("currency", "INR"),
        ("line_items", []),
        ("tax_total", 0.0),
        ("total", 0.0),
    ],
)
def test_non_invoice_with_populated_field_is_rejected(
    field_name: str, value: object
) -> None:
    payload = _non_invoice_payload()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)
