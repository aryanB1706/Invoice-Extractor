"""Validate the downloaded assignment ground truth against our schema."""

import json
from pathlib import Path
from typing import Any

import pytest

from invoiceiq.schemas import Invoice

LABELLED_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "samples" / "assignment" / "labelled"
)
LABEL_FILES = sorted(LABELLED_DIR.glob("*.json"))
REQUIRED_KEYS = {
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
}

pytestmark = pytest.mark.skipif(
    not LABEL_FILES, reason="assignment labelled data is not downloaded"
)


@pytest.mark.parametrize("label_path", LABEL_FILES, ids=lambda path: path.stem)
def test_assignment_ground_truth_is_schema_compatible(label_path: Path) -> None:
    payload: dict[str, Any] = json.loads(label_path.read_text(encoding="utf-8"))

    invoice = Invoice.model_validate(payload)
    dumped = invoice.model_dump(mode="json")

    assert set(dumped) == REQUIRED_KEYS
    if not dumped["is_invoice"]:
        assert all(
            value is None for key, value in dumped.items() if key != "is_invoice"
        )
