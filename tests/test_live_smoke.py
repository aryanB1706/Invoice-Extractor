"""Optional live smoke test against the real Groq API.

Skipped by default so the regular suite stays offline and deterministic. Run
with ``INVOICEIQ_LIVE_TESTS=1`` when a valid ``GROQ_API_KEY`` is configured.
This test checks pipeline wiring only; it is not an accuracy measurement.
"""

import os
from pathlib import Path

import pytest

from invoiceiq.api.service import process_document
from invoiceiq.config import get_settings

SAMPLE_PDF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "samples"
    / "assignment"
    / "labelled"
    / "L01.pdf"
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("INVOICEIQ_LIVE_TESTS") != "1",
        reason="set INVOICEIQ_LIVE_TESTS=1 to call the live Groq API",
    ),
]


def test_live_groq_pipeline_returns_schema_valid_invoice() -> None:
    if not get_settings().groq_configured:
        pytest.skip("Groq is not configured")
    if not SAMPLE_PDF.is_file():
        pytest.skip("assignment sample is not downloaded")

    result = process_document(SAMPLE_PDF)

    assert result.invoice.is_invoice is True
    assert result.invoice.invoice_number
    assert result.invoice.line_items
    assert result.model
