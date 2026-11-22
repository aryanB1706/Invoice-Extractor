"""Batch extraction output and list-price cost calculations."""

import json
from pathlib import Path

from pydantic import SecretStr

from invoiceiq.batch import (
    DocumentOutput,
    _build_cost_analysis,
    estimate_list_price_cost,
    run_batch,
)
from invoiceiq.config import Settings
from invoiceiq.schemas import (
    DocumentQuality,
    ExtractionResponse,
    Invoice,
    TokenUsage,
    ValidationReport,
)

INVOICE = {
    "is_invoice": True,
    "vendor": "Acme Supplies",
    "invoice_number": "INV-2041",
    "invoice_date": "2026-03-14",
    "currency": "INR",
    "line_items": [{"description": "A4 paper", "quantity": 10, "amount": 3200.0}],
    "tax_total": 576.0,
    "total": 3776.0,
}


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("test-key"),
        groq_model="openai/gpt-oss-120b",
    )


def _write_assignment(root: Path) -> None:
    labelled = root / "labelled"
    unlabelled = root / "unlabelled"
    labelled.mkdir(parents=True)
    unlabelled.mkdir()
    for index in range(1, 11):
        stem = f"L{index:02}"
        (labelled / f"{stem}.pdf").write_bytes(b"placeholder")
        (labelled / f"{stem}.json").write_text(json.dumps(INVOICE), encoding="utf-8")
        (unlabelled / f"U{index:02}.pdf").write_bytes(b"placeholder")


def test_list_price_cost_uses_configured_input_and_output_rates() -> None:
    settings = _settings()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    assert estimate_list_price_cost(usage, settings) == 0.75


def test_list_price_cost_applies_reported_cached_input_tokens() -> None:
    settings = _settings()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cached_prompt_tokens=500_000,
    )

    assert estimate_list_price_cost(usage, settings) == 0.7125


def test_missing_usage_metadata_is_unknown_not_zero() -> None:
    settings = _settings()
    assert estimate_list_price_cost(None, settings) is None

    document = DocumentOutput(
        source_filename="L01.pdf",
        source_relative_path="labelled/L01.pdf",
        dataset_split="labelled",
        ground_truth_reference="L01.json",
        status="success",
        model=settings.groq_model,
        invoice=Invoice.model_validate(INVOICE),
        needs_review=False,
        token_usage=TokenUsage(prompt_tokens=10),
    )
    cost = _build_cost_analysis([document], settings)

    assert cost.total_api_cost_usd is None
    assert cost.average_cost_per_successful_document_usd is None
    assert cost.documents_missing_output_usage == ["L01.pdf"]


def test_batch_writes_20_traceable_outputs_and_uses_labels_only_for_eval(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "assignment"
    _write_assignment(source_dir)

    def fake_process(path: Path, *, on_pages_loaded=None) -> ExtractionResponse:
        if on_pages_loaded is not None:
            on_pages_loaded(
                DocumentQuality(page_count=1, ocr_pages=0, text_characters=300)
            )
        return ExtractionResponse(
            invoice=Invoice.model_validate(INVOICE),
            validation=ValidationReport(is_valid=True),
            needs_review=False,
            model="openai/gpt-oss-120b",
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cached_prompt_tokens=0,
            ),
        )

    output_dir = tmp_path / "outputs"
    report_path = tmp_path / "REPORT.md"
    summary = run_batch(
        data_dir=source_dir,
        output_dir=output_dir,
        settings=_settings(),
        process=fake_process,
        report_path=report_path,
    )

    per_document = list(output_dir.glob("*.json"))
    assert summary.total_documents == 20
    assert summary.successful_documents == 20
    assert len(per_document) == 22
    assert (output_dir / "L01.json").is_file()
    assert (output_dir / "U10.json").is_file()

    unlabelled = json.loads((output_dir / "U01.json").read_text(encoding="utf-8"))
    assert unlabelled["dataset_split"] == "unlabelled"
    assert unlabelled["ground_truth_reference"] is None
    assert unlabelled["invoice"]["vendor"] == "Acme Supplies"
    assert summary.pricing.total_api_cost_usd == 0.00054
    assert summary.pricing.average_cost_per_successful_document_usd == 0.000027
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Human-review effectiveness" in report_text
    assert "Mean wrong fields/document" in report_text
    assert (output_dir / "evaluation" / "evaluation.json").is_file()
