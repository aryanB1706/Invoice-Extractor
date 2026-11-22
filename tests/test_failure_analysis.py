"""Failure-analysis categories and human-review evaluation."""

import json
from pathlib import Path

import pymupdf

from invoiceiq.evaluation.failure_analysis import (
    analyze_saved_evaluation,
    build_failure_analysis,
)
from invoiceiq.evaluation.runner import (
    discover_labelled_documents,
    evaluate_dataset,
)
from invoiceiq.schemas import (
    DocumentQuality,
    ExtractionResponse,
    Invoice,
    ValidationIssue,
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


def _write_dataset(directory: Path) -> None:
    for stem in ("L01", "L02", "L03"):
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Invoice quality test " * 10)
        pdf.save(directory / f"{stem}.pdf")
        pdf.close()
        payload = json.loads(json.dumps(INVOICE))
        if stem == "L03":
            payload["line_items"][0]["quantity"] = None
        (directory / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_failure_analysis_reports_observed_categories_and_review_signal(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path)
    documents = discover_labelled_documents(tmp_path)
    bad_prediction = dict(INVOICE)
    bad_prediction["line_items"] = [
        {"description": "A4 paper", "quantity": None, "amount": 3200.0},
        {"description": "Unlisted service", "quantity": 1, "amount": 42.0},
    ]
    bad_prediction["tax_total"] = 0.0
    bad_prediction["total"] = 3200.0

    def process(path: Path) -> ExtractionResponse:
        if path.stem == "L01":
            return ExtractionResponse(
                invoice=Invoice.model_validate(bad_prediction),
                validation=ValidationReport(
                    is_valid=False,
                    issues=[
                        ValidationIssue(
                            field="line_items",
                            message="Missing quantity.",
                        )
                    ],
                ),
                needs_review=True,
                model="fake-model",
            )
        if path.stem == "L03":
            payload = json.loads(json.dumps(INVOICE))
            payload["line_items"][0]["quantity"] = None
            return ExtractionResponse(
                invoice=Invoice.model_validate(payload),
                validation=ValidationReport(
                    is_valid=False,
                    issues=[
                        ValidationIssue(
                            field="line_items",
                            message="Missing quantity.",
                        )
                    ],
                ),
                needs_review=True,
                model="fake-model",
            )
        return ExtractionResponse(
            invoice=Invoice.model_validate(INVOICE),
            validation=ValidationReport(is_valid=True),
            needs_review=False,
            model="fake-model",
        )

    evaluation = evaluate_dataset(
        documents,
        process=process,
        data_dir=tmp_path,
        output_dir=tmp_path / "output",
    )
    failed_document = evaluation.documents[0].model_copy(
        update={
            "document_quality": DocumentQuality(
                page_count=1,
                ocr_pages=1,
                text_characters=35,
            )
        }
    )
    enriched_evaluation = evaluation.model_copy(
        update={
            "documents": [
                failed_document,
                evaluation.documents[1],
                evaluation.documents[2],
            ]
        }
    )
    analysis = build_failure_analysis(enriched_evaluation)
    categories = {category.code: category for category in analysis.categories}

    assert categories["ocr_errors"].documents == ["L01.pdf"]
    assert categories["line_item_errors"].documents == ["L01.pdf"]
    assert categories["tax_total_errors"].documents == ["L01.pdf"]
    assert categories["missing_information"].documents == ["L01.pdf"]
    assert categories["hallucinations"].documents == ["L01.pdf"]
    assert categories["vendor_errors"].document_count == 0
    assert analysis.review_effectiveness.flagged_documents == 2
    assert analysis.review_effectiveness.flagged_error_rate == 0.5
    assert analysis.review_effectiveness.flagged_mean_field_errors == 1.5
    assert analysis.review_effectiveness.unflagged_error_rate == 0.0
    assert [
        item.document for item in analysis.review_effectiveness.flagged_without_errors
    ] == ["L03.pdf"]
    assert "quantity" in analysis.review_effectiveness.flagged_without_errors[0].reasons[0]
    assert (tmp_path / "output" / "failure_analysis.md").is_file()
    assert (tmp_path / "output" / "failure_analysis.json").is_file()

    saved_analysis = analyze_saved_evaluation(
        tmp_path / "output" / "evaluation.json",
        data_dir=tmp_path,
        output_dir=tmp_path / "saved-analysis",
    )
    assert saved_analysis.review_effectiveness.flagged_documents == 2
    assert saved_analysis.review_effectiveness.unflagged_documents == 1
