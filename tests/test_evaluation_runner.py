"""Runner, output, and CLI integration tests using an injected pipeline."""

import json
from pathlib import Path
from typing import Any

import pytest

from invoiceiq.evaluation.runner import (
    build_summary,
    discover_labelled_documents,
    evaluate_dataset,
    run_evaluation,
)
from invoiceiq.schemas import (
    ExtractionResponse,
    Invoice,
    TokenUsage,
    ValidationReport,
)


def _invoice_payload() -> dict[str, Any]:
    return {
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


def _response(payload: dict[str, Any]) -> ExtractionResponse:
    return ExtractionResponse(
        invoice=Invoice.model_validate(payload),
        validation=ValidationReport(is_valid=True),
        needs_review=False,
        warnings=[],
        model="fake-model",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _fake_process(payloads: dict[str, dict[str, Any]]):
    def process(path: Path) -> ExtractionResponse:
        return _response(payloads[path.stem])

    return process


def _write_dataset(directory: Path, payloads: dict[str, dict[str, Any]]) -> None:
    for stem, payload in payloads.items():
        (directory / f"{stem}.pdf").write_bytes(b"%PDF-1.4 placeholder")
        (directory / f"{stem}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_discover_pairs_documents_with_ground_truth(tmp_path: Path) -> None:
    _write_dataset(tmp_path, {"L01": _invoice_payload(), "L02": _non_invoice_payload()})

    documents = discover_labelled_documents(tmp_path)

    assert [document.stem for document in documents] == ["L01", "L02"]
    assert all(document.ground_truth.suffix == ".json" for document in documents)


def test_discover_rejects_document_without_ground_truth(tmp_path: Path) -> None:
    (tmp_path / "L01.pdf").write_bytes(b"%PDF-1.4 placeholder")

    with pytest.raises(FileNotFoundError, match="Missing ground-truth"):
        discover_labelled_documents(tmp_path)


def test_discover_rejects_ground_truth_without_document(tmp_path: Path) -> None:
    (tmp_path / "L01.json").write_text(json.dumps(_invoice_payload()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing document"):
        discover_labelled_documents(tmp_path)


def test_discover_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No labelled documents"):
        discover_labelled_documents(tmp_path)


def test_run_evaluation_reports_perfect_predictions(tmp_path: Path) -> None:
    payloads = {"L01": _invoice_payload(), "L02": _non_invoice_payload()}
    _write_dataset(tmp_path, payloads)
    documents = discover_labelled_documents(tmp_path)

    results = run_evaluation(
        documents, process=_fake_process(payloads), tolerance=0.01
    )
    summary = build_summary(results, tolerance=0.01)

    assert all(document.fully_correct for document in results)
    assert summary.overall_accuracy == 1.0
    assert summary.fully_correct_documents == 2
    assert summary.failed_documents == 0
    assert summary.invoice_confusion == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 1,
    }
    assert summary.line_items is not None
    assert summary.line_items.matched_items == 1
    assert summary.total_usage == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }


def test_run_evaluation_counts_pipeline_failures_as_incorrect(tmp_path: Path) -> None:
    payloads = {"L01": _invoice_payload()}
    _write_dataset(tmp_path, payloads)
    documents = discover_labelled_documents(tmp_path)

    def failing_process(path: Path) -> ExtractionResponse:
        raise RuntimeError("pipeline exploded")

    results = run_evaluation(documents, process=failing_process, tolerance=0.01)
    summary = build_summary(results, tolerance=0.01)

    assert results[0].error == "RuntimeError: pipeline exploded"
    assert results[0].predicted is None
    assert results[0].needs_review is True
    assert all(comparison.match is False for comparison in results[0].fields.values())
    assert summary.overall_accuracy == 0.0
    assert summary.failed_documents == 1
    assert summary.invoice_confusion == {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }


def test_evaluate_dataset_writes_predictions_and_reports(tmp_path: Path) -> None:
    payloads = {"L01": _invoice_payload()}
    _write_dataset(tmp_path, payloads)
    documents = discover_labelled_documents(tmp_path)
    output_dir = tmp_path / "output"

    report = evaluate_dataset(
        documents,
        process=_fake_process(payloads),
        tolerance=0.01,
        data_dir=tmp_path,
        output_dir=output_dir,
    )

    assert report.summary.overall_accuracy == 1.0
    assert (output_dir / "predictions.json").is_file()
    assert (output_dir / "evaluation.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "predictions" / "L01.json").is_file()

    evaluation = json.loads(
        (output_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["summary"]["overall_accuracy"] == 1.0
    assert evaluation["documents"][0]["fully_correct"] is True

    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Matching rules" in markdown
    assert "## Tuning disclosure" in markdown
    assert "Overall accuracy" in markdown
