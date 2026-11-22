"""Dataset-level evaluation runner and output writers."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from invoiceiq.api.service import process_document
from invoiceiq.evaluation.failure_analysis import write_failure_analysis
from invoiceiq.evaluation.metrics import compare_invoices, failed_field_comparisons
from invoiceiq.evaluation.models import (
    DocumentEvaluation,
    EvaluationReport,
    EvaluationSummary,
    FieldSummary,
    LineItemSummary,
)
from invoiceiq.evaluation.report import format_markdown
from invoiceiq.schemas import DocumentQuality, ExtractionResponse, Invoice

DOCUMENT_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

MATCHING_RULES = [
    "Text (vendor, invoice_number, currency): Unicode NFKC normalization, "
    "whitespace collapsing, and case folding. Punctuation and digits are kept.",
    "Dates (invoice_date): parsed as ISO YYYY-MM-DD and compared as dates.",
    "Numbers (tax_total, total, quantity, amount): compared with an absolute "
    "tolerance of {tolerance}. 1200 and 1200.00 are equivalent.",
    "is_invoice: exact boolean comparison.",
    "line_items: compared in printed order. The field matches only when the "
    "item count matches and every item matches on description, quantity, and "
    "amount. Extra or missing items are errors, not ignored.",
    "Null handling: null matches only null. A null prediction never matches a "
    "populated ground-truth value.",
    "Overall accuracy: unweighted mean of the eight field-level accuracies. "
    "Each field has equal weight; line_items is a strict whole-list match.",
    "Failed extractions count as incorrect for every field and are reported "
    "separately. They are not excluded from the denominator.",
]

TUNING_DISCLOSURE = (
    "The baseline evaluation preceded this revision. After reviewing aggregate "
    "failure categories, the prompt was updated globally to distinguish quantity, "
    "unit price, line amount, tax, and grand total by their printed table headers, "
    "and to map only unambiguous currency symbols. No sample-specific values were "
    "added. OCR_RENDER_DPI was raised globally from 200 to 300 and "
    "GROQ_MAX_TOKENS from 2048 to 8192. Because these changes follow analysis of "
    "the labelled set, any rerun on those same 10 documents is a post-analysis "
    "diagnostic, not an independent holdout score. The earlier tax_total=0 "
    "convention comes from the assignment instructions."
)


@dataclass(frozen=True)
class LabelledDocument:
    """A document paired with its ground-truth JSON file."""

    stem: str
    document: Path
    ground_truth: Path


def discover_labelled_documents(directory: str | Path) -> list[LabelledDocument]:
    """Pair every supported document with its same-stem ground-truth JSON."""
    resolved = Path(directory)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Labelled directory not found: {resolved}")

    documents = {
        path.stem: path
        for path in resolved.iterdir()
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    }
    ground_truths = {
        path.stem: path
        for path in resolved.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    }

    missing_ground_truth = sorted(set(documents) - set(ground_truths))
    missing_documents = sorted(set(ground_truths) - set(documents))
    if missing_ground_truth:
        raise FileNotFoundError(
            "Missing ground-truth JSON for: " + ", ".join(missing_ground_truth)
        )
    if missing_documents:
        raise FileNotFoundError(
            "Missing document for ground-truth JSON: " + ", ".join(missing_documents)
        )
    if not documents:
        raise FileNotFoundError(
            f"No labelled documents found in {resolved}. Expected PDFs or images "
            "paired with same-stem .json files."
        )

    return [
        LabelledDocument(
            stem=stem,
            document=documents[stem],
            ground_truth=ground_truths[stem],
        )
        for stem in sorted(documents)
    ]


def run_evaluation(
    documents: list[LabelledDocument],
    *,
    process: Callable[
        [Path], ExtractionResponse | tuple[ExtractionResponse, DocumentQuality]
    ]
    | None = None,
    tolerance: float = 0.01,
    progress: Callable[[int, int, LabelledDocument], None] | None = None,
) -> list[DocumentEvaluation]:
    """Run the pipeline over every labelled document and compare results."""
    results: list[DocumentEvaluation] = []

    for index, item in enumerate(documents, start=1):
        if progress is not None:
            progress(index, len(documents), item)

        expected_data = json.loads(item.ground_truth.read_text(encoding="utf-8"))
        expected_invoice = Invoice.model_validate(expected_data)
        document_quality: DocumentQuality | None = None

        def capture_quality(quality: DocumentQuality) -> None:
            nonlocal document_quality
            document_quality = quality

        try:
            if process is None:
                response = process_document(
                    item.document,
                    on_pages_loaded=capture_quality,
                )
            else:
                processed = process(item.document)
                if isinstance(processed, tuple):
                    response, document_quality = processed
                else:
                    response = processed
        except Exception as exc:  # noqa: BLE001 - record any pipeline failure
            results.append(
                DocumentEvaluation(
                    document=item.document.name,
                    ground_truth=item.ground_truth.name,
                    error=f"{type(exc).__name__}: {exc}",
                    fields=failed_field_comparisons(expected_data),
                    fully_correct=False,
                    expected=expected_data,
                    predicted=None,
                    document_quality=document_quality,
                    needs_review=True,
                )
            )
            continue

        comparisons, line_item_metrics = compare_invoices(
            expected_invoice, response.invoice, tolerance=tolerance
        )
        results.append(
            DocumentEvaluation(
                document=item.document.name,
                ground_truth=item.ground_truth.name,
                error=None,
                fields=comparisons,
                line_item_metrics=line_item_metrics,
                fully_correct=all(comparison.match for comparison in comparisons.values()),
                expected=expected_data,
                predicted=response.invoice.model_dump(mode="json"),
                validation=response.validation.model_dump(mode="json"),
                document_quality=document_quality,
                needs_review=response.needs_review,
                warnings=response.warnings,
                model=response.model,
                usage=(
                    response.usage.model_dump(mode="json")
                    if response.usage is not None
                    else None
                ),
            )
        )

    return results


def build_summary(
    documents: list[DocumentEvaluation], *, tolerance: float
) -> EvaluationSummary:
    """Aggregate field-level, classification, and line-item metrics."""
    total = len(documents)
    fields = {
        field: FieldSummary(
            correct=sum(1 for doc in documents if doc.fields[field].match),
            total=total,
            accuracy=(
                sum(1 for doc in documents if doc.fields[field].match) / total
                if total
                else 0.0
            ),
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
    overall_accuracy = (
        sum(summary.accuracy for summary in fields.values()) / len(fields)
        if fields
        else 0.0
    )

    return EvaluationSummary(
        total_documents=total,
        failed_documents=sum(1 for doc in documents if doc.error is not None),
        fully_correct_documents=sum(1 for doc in documents if doc.fully_correct),
        overall_accuracy=overall_accuracy,
        fields=fields,
        invoice_confusion=_invoice_confusion(documents),
        line_items=_line_item_summary(documents),
        tolerance=tolerance,
        model=next((doc.model for doc in documents if doc.model), None),
        total_usage=_total_usage(documents),
    )


def build_report(
    documents: list[DocumentEvaluation],
    summary: EvaluationSummary,
    *,
    data_dir: Path,
    output_dir: Path,
) -> EvaluationReport:
    """Assemble the persisted report object."""
    rules = [
        rule.format(tolerance=summary.tolerance) if "{tolerance}" in rule else rule
        for rule in MATCHING_RULES
    ]
    return EvaluationReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        model=summary.model,
        summary=summary,
        documents=documents,
        matching_rules=rules,
        tuning_disclosure=TUNING_DISCLOSURE,
    )


def evaluate_dataset(
    documents: list[LabelledDocument],
    *,
    process: Callable[
        [Path], ExtractionResponse | tuple[ExtractionResponse, DocumentQuality]
    ]
    | None = None,
    tolerance: float = 0.01,
    data_dir: Path,
    output_dir: Path,
    progress: Callable[[int, int, LabelledDocument], None] | None = None,
) -> EvaluationReport:
    """Run, summarize, and persist a complete evaluation."""
    results = run_evaluation(
        documents, process=process, tolerance=tolerance, progress=progress
    )
    summary = build_summary(results, tolerance=tolerance)
    report = build_report(
        results, summary, data_dir=data_dir, output_dir=output_dir
    )
    write_outputs(report, output_dir)
    return report


def write_outputs(report: EvaluationReport, output_dir: str | Path) -> dict[str, Path]:
    """Write combined predictions, per-document predictions, and the report."""
    resolved = Path(output_dir)
    predictions_dir = resolved / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions = [
        {
            "document": document.document,
            "ground_truth": document.ground_truth,
            "model": document.model,
            "usage": document.usage,
            "error": document.error,
            "document_quality": (
                document.document_quality.model_dump(mode="json")
                if document.document_quality
                else None
            ),
            "needs_review": document.needs_review,
            "warnings": document.warnings,
            "prediction": document.predicted,
        }
        for document in report.documents
    ]
    paths = {
        "predictions": resolved / "predictions.json",
        "evaluation": resolved / "evaluation.json",
        "report": resolved / "report.md",
        "failure_analysis": resolved / "failure_analysis.md",
    }

    _write_json(paths["predictions"], predictions)
    _write_json(paths["evaluation"], report.model_dump(mode="json"))
    paths["report"].write_text(format_markdown(report), encoding="utf-8")
    write_failure_analysis(report, resolved)

    for document in report.documents:
        stem = Path(document.document).stem
        _write_json(
            predictions_dir / f"{stem}.json",
            {
                "document": document.document,
                "ground_truth": document.ground_truth,
                "model": document.model,
                "usage": document.usage,
                "error": document.error,
                "document_quality": (
                    document.document_quality.model_dump(mode="json")
                    if document.document_quality
                    else None
                ),
                "needs_review": document.needs_review,
                "warnings": document.warnings,
                "prediction": document.predicted,
            },
        )

    return paths


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _invoice_confusion(documents: list[DocumentEvaluation]) -> dict[str, int]:
    confusion = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    for document in documents:
        if document.predicted is None or document.expected is None:
            continue
        expected = bool(document.expected.get("is_invoice"))
        predicted = bool(document.predicted.get("is_invoice"))
        if expected and predicted:
            confusion["true_positive"] += 1
        elif not expected and predicted:
            confusion["false_positive"] += 1
        elif expected and not predicted:
            confusion["false_negative"] += 1
        else:
            confusion["true_negative"] += 1
    return confusion


def _line_item_summary(
    documents: list[DocumentEvaluation],
) -> LineItemSummary | None:
    metrics = [
        document.line_item_metrics
        for document in documents
        if document.line_item_metrics is not None
    ]
    if not metrics:
        return None

    aligned = sum(metric.aligned_items for metric in metrics)
    expected_items = sum(metric.expected_count for metric in metrics)
    predicted_items = sum(metric.predicted_count for metric in metrics)
    matched_items = sum(metric.matched_items for metric in metrics)

    return LineItemSummary(
        documents_with_line_items=len(metrics),
        expected_items=expected_items,
        predicted_items=predicted_items,
        matched_items=matched_items,
        aligned_items=aligned,
        item_match_rate=matched_items / aligned if aligned else 1.0,
        description_accuracy=(
            sum(metric.description_matches for metric in metrics) / aligned
            if aligned
            else 1.0
        ),
        quantity_accuracy=(
            sum(metric.quantity_matches for metric in metrics) / aligned
            if aligned
            else 1.0
        ),
        amount_accuracy=(
            sum(metric.amount_matches for metric in metrics) / aligned
            if aligned
            else 1.0
        ),
        count_match_accuracy=(
            sum(1 for metric in metrics if metric.count_match) / len(metrics)
        ),
    )


def _total_usage(documents: list[DocumentEvaluation]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for document in documents:
        if not document.usage:
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = document.usage.get(key)
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals
