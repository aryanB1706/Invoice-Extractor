"""Evidence-based failure categorization and human-review analysis."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from invoiceiq.config import get_settings
from invoiceiq.document_processing.loader import load_document
from invoiceiq.evaluation.models import DocumentEvaluation, EvaluationReport
from invoiceiq.schemas import DocumentQuality, Invoice
from invoiceiq.validation import validate_invoice

CATEGORIES = (
    ("ocr_errors", "OCR-associated errors"),
    ("poor_document_quality", "Poor document quality"),
    ("vendor_errors", "Incorrect vendor extraction"),
    ("invoice_number_errors", "Invoice number extraction errors"),
    ("date_errors", "Date extraction errors"),
    ("line_item_errors", "Line-item extraction errors"),
    ("tax_total_errors", "Tax and total mismatches"),
    ("classification_errors", "Non-invoice classification errors"),
    ("hallucinations", "LLM hallucinations / unsupported values"),
    ("missing_information", "Missing information"),
)


class FailureExample(BaseModel):
    """A representative failure with observed evidence and cautious attribution."""

    document: str
    evidence: list[str]
    assessment: str


class FailureCategory(BaseModel):
    """Documents assigned to one failure category."""

    code: str
    label: str
    document_count: int
    documents: list[str]
    examples: list[FailureExample]


class ReviewOnlyDocument(BaseModel):
    """A correctly extracted document conservatively flagged for review."""

    document: str
    reasons: list[str]


class ReviewEffectiveness(BaseModel):
    """Comparison of extraction errors for flagged and unflagged documents."""

    flagged_documents: int
    flagged_with_errors: int
    flagged_error_rate: float
    flagged_mean_field_errors: float
    unflagged_documents: int
    unflagged_with_errors: int
    unflagged_error_rate: float
    unflagged_mean_field_errors: float
    flagged_without_errors: list[ReviewOnlyDocument]


class FailureAnalysisReport(BaseModel):
    """Failure categories, examples, review comparison, and priorities."""

    generated_at: str
    source_evaluation_generated_at: str
    model: str | None
    total_documents: int
    documents_with_errors: int
    categories: list[FailureCategory]
    review_effectiveness: ReviewEffectiveness
    highest_priority_improvements: list[str]
    attribution_note: str


def build_failure_analysis(report: EvaluationReport) -> FailureAnalysisReport:
    """Categorize observed mismatches without overstating root-cause evidence."""
    examples: dict[str, dict[str, FailureExample]] = {
        code: {} for code, _ in CATEGORIES
    }

    for document in report.documents:
        mismatches = {
            field for field, comparison in document.fields.items() if not comparison.match
        }
        if not mismatches:
            continue

        if (
            document.document_quality
            and document.document_quality.ocr_pages
            and mismatches.intersection({"currency", "line_items", "tax_total", "total"})
        ):
            quality = document.document_quality
            _add_example(
                examples,
                "ocr_errors",
                document,
                [
                    f"OCR used on {quality.ocr_pages}/{quality.page_count} pages.",
                    f"Text extracted: {quality.text_characters} characters.",
                    "Mismatched fields: "
                    + ", ".join(sorted(mismatches.intersection(
                        {"currency", "line_items", "tax_total", "total"}
                    ))),
                    *quality.page_errors,
                ],
                "The error followed OCR processing. OCR errors and downstream table "
                "interpretation are both plausible; the comparison alone cannot "
                "attribute causality to one stage.",
            )

        sparse_ocr = (
            document.document_quality
            and document.document_quality.ocr_pages
            and document.document_quality.text_characters
            < get_settings().ocr_min_text_chars
        )
        if document.document_quality and (
            document.document_quality.page_errors
            or document.document_quality.text_characters == 0
            or sparse_ocr
        ):
            _add_example(
                examples,
                "poor_document_quality",
                document,
                [
                    *document.document_quality.page_errors,
                    f"Extracted text: {document.document_quality.text_characters} characters.",
                    *(
                        ["OCR text is below the configured minimum."]
                        if sparse_ocr
                        else []
                    ),
                ],
                "A page-processing error, empty text, or sparse OCR output was "
                "recorded; this is direct evidence of a readability concern.",
            )

        for field, code in (
            ("vendor", "vendor_errors"),
            ("invoice_number", "invoice_number_errors"),
            ("invoice_date", "date_errors"),
            ("line_items", "line_item_errors"),
            ("is_invoice", "classification_errors"),
        ):
            if field in mismatches:
                assessment = (
                    "OCR was used and item-level fields disagree with Ground Truth. "
                    "OCR damage and downstream table interpretation are both "
                    "plausible; this comparison cannot isolate the cause."
                    if field == "line_items"
                    and document.document_quality
                    and document.document_quality.ocr_pages
                    else "The field differs from Ground Truth. The available result "
                    "does not establish why the extractor made this mistake."
                )
                _add_example(
                    examples,
                    code,
                    document,
                    _field_evidence(document, field),
                    assessment,
                )

        if mismatches.intersection({"tax_total", "total"}):
            evidence = [
                *(
                    _field_evidence(document, "tax_total")
                    if "tax_total" in mismatches
                    else []
                ),
                *(
                    _field_evidence(document, "total")
                    if "total" in mismatches
                    else []
                ),
            ]
            issues = _validation_messages(document, {"tax_total", "total"})
            evidence.extend(f"Validation: {issue}" for issue in issues)
            _add_example(
                examples,
                "tax_total_errors",
                document,
                evidence,
                _tax_assessment(document),
            )

        missing = _missing_information(document)
        if missing:
            _add_example(
                examples,
                "missing_information",
                document,
                missing,
                _missing_assessment(document),
            )

        unsupported = _unsupported_values(document)
        if unsupported:
            _add_example(
                examples,
                "hallucinations",
                document,
                unsupported,
                "A value is populated where Ground Truth is null. This is a "
                "candidate unsupported value, not proof of hallucination without "
                "checking the source document.",
            )

    categories = [
        FailureCategory(
            code=code,
            label=label,
            document_count=len(examples[code]),
            documents=sorted(examples[code]),
            examples=list(examples[code].values())[:5],
        )
        for code, label in CATEGORIES
    ]
    review = _review_effectiveness(report.documents)
    return FailureAnalysisReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source_evaluation_generated_at=report.generated_at,
        model=report.model,
        total_documents=len(report.documents),
        documents_with_errors=sum(not document.fully_correct for document in report.documents),
        categories=categories,
        review_effectiveness=review,
        highest_priority_improvements=_priorities(categories, review),
        attribution_note=(
            "Categories describe observed mismatches, not guaranteed root causes. "
            "OCR use and page warnings are recorded when available. Ground Truth "
            "comparison cannot prove an LLM hallucination or distinguish OCR "
            "damage from downstream interpretation without checking source text."
        ),
    )


def format_failure_markdown(analysis: FailureAnalysisReport) -> str:
    """Format the analysis as a standalone Markdown report."""
    review = analysis.review_effectiveness
    lines = [
        "# InvoiceIQ Failure Analysis",
        "",
        f"- Generated: {analysis.generated_at}",
        f"- Source evaluation generated: {analysis.source_evaluation_generated_at}",
        f"- Model: `{analysis.model or '<unknown>'}`",
        f"- Documents analyzed: {analysis.total_documents}",
        f"- Documents with at least one extraction error: "
        f"{analysis.documents_with_errors}",
        "",
        "## Failure categories",
        "",
        "Categories can overlap. Counts are documents with evidence for each "
        "category; they are not mutually exclusive.",
        "",
        "| Category | Documents | Examples |",
        "| --- | ---: | --- |",
    ]
    for category in analysis.categories:
        examples = ", ".join(category.documents) or "None observed"
        lines.append(
            f"| {category.label} | {category.document_count} | {examples} |"
        )

    lines.extend(["", "## Representative examples", ""])
    for category in analysis.categories:
        if not category.examples:
            continue
        lines.extend([f"### {category.label}", ""])
        for example in category.examples:
            lines.append(f"**{example.document}**")
            lines.extend(f"- Evidence: {_escape(value)}" for value in example.evidence)
            lines.append(f"- Assessment: {_escape(example.assessment)}")
            lines.append("")

    lines.extend(
        [
            "## Human-review signal",
            "",
            "This compares deterministic `needs_review` status with Ground Truth "
            "errors. For older saved results, status is recalculated from the saved "
            "prediction and current validator. This is an empirical check on this "
            "small labelled set, not a calibrated confidence score.",
            "",
            "| Group | Docs | With errors | Error rate | Mean wrong fields/document |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Flagged | {review.flagged_documents} | {review.flagged_with_errors} | "
            f"{_percent_or_na(review.flagged_error_rate, review.flagged_documents)} | "
            f"{review.flagged_mean_field_errors:.2f} |",
            f"| Not flagged | {review.unflagged_documents} | "
            f"{review.unflagged_with_errors} | "
            f"{_percent_or_na(review.unflagged_error_rate, review.unflagged_documents)} | "
            f"{review.unflagged_mean_field_errors:.2f} |",
            "",
        ]
    )
    if review.flagged_without_errors:
        lines.extend(
            [
                "Some flagged documents have no Ground Truth field mismatch but "
                "were routed for review because information was missing or quality "
                "signals were present:",
                "",
            ]
        )
        for item in review.flagged_without_errors:
            lines.append(
                f"- {item.document}: {'; '.join(_escape(reason) for reason in item.reasons)}"
            )
        lines.append("")

    lines.extend(["## Highest-priority improvements", ""])
    lines.extend(
        f"{index}. {recommendation}"
        for index, recommendation in enumerate(
            analysis.highest_priority_improvements, start=1
        )
    )
    lines.extend(
        [
            "",
            "## Attribution limits",
            "",
            analysis.attribution_note,
            "",
        ]
    )
    return "\n".join(lines)


def write_failure_analysis(
    report: EvaluationReport, output_dir: str | Path
) -> FailureAnalysisReport:
    """Persist JSON and Markdown failure-analysis artifacts."""
    analysis = build_failure_analysis(report)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "failure_analysis.json").write_text(
        analysis.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (directory / "failure_analysis.md").write_text(
        format_failure_markdown(analysis), encoding="utf-8"
    )
    return analysis


def analyze_saved_evaluation(
    evaluation_file: str | Path,
    *,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> FailureAnalysisReport:
    """Analyze Prompt 8 output without making new LLM calls."""
    source = Path(evaluation_file)
    report = EvaluationReport.model_validate_json(source.read_text(encoding="utf-8"))
    documents_dir = Path(data_dir) if data_dir is not None else Path(report.data_dir)
    enriched_documents = []

    for document in report.documents:
        quality = _read_document_quality(documents_dir / document.document)
        if quality is None:
            quality = document.document_quality
        needs_review = _current_review_status(document, quality)
        enriched_documents.append(
            document.model_copy(
                update={
                    "document_quality": quality,
                    "needs_review": needs_review,
                }
            )
        )

    enriched_report = report.model_copy(update={"documents": enriched_documents})
    destination = Path(output_dir) if output_dir is not None else source.parent
    return write_failure_analysis(enriched_report, destination)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.analyze_failures",
        description="Analyze saved InvoiceIQ evaluation results without new LLM calls.",
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=settings.output_dir / "evaluation" / "evaluation.json",
        help="Saved evaluation.json from Prompt 8.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the evaluated documents (defaults to the report path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Failure report directory (defaults next to evaluation.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_saved_evaluation(
            args.evaluation_file,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    review = analysis.review_effectiveness
    print(
        f"Analyzed {analysis.total_documents} documents; "
        f"{analysis.documents_with_errors} contain extraction errors."
    )
    print(
        f"Review flagged: {review.flagged_documents}; "
        f"flagged error rate: "
        f"{_percent_or_na(review.flagged_error_rate, review.flagged_documents)}; "
        f"unflagged error rate: "
        f"{_percent_or_na(review.unflagged_error_rate, review.unflagged_documents)}"
    )
    output = Path(args.output_dir) if args.output_dir else args.evaluation_file.parent
    print(f"Saved failure_analysis.md and failure_analysis.json to {output}")
    return 0


def _read_document_quality(path: Path) -> DocumentQuality | None:
    if not path.is_file():
        return None
    try:
        pages = load_document(path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return DocumentQuality(
            page_count=0,
            ocr_pages=0,
            text_characters=0,
            page_errors=[f"{type(exc).__name__}: {exc}"],
        )
    return DocumentQuality(
        page_count=len(pages),
        ocr_pages=sum(page.used_ocr for page in pages),
        text_characters=sum(
            not character.isspace()
            for page in pages
            for character in page.text
        ),
        page_errors=[page.error for page in pages if page.error],
    )


def _current_review_status(
    document: DocumentEvaluation, quality: DocumentQuality | None
) -> bool:
    return bool(_review_reasons(document, quality))


def _review_reasons(
    document: DocumentEvaluation, quality: DocumentQuality | None = None
) -> list[str]:
    reasons = []
    if document.error:
        reasons.append(f"Extraction failed: {document.error}")
    reasons.extend(document.warnings)
    if quality:
        reasons.extend(quality.page_errors)
        if (
            quality.ocr_pages
            and quality.text_characters < get_settings().ocr_min_text_chars
        ):
            reasons.append("OCR output is below the configured text threshold.")
    if document.predicted is None:
        return reasons or ["No prediction was produced."]
    try:
        invoice = Invoice.model_validate(document.predicted)
    except ValueError:
        return reasons + ["Prediction does not satisfy the invoice schema."]
    if invoice.is_invoice:
        validation = validate_invoice(invoice)
        reasons.extend(issue.message for issue in validation.issues)
    return reasons


def _add_example(
    examples: dict[str, dict[str, FailureExample]],
    category: str,
    document: DocumentEvaluation,
    evidence: list[str],
    assessment: str,
) -> None:
    examples[category][document.document] = FailureExample(
        document=document.document,
        evidence=evidence,
        assessment=assessment,
    )


def _field_evidence(document: DocumentEvaluation, field: str) -> list[str]:
    if field == "line_items" and document.line_item_metrics is not None:
        metrics = document.line_item_metrics
        evidence = [
            f"Expected/predicted item count: "
            f"{metrics.expected_count}/{metrics.predicted_count}.",
            f"Matching descriptions/quantities/amounts: "
            f"{metrics.description_matches}/{metrics.aligned_items}, "
            f"{metrics.quantity_matches}/{metrics.aligned_items}, "
            f"{metrics.amount_matches}/{metrics.aligned_items}.",
        ]
        evidence.extend(metrics.details[:5])
        if len(metrics.details) > 5:
            evidence.append(f"{len(metrics.details) - 5} additional item mismatch(es).")
        return evidence
    comparison = document.fields[field]
    return [
        f"{field}: expected {_short(comparison.expected)}; "
        f"predicted {_short(comparison.predicted)}."
    ]


def _missing_assessment(document: DocumentEvaluation) -> str:
    if document.document_quality and document.document_quality.ocr_pages:
        return (
            "Ground Truth contains values absent from the prediction, and OCR was "
            "used. OCR/table recovery is a plausible contributor, but downstream "
            "field selection cannot be ruled out."
        )
    if document.document_quality:
        return (
            "Ground Truth contains values absent from the prediction. The document "
            "was read without OCR, so the mismatch is in extraction/field selection; "
            "the exact model-side cause is not established."
        )
    return (
        "Ground Truth contains required data that is absent from the prediction; "
        "the saved result has no document-processing evidence to identify a cause."
    )


def _validation_messages(
    document: DocumentEvaluation, fields: set[str]
) -> list[str]:
    if not document.validation:
        return []
    return [
        str(issue.get("message", ""))
        for issue in document.validation.get("issues", [])
        if issue.get("field") in fields
    ]


def _tax_assessment(document: DocumentEvaluation) -> str:
    expected_tax = document.fields["tax_total"].expected
    predicted_tax = document.fields["tax_total"].predicted
    expected_total = document.fields["total"].expected
    if (
        predicted_tax is not None
        and expected_total is not None
        and predicted_tax == expected_total
        and predicted_tax != expected_tax
    ):
        return (
            "Predicted tax_total equals the Ground Truth total while differing "
            "from Ground Truth tax_total, consistent with a tax/total field mix-up."
        )
    messages = _validation_messages(document, {"tax_total", "total"})
    if messages:
        return "Validation detected incomplete or inconsistent totals: " + "; ".join(messages)
    return (
        "Tax or total differs from Ground Truth; the values alone do not prove "
        "whether this came from OCR, extraction, or arithmetic."
    )


def _missing_information(document: DocumentEvaluation) -> list[str]:
    missing = []
    for field, comparison in document.fields.items():
        if comparison.expected is not None and comparison.predicted is None:
            missing.append(f"{field} is null; Ground Truth is {_short(comparison.expected)}.")

    expected_items = (document.expected or {}).get("line_items") or []
    predicted_items = (document.predicted or {}).get("line_items") or []
    for index, expected in enumerate(expected_items):
        predicted = predicted_items[index] if index < len(predicted_items) else {}
        for field in ("description", "quantity", "amount"):
            if expected.get(field) is not None and predicted.get(field) is None:
                missing.append(f"line_items[{index + 1}].{field} is missing.")
    if len(predicted_items) < len(expected_items):
        missing.append(
            f"{len(expected_items) - len(predicted_items)} expected line item(s) are absent."
        )
    return missing


def _unsupported_values(document: DocumentEvaluation) -> list[str]:
    unsupported = [
        f"{field} is populated as {_short(comparison.predicted)} while Ground Truth is null."
        for field, comparison in document.fields.items()
        if comparison.expected is None and comparison.predicted is not None
    ]
    expected_items = (document.expected or {}).get("line_items") or []
    predicted_items = (document.predicted or {}).get("line_items") or []
    for index, predicted in enumerate(predicted_items):
        if index >= len(expected_items):
            unsupported.append(f"Unexpected predicted line item {index + 1}.")
            continue
        expected = expected_items[index]
        for field in ("description", "quantity", "amount"):
            if expected.get(field) is None and predicted.get(field) is not None:
                unsupported.append(
                    f"line_items[{index + 1}].{field} is populated as "
                    f"{_short(predicted[field])} while Ground Truth is null."
                )
    return unsupported


def _review_effectiveness(
    documents: list[DocumentEvaluation],
) -> ReviewEffectiveness:
    flagged = [document for document in documents if document.needs_review]
    unflagged = [document for document in documents if not document.needs_review]
    flagged_errors = [document for document in flagged if not document.fully_correct]
    unflagged_errors = [document for document in unflagged if not document.fully_correct]
    flagged_without_errors = [
        ReviewOnlyDocument(
            document=document.document,
            reasons=_review_reasons(document, document.document_quality)
            or ["The pipeline requested review despite no recorded validation issue."],
        )
        for document in flagged
        if document.fully_correct
    ]

    return ReviewEffectiveness(
        flagged_documents=len(flagged),
        flagged_with_errors=len(flagged_errors),
        flagged_error_rate=len(flagged_errors) / len(flagged) if flagged else 0.0,
        flagged_mean_field_errors=_mean_field_errors(flagged),
        unflagged_documents=len(unflagged),
        unflagged_with_errors=len(unflagged_errors),
        unflagged_error_rate=(
            len(unflagged_errors) / len(unflagged) if unflagged else 0.0
        ),
        unflagged_mean_field_errors=_mean_field_errors(unflagged),
        flagged_without_errors=flagged_without_errors,
    )


def _mean_field_errors(documents: list[DocumentEvaluation]) -> float:
    if not documents:
        return 0.0
    errors = sum(
        sum(not comparison.match for comparison in document.fields.values())
        for document in documents
    )
    return errors / len(documents)


def _priorities(
    categories: list[FailureCategory], review: ReviewEffectiveness
) -> list[str]:
    counts = {category.code: category.document_count for category in categories}
    priorities = []
    if counts["line_item_errors"] or counts["ocr_errors"]:
        priorities.append(
            "Improve OCR/table row and column recovery, especially quantities and "
            "item boundaries; route OCR-derived invoices with missing line-item "
            "fields to human review."
        )
    if counts["tax_total_errors"]:
        priorities.append(
            "Strengthen tax-versus-grand-total separation and arithmetic checks. "
            "Keep tax missing/field mix-ups visible instead of filling them by inference."
        )
    if review.flagged_mean_field_errors > review.unflagged_mean_field_errors:
        priorities.append(
            "Retain the deterministic review gates: flagged documents currently "
            "have more mismatched fields on average than unflagged documents."
        )
    if counts["vendor_errors"] or counts["invoice_number_errors"] or counts["date_errors"]:
        priorities.append(
            "Add targeted checks for vendor, invoice number, and date extraction."
        )
    priorities.append(
        "Use a separate development set for any prompt/model changes; preserve "
        "the labelled assignment set as a held-out evaluation."
    )
    return priorities


def _short(value: object, limit: int = 120) -> str:
    text = "null" if value is None else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def _percent_or_na(value: float, count: int) -> str:
    return f"{value * 100:.1f}%" if count else "n/a"
