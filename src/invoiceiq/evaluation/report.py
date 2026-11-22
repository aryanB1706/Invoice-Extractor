"""Human-readable console and Markdown reports."""

from invoiceiq.evaluation.models import EvaluationReport

FIELD_ORDER = (
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
)

FIELD_LABELS = {
    "is_invoice": "inv",
    "vendor": "vendor",
    "invoice_number": "number",
    "invoice_date": "date",
    "currency": "curr",
    "line_items": "items",
    "tax_total": "tax",
    "total": "total",
}


def format_console_report(report: EvaluationReport) -> str:
    """Return a readable plain-text evaluation summary."""
    summary = report.summary
    lines = [
        f"Model: {report.model or '<unknown>'}",
        f"Documents: {summary.total_documents}  "
        f"Failed: {summary.failed_documents}  "
        f"Fully correct: {summary.fully_correct_documents}",
        f"Overall accuracy (macro mean of 8 fields): "
        f"{_percent(summary.overall_accuracy)}",
        "",
        f"{'Field':<16}{'Correct':>8}{'Total':>8}{'Accuracy':>10}",
        f"{'-' * 16}{'-' * 8}{'-' * 8}{'-' * 10}",
    ]
    for field in FIELD_ORDER:
        field_summary = summary.fields[field]
        lines.append(
            f"{field:<16}{field_summary.correct:>8}{field_summary.total:>8}"
            f"{_percent(field_summary.accuracy):>10}"
        )

    confusion = summary.invoice_confusion
    lines.extend(
        [
            "",
            "Invoice classification: "
            f"TP={confusion['true_positive']} "
            f"FP={confusion['false_positive']} "
            f"FN={confusion['false_negative']} "
            f"TN={confusion['true_negative']}",
        ]
    )

    if summary.line_items is not None:
        items = summary.line_items
        lines.extend(
            [
                "",
                f"Line items: {items.matched_items}/{items.aligned_items} matched "
                f"({_percent(items.item_match_rate)})",
                f"  description {_percent(items.description_accuracy)} | "
                f"quantity {_percent(items.quantity_accuracy)} | "
                f"amount {_percent(items.amount_accuracy)} | "
                f"count {_percent(items.count_match_accuracy)}",
            ]
        )

    header = f"{'Document':<12}" + "".join(
        f"{FIELD_LABELS[field]:>8}" for field in FIELD_ORDER
    )
    lines.extend(["", header, "-" * len(header)])
    for document in report.documents:
        row = f"{document.document:<12}" + "".join(
            f"{'Y' if document.fields[field].match else 'N':>8}"
            for field in FIELD_ORDER
        )
        if document.error:
            row += f"  ERROR: {_short(document.error)}"
        lines.append(row)

    return "\n".join(lines)


def format_markdown(report: EvaluationReport) -> str:
    """Return the persisted Markdown report."""
    summary = report.summary
    lines = [
        "# InvoiceIQ Evaluation Report",
        "",
        f"- Generated: {report.generated_at}",
        f"- Data directory: `{report.data_dir}`",
        f"- Output directory: `{report.output_dir}`",
        f"- Model: `{report.model or '<unknown>'}`",
        f"- Documents: {summary.total_documents}",
        f"- Failed extractions: {summary.failed_documents}",
        f"- Numeric tolerance: {summary.tolerance}",
        "",
        "## Overall accuracy",
        "",
        "Overall accuracy is the unweighted mean of the eight field-level "
        "accuracies. Each field has equal weight. `line_items` is a strict "
        "whole-list match: the item count must match and every item must match "
        "on description, quantity, and amount.",
        "",
        f"**Overall accuracy: {_percent(summary.overall_accuracy)}**",
        "",
        f"- Fully correct documents: {summary.fully_correct_documents}"
        f"/{summary.total_documents}",
        f"- Failed extractions counted as incorrect: {summary.failed_documents}",
        "",
        "## Field-level accuracy",
        "",
        "| Field | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in FIELD_ORDER:
        field_summary = summary.fields[field]
        lines.append(
            f"| {field} | {field_summary.correct} | {field_summary.total} | "
            f"{_percent(field_summary.accuracy)} |"
        )

    confusion = summary.invoice_confusion
    lines.extend(
        [
            "",
            "## Invoice classification",
            "",
            "| True positive | False positive | False negative | True negative |",
            "| ---: | ---: | ---: | ---: |",
            f"| {confusion['true_positive']} | {confusion['false_positive']} | "
            f"{confusion['false_negative']} | {confusion['true_negative']} |",
        ]
    )

    if summary.line_items is not None:
        items = summary.line_items
        lines.extend(
            [
                "",
                "## Line-item metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Documents with line items | {items.documents_with_line_items} |",
                f"| Expected items | {items.expected_items} |",
                f"| Predicted items | {items.predicted_items} |",
                f"| Matched items | {items.matched_items} |",
                f"| Item match rate | {_percent(items.item_match_rate)} |",
                f"| Description accuracy | {_percent(items.description_accuracy)} |",
                f"| Quantity accuracy | {_percent(items.quantity_accuracy)} |",
                f"| Amount accuracy | {_percent(items.amount_accuracy)} |",
                f"| Item-count accuracy | {_percent(items.count_match_accuracy)} |",
            ]
        )

    header = "| Document | " + " | ".join(FIELD_ORDER) + " | Fully correct | Error |"
    separator = "| --- | " + " | ".join("---" for _ in FIELD_ORDER) + " | --- | --- |"
    lines.extend(
        [
            "",
            "## Per-document results",
            "",
            header,
            separator,
        ]
    )
    for document in report.documents:
        cells = [
            "Y" if document.fields[field].match else "N" for field in FIELD_ORDER
        ]
        lines.append(
            f"| {document.document} | " + " | ".join(cells) + " | "
            f"{'Y' if document.fully_correct else 'N'} | "
            f"{_escape(document.error or '')} |"
        )

    lines.extend(["", "## Matching rules", ""])
    lines.extend(f"- {rule}" for rule in report.matching_rules)
    lines.extend(["", "## Tuning disclosure", "", report.tuning_disclosure, ""])
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _short(value: str, limit: int = 60) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _escape(value: str) -> str:
    return _short(value).replace("|", "\\|")
