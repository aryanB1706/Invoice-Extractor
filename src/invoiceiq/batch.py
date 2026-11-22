"""Process the complete assignment pack and write final deliverables."""

import argparse
import json
import math
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from invoiceiq.api.service import process_document
from invoiceiq.config import PROJECT_ROOT, Settings, get_settings
from invoiceiq.evaluation.failure_analysis import FailureAnalysisReport
from invoiceiq.evaluation.runner import (
    discover_labelled_documents,
    evaluate_dataset,
)
from invoiceiq.schemas import (
    DocumentQuality,
    ExtractionResponse,
    Invoice,
    TokenUsage,
    ValidationReport,
)

DOCUMENT_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
GROQ_MODEL_DOCS = "https://console.groq.com/docs/models"


class AssignmentDocument(BaseModel):
    """A source document from one of the two provided assignment splits."""

    path: Path
    split: Literal["labelled", "unlabelled"]
    ground_truth_reference: str | None = None


class DocumentOutput(BaseModel):
    """Traceable extraction output for one source document."""

    source_filename: str
    source_relative_path: str
    dataset_split: Literal["labelled", "unlabelled"]
    ground_truth_reference: str | None = None
    status: Literal["success", "failed"]
    model: str
    invoice: Invoice | None = None
    validation: ValidationReport | None = None
    needs_review: bool
    warnings: list[str] = Field(default_factory=list)
    document_quality: DocumentQuality | None = None
    token_usage: TokenUsage | None = None
    list_price_estimate_usd: float | None = None
    error: str | None = None


class CostAnalysis(BaseModel):
    """Observed usage and transparent list-price calculation."""

    model: str
    pricing_source_url: str
    pricing_as_of: str
    input_price_per_million_usd: float
    cached_input_price_per_million_usd: float
    output_price_per_million_usd: float
    total_documents: int
    successful_documents: int
    failed_documents: int
    documents_with_complete_usage: int
    documents_missing_input_usage: list[str]
    documents_missing_output_usage: list[str]
    documents_missing_cached_usage: list[str]
    documents_with_invalid_usage: list[str]
    observed_input_tokens: int
    observed_output_tokens: int
    observed_cached_input_tokens: int
    cost_for_calls_with_complete_usage_usd: float
    total_api_cost_usd: float | None
    average_cost_per_successful_document_usd: float | None
    average_cost_per_successful_document_with_usage_usd: float | None
    billing_note: str
    calculation_method: str


class BatchSummary(BaseModel):
    """Counts for the 20-document batch."""

    generated_at: str
    source_directory: str
    output_directory: str
    total_documents: int
    labelled_documents: int
    unlabelled_documents: int
    successful_documents: int
    failed_documents: int
    model: str
    pricing: CostAnalysis


def discover_assignment_documents(data_dir: str | Path) -> list[AssignmentDocument]:
    """Find all provided labelled and unlabelled documents without making labels."""
    root = Path(data_dir)
    labelled = discover_labelled_documents(root / "labelled")
    unlabelled_dir = root / "unlabelled"
    if not unlabelled_dir.is_dir():
        raise FileNotFoundError(f"Unlabelled directory not found: {unlabelled_dir}")

    unlabelled = sorted(
        (
            path
            for path in unlabelled_dir.iterdir()
            if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(labelled) != 10 or len(unlabelled) != 10:
        raise ValueError(
            "Expected 10 labelled and 10 unlabelled assignment documents; found "
            f"{len(labelled)} labelled and {len(unlabelled)} unlabelled."
        )

    documents = [
        AssignmentDocument(
            path=item.document,
            split="labelled",
            ground_truth_reference=item.ground_truth.name,
        )
        for item in labelled
    ]
    documents.extend(
        AssignmentDocument(path=path, split="unlabelled")
        for path in unlabelled
    )
    stems = [item.path.stem.casefold() for item in documents]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate document stems would overwrite output JSON files.")
    return documents


def estimate_list_price_cost(
    usage: TokenUsage | None, settings: Settings
) -> float | None:
    """Price a request only when both input and output counts are reported."""
    if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        return None

    cached = usage.cached_prompt_tokens or 0
    if cached > usage.prompt_tokens:
        return None
    uncached = usage.prompt_tokens - cached
    cost = (
        Decimal(uncached)
        * Decimal(str(settings.groq_input_price_per_million_usd))
        + Decimal(cached)
        * Decimal(str(settings.groq_cached_input_price_per_million_usd))
        + Decimal(usage.completion_tokens)
        * Decimal(str(settings.groq_output_price_per_million_usd))
    ) / Decimal(1_000_000)
    return round(float(cost), 10)


def run_batch(
    *,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    process: Callable[..., ExtractionResponse] | None = None,
    progress: Callable[[int, int, AssignmentDocument], None] | None = None,
    report_path: str | Path | None = None,
) -> BatchSummary:
    """Extract all 20 documents, then evaluate the labelled predictions."""
    resolved_settings = settings or get_settings()
    if not resolved_settings.groq_configured:
        raise RuntimeError("Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL.")
    if resolved_settings.groq_model != resolved_settings.groq_pricing_model:
        raise ValueError(
            f"Pricing is configured for {resolved_settings.groq_pricing_model!r}, "
            f"but GROQ_MODEL is {resolved_settings.groq_model!r}. Update the pricing "
            "model and rates before running cost analysis."
        )
    _validate_pricing(resolved_settings)

    source_dir = (
        Path(data_dir)
        if data_dir
        else resolved_settings.data_dir / "samples" / "assignment"
    )
    destination = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"
    destination.mkdir(parents=True, exist_ok=True)
    documents = discover_assignment_documents(source_dir)
    processor = process or process_document
    outputs: list[DocumentOutput] = []

    for index, item in enumerate(documents, start=1):
        if progress is not None:
            progress(index, len(documents), item)
        captured_quality: list[DocumentQuality] = []

        try:
            response = processor(
                item.path,
                on_pages_loaded=captured_quality.append,
            )
            quality = captured_quality[0] if captured_quality else None
            result = DocumentOutput(
                source_filename=item.path.name,
                source_relative_path=item.path.relative_to(source_dir).as_posix(),
                dataset_split=item.split,
                ground_truth_reference=item.ground_truth_reference,
                status="success",
                model=response.model or resolved_settings.groq_model,
                invoice=response.invoice,
                validation=response.validation,
                needs_review=response.needs_review,
                warnings=response.warnings,
                document_quality=quality,
                token_usage=response.usage,
                list_price_estimate_usd=estimate_list_price_cost(
                    response.usage, resolved_settings
                ),
            )
        except Exception as exc:  # noqa: BLE001 - record failures and continue the batch
            result = DocumentOutput(
                source_filename=item.path.name,
                source_relative_path=item.path.relative_to(source_dir).as_posix(),
                dataset_split=item.split,
                ground_truth_reference=item.ground_truth_reference,
                status="failed",
                model=resolved_settings.groq_model,
                needs_review=True,
                document_quality=captured_quality[0] if captured_quality else None,
                error=f"{type(exc).__name__}: {exc}",
            )

        _write_json(destination / f"{item.path.stem}.json", result.model_dump(mode="json"))
        outputs.append(result)

    pricing = _build_cost_analysis(outputs, resolved_settings)
    evaluation = _evaluate_saved_labelled_outputs(
        source_dir=source_dir,
        output_dir=destination,
        evaluation_dir=destination / "evaluation",
        settings=resolved_settings,
    )
    failure_analysis = FailureAnalysisReport.model_validate_json(
        (destination / "evaluation" / "failure_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    summary = BatchSummary(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source_directory=str(source_dir),
        output_directory=str(destination),
        total_documents=len(outputs),
        labelled_documents=sum(item.dataset_split == "labelled" for item in outputs),
        unlabelled_documents=sum(item.dataset_split == "unlabelled" for item in outputs),
        successful_documents=sum(item.status == "success" for item in outputs),
        failed_documents=sum(item.status == "failed" for item in outputs),
        model=resolved_settings.groq_model,
        pricing=pricing,
    )
    _write_json(destination / "cost_analysis.json", pricing.model_dump(mode="json"))
    _write_json(destination / "batch_summary.json", summary.model_dump(mode="json"))
    final_report_path = Path(report_path) if report_path else PROJECT_ROOT / "REPORT.md"
    final_report_path.write_text(
        _format_final_report(summary, evaluation, failure_analysis, outputs),
        encoding="utf-8",
    )
    return summary


def _validate_pricing(settings: Settings) -> None:
    rates = (
        settings.groq_input_price_per_million_usd,
        settings.groq_cached_input_price_per_million_usd,
        settings.groq_output_price_per_million_usd,
    )
    if not all(math.isfinite(rate) and rate >= 0 for rate in rates):
        raise ValueError("Configured Groq token prices must be finite and non-negative.")
    if not settings.groq_pricing_source_url.strip():
        raise ValueError("GROQ_PRICING_SOURCE_URL cannot be empty.")


def _build_cost_analysis(
    outputs: list[DocumentOutput], settings: Settings
) -> CostAnalysis:
    missing_input = [
        item.source_filename
        for item in outputs
        if item.token_usage is None or item.token_usage.prompt_tokens is None
    ]
    missing_output = [
        item.source_filename
        for item in outputs
        if item.token_usage is None or item.token_usage.completion_tokens is None
    ]
    missing_cached = [
        item.source_filename
        for item in outputs
        if item.token_usage is None or item.token_usage.cached_prompt_tokens is None
    ]
    invalid_usage = [
        item.source_filename
        for item in outputs
        if item.token_usage
        and item.token_usage.prompt_tokens is not None
        and item.token_usage.cached_prompt_tokens is not None
        and item.token_usage.cached_prompt_tokens > item.token_usage.prompt_tokens
    ]
    priced = [
        item for item in outputs if item.list_price_estimate_usd is not None
    ]
    successful = [item for item in outputs if item.status == "success"]
    priced_successful = [
        item for item in successful if item.list_price_estimate_usd is not None
    ]
    complete_usage = len(priced)
    known_cost = sum(
        (Decimal(str(item.list_price_estimate_usd)) for item in priced),
        start=Decimal(0),
    )
    successful_cost = sum(
        (Decimal(str(item.list_price_estimate_usd)) for item in priced_successful),
        start=Decimal(0),
    )
    usage_complete_for_all = (
        not missing_input
        and not missing_output
        and not invalid_usage
        and len(priced) == len(outputs)
    )
    usage_complete_for_successes = len(priced_successful) == len(successful)
    input_tokens = sum(
        item.token_usage.prompt_tokens
        for item in outputs
        if item.token_usage and item.token_usage.prompt_tokens is not None
    )
    output_tokens = sum(
        item.token_usage.completion_tokens
        for item in outputs
        if item.token_usage and item.token_usage.completion_tokens is not None
    )
    cached_tokens = sum(
        item.token_usage.cached_prompt_tokens
        for item in outputs
        if item.token_usage and item.token_usage.cached_prompt_tokens is not None
    )

    return CostAnalysis(
        model=settings.groq_model,
        pricing_source_url=settings.groq_pricing_source_url,
        pricing_as_of=settings.groq_pricing_as_of.isoformat(),
        input_price_per_million_usd=settings.groq_input_price_per_million_usd,
        cached_input_price_per_million_usd=(
            settings.groq_cached_input_price_per_million_usd
        ),
        output_price_per_million_usd=settings.groq_output_price_per_million_usd,
        total_documents=len(outputs),
        successful_documents=len(successful),
        failed_documents=len(outputs) - len(successful),
        documents_with_complete_usage=complete_usage,
        documents_missing_input_usage=missing_input,
        documents_missing_output_usage=missing_output,
        documents_missing_cached_usage=missing_cached,
        documents_with_invalid_usage=invalid_usage,
        observed_input_tokens=input_tokens,
        observed_output_tokens=output_tokens,
        observed_cached_input_tokens=cached_tokens,
        cost_for_calls_with_complete_usage_usd=round(float(known_cost), 10),
        total_api_cost_usd=(
            round(float(known_cost), 10) if usage_complete_for_all else None
        ),
        average_cost_per_successful_document_usd=(
            round(float(successful_cost / len(successful)), 10)
            if successful and usage_complete_for_successes
            else None
        ),
        average_cost_per_successful_document_with_usage_usd=(
            round(float(successful_cost / len(priced_successful)), 10)
            if priced_successful
            else None
        ),
        billing_note=(
            "The batch used a Groq free-tier API key. The listed-price estimate is "
            "provided for comparison; the inference API does not expose account "
            "billing, so the actual charge is not independently verified. Usage "
            "within the free quota is expected to cost $0."
        ),
        calculation_method=(
            "Per request: (uncached input tokens × input USD/1M + cached input "
            "tokens × cached-input USD/1M + output tokens × output USD/1M) / "
            "1,000,000. A cached-input discount is applied only when the provider "
            "reports cached token counts. Missing input/output metadata is never "
            "treated as zero; if any request is missing either count, total API "
            "cost is reported as unknown."
        ),
    )


def _evaluate_saved_labelled_outputs(
    *,
    source_dir: Path,
    output_dir: Path,
    evaluation_dir: Path,
    settings: Settings,
):
    labelled_dir = source_dir / "labelled"
    labelled_documents = discover_labelled_documents(labelled_dir)

    def process_saved(
        path: Path,
    ) -> ExtractionResponse | tuple[ExtractionResponse, DocumentQuality]:
        record = json.loads((output_dir / f"{path.stem}.json").read_text(encoding="utf-8"))
        if record["status"] != "success":
            raise RuntimeError(record["error"] or "Extraction failed.")
        response = ExtractionResponse(
            invoice=Invoice.model_validate(record["invoice"]),
            validation=ValidationReport.model_validate(record["validation"]),
            needs_review=record["needs_review"],
            warnings=record["warnings"],
            model=record["model"],
            usage=(
                TokenUsage.model_validate(record["token_usage"])
                if record["token_usage"] is not None
                else None
            ),
        )
        quality_data = record["document_quality"]
        if quality_data is None:
            return response
        return response, DocumentQuality.model_validate(quality_data)

    return evaluate_dataset(
        labelled_documents,
        process=process_saved,
        tolerance=settings.invoice_amount_tolerance,
        data_dir=labelled_dir,
        output_dir=evaluation_dir,
    )


def _format_final_report(
    summary: BatchSummary,
    evaluation: Any,
    failure: FailureAnalysisReport,
    outputs: list[DocumentOutput],
) -> str:
    cost = summary.pricing
    lines = [
        "# InvoiceIQ Final Report",
        "",
        f"- Generated: {summary.generated_at}",
        f"- Model: `{summary.model}`",
        f"- Documents processed: {summary.total_documents} "
        f"({summary.successful_documents} successful, {summary.failed_documents} failed)",
        "",
        "## Project overview",
        "",
        "InvoiceIQ accepts PDF and image invoices, extracts structured invoice "
        "fields, validates required values and arithmetic, and exposes the pipeline "
        "through a FastAPI endpoint and command-line tools.",
        "",
        "## Architecture and workflow",
        "",
        "1. PyMuPDF reads embedded PDF text; Tesseract OCR handles images and scanned "
        "or low-text PDF pages.",
        "2. A LangChain `ChatGroq` structured-output call maps document text to the "
        "Pydantic `Invoice` schema.",
        "3. Schema checks reject invalid dates, non-finite numbers, extra fields, "
        "and populated fields on non-invoices.",
        "4. Business validation checks required invoice fields, line-item "
        "completeness, and whether line amounts plus tax equal the printed total.",
        "5. The batch command writes one JSON file per source document; labelled "
        "predictions are compared to Ground Truth from that same run.",
        "",
        "## Model selection",
        "",
        f"`{summary.model}` is the configured Groq model for this run. Groq's model "
        "documentation lists tool use, JSON Schema mode, and reasoning capabilities, "
        "which fit the structured invoice response. No comparative model benchmark "
        "was run, so model selection is configuration-driven rather than a claim "
        "that this model outperforms alternatives.",
        "",
        f"Source: {GROQ_MODEL_DOCS} (checked {cost.pricing_as_of}).",
        "",
        "## Extraction approach and prompting",
        "",
        "The extractor uses a strict Pydantic schema with nullable fields. The "
        "system prompt directs the model to use printed evidence, preserve every "
        "line-item row, map values by table headers, distinguish quantity/unit "
        "price/line amount/tax/grand total, normalize dates, avoid unsupported "
        "calculations, and return null for missing or ambiguous values. Unambiguous "
        "currency symbols may map to a currency code; address-based currency "
        "guessing is disallowed.",
        "",
        "### Approach that failed",
        "",
        "An earlier long-invoice request using Groq function-calling structured "
        "output returned malformed tool-call JSON for L04 once. A later full run "
        "succeeded on that document. This was an observed output-format failure, "
        "not evidence that extraction always fails for long invoices. The global "
        "completion-token setting was raised to 8192; failed documents are now "
        "recorded explicitly rather than receiving fabricated values.",
        "",
        "## Field-level accuracy",
        "",
        f"Accuracy is measured on the {evaluation.summary.total_documents} labelled "
        "documents using the batch predictions in this submission.",
        "",
        "| Field | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in (
        "is_invoice",
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    ):
        metric = evaluation.summary.fields[field]
        lines.append(
            f"| {field} | {metric.correct} | {metric.total} | "
            f"{_percent(metric.accuracy)} |"
        )
    lines.extend(
        [
            "",
            f"**Macro overall accuracy:** {_percent(evaluation.summary.overall_accuracy)} "
            "(unweighted mean of the eight field accuracies).",
            "",
            f"Fully correct documents: {evaluation.summary.fully_correct_documents}/"
            f"{evaluation.summary.total_documents}. This is a small labelled set; "
            "the measured score is descriptive, not a population guarantee.",
            "",
            "### Matching rules",
            "",
        ]
    )
    lines.extend(f"- {rule}" for rule in evaluation.matching_rules)
    if evaluation.summary.line_items:
        item = evaluation.summary.line_items
        lines.extend(
            [
                "",
                "Line-item detail: "
                f"{item.matched_items}/{item.aligned_items} aligned items fully match; "
                f"description {_percent(item.description_accuracy)}, "
                f"quantity {_percent(item.quantity_accuracy)}, "
                f"amount {_percent(item.amount_accuracy)}, "
                f"item-count {_percent(item.count_match_accuracy)}.",
            ]
        )

    lines.extend(["", "## Failure analysis", ""])
    lines.append(
        f"{failure.documents_with_errors}/{failure.total_documents} labelled "
        "documents contain at least one field mismatch. Categories may overlap."
    )
    lines.extend(["", "| Category | Documents | Examples |", "| --- | ---: | --- |"])
    for category in failure.categories:
        examples = ", ".join(category.documents) or "None observed"
        lines.append(f"| {category.label} | {category.document_count} | {examples} |")
    review = failure.review_effectiveness
    lines.extend(
        [
            "",
            "### Human-review effectiveness",
            "",
            "A document counts as having an extraction error when any evaluated "
            "field differs from Ground Truth. This is an empirical comparison on "
            "the small labelled set, not a calibrated confidence measure.",
            "",
            "| Review status | Documents | With errors | Error rate | Mean wrong fields/document |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Flagged | {review.flagged_documents} | {review.flagged_with_errors} | "
            f"{_percent(review.flagged_error_rate)} | "
            f"{review.flagged_mean_field_errors:.2f} |",
            f"| Not flagged | {review.unflagged_documents} | "
            f"{review.unflagged_with_errors} | "
            f"{_percent(review.unflagged_error_rate)} | "
            f"{review.unflagged_mean_field_errors:.2f} |",
            "",
            f"Flagged documents without a Ground Truth mismatch: "
            f"{len(review.flagged_without_errors)}.",
        ]
    )
    for code in ("line_item_errors", "tax_total_errors", "missing_information"):
        category = next((item for item in failure.categories if item.code == code), None)
        if not category or not category.examples:
            continue
        lines.extend(["", f"### {category.label}", ""])
        for example in category.examples[:3]:
            lines.append(f"**{example.document}** — {example.assessment}")
            lines.extend(f"- {evidence}" for evidence in example.evidence[:5])
            lines.append("")

    lines.extend(
        [
            "### Highest-priority improvements",
            "",
        ]
    )
    lines.extend(
        f"{index}. {recommendation}"
        for index, recommendation in enumerate(
            failure.highest_priority_improvements, start=1
        )
    )
    lines.extend(
        [
            "",
            "## Cost analysis",
            "",
            f"- Pricing model: `{cost.model}`",
            f"- Source: {cost.pricing_source_url} (checked {cost.pricing_as_of})",
            f"- Standard input: ${cost.input_price_per_million_usd:.3f}/1M tokens; "
            f"cached input: ${cost.cached_input_price_per_million_usd:.3f}/1M; "
            f"output: ${cost.output_price_per_million_usd:.3f}/1M.",
            f"- Input tokens observed: {cost.observed_input_tokens:,}; output tokens "
            f"observed: {cost.observed_output_tokens:,}; cached input tokens "
            f"observed: {cost.observed_cached_input_tokens:,}.",
            f"- Documents with complete input/output usage: "
            f"{cost.documents_with_complete_usage}/{cost.total_documents}.",
            f"- Cost for documents with complete usage: "
            f"${cost.cost_for_calls_with_complete_usage_usd:.8f}.",
            f"- Total API list-price estimate: "
            f"{_money_or_unknown(cost.total_api_cost_usd)}.",
            f"- Average per successfully processed document: "
            f"{_money_or_unknown(cost.average_cost_per_successful_document_usd)}.",
            "",
            cost.calculation_method,
            "",
            cost.billing_note,
            "",
            "Missing token metadata is listed in `outputs/cost_analysis.json`; "
            "missing counts are not treated as zero. Invalid cached-token counts "
            "are listed separately and leave total cost unknown.",
            "",
            "## Outputs and Ground Truth handling",
            "",
            f"The batch wrote {len(outputs)} JSON files to `outputs/`, named from "
            "the source stems. Each record stores the original filename, split, "
            "prediction, validation, review flag, token usage, and any error. "
            "Unlabelled records have no Ground Truth field or fabricated labels. "
            "Evaluation uses only the 10 actual same-stem labelled JSON files.",
            "",
            "## Known limitations",
            "",
            "- Only 10 documents have Ground Truth; unlabelled documents cannot be "
            "scored for accuracy.",
            "- OCR can lose table structure or confuse small text; OCR use does not "
            "prove that OCR caused a specific mismatch.",
            "- Structured extraction may omit quantities or tax fields even when "
            "text is available. Validation flags incomplete fields but does not "
            "invent or derive them.",
            "- The batch used a Groq free-tier API key. The inference response does "
            "not provide an account billing statement, so the actual charge is "
            "unverified.",
            "- Prompt/OCR changes were made after reviewing the labelled-set failure "
            "categories. This run is a post-analysis diagnostic, not an independent "
            "holdout measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _money_or_unknown(value: float | None) -> str:
    return f"${value:.8f}" if value is not None else "unknown (token metadata incomplete)"


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_all",
        description="Extract all 20 assignment documents and generate final reports.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir / "samples" / "assignment",
        help="Assignment directory containing labelled/ and unlabelled/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for one JSON per source, batch metadata, and cost analysis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_batch(data_dir=args.data_dir, output_dir=args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Processed {summary.total_documents} documents: "
        f"{summary.successful_documents} successful, {summary.failed_documents} failed."
    )
    print(f"Saved per-document JSON and cost analysis to {summary.output_directory}")
    print(f"Saved final report to {PROJECT_ROOT / 'REPORT.md'}")
    return 1 if summary.failed_documents else 0
