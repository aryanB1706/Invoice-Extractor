"""Models for evaluation results and reports."""

from typing import Any

from pydantic import BaseModel, Field

FIELDS: tuple[str, ...] = (
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
)


class FieldComparison(BaseModel):
    """Comparison of one expected field against its prediction."""

    expected: Any = None
    predicted: Any = None
    match: bool


class LineItemMetrics(BaseModel):
    """Item-level comparison details for one document."""

    expected_count: int
    predicted_count: int
    count_match: bool
    matched_items: int
    aligned_items: int
    item_match_rate: float
    description_matches: int
    quantity_matches: int
    amount_matches: int
    details: list[str] = Field(default_factory=list)


class DocumentEvaluation(BaseModel):
    """Full evaluation result for one labelled document."""

    document: str
    ground_truth: str
    error: str | None = None
    fields: dict[str, FieldComparison]
    line_item_metrics: LineItemMetrics | None = None
    fully_correct: bool
    expected: dict[str, Any] | None = None
    predicted: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    needs_review: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] | None = None


class FieldSummary(BaseModel):
    """Aggregate accuracy for one field."""

    correct: int
    total: int
    accuracy: float


class LineItemSummary(BaseModel):
    """Aggregate line-item metrics across the labelled set."""

    documents_with_line_items: int
    expected_items: int
    predicted_items: int
    matched_items: int
    aligned_items: int
    item_match_rate: float
    description_accuracy: float
    quantity_accuracy: float
    amount_accuracy: float
    count_match_accuracy: float


class EvaluationSummary(BaseModel):
    """Aggregate metrics for the whole labelled set."""

    total_documents: int
    failed_documents: int
    fully_correct_documents: int
    overall_accuracy: float
    fields: dict[str, FieldSummary]
    invoice_confusion: dict[str, int]
    line_items: LineItemSummary | None = None
    tolerance: float
    model: str | None = None
    total_usage: dict[str, int] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Everything written to ``evaluation.json`` and ``report.md``."""

    generated_at: str
    data_dir: str
    output_dir: str
    model: str | None
    summary: EvaluationSummary
    documents: list[DocumentEvaluation]
    matching_rules: list[str]
    tuning_disclosure: str
