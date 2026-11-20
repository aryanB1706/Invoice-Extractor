"""Shared Pydantic models used across the pipeline."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    """A single line on an invoice."""

    description: str | None = None
    quantity: float | None = None
    amount: float | None = None


class Invoice(BaseModel):
    """Invoice output matching the assignment's required JSON schema."""

    model_config = ConfigDict(extra="ignore")

    is_invoice: bool = Field(
        description=(
            "True for invoices, bills, or tax invoices. False for bank statements, "
            "purchase orders, cover letters, and other documents."
        )
    )
    vendor: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    currency: str | None = None
    line_items: list[LineItem] | None = Field(
        default=None,
        description="All invoice line items. Null for non-invoices or when absent.",
    )
    tax_total: float | None = None
    total: float | None = None


class PageText(BaseModel):
    """Text extracted from one document page."""

    page_number: int
    text: str
    used_ocr: bool = False
    error: str | None = None


class ValidationIssue(BaseModel):
    """A single validation problem for an extracted invoice."""

    field: str
    message: str


class ValidationReport(BaseModel):
    """Result of validating an extracted invoice."""

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token counts reported by the LLM provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ExtractionResult(BaseModel):
    """Invoice plus metadata produced by the extraction layer."""

    invoice: Invoice
    model: str | None = None
    usage: TokenUsage | None = None
