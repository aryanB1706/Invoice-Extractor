"""Shared Pydantic models used across the pipeline."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    """A single line on an invoice."""

    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class Invoice(BaseModel):
    """Structured invoice returned by the extraction step."""

    model_config = ConfigDict(extra="ignore")

    is_invoice: bool = Field(
        default=True,
        description=(
            "True when the document is an invoice, bill, or tax invoice. "
            "False for contracts, letters, statements, or unrelated documents."
        ),
    )
    extraction_notes: str | None = Field(
        default=None,
        description="Brief note when the document is not an invoice or extraction is limited.",
    )
    vendor_name: str | None = None
    vendor_address: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    currency: str | None = None
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    raw_text: str | None = Field(default=None, exclude=True)


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
