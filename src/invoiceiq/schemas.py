"""Shared Pydantic models used across the pipeline."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STRICT_SCHEMA = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class LineItem(BaseModel):
    """A single line on an invoice."""

    model_config = STRICT_SCHEMA

    description: str | None
    quantity: float | None
    amount: float | None


class Invoice(BaseModel):
    """Invoice output matching the assignment's required JSON schema."""

    model_config = STRICT_SCHEMA

    is_invoice: bool = Field(
        description=(
            "True for invoices, bills, or tax invoices. False for bank statements, "
            "purchase orders, cover letters, and other documents."
        )
    )
    vendor: str | None
    invoice_number: str | None
    invoice_date: date | None
    currency: str | None
    line_items: list[LineItem] | None = Field(
        description="All invoice line items. Null for non-invoices or when absent."
    )
    tax_total: float | None
    total: float | None

    @field_validator("invoice_date", mode="before")
    @classmethod
    def validate_invoice_date(cls, value: object) -> object:
        """Require dates in exactly YYYY-MM-DD format."""
        if value is None:
            return None
        if isinstance(value, datetime):
            raise ValueError("invoice_date must use YYYY-MM-DD format without a time")
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("invoice_date must use YYYY-MM-DD format") from exc
            if parsed.isoformat() != value:
                raise ValueError("invoice_date must use YYYY-MM-DD format")
            return parsed
        raise ValueError("invoice_date must be a YYYY-MM-DD string")

    @model_validator(mode="after")
    def validate_non_invoice_fields(self) -> "Invoice":
        """Require every non-invoice field except is_invoice to be null."""
        if self.is_invoice:
            return self

        populated = [
            field_name
            for field_name in (
                "vendor",
                "invoice_number",
                "invoice_date",
                "currency",
                "line_items",
                "tax_total",
                "total",
            )
            if getattr(self, field_name) is not None
        ]
        if populated:
            raise ValueError(
                "Non-invoice documents must have null values for: "
                + ", ".join(populated)
            )
        return self


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


class DocumentQuality(BaseModel):
    """Document-processing signals recorded for evaluation diagnostics."""

    page_count: int
    ocr_pages: int
    text_characters: int
    page_errors: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token counts reported by the LLM provider."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_prompt_tokens: int | None = Field(default=None, ge=0)


class ExtractionResult(BaseModel):
    """Invoice plus metadata produced by the extraction layer."""

    invoice: Invoice
    model: str | None = None
    usage: TokenUsage | None = None


class ExtractionResponse(BaseModel):
    """Public API response for a processed document."""

    invoice: Invoice
    validation: ValidationReport
    needs_review: bool
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    usage: TokenUsage | None = None
