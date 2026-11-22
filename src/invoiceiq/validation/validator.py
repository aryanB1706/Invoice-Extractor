"""Rule-based validation for extracted invoices."""

import re
from math import isclose, isfinite

from invoiceiq.schemas import Invoice, ValidationIssue, ValidationReport

AMOUNT_TOLERANCE = 0.01


def validate_invoice(
    invoice: Invoice, *, tolerance: float = AMOUNT_TOLERANCE
) -> ValidationReport:
    """Check required fields and arithmetic consistency."""
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Validation tolerance must be a finite non-negative number.")

    issues: list[ValidationIssue] = []

    if not invoice.is_invoice:
        issues.append(
            ValidationIssue(field="is_invoice", message="Document is not an invoice.")
        )
        return ValidationReport(is_valid=False, issues=issues)

    if not invoice.vendor or not invoice.vendor.strip():
        issues.append(ValidationIssue(field="vendor", message="Vendor is missing."))
    if not invoice.invoice_number or not invoice.invoice_number.strip():
        issues.append(
            ValidationIssue(field="invoice_number", message="Invoice number is missing.")
        )
    if invoice.invoice_date is None:
        issues.append(ValidationIssue(field="invoice_date", message="Invoice date is missing."))
    if not invoice.currency:
        issues.append(ValidationIssue(field="currency", message="Currency is missing."))
    elif re.fullmatch(r"[A-Za-z]{3}", invoice.currency.strip()) is None:
        issues.append(
            ValidationIssue(
                field="currency",
                message="Currency must be a three-letter code.",
            )
        )
    if invoice.total is None:
        issues.append(ValidationIssue(field="total", message="Total is missing."))
    elif invoice.total < 0:
        issues.append(ValidationIssue(field="total", message="Total must not be negative."))

    if not invoice.line_items:
        issues.append(
            ValidationIssue(
                field="line_items",
                message="Cannot verify total because line items are missing.",
            )
        )
    else:
        missing = []
        for index, item in enumerate(invoice.line_items, start=1):
            fields = [
                field
                for field, value in (
                    ("description", item.description),
                    ("quantity", item.quantity),
                    ("amount", item.amount),
                )
                if value is None or (field == "description" and not value.strip())
            ]
            if fields:
                missing.append(f"item {index}: {', '.join(fields)}")
        if missing:
            issues.append(
                ValidationIssue(
                    field="line_items",
                    message="Missing line-item information: " + "; ".join(missing),
                )
            )

    if invoice.is_invoice and invoice.tax_total is None:
        issues.append(
            ValidationIssue(
                field="tax_total",
                message="Cannot verify total because the tax amount is missing.",
            )
        )
    elif (
        invoice.is_invoice
        and invoice.line_items
        and all(item.amount is not None for item in invoice.line_items)
        and invoice.total is not None
    ):
        expected_total = sum(item.amount for item in invoice.line_items) + invoice.tax_total
        if not isclose(expected_total, invoice.total, rel_tol=0, abs_tol=tolerance):
            issues.append(
                ValidationIssue(
                    field="total",
                    message="Total does not equal line-item amounts plus tax.",
                )
            )

    return ValidationReport(is_valid=not issues, issues=issues)
