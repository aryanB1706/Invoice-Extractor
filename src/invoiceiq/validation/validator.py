"""Rule-based validation for extracted invoices."""

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

    if not invoice.vendor:
        issues.append(ValidationIssue(field="vendor", message="Vendor is missing."))
    if not invoice.invoice_number:
        issues.append(
            ValidationIssue(field="invoice_number", message="Invoice number is missing.")
        )
    if invoice.invoice_date is None:
        issues.append(ValidationIssue(field="invoice_date", message="Invoice date is missing."))
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
    elif any(item.amount is None for item in invoice.line_items):
        issues.append(
            ValidationIssue(
                field="line_items",
                message="Cannot verify total because one or more line-item amounts are missing.",
            )
        )
    elif invoice.tax_total is None:
        issues.append(
            ValidationIssue(
                field="tax_total",
                message="Cannot verify total because the tax amount is missing.",
            )
        )
    elif invoice.total is not None:
        expected_total = sum(item.amount for item in invoice.line_items) + invoice.tax_total
        if not isclose(expected_total, invoice.total, rel_tol=0, abs_tol=tolerance):
            issues.append(
                ValidationIssue(
                    field="total",
                    message="Total does not equal line-item amounts plus tax.",
                )
            )

    return ValidationReport(is_valid=not issues, issues=issues)
