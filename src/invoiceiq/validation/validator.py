"""Rule-based validation for extracted invoices."""

from math import isclose

from invoiceiq.schemas import Invoice, ValidationIssue, ValidationReport

AMOUNT_TOLERANCE = 0.01


def validate_invoice(invoice: Invoice) -> ValidationReport:
    """Check required fields and arithmetic consistency."""
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

    amounts = [item.amount for item in invoice.line_items or []]
    if (
        amounts
        and all(amount is not None for amount in amounts)
        and invoice.tax_total is not None
        and invoice.total is not None
    ):
        expected_total = sum(amount for amount in amounts if amount is not None) + invoice.tax_total
        if not isclose(expected_total, invoice.total, abs_tol=AMOUNT_TOLERANCE):
            issues.append(
                ValidationIssue(
                    field="total",
                    message="Total does not equal line-item amounts plus tax.",
                )
            )

    return ValidationReport(is_valid=not issues, issues=issues)
