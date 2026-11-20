"""Rule-based validation for extracted invoices."""

from decimal import Decimal

from invoiceiq.schemas import Invoice, ValidationIssue, ValidationReport

AMOUNT_TOLERANCE = Decimal("0.01")


def validate_invoice(invoice: Invoice) -> ValidationReport:
    """Check required fields and basic arithmetic consistency."""
    issues: list[ValidationIssue] = []

    if not invoice.invoice_number:
        issues.append(ValidationIssue(field="invoice_number", message="Invoice number is missing."))
    if not invoice.vendor_name:
        issues.append(ValidationIssue(field="vendor_name", message="Vendor name is missing."))

    if (
        invoice.subtotal is not None
        and invoice.tax is not None
        and invoice.total is not None
        and abs(invoice.subtotal + invoice.tax - invoice.total) > AMOUNT_TOLERANCE
    ):
        issues.append(
            ValidationIssue(
                field="total",
                message="Total does not equal subtotal plus tax.",
            )
        )

    if (
        invoice.invoice_date is not None
        and invoice.due_date is not None
        and invoice.due_date < invoice.invoice_date
    ):
        issues.append(
            ValidationIssue(
                field="due_date",
                message="Due date is earlier than the invoice date.",
            )
        )

    if invoice.total is not None and invoice.total < 0:
        issues.append(ValidationIssue(field="total", message="Total must not be negative."))

    return ValidationReport(is_valid=not issues, issues=issues)
