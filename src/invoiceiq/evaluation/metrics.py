"""Field-level comparison between expected and predicted invoices."""

from invoiceiq.schemas import Invoice

SCORED_FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
)


def evaluate_invoice(expected: Invoice, predicted: Invoice) -> dict[str, bool]:
    """Return a per-field match report for a single invoice."""
    return {
        field: getattr(expected, field) == getattr(predicted, field) for field in SCORED_FIELDS
    }
