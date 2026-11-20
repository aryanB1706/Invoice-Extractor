"""InvoiceIQ - AI invoice extraction pipeline."""

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main() -> None:
    """Entry point for the ``invoiceiq`` command."""
    from invoiceiq.cli import main as cli_main

    cli_main()
