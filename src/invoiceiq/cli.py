"""Minimal command-line entry point for local smoke testing."""

import argparse
from pathlib import Path

from invoiceiq import __version__
from invoiceiq.document_processing.loader import load_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="invoiceiq", description="InvoiceIQ CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("document", nargs="?", type=Path, help="PDF or image to inspect")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load a document and print a short processing summary."""
    args = build_parser().parse_args(argv)

    if args.document is None:
        build_parser().print_help()
        return

    pages = load_document(args.document)
    characters = sum(len(page.text) for page in pages)
    print(f"Loaded {len(pages)} page(s) with {characters} extracted character(s).")
