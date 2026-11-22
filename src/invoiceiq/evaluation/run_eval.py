"""Command-line evaluation harness.

Run from the project root:

    python -m evaluation.run_eval

or, without installing the shim:

    python -m invoiceiq.evaluation.run_eval
"""

import argparse
import math
import sys
from pathlib import Path

from invoiceiq.config import get_settings
from invoiceiq.evaluation.report import format_console_report
from invoiceiq.evaluation.runner import (
    LabelledDocument,
    discover_labelled_documents,
    evaluate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description="Evaluate invoice extraction against labelled ground truth.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with paired labelled documents and .json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for predictions.json, evaluation.json, and report.md.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Absolute tolerance for numeric fields (default from settings).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    data_dir = args.data_dir or settings.data_dir / "samples" / "assignment" / "labelled"
    output_dir = args.output_dir or settings.output_dir / "evaluation"
    tolerance = (
        settings.invoice_amount_tolerance if args.tolerance is None else args.tolerance
    )

    if not settings.groq_configured:
        print(
            "error: Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL in .env.",
            file=sys.stderr,
        )
        return 1
    if not math.isfinite(tolerance) or tolerance < 0:
        print("error: --tolerance must be a finite non-negative number.", file=sys.stderr)
        return 1

    try:
        documents = discover_labelled_documents(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"InvoiceIQ evaluation: {len(documents)} labelled documents")
    print(f"  data:      {data_dir}")
    print(f"  output:    {output_dir}")
    print(f"  model:     {settings.groq_model}")
    print(f"  tolerance: {tolerance}")
    print()

    def progress(index: int, total: int, document: LabelledDocument) -> None:
        print(f"[{index}/{total}] {document.document.name}", flush=True)

    report = evaluate_dataset(
        documents,
        tolerance=tolerance,
        data_dir=data_dir,
        output_dir=output_dir,
        progress=progress,
    )

    print()
    print(format_console_report(report))
    print()
    print(f"Saved predictions and results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
