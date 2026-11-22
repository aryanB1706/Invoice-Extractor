"""Compatibility launcher for the InvoiceIQ evaluation harness.

This makes the documented command work from the project root:

    python -m evaluation.run_eval
"""

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from invoiceiq.evaluation.run_eval import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
