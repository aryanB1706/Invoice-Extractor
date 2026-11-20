"""Document loader tests."""

from pathlib import Path

import pytest

from invoiceiq.document_processing.loader import load_document


def test_missing_document_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.pdf")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "invoice.txt"
    path.write_text("not a document", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(path)
