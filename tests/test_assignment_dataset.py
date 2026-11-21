"""Optional checks over the downloaded 20-document assignment pack.

These tests are skipped by default because the dataset is not committed. Run
them with ``INVOICEIQ_DATASET_TESTS=1`` after downloading the pack into
``data/samples/assignment``.
"""

import os
from pathlib import Path

import pytest

from invoiceiq.config import get_settings
from invoiceiq.document_processing.loader import load_document
from invoiceiq.document_processing.ocr import _configure_tesseract

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "samples" / "assignment"
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DOCUMENTS = sorted(
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
)


def _tesseract_available() -> bool:
    settings = get_settings()
    if settings.tesseract_cmd:
        return Path(settings.tesseract_cmd).is_file()
    import shutil

    return shutil.which("tesseract") is not None


pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(
        os.getenv("INVOICEIQ_DATASET_TESTS") != "1",
        reason="set INVOICEIQ_DATASET_TESTS=1 to run dataset checks",
    ),
    pytest.mark.skipif(not DOCUMENTS, reason="assignment dataset is not downloaded"),
    pytest.mark.skipif(not _tesseract_available(), reason="Tesseract is not installed"),
]


@pytest.mark.parametrize("document_path", DOCUMENTS, ids=lambda path: path.stem)
def test_assignment_document_is_loadable(document_path: Path) -> None:
    _configure_tesseract(get_settings())

    pages = load_document(document_path)

    assert pages
    assert any(page.text.strip() for page in pages)
