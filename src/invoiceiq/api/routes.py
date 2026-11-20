"""HTTP endpoints for invoice processing."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from invoiceiq.api.service import UnreadableDocumentError, process_document
from invoiceiq.config import get_settings
from invoiceiq.document_processing.loader import DocumentProcessingError
from invoiceiq.extraction import (
    LLMConfigurationError,
    LLMError,
    LLMTimeoutError,
)
from invoiceiq.schemas import ExtractionResponse

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
router = APIRouter(tags=["invoices"])


@router.post("/extract")
async def extract_invoice(file: Annotated[UploadFile, File()]) -> ExtractionResponse:
    """Accept an invoice document and return extracted and validated data."""
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PDF, JPG, JPEG, or PNG file.",
        )

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {get_settings().max_upload_mb} MB upload limit.",
        )
    _validate_file_content(content, extension)

    try:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / f"upload{extension}"
            path.write_bytes(content)
            return await run_in_threadpool(process_document, path)
    except UnreadableDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _validate_file_content(content: bytes, extension: str) -> None:
    if extension == ".pdf":
        if b"%PDF-" not in content[:1024]:
            raise HTTPException(status_code=422, detail="Uploaded file is not a valid PDF.")
        return

    try:
        with Image.open(BytesIO(content)) as image:
            if image.format != IMAGE_FORMATS[extension]:
                raise HTTPException(
                    status_code=422,
                    detail=f"File extension does not match its {image.format} image content.",
                )
            image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Uploaded image is unreadable.") from exc
