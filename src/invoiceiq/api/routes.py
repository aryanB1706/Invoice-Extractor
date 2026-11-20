"""HTTP endpoints for invoice processing."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(tags=["invoices"])


@router.post("/extract")
async def extract_invoice(file: Annotated[UploadFile, File()]) -> dict[str, str]:
    """Accept an invoice document and return structured data.

    The endpoint is reserved for the pipeline milestone.
    """
    raise HTTPException(status_code=501, detail="Extraction pipeline is not implemented yet.")
