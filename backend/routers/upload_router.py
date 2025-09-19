# backend/routers/upload_router.py
"""
Upload router - thin web layer that delegates to UploadService.
Handles HTTP concerns only, business logic is in services.upload.UploadService.
"""

from fastapi import APIRouter, Depends, File, UploadFile
from langchain_chroma import Chroma as LangchainChroma

from backend.deps import get_upload_service
from backend.rag_components import get_vectorstore
from backend.schemas import UploadResponse
from backend.services.upload import UploadService

router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)


@router.post("", response_model=UploadResponse)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    upload_service: UploadService = Depends(get_upload_service),
    vectorstore: LangchainChroma = Depends(get_vectorstore),
):
    """
    Upload and process a document for vector storage.

    Args:
        file: The uploaded file
        upload_service: Injected UploadService instance
        vectorstore: Injected vectorstore instance

    Returns:
        UploadResponse with processing results
    """
    # Delegate all business logic to the service
    return await upload_service.process_document_upload(file, vectorstore)
