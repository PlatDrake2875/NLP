# backend/routers/document_router.py
"""
Document router - thin web layer that delegates to DocumentService.
Handles HTTP concerns only, business logic is in services.document.DocumentService.
"""

from fastapi import APIRouter, Depends
from langchain_chroma import Chroma as LangchainChroma

from backend.deps import get_document_service
from backend.rag_components import get_vectorstore
from backend.schemas import DocumentListResponse
from backend.services.document import DocumentService

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.get("", response_model=DocumentListResponse)
async def get_all_documents(
    document_service: DocumentService = Depends(get_document_service),
    vectorstore: LangchainChroma = Depends(get_vectorstore),
):
    """
    Retrieves all document chunks currently stored in the vector database.

    Args:
        document_service: Injected DocumentService instance
        vectorstore: Injected vectorstore instance

    Returns:
        DocumentListResponse with all documents and metadata
    """
    return await document_service.get_all_documents(vectorstore)
