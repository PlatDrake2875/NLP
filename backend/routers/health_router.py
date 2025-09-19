# backend/routers/health_router.py
"""
Health router - thin web layer that delegates to HealthService.
Handles HTTP concerns only, business logic is in services.health.HealthService.
"""

from typing import Optional

import chromadb
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from langchain_ollama import ChatOllama

from deps import get_health_service
from rag_components import get_optional_chroma_client, get_optional_ollama_chat_for_rag
from schemas import HealthResponse
from services.health import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check_endpoint(
    health_service: HealthService = Depends(get_health_service),
    chroma_client: Optional[chromadb.HttpClient] = Depends(get_optional_chroma_client),
    ollama_chat_for_rag: Optional[ChatOllama] = Depends(
        get_optional_ollama_chat_for_rag
    ),
):
    """
    Check the health status of all system components.

    Args:
        health_service: Injected HealthService instance
        chroma_client: Optional ChromaDB client instance
        ollama_chat_for_rag: Optional Ollama chat instance

    Returns:
        HealthResponse with component status details
        HTTP 503 status code if any component is unhealthy
    """
    # Delegate all business logic to the service
    response_payload = await health_service.perform_health_check(
        chroma_client, ollama_chat_for_rag
    )

    # Handle HTTP status code based on overall health
    if response_payload.status == "error":
        detail_payload = (
            response_payload.model_dump()
            if hasattr(response_payload, "model_dump")
            else response_payload.dict()
        )
        return JSONResponse(status_code=503, content=detail_payload)

    return response_payload
