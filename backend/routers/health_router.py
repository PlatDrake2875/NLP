# backend/routers/health_router.py
import logging
import ollama
from fastapi import APIRouter, HTTPException, Depends # Import Depends
from fastapi.responses import JSONResponse
from typing import Optional

from schemas import HealthResponse, HealthStatusDetail
from config import OLLAMA_BASE_URL, logger
# Import the OPTIONAL dependency functions
from rag_components import get_optional_chroma_client, get_optional_ollama_chat_for_rag
import chromadb
from langchain_ollama import ChatOllama


router = APIRouter(
    prefix="/health",
    tags=["health"]
)

@router.get("", response_model=HealthResponse)
async def health_check_endpoint(
    # Inject optional dependencies using Depends
    chroma_client: Optional[chromadb.HttpClient] = Depends(get_optional_chroma_client),
    ollama_chat_for_rag: Optional[ChatOllama] = Depends(get_optional_ollama_chat_for_rag)
):
    logger.info("Health check endpoint called.")

    # --- Check Ollama Status ---
    ollama_status = HealthStatusDetail(status="disconnected", details="ChatOllama (for RAG) component failed to initialize or OLLAMA_BASE_URL not set.")
    if ollama_chat_for_rag and OLLAMA_BASE_URL: # Check if the injected instance exists
        try:
            client = ollama.AsyncClient(host=OLLAMA_BASE_URL, timeout=5.0)
            await client.list()
            ollama_status = HealthStatusDetail(status="connected", details="Ollama service is responsive.")
            logger.info("Ollama health check: PASSED")
        except Exception as e:
            error_detail = f"Ollama connection or list models failed: {str(e)}"
            ollama_status = HealthStatusDetail(status="disconnected", details=error_detail)
            logger.warning(f"Ollama health check: FAILED - {error_detail}")
    else:
        logger.warning(f"Ollama health check: SKIPPED - ollama_chat_for_rag is {'available' if ollama_chat_for_rag else 'unavailable'}, OLLAMA_BASE_URL is {OLLAMA_BASE_URL}")

    # --- Check ChromaDB Status ---
    chroma_status = HealthStatusDetail(status="disconnected", details="ChromaDB client not initialized or connection failed.")
    if chroma_client: # Check if the injected instance exists
        try:
            chroma_client.heartbeat()
            chroma_status = HealthStatusDetail(status="connected", details="ChromaDB service is responsive.")
            logger.info("ChromaDB health check: PASSED")
        except Exception as e:
            error_detail = f"ChromaDB heartbeat error: {str(e)}"
            chroma_status = HealthStatusDetail(status="disconnected", details=error_detail)
            logger.warning(f"ChromaDB health check: FAILED - {error_detail}")
    else:
        logger.warning("ChromaDB health check: SKIPPED - chroma_client is not initialized.")

    # --- Determine Overall Status ---
    overall_status = "ok" if ollama_status.status == "connected" and chroma_status.status == "connected" else "error"

    response_payload = HealthResponse(
        status=overall_status,
        ollama=ollama_status,
        chromadb=chroma_status
    )

    if overall_status == "error":
        logger.warning(f"Health check returning status: ERROR. Details: {response_payload}")
        detail_payload = response_payload.model_dump() if hasattr(response_payload, 'model_dump') else response_payload.dict()
        return JSONResponse(status_code=503, content=detail_payload)

    logger.info(f"Health check returning status: OK. Details: {response_payload}")
    return response_payload
