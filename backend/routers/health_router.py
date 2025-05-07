# backend/routers/health_router.py
import logging
import ollama # For AsyncClient
from fastapi import APIRouter, HTTPException

# Import schemas, config, and RAG components using direct absolute paths
from schemas import HealthResponse, HealthStatusDetail
from config import OLLAMA_BASE_URL, logger
from rag_components import ollama_chat_for_rag, chroma_client # Access initialized components


router = APIRouter(
    prefix="/health", # Prefix relative to /api -> /api/health
    tags=["health"]
)

@router.get("", response_model=HealthResponse) # Path relative to prefix -> /api/health
async def health_check_endpoint():
    logger.info("Health check endpoint called.")
    
    ollama_status = HealthStatusDetail(status="disconnected", details="ChatOllama (for RAG) component failed to initialize or OLLAMA_BASE_URL not set.")
    if ollama_chat_for_rag and OLLAMA_BASE_URL: 
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
        logger.warning(f"Ollama health check: SKIPPED - ollama_chat_for_rag is {ollama_chat_for_rag}, OLLAMA_BASE_URL is {OLLAMA_BASE_URL}")
    
    chroma_status = HealthStatusDetail(status="disconnected", details="ChromaDB client not initialized or connection failed.")
    if chroma_client:
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
            
    overall_status = "ok" if ollama_status.status == "connected" and chroma_status.status == "connected" else "error"
    
    response_payload = HealthResponse(
        status=overall_status,
        ollama=ollama_status,
        chromadb=chroma_status
    )
    
    if overall_status == "error":
        logger.warning(f"Health check returning status: ERROR. Details: {response_payload}")
        detail_payload = response_payload.model_dump() if hasattr(response_payload, 'model_dump') else response_payload.dict()
        raise HTTPException(status_code=503, detail=detail_payload) 
        
    logger.info(f"Health check returning status: OK. Details: {response_payload}")
    return response_payload 
