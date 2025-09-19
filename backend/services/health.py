"""
Health service layer containing all health check business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Optional

import chromadb
import ollama
from fastapi.responses import JSONResponse
from langchain_ollama import ChatOllama

from backend.config import OLLAMA_BASE_URL
from backend.schemas import HealthResponse, HealthStatusDetail


class HealthService:
    """Service class handling all health check business logic."""

    def __init__(self):
        self.ollama_base_url = OLLAMA_BASE_URL

    async def perform_health_check(
        self,
        chroma_client: Optional[chromadb.HttpClient] = None,
        ollama_chat_for_rag: Optional[ChatOllama] = None,
    ) -> HealthResponse | JSONResponse:
        # Check Ollama status
        ollama_status = await self._check_ollama_status(ollama_chat_for_rag)

        # Check ChromaDB status
        chroma_status = self._check_chroma_status(chroma_client)

        # Determine overall status
        overall_status = (
            "ok"
            if ollama_status.status == "connected"
            and chroma_status.status == "connected"
            else "error"
        )

        response_payload = HealthResponse(
            status=overall_status, ollama=ollama_status, chromadb=chroma_status
        )

        if overall_status == "error":
            detail_payload = (
                response_payload.model_dump()
                if hasattr(response_payload, "model_dump")
                else response_payload.dict()
            )
            return JSONResponse(status_code=503, content=detail_payload)

        return response_payload

    async def _check_ollama_status(
        self, ollama_chat_for_rag: Optional[ChatOllama]
    ) -> HealthStatusDetail:
        """Check the status of the Ollama service."""
        if not ollama_chat_for_rag or not self.ollama_base_url:
            return HealthStatusDetail(
                status="disconnected",
                details="ChatOllama (for RAG) component failed to initialize or OLLAMA_BASE_URL not set.",
            )

        try:
            client = ollama.AsyncClient(host=self.ollama_base_url, timeout=5.0)
            await client.list()
            return HealthStatusDetail(
                status="connected", details="Ollama service is responsive."
            )
        except Exception as e:
            error_detail = f"Ollama connection or list models failed: {str(e)}"
            return HealthStatusDetail(status="disconnected", details=error_detail)

    def _check_chroma_status(
        self, chroma_client: Optional[chromadb.HttpClient]
    ) -> HealthStatusDetail:
        """Check the status of the ChromaDB service."""
        if not chroma_client:
            return HealthStatusDetail(
                status="disconnected",
                details="ChromaDB client not initialized or connection failed.",
            )

        try:
            chroma_client.heartbeat()
            return HealthStatusDetail(
                status="connected", details="ChromaDB service is responsive."
            )
        except Exception as e:
            error_detail = f"ChromaDB heartbeat error: {str(e)}"
            return HealthStatusDetail(status="disconnected", details=error_detail)
