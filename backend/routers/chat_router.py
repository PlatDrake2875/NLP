# backend/routers/chat_router.py
"""
Chat router - thin web layer that delegates to ChatService.
Handles HTTP concerns only, business logic is in services.chat.ChatService.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.deps import get_chat_service
from backend.schemas import ChatRequest
from backend.services.chat import ChatService

# --- Router Setup ---
router = APIRouter(
    tags=["chat"],
)


# --- API Endpoints ---
@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)
):
    """
    Chat endpoint that streams responses from the configured LLM.

    Args:
        request: The chat request containing query, model, history, etc.
        chat_service: Injected ChatService instance

    Returns:
        StreamingResponse with Server-Sent Events (SSE) format
    """
    # Convert Pydantic models to simple dicts for the service layer
    history_dicts = []
    if request.history:
        history_dicts = [
            {"sender": msg.sender, "text": msg.text} for msg in request.history
        ]

    # Delegate all business logic to the service
    generator = chat_service.process_chat_request(
        query=request.query,
        model_name=request.model,
        history=history_dicts,
        use_rag=request.use_rag,
    )

    return StreamingResponse(generator, media_type="text/event-stream")
