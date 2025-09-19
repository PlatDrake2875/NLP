# backend/routers/model_router.py
"""
Model router - thin web layer that delegates to ModelService.
Handles HTTP concerns only, business logic is in services.model.ModelService.
"""

from fastapi import APIRouter, Depends

from backend.deps import get_model_service
from backend.schemas import OllamaModelInfo
from backend.services.model import ModelService

router = APIRouter(
    prefix="/models",  # Prefix relative to the /api added in main.py
    tags=["models"],
)


@router.get(
    "", response_model=list[OllamaModelInfo]
)  # Path is relative to prefix -> /api/models
async def list_ollama_models_endpoint(
    model_service: ModelService = Depends(get_model_service),
):
    """
    List available Ollama models.

    Args:
        model_service: Injected ModelService instance

    Returns:
        List of available models
    """
    # Delegate all business logic to the service
    return await model_service.list_available_models()
