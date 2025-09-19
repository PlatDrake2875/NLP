# backend/routers/automate_router.py
"""
Automate router - thin web layer that delegates to AutomateService.
Handles HTTP concerns only, business logic is in services.automate.AutomateService.
"""

from fastapi import APIRouter, Depends

from deps import get_automate_service
from schemas import AutomateRequest, AutomateResponse
from services.automate import AutomateService

router = APIRouter()


@router.post("/automate_conversation", response_model=AutomateResponse)
async def automate_conversation_endpoint(
    payload: AutomateRequest,
    automate_service: AutomateService = Depends(get_automate_service),
):
    """
    Endpoint to automate a conversation based on the provided history and parameters.

    Args:
        payload: AutomateRequest with conversation history and automation task
        automate_service: Injected AutomateService instance

    Returns:
        AutomateResponse with automation results
    """
    # Delegate all business logic to the service
    return await automate_service.process_automation_request(payload)
