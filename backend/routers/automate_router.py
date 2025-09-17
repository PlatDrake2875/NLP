# backend/routers/automate_router.py
from typing import TYPE_CHECKING, Any, Optional  # Added Optional

from fastapi import APIRouter, HTTPException

# Attempt to import logger from config, fallback if necessary
try:
    from config import logger
except (ImportError, ModuleNotFoundError):
    import logging

    logger = logging.getLogger("automate_router_fallback")  # type: ignore
    logger.warning(
        "Could not import logger from 'config'. Using fallback logger for automate_router."
    )

# Import Pydantic schemas directly
try:
    from schemas import AutomateRequest, AutomateResponse
    from schemas import Message as PydanticMessage
except (ImportError, ModuleNotFoundError) as e_schemas:
    logger.error(
        f"CRITICAL: Could not import Pydantic schemas (AutomateRequest, AutomateResponse, Message) from 'schemas': {e_schemas}. Endpoint will likely fail."
    )

    # Define dummy classes to allow module parsing, but endpoint will be non-functional
    class PydanticMessage:  # type: ignore
        pass

    class AutomateRequest:  # type: ignore
        pass

    class AutomateResponse:  # type: ignore
        pass


# Import ONLY the specific function needed from rag_components at the module level
try:
    from rag_components import get_llm_for_automation
except ImportError as e_rag_import:
    logger.error(
        f"CRITICAL: Could not import 'get_llm_for_automation' from 'rag_components': {e_rag_import}. Automation endpoint will fail."
    )

    # Define a dummy function if the import fails, to allow the rest of the module to load
    # but the endpoint will not work.
    def get_llm_for_automation(model_name: Optional[str] = None):  # type: ignore
        raise ImportError(
            "Dummy get_llm_for_automation due to critical import error from rag_components"
        )


# For type hinting ChatOllama if needed elsewhere in this file, use TYPE_CHECKING
if TYPE_CHECKING:
    pass


router = APIRouter()


@router.post("/automate_conversation", response_model=AutomateResponse)
async def automate_conversation_endpoint(
    payload: AutomateRequest,
    # fastapi_request: Request # Kept for potential future use, but not used currently
):
    """
    Endpoint to automate a conversation based on the provided history and parameters.
    Uses a dedicated getter for the LLM from rag_components.
    """
    request_id_suffix = str(id(payload))[-6:]
    logger.info(
        f"[automate_{request_id_suffix}] Received request for /api/automate_conversation."
    )

    if not hasattr(AutomateRequest, "model_fields"):  # Check if it's the dummy
        logger.error(
            f"[automate_{request_id_suffix}] AutomateRequest schema appears to be a dummy. Schema imports failed critically."
        )
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: Automation schemas not loaded.",
        )
    if not isinstance(payload, AutomateRequest):
        logger.error(
            f"[automate_{request_id_suffix}] Payload is not an instance of the expected AutomateRequest schema. Type: {type(payload)}"
        )
        raise HTTPException(
            status_code=422, detail="Invalid payload structure for automation request."
        )

    logger.info(
        f"[automate_{request_id_suffix}] Requested Model: {payload.model}, Task: {payload.automation_task or 'N/A'}"
    )
    logger.debug(
        f"[automate_{request_id_suffix}] Payload: {payload.model_dump_json(indent=2)}"
    )

    try:
        # Get the LLM instance using the specific getter, passing the desired model from payload
        llm_to_use = get_llm_for_automation(model_name=payload.model)
        logger.info(
            f"[automate_{request_id_suffix}] LLM for task resolved to: {llm_to_use.model if hasattr(llm_to_use, 'model') else 'Unknown'}"
        )
    except RuntimeError as e:
        logger.error(
            f"[automate_{request_id_suffix}] Failed to get LLM for automation (RuntimeError from getter): {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"LLM not available: {str(e)}"
        ) from e
    except ImportError as e_imp:  # Catch if the dummy get_llm_for_automation was called
        logger.error(
            f"[automate_{request_id_suffix}] Failed to get LLM due to earlier import error for get_llm_for_automation: {e_imp}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: LLM accessor not loaded.",
        ) from e_imp
    except Exception as e_get_llm:
        logger.error(
            f"[automate_{request_id_suffix}] Unexpected error getting LLM: {e_get_llm}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve LLM due to an unexpected error."
        ) from e_get_llm

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    langchain_messages: list[SystemMessage | HumanMessage | AIMessage] = []
    for msg in payload.conversation_history:
        if (
            not isinstance(msg, PydanticMessage)
            or not hasattr(msg, "role")
            or not hasattr(msg, "content")
        ):
            logger.warning(
                f"[automate_{request_id_suffix}] Invalid message object in conversation_history: {msg}"
            )
            continue
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))
        else:
            logger.warning(
                f"Unknown role '{msg.role}' in conversation history, treating as human message."
            )
            langchain_messages.append(HumanMessage(content=msg.content))

    automated_data: dict[str, Any] = {}
    response_message = "Automation task processed."
    current_automation_task = payload.automation_task

    try:
        if current_automation_task == "summarize_conversation":
            logger.info(
                f"[automate_{request_id_suffix}] Performing REAL summarization task..."
            )
            conversation_text = "\n".join(
                [f"{msg.role}: {msg.content}" for msg in payload.conversation_history]
            )
            summarization_prompt_messages = [
                SystemMessage(
                    content="You are a helpful assistant that summarizes conversations."
                ),
                HumanMessage(
                    content=f"Please summarize the following conversation:\n\n{conversation_text}\n\nSummary:"
                ),
            ]
            summary_response = await llm_to_use.ainvoke(summarization_prompt_messages)
            automated_data["summary"] = summary_response.content
            response_message = "Conversation summarized successfully by the LLM."

        elif current_automation_task == "suggest_next_reply":
            logger.info(
                f"[automate_{request_id_suffix}] Performing REAL next reply suggestion..."
            )
            if not langchain_messages:
                automated_data["suggested_reply"] = (
                    "Cannot suggest a reply for an empty conversation."
                )
            else:
                suggestion_response = await llm_to_use.ainvoke(langchain_messages)
                automated_data["suggested_reply"] = suggestion_response.content
            response_message = "Next reply suggested successfully by the LLM."

        elif not current_automation_task:
            logger.info(
                f"[automate_{request_id_suffix}] No specific automation_task provided. Returning generic response."
            )
            automated_data["info"] = (
                "No specific task was requested, but the payload was received."
            )
            response_message = "Automation endpoint processed with no specific task."

        else:
            logger.warning(
                f"[automate_{request_id_suffix}] Unknown automation_task: {current_automation_task}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unknown automation_task: '{current_automation_task}'. Supported tasks: 'summarize_conversation', 'suggest_next_reply' or no task for generic processing.",
            )

    except Exception as e:
        logger.error(
            f"[automate_{request_id_suffix}] Error during LLM call or automation logic: {e}",
            exc_info=True,
        )
        error_detail = f"Error processing automation task: {str(e)}"
        if hasattr(e, "response") and hasattr(e.response, "text"):
            error_detail += f" | LLM Response: {e.response.text[:500]}"
        raise HTTPException(status_code=500, detail=error_detail) from e

    logger.info(f"[automate_{request_id_suffix}] Automation processing complete.")

    if not hasattr(AutomateResponse, "model_fields"):
        logger.error(
            f"[automate_{request_id_suffix}] AutomateResponse schema is a dummy. Returning raw dict due to import failure."
        )
        return {
            "status": "success",
            "message": response_message,
            "data": automated_data,
            "error_details": "Response schema load error",
        }

    return AutomateResponse(
        status="success", message=response_message, data=automated_data
    )
