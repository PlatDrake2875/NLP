"""
Automate service layer containing all automation business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_components import get_llm_for_automation
from schemas import AutomateRequest, AutomateResponse
from schemas import Message as PydanticMessage


class AutomateService:
    """Service class handling all automation business logic."""

    def __init__(self):
        self.supported_tasks = {"summarize_conversation", "suggest_next_reply"}

    async def process_automation_request(
        self, payload: AutomateRequest
    ) -> AutomateResponse:
        # Validate request structure
        self._validate_request(payload)

        # Get LLM instance for the specified model
        llm_to_use = get_llm_for_automation()

        # Convert conversation history to LangChain messages
        langchain_messages = self._convert_to_langchain_messages(
            payload.conversation_history
        )

        # Process the automation task
        automated_data, response_message = await self._process_task(
            payload.automation_task,
            payload.conversation_history,
            langchain_messages,
            llm_to_use,
        )

        return AutomateResponse(
            status="success", message=response_message, data=automated_data
        )

    def _validate_request(self, payload: AutomateRequest) -> None:
        """Validate the automation request."""
        if not hasattr(AutomateRequest, "model_fields"):
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: Automation schemas not loaded.",
            )

        if not isinstance(payload, AutomateRequest):
            raise HTTPException(
                status_code=422,
                detail="Invalid payload structure for automation request.",
            )

    def _convert_to_langchain_messages(
        self, conversation_history: list[PydanticMessage]
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """Convert Pydantic messages to LangChain messages."""
        langchain_messages = []

        for msg in conversation_history:
            if (
                not isinstance(msg, PydanticMessage)
                or not hasattr(msg, "role")
                or not hasattr(msg, "content")
            ):
                continue

            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            else:
                langchain_messages.append(HumanMessage(content=msg.content))

        return langchain_messages

    async def _process_task(
        self,
        automation_task: str,
        conversation_history: list[PydanticMessage],
        langchain_messages: list,
        llm_to_use: Any,
    ) -> tuple[dict[str, Any], str]:
        """Process the specific automation task."""
        if automation_task == "summarize_conversation":
            return await self._summarize_conversation(conversation_history, llm_to_use)
        elif automation_task == "suggest_next_reply":
            return await self._suggest_next_reply(langchain_messages, llm_to_use)
        elif not automation_task:
            return self._handle_no_task()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown automation_task: '{automation_task}'. Supported tasks: {', '.join(self.supported_tasks)} or no task for generic processing.",
            )

    async def _summarize_conversation(
        self, conversation_history: list[PydanticMessage], llm_to_use: Any
    ) -> tuple[dict[str, Any], str]:
        """Summarize the conversation using the LLM."""
        conversation_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in conversation_history]
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

        automated_data = {"summary": summary_response.content}
        response_message = "Conversation summarized successfully by the LLM."

        return automated_data, response_message

    async def _suggest_next_reply(
        self, langchain_messages: list, llm_to_use: Any
    ) -> tuple[dict[str, Any], str]:
        """Suggest the next reply using the LLM."""
        if not langchain_messages:
            automated_data = {
                "suggested_reply": "Cannot suggest a reply for an empty conversation."
            }
        else:
            suggestion_response = await llm_to_use.ainvoke(langchain_messages)
            automated_data = {"suggested_reply": suggestion_response.content}

        response_message = "Next reply suggested successfully by the LLM."

        return automated_data, response_message

    def _handle_no_task(self) -> tuple[dict[str, Any], str]:
        """Handle requests with no specific task."""
        automated_data = {
            "info": "No specific task was requested, but the payload was received."
        }
        response_message = "Automation endpoint processed with no specific task."

        return automated_data, response_message
