# backend/schemas.py
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("nlp_backend.schemas") # Logger for this module

# --- Generic Message Model (useful for conversation histories) ---
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }

# --- Pydantic Models (Existing) ---
class OllamaModelInfo(BaseModel):
    name: str
    modified_at: str # Storing as string as per your original model
    size: int

    @classmethod
    def from_ollama(cls, raw: Dict[str, Any]) -> Optional["OllamaModelInfo"]:
        # raw dictionary comes from an ollama.Model object converted to dict
        model_name = raw.get("name") or raw.get("model") # 'model' is the key in ollama.Model
        if not model_name:
            logger.warning(f"Skipping model entry in from_ollama due to missing 'name'/'model': {raw}")
            return None

        modified_at_val = raw.get("modified_at")
        modified_at_str = ""
        if isinstance(modified_at_val, datetime):
            modified_at_str = modified_at_val.isoformat()
        elif isinstance(modified_at_val, str):
            # Attempt to parse if it's a string that might need conversion
            try:
                # Ensure timezone info is handled correctly for ISO format
                dt_obj = datetime.fromisoformat(modified_at_val.replace('Z', '+00:00'))
                modified_at_str = dt_obj.isoformat()
            except ValueError:
                logger.warning(f"Could not parse modified_at string '{modified_at_val}' to datetime. Keeping as is.")
                modified_at_str = modified_at_val # Keep as is if parsing fails
        else:
            logger.warning(f"modified_at value is not a datetime object or string: {modified_at_val}. Setting to 'N/A'.")
            modified_at_str = "N/A"

        return cls(
            name=model_name,
            modified_at=modified_at_str,
            size=raw.get("size", 0)
        )

class LegacyChatRequest(BaseModel):
    query: str
    model: Optional[str] = None # Made model optional to align with usage

class RAGChatRequest(BaseModel):
    session_id: str
    prompt: str
    # Consider using List[Message] here if it fits your history structure
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history as a list of role/content dicts.") 
    use_rag: bool = True

class RAGChatResponse(BaseModel):
    session_id: str
    response: str

class RAGStreamRequest(BaseModel):
    session_id: str
    prompt: str
    # Consider using List[Message] here
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history for RAG stream.")
    use_rag: bool = True

# Model for the response from /api/chat (streaming uses dicts, but this can be for non-streaming if needed)
class ChatResponseToken(BaseModel):
    token: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    filename: str # Assuming one file per response, adjust if multiple
    chunks_added: Optional[int] = None

class HealthStatusDetail(BaseModel):
    status: str
    details: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    ollama: HealthStatusDetail
    chromadb: HealthStatusDetail

# --- Schemas for Document Chunks (Existing) ---
class DocumentChunk(BaseModel):
    """Represents a single document chunk retrieved from the vector store."""
    id: str
    content: str = Field(..., alias="page_content") # Use alias if field name differs
    metadata: Dict[str, Any] = Field(default_factory=dict) # Ensure default is a factory

    class Config:
        # Correct Pydantic V2 config key
        populate_by_name = True


class DocumentListResponse(BaseModel):
    """Response model for listing all document chunks."""
    count: int
    documents: List[DocumentChunk]

# --- Schemas for Automation Endpoint ---
class AutomateRequest(BaseModel):
    conversation_history: List[Message] = Field(..., description="The current conversation history to be automated.")
    model: str = Field(..., description="The model to use for the automation task.")
    automation_task: Optional[str] = Field(None, description="Specific automation task to perform (e.g., 'summarize', 'generate_next_steps').")
    config_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional configuration parameters for automation.")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_history": [
                    {"role": "user", "content": "We discussed the project timeline and deliverables."},
                    {"role": "assistant", "content": "Okay, I've noted that. The key deliverables are X, Y, and Z due by next Friday."},
                    {"role": "user", "content": "Correct. Also, remember to schedule the follow-up meeting."}
                ],
                "model": "llama3:latest", # Example model
                "automation_task": "generate_meeting_summary_and_actions",
                "config_params": {"max_summary_length": 200}
            }
        }

class AutomateResponse(BaseModel):
    status: str = Field(..., description="Status of the automation request (e.g., 'success', 'error').")
    message: Optional[str] = Field(None, description="A message providing details about the outcome.")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Output data from the automation process.")
    error_details: Optional[str] = Field(None, description="Details if an error occurred.")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Conversation automated successfully.",
                "data": {
                    "summary": "The project timeline and deliverables were discussed. Key items are X, Y, Z due next Friday. A follow-up meeting needs to be scheduled.",
                    "action_items": ["Schedule follow-up meeting."]
                }
            }
        }
