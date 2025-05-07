# backend/schemas.py
import logging
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("nlp_backend.schemas") # Logger for this module

# --- Pydantic Models ---
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
    history: List[Dict[str, str]] = []
    use_rag: bool = True

class RAGChatResponse(BaseModel):
    session_id: str
    response: str

class RAGStreamRequest(BaseModel):
    session_id: str
    prompt: str
    history: List[Dict[str, str]] = []
    use_rag: bool = True

# Model for the response from /api/chat (streaming uses dicts, but this can be for non-streaming if needed)
class ChatResponseToken(BaseModel):
    token: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: Optional[int] = None

class HealthStatusDetail(BaseModel):
    status: str
    details: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    ollama: HealthStatusDetail
    chromadb: HealthStatusDetail
