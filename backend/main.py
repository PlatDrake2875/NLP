from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse # Import StreamingResponse
from pydantic import BaseModel
import os
import json
import httpx # Using httpx for async benefits and streaming
from typing import List, Dict, Any, AsyncGenerator, Optional # Import Optional

app = FastAPI(title="Accessibility Navigator API")
# Ensure OLLAMA_URL is correctly set in your environment or default
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") # Default to localhost if not set

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more specific in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str
    model: str

class OllamaModel(BaseModel):
    """Model representing a single Ollama model tag."""
    name: str
    modified_at: str
    size: int

class OllamaTagsResponse(BaseModel):
    """Response model for Ollama's /api/tags endpoint."""
    models: List[OllamaModel]

# New models for automation endpoint
class AutomateRequest(BaseModel):
    """Request model for the automation endpoint."""
    inputs: List[str]
    model: str

class ConversationTurn(BaseModel):
    """Represents one turn (user input + bot response) in the conversation."""
    sender: str # 'user' or 'bot'
    text: str

# --- Helper Functions ---

async def get_ollama_models() -> List[OllamaModel]:
    """Fetches the list of available models from Ollama."""
    api_url = f"{OLLAMA_URL}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()
            tags_response = OllamaTagsResponse(**data)
            return tags_response.models
    except httpx.RequestError as e:
        print(f"Error fetching Ollama models: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to Ollama at {api_url}: {e}")
    except httpx.HTTPStatusError as e:
        print(f"Error response from Ollama API ({e.response.status_code}): {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from Ollama API: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching models: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching models.")


async def ollama_stream_generator(query: str, model: str) -> AsyncGenerator[str, None]:
    """
    Async generator that yields chunks from Ollama's streaming API.
    (Used by the interactive chat endpoint)
    """
    api_url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": True
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", api_url, json=payload) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_text = error_content.decode()
                    print(f"Error from Ollama API: Status {response.status_code}, Body: {error_text}")
                    detail = f"Status {response.status_code}"
                    try:
                        error_json = json.loads(error_text)
                        detail = error_json.get("error", detail)
                    except json.JSONDecodeError:
                        detail = f"{detail}: {error_text[:100]}"
                    response.raise_for_status() # Raise HTTPStatusError

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data_line = json.loads(line)
                            if "message" in data_line and "content" in data_line["message"]:
                                yield data_line["message"]["content"]
                            if "error" in data_line:
                                print(f"Error in Ollama stream: {data_line['error']}")
                                break
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {line}")
                        except Exception as e:
                            print(f"Error processing stream line: {line}, Error: {e}")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error connecting to Ollama: {e}")
        pass
    except httpx.RequestError as e:
        print(f"Request Error connecting to Ollama: {e}")
        pass
    except Exception as e:
        print(f"An unexpected error occurred during streaming generation: {e}")
        pass

async def get_ollama_response(messages: List[Dict[str, str]], model: str) -> Optional[str]:
    """
    Gets a single, non-streaming response from Ollama given a conversation history.
    (Used by the automation endpoint)
    """
    api_url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages, # Pass the whole history
        "stream": False # Request non-streaming response
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client: # Longer timeout for potentially longer generation
            response = await client.post(api_url, json=payload)
            response.raise_for_status() # Raise exception for bad status codes
            data = response.json()
            if data and "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            else:
                print(f"Warning: Unexpected response structure from Ollama: {data}")
                return None # Or raise an error
    except httpx.RequestError as e:
        print(f"Request Error connecting to Ollama for automation: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to Ollama: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error from Ollama API ({e.response.status_code}): {e.response.text}")
        detail = f"Ollama API Error: Status {e.response.status_code}"
        try:
            error_json = e.response.json()
            detail = error_json.get("error", detail)
        except json.JSONDecodeError:
             detail = f"{detail}: {e.response.text[:150]}" # Include snippet of non-JSON error
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        print(f"An unexpected error occurred getting Ollama response: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during Ollama communication.")


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Accessibility Navigator API"}

@app.get("/api/models", response_model=List[OllamaModel])
async def list_models():
    """Endpoint to get the list of available Ollama models."""
    models = await get_ollama_models()
    return models

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Interactive chat endpoint that streams the response from Ollama token by token.
    """
    try:
        return StreamingResponse(
            ollama_stream_generator(request.query, request.model),
            media_type="text/plain"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error setting up chat stream: {e}")
        raise HTTPException(status_code=500, detail="Failed to start chat stream.")

@app.post("/api/automate_conversation", response_model=List[ConversationTurn])
async def automate_conversation_endpoint(request: AutomateRequest):
    """
    Endpoint to run an automated conversation based on a list of user inputs.
    It processes inputs sequentially, maintaining context, and returns the full history.
    """
    conversation_history: List[Dict[str, str]] = [] # Stores Ollama-compatible history [{role: 'user'/'assistant', content: '...'}]
    full_result: List[ConversationTurn] = [] # Stores the final result for the frontend

    if not request.inputs:
        raise HTTPException(status_code=400, detail="Input list cannot be empty.")

    print(f"Starting automated conversation with model {request.model} for {len(request.inputs)} inputs.")

    for user_input in request.inputs:
        # Add user input to Ollama history and final result
        user_message_ollama = {"role": "user", "content": user_input}
        user_message_frontend = ConversationTurn(sender="user", text=user_input)

        conversation_history.append(user_message_ollama)
        full_result.append(user_message_frontend)

        print(f"  Processing user input: '{user_input[:50]}...'")

        # Get bot response from Ollama using the current history
        try:
            bot_response_text = await get_ollama_response(conversation_history, request.model)

            if bot_response_text is None:
                # Handle case where Ollama didn't return content as expected
                bot_response_text = "[Error: No response content received from Ollama]"
                # Decide if we should stop or continue
                # For now, add the error and continue
                print(f"  Warning: No content received from Ollama for input '{user_input[:50]}...'")

            # Add bot response to Ollama history and final result
            bot_message_ollama = {"role": "assistant", "content": bot_response_text}
            bot_message_frontend = ConversationTurn(sender="bot", text=bot_response_text)

            conversation_history.append(bot_message_ollama)
            full_result.append(bot_message_frontend)
            print(f"  Received bot response: '{bot_response_text[:50]}...'")

        except HTTPException as e:
            # If get_ollama_response raises an HTTPException, append an error message and stop
            print(f"  Error during Ollama call: {e.detail}")
            error_message_frontend = ConversationTurn(sender="bot", text=f"[Automation Error: {e.detail}]")
            full_result.append(error_message_frontend)
            # Optionally re-raise or just return the partial result with the error
            # raise e # Re-raise to send error status code back
            break # Stop processing further inputs on error
        except Exception as e:
            # Catch unexpected errors during the loop
            print(f"  Unexpected error during automation loop: {e}")
            error_message_frontend = ConversationTurn(sender="bot", text=f"[Unexpected Automation Error: {e}]")
            full_result.append(error_message_frontend)
            break # Stop processing

    print(f"Finished automated conversation. Returning {len(full_result)} turns.")
    return full_result

# --- Uvicorn runner (for local development) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
