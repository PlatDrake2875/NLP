from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse # Import StreamingResponse
from pydantic import BaseModel
import os
import json
import httpx # Using httpx for async benefits and streaming
from typing import List, Dict, Any, AsyncGenerator # Import AsyncGenerator

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
                # Raise exception immediately for bad status codes
                if response.status_code != 200:
                    error_content = await response.aread()
                    error_text = error_content.decode()
                    print(f"Error from Ollama API: Status {response.status_code}, Body: {error_text}")
                    # Try to parse detail from Ollama's error JSON
                    detail = f"Status {response.status_code}"
                    try:
                        error_json = json.loads(error_text)
                        detail = error_json.get("error", detail)
                    except json.JSONDecodeError:
                        detail = f"{detail}: {error_text[:100]}"
                    # Yield an error message chunk (optional, frontend needs to handle)
                    # yield json.dumps({"error": f"Error contacting model '{model}': {detail}"}) + "\n"
                    # Or just raise, letting the main endpoint handle it before streaming starts
                    response.raise_for_status() # This will likely raise an HTTPStatusError

                # Stream the response content chunk by chunk
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data_line = json.loads(line)
                            if "message" in data_line and "content" in data_line["message"]:
                                yield data_line["message"]["content"] # Yield only the content part
                            if "error" in data_line:
                                print(f"Error in Ollama stream: {data_line['error']}")
                                # Yield an error chunk (optional, frontend needs to handle)
                                # yield json.dumps({"error": f"Error from model '{model}': {data_line['error']}"}) + "\n"
                                # Or break/raise depending on desired behavior
                                break # Stop streaming on error
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {line}")
                        except Exception as e:
                            print(f"Error processing stream line: {line}, Error: {e}")
                            # Decide whether to continue or stop on processing errors

    except httpx.HTTPStatusError as e:
        # Handle errors that occur before or during the initial connection/response headers
        print(f"HTTP Error connecting to Ollama: {e}")
        # Yielding error here might be tricky as the response status is already set
        # Better to let the main endpoint catch this before returning StreamingResponse
        # For simplicity, we'll just log and the stream will end prematurely.
        # A more robust solution might involve a wrapper or different error signaling.
        pass # The stream will simply end if connection fails or status is bad initially.
    except httpx.RequestError as e:
        print(f"Request Error connecting to Ollama: {e}")
        pass # Stream ends
    except Exception as e:
        print(f"An unexpected error occurred during streaming generation: {e}")
        pass # Stream ends


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Accessibility Navigator API"}

@app.get("/api/models", response_model=List[OllamaModel])
async def list_models():
    """Endpoint to get the list of available Ollama models."""
    # This endpoint remains unchanged
    models = await get_ollama_models()
    return models

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that streams the response from Ollama token by token.
    """
    try:
        # Return a StreamingResponse, passing the async generator
        return StreamingResponse(
            ollama_stream_generator(request.query, request.model),
            media_type="text/plain" # Or "text/event-stream" if formatting that way
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions (e.g., from model fetching if called here)
        raise e
    except Exception as e:
        # Catch unexpected errors before starting the stream
        print(f"Error setting up chat stream: {e}")
        raise HTTPException(status_code=500, detail="Failed to start chat stream.")

# --- Uvicorn runner (for local development) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

