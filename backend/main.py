from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests # Keep requests for API call
import subprocess # Keep subprocess if you might switch back
import os
import json
import httpx # Using httpx for potential async benefits and streaming

app = FastAPI(title="Accessibility Navigator API")
# Ensure OLLAMA_URL is correctly set in your environment or default
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") # Default to localhost if not docker internal

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more specific in production! e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Accessibility Navigator API"}

class ChatRequest(BaseModel):
    query: str

async def call_gemma3_streaming(query: str) -> str:
    """
    Calls Ollama's API using streaming and accumulates the response.
    Uses httpx for async request.
    """
    api_url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": "gemma3", # Make sure 'gemma3' is the correct model name served by Ollama
        "messages": [{"role": "user", "content": query}],
        "stream": True # Explicitly request streaming
    }
    accumulated_content = ""
    try:
        # Use httpx.AsyncClient for async streaming request
        async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout
            async with client.stream("POST", api_url, json=payload) as response:
                # Check for non-200 status codes before trying to read
                if response.status_code != 200:
                    error_content = await response.aread() # Read error body
                    print(f"Error from Ollama API: Status {response.status_code}, Body: {error_content.decode()}")
                    return f"Error contacting model: Status {response.status_code}"

                # Process the stream line by line
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data_line = json.loads(line)
                            # Check if 'message' and 'content' exist and accumulate
                            if "message" in data_line and "content" in data_line["message"]:
                                accumulated_content += data_line["message"]["content"]
                            # Optional: Check for final 'done' status if needed
                            # if data_line.get("done"):
                            #     break
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {line}")
                        except Exception as e:
                            print(f"Error processing stream line: {line}, Error: {e}")

    except httpx.RequestError as e:
        print(f"Error calling Ollama API: {e}")
        return f"Error calling Ollama service: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

    # Return accumulated content or a default message if empty
    return accumulated_content if accumulated_content else "No response content received."


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Use the async streaming function
    response_text = await call_gemma3_streaming(request.query)
    return {"answer": response_text}