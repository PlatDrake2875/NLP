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

# --- If you still need the non-streaming version for fallback or testing ---
# def call_gemma3_non_streaming(query: str) -> str:
#     api_url = f"{OLLAMA_URL}/api/chat"
#     payload = {
#         "model": "gemma3",
#         "messages": [{"role": "user", "content": query}],
#         "stream": False # Explicitly non-streaming
#     }
#     try:
#         response = requests.post(api_url, json=payload, timeout=60) # Add timeout
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
#         data = response.json()
#         # Adjust based on actual non-streaming response structure
#         if "message" in data and "content" in data["message"]:
#             return data["message"]["content"]
#         elif "response" in data: # Some Ollama versions might use 'response' key
#             return data["response"]
#         else:
#              print("Unexpected non-streaming response format:", data)
#              return "Unexpected response format from model."
#     except requests.exceptions.RequestException as e:
#         print(f"Error calling Ollama (non-streaming): {e}")
#         return f"Error calling Ollama service: {e}"
#     except Exception as e:
#         print(f"An unexpected error occurred (non-streaming): {e}")
#         return f"An unexpected error occurred: {e}"

# @app.post("/api/chat_non_stream")
# async def chat_endpoint_non_stream(request: ChatRequest):
#     response_text = call_gemma3_non_streaming(request.query)
#     return {"answer": response_text}
# --- End of non-streaming version ---

# To run (if this file is named main.py): uvicorn main:app --reload