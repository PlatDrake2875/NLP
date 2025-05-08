# backend/routers/chat_router.py
import logging
import json
import time
import httpx
from fastapi import APIRouter, HTTPException, Depends, Request as FastAPIRequest
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator, Optional, List, Dict, Any

# --- Direct Imports ---
from schemas import (
    LegacyChatRequest, RAGChatRequest, RAGChatResponse,
    RAGStreamRequest, ChatResponseToken
)
try:
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG, logger, RAG_ENABLED
except ImportError as e:
    logger.critical(f"Failed to import from config: {e}. Ensure config.py exists.")
    raise

try:
    from rag_components import get_rag_context_prefix # Import the new function
except ImportError as e:
     logger.critical(f"Failed to import from rag_components: {e}. Ensure rag_components.py exists.")
     raise

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# Type hints (optional but good practice)
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma as LangchainChroma
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate


# --- Router Setup ---
router = APIRouter(
    tags=["chat"],
)

# --- Helper Async Generators for Streaming ---

async def _direct_ollama_stream(
    model_name: str,
    messages_payload: List[Dict[str, str]], # Accept pre-formatted messages
    stream_id: str
) -> AsyncGenerator[str, None]:
    """Handles direct streaming call to Ollama API with pre-formatted messages."""
    logger.info(f"[{stream_id}] Calling Ollama API directly for model '{model_name}'.")
    logger.info(f"[{stream_id}] Sending messages payload (count: {len(messages_payload)}) to Ollama API.")
    if messages_payload:
         logger.debug(f"[{stream_id}] Last message role: {messages_payload[-1].get('role')}, Content length: {len(messages_payload[-1].get('content', ''))}")
    else:
         logger.warning(f"[{stream_id}] Sending empty messages payload to Ollama API.")

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json={"model": model_name, "messages": messages_payload, "stream": True}) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    err_msg = f"Ollama API error {response.status_code}: {error_content.decode()}"
                    logger.error(f"[{stream_id}] {err_msg}")
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"; return

                logger.info(f"[{stream_id}] Successfully connected to Ollama stream.")
                chunk_index = 0
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            token = chunk_data.get("message", {}).get("content", "")
                            is_done = chunk_data.get("done", False)

                            if token:
                                 yield f"data: {json.dumps({'token': token})}\n\n"; chunk_index += 1
                            if is_done:
                                logger.info(f"[{stream_id}] Ollama stream reported done=true after {chunk_index} token chunks.")
                                break
                        except json.JSONDecodeError: logger.warning(f"[{stream_id}] Failed to decode JSON line: {line}")
                        except Exception as e_parse: logger.error(f"[{stream_id}] Error processing chunk: {line}. Error: {e_parse}", exc_info=True)

        logger.info(f"[{stream_id}] Yielding final done status.")
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    except httpx.RequestError as e_req:
        logger.error(f"[{stream_id}] HTTP request error: {e_req}", exc_info=True)
        yield f"data: {json.dumps({'error': f'Could not connect to Ollama service: {str(e_req)}'})}\n\n"
    except Exception as e_direct:
        logger.error(f"[{stream_id}] Unhandled exception in _direct_ollama_stream: {e_direct}", exc_info=True)
        yield f"data: {json.dumps({'error': f'Server error during direct Ollama call: {str(e_direct)}'})}\n\n"
    finally:
        logger.info(f"[{stream_id}] Direct Ollama event stream generator finished.")


# --- API Endpoints ---
@router.post("/chat")
async def legacy_compatible_chat_stream(fastapi_req: FastAPIRequest):
    request_id = f"req_{int(time.time()*1000)}"
    try:
        actual_body = await fastapi_req.json()
    except json.JSONDecodeError:
        logger.error(f"[{request_id}] /api/chat: Invalid JSON in request body.")
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON in request body."})

    logger.info(f"[{request_id}] Received request for /api/chat. RAG_ENABLED={RAG_ENABLED}. Raw body: {actual_body}")

    query = actual_body.get("query")
    model_name_from_request = actual_body.get("model")
    history = actual_body.get("history", [])
    use_rag_for_this_request = actual_body.get("use_rag", RAG_ENABLED)

    if not query:
        logger.error(f"[{request_id}] /api/chat: 'query' field is missing.")
        return JSONResponse(status_code=400, content={"detail": "'query' field is required."})
    if not isinstance(history, list):
         logger.warning(f"[{request_id}] /api/chat: Invalid or missing 'history', defaulting to empty list.")
         history = []

    effective_model_name = model_name_from_request or OLLAMA_MODEL_FOR_RAG
    logger.info(f"[{request_id}] Effective LLM for final generation: {effective_model_name}")

    # Prepare messages for the LLM
    messages_for_llm: List[Dict[str, str]] = []
    rag_prefix_generated = False

    # Add existing history
    for msg in history:
        role = msg.get("sender", "user").lower()
        content = msg.get("text", "")
        if role == "bot":
            messages_for_llm.append({"role": "assistant", "content": content})
        else:
            messages_for_llm.append({"role": "user", "content": content})

    # Attempt RAG prefix generation
    if RAG_ENABLED and use_rag_for_this_request:
        logger.info(f"[{request_id}] Attempting to generate RAG context prefix.")
        try:
            rag_prefixed_prompt_string = await get_rag_context_prefix(query)
        except Exception as e_rag:
            logger.error(f"[{request_id}] Error during get_rag_context_prefix: {e_rag}", exc_info=True)
            rag_prefixed_prompt_string = None # Ensure it's None on error

        if rag_prefixed_prompt_string:
            messages_for_llm.append({"role": "user", "content": rag_prefixed_prompt_string})
            rag_prefix_generated = True
            logger.info(f"[{request_id}] Using RAG-generated prefix as final user message.")
        else:
            logger.info(f"[{request_id}] Failed to generate RAG prefix or no documents found. Using original query.")
            messages_for_llm.append({"role": "user", "content": query})
    else:
        logger.info(f"[{request_id}] RAG not used for this request. Using original query.")
        messages_for_llm.append({"role": "user", "content": query})

    # Streaming
    stream_id = f"stream_{request_id}"
    logger.info(f"[{stream_id}] Starting stream. RAG prefix used: {rag_prefix_generated}")

    # --- FIX: Remove the unexpected 'query' keyword argument ---
    generator = _direct_ollama_stream(
        model_name=effective_model_name,
        messages_payload=messages_for_llm,
        stream_id=stream_id
        # query="" # REMOVED this line
    )
    # --- END FIX ---

    return StreamingResponse(generator, media_type="text/event-stream")


# --- Commented out deprecated endpoints ---
# @router.post("/chat_rag", response_model=RAGChatResponse)
# async def rag_specific_chat_endpoint(request_body: RAGChatRequest):
#     logger.warning("/api/chat_rag endpoint is potentially deprecated. Use /api/chat with use_rag flag.")
#     raise HTTPException(status_code=404, detail="Endpoint deprecated. Use /api/chat.")

# @router.post("/stream_rag")
# async def rag_specific_stream_endpoint(request_body: RAGStreamRequest):
#     logger.warning("/api/stream_rag endpoint is potentially deprecated. Use /api/chat with use_rag flag.")
#     raise HTTPException(status_code=404, detail="Endpoint deprecated. Use /api/chat.")
