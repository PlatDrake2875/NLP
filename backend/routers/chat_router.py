# backend/routers/chat_router.py
import logging
import json
import time
import httpx
from fastapi import APIRouter, HTTPException, Depends, Request as FastAPIRequest # Import Depends
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator, Optional, List, Dict, Any

# --- Direct Imports ---
from schemas import (
    LegacyChatRequest, RAGChatRequest, RAGChatResponse,
    RAGStreamRequest, ChatResponseToken
)
from config import OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG, logger
# Import necessary functions and the main chain getter
from rag_components import (
    format_history_for_lc,
    get_rag_or_simple_chain, # Import the main chain getter
    RAG_ENABLED # Import the RAG_ENABLED flag
)
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
    query: str,
    history: List[Dict[str, str]],
    stream_id: str
) -> AsyncGenerator[str, None]:
    """Handles direct streaming call to Ollama API. (No changes needed here)"""
    logger.info(f"[{stream_id}] Calling Ollama API directly for model '{model_name}'.")
    messages_payload = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for msg in history:
         role = msg.get("sender", "user")
         if role == "bot": role = "assistant"
         content = msg.get("text", "")
         if not isinstance(content, str): content = str(content)
         messages_payload.append({"role": role, "content": content})
    messages_payload.append({"role": "user", "content": query})
    logger.info(f"[{stream_id}] Sending messages to Ollama API: {messages_payload}")
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
                            if token:
                                 logger.debug(f"[{stream_id}] Yielding token {chunk_index}: '{token}'")
                                 yield f"data: {json.dumps({'token': token})}\n\n"; chunk_index += 1
                            if chunk_data.get("done"): logger.info(f"[{stream_id}] Ollama stream reported done=true."); break
                        except json.JSONDecodeError: logger.warning(f"[{stream_id}] Failed to decode JSON line: {line}")
                        except Exception as e_parse: logger.error(f"[{stream_id}] Error processing chunk: {line}. Error: {e_parse}", exc_info=True)
        logger.info(f"[{stream_id}] Yielding final done status after {chunk_index} token chunks.")
        yield f"data: {json.dumps({'status': 'done'})}\n\n"
    except httpx.RequestError as e_req:
        logger.error(f"[{stream_id}] HTTP request error: {e_req}", exc_info=True)
        yield f"data: {json.dumps({'error': f'Could not connect to Ollama service: {str(e_req)}'})}\n\n"
    except Exception as e_direct:
        logger.error(f"[{stream_id}] Unhandled exception: {e_direct}", exc_info=True)
        yield f"data: {json.dumps({'error': f'Server error during direct Ollama call: {str(e_direct)}'})}\n\n"
    finally: logger.info(f"[{stream_id}] Direct Ollama event stream generator finished.")


async def _langchain_stream(
    # No request object needed here anymore
    use_rag: bool,
    model_name: Optional[str],
    query: str,
    history: List[Dict[str, str]],
    stream_id: str
) -> AsyncGenerator[str, None]:
    """Handles streaming using the LangChain chain obtained via get_rag_or_simple_chain."""
    logger.info(f"[{stream_id}] Using LangChain chain (use_rag={use_rag}) for model '{model_name or 'Default'}'.")
    try:
        # Get the appropriate chain (RAG or Simple) using the main function
        # This function now internally handles getting components via dependencies
        chain_to_use = get_rag_or_simple_chain(use_rag_flag=use_rag, specific_ollama_model_name=model_name)

        # Prepare input and stream
        chain_input = {"question": query, "chat_history": history}
        logger.info(f"[{stream_id}] Streaming response via LangChain...")
        chunk_index = 0
        async for chunk in chain_to_use.astream(chain_input):
            token = chunk if isinstance(chunk, str) else str(chunk)
            logger.debug(f"[{stream_id}] Yielding LangChain chunk {chunk_index}: '{token}'")
            yield f"data: {json.dumps({'token': token})}\n\n"
            chunk_index += 1

        logger.info(f"[{stream_id}] Yielding LangChain done status after {chunk_index} chunks.")
        yield f"data: {json.dumps({'status': 'done'})}\n\n"
        logger.info(f"[{stream_id}] LangChain stream finished for query: '{query[:50]}...'")

    except HTTPException as e_http:
         logger.error(f"[{stream_id}] HTTP Exception during LangChain stream: {e_http.detail}")
         yield f"data: {json.dumps({'error': e_http.detail})}\n\n"
    except Exception as e_stream:
        logger.error(f"[{stream_id}] Unhandled exception during LangChain stream: {e_stream}", exc_info=True)
        yield f"data: {json.dumps({'error': f'Failed to generate LangChain streaming response: {str(e_stream)}'})}\n\n"
    finally:
         logger.info(f"[{stream_id}] LangChain event stream generator finished.")


# --- API Endpoints ---

@router.post("/chat")
async def legacy_compatible_chat_stream(fastapi_req: FastAPIRequest): # Keep request for body parsing
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

    if not isinstance(history, list) or not all(isinstance(msg, dict) for msg in history):
         logger.error(f"[{request_id}] /api/chat: Invalid 'history' format.")
         return JSONResponse(status_code=400, content={"detail": "Invalid 'history' format."})
    if not query:
        logger.error(f"[{request_id}] /api/chat: 'query' field is missing.")
        return JSONResponse(status_code=400, content={"detail": "'query' field is required."})

    effective_model_name = OLLAMA_MODEL_FOR_RAG
    if model_name_from_request and isinstance(model_name_from_request, str) and model_name_from_request.strip() and model_name_from_request != "Unknown Model":
        effective_model_name = model_name_from_request
        logger.info(f"[{request_id}] /api/chat using requested model: {effective_model_name}")
    else:
        logger.warning(f"[{request_id}] /api/chat using default RAG model: {OLLAMA_MODEL_FOR_RAG}")

    logger.info(f"[{request_id}] /api/chat final model: {effective_model_name}, query: '{query[:50]}...'")
    stream_id = f"stream_{request_id}"

    if not RAG_ENABLED:
        generator = _direct_ollama_stream(effective_model_name, query, history, stream_id)
    else:
        # Use LangChain stream, assuming RAG=True for this endpoint if RAG_ENABLED
        generator = _langchain_stream(True, effective_model_name, query, history, stream_id)

    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("/chat_rag", response_model=RAGChatResponse)
async def rag_specific_chat_endpoint(request_body: RAGChatRequest): # No request object needed
    request_id = f"req_{int(time.time()*1000)}"
    logger.info(f"[{request_id}] RAG chat_rag: session {request_body.session_id}, use_rag={request_body.use_rag}, prompt: '{request_body.prompt[:50]}...'")
    try:
        # Get the appropriate chain using the main function
        # Pass None for model_name to use the default RAG model
        chain = get_rag_or_simple_chain(use_rag_flag=request_body.use_rag, specific_ollama_model_name=None)

        # Prepare input and invoke
        chain_input = {"question": request_body.prompt, "chat_history": request_body.history}
        response_content = await chain.ainvoke(chain_input)

        logger.info(f"[{request_id}] RAG chat_rag successful for session {request_body.session_id}")
        return RAGChatResponse(session_id=request_body.session_id, response=response_content)

    except HTTPException as e_http:
        logger.error(f"[{request_id}] RAG chat_rag HTTP error: {e_http.detail}")
        raise e_http
    except Exception as e:
        logger.error(f"[{request_id}] RAG chat_rag unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")


@router.post("/stream_rag")
async def rag_specific_stream_endpoint(request_body: RAGStreamRequest): # No request object needed
    request_id = f"req_{int(time.time()*1000)}"
    logger.info(f"[{request_id}] RAG stream_rag: session {request_body.session_id}, use_rag={request_body.use_rag}, prompt: '{request_body.prompt[:50]}...'")

    stream_id = f"stream_{request_id}"
    # Use the LangChain stream helper, respecting use_rag flag
    # Pass None for model_name to use the default RAG model
    generator = _langchain_stream(
        request_body.use_rag,
        None, # Use default model configured for RAG
        request_body.prompt,
        request_body.history,
        stream_id
    )

    return StreamingResponse(generator, media_type="text/event-stream")
