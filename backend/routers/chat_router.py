# backend/routers/chat_router.py
import logging
import json
import time 
import httpx # Import httpx for direct API calls
from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator, Optional, List, Dict

# Import schemas, config, and RAG components
try:
    from schemas import (
        LegacyChatRequest, RAGChatRequest, RAGChatResponse, 
        RAGStreamRequest, ChatResponseToken
    )
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG, logger
    from rag_components import get_rag_or_simple_chain, format_history_for_lc # Import formatting func
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage # Import message types
except ImportError:
    # Fallback imports (less likely needed now)
    from backend.schemas import ( 
        LegacyChatRequest, RAGChatRequest, RAGChatResponse, 
        RAGStreamRequest, ChatResponseToken
    )
    from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG, logger 
    from backend.rag_components import get_rag_or_simple_chain, format_history_for_lc
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# --- Control Flag ---
# Set this to True to use the LangChain RAG/Simple chain from rag_components.py
# Set this to False to bypass LangChain and call Ollama API directly in /api/chat
RAG_ENABLED = False # Keep RAG disabled for now as requested

router = APIRouter(
    tags=["chat"],
    # No prefix here, it's handled in main.py
)

# Path is relative to the prefix defined in main.py (e.g., /api/chat)
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
    
    if not query:
        logger.error(f"[{request_id}] /api/chat: 'query' field is missing or empty in request.")
        return JSONResponse(status_code=400, content={"detail": "'query' field is required."})
    
    effective_model_name = OLLAMA_MODEL_FOR_RAG 
    if model_name_from_request and model_name_from_request.strip() and model_name_from_request != "Unknown Model": 
        effective_model_name = model_name_from_request
        logger.info(f"[{request_id}] /api/chat will use model: {effective_model_name}")
    else:
        logger.warning(f"[{request_id}] /api/chat: 'model' field missing or invalid ('{model_name_from_request}'). Using default model: {OLLAMA_MODEL_FOR_RAG}")
        effective_model_name = OLLAMA_MODEL_FOR_RAG 

    logger.info(f"[{request_id}] /api/chat final model: {effective_model_name}, query: '{query[:50]}...'")
    
    async def event_stream() -> AsyncGenerator[str, None]:
        stream_id = f"stream_{request_id}" 
        
        if not RAG_ENABLED:
            # --- Direct Ollama API Call Path ---
            logger.info(f"[{stream_id}] RAG is disabled. Calling Ollama API directly.")
            messages_payload = [{"role": "system", "content": "You are a helpful AI assistant."}]
            for msg in history:
                 role = msg.get("sender", "user") 
                 if role == "bot":
                     role = "assistant"
                 messages_payload.append({"role": role, "content": msg.get("text", "")})
            messages_payload.append({"role": "user", "content": query})
            logger.info(f"[{stream_id}] Sending messages to Ollama API: {messages_payload}")
            try:
                async with httpx.AsyncClient(timeout=None) as client: 
                    async with client.stream(
                        "POST", 
                        f"{OLLAMA_BASE_URL}/api/chat", 
                        json={
                            "model": effective_model_name,
                            "messages": messages_payload,
                            "stream": True
                        }
                    ) as response:
                        if response.status_code != 200:
                            error_content = await response.aread()
                            logger.error(f"[{stream_id}] Ollama API error ({response.status_code}): {error_content.decode()}")
                            yield f"data: {json.dumps({'error': f'Ollama API error {response.status_code}: {error_content.decode()}'})}\n\n"
                            return 
                        logger.info(f"[{stream_id}] Successfully connected to Ollama stream.")
                        chunk_index = 0
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    chunk_data = json.loads(line)
                                    token = chunk_data.get("message", {}).get("content", "")
                                    if token:
                                         logger.debug(f"[{stream_id}] Yielding token {chunk_index}: '{token}'")
                                         sse_data = json.dumps({'token': token})
                                         yield f"data: {sse_data}\n\n"
                                         chunk_index += 1
                                    if chunk_data.get("done"):
                                        logger.info(f"[{stream_id}] Ollama stream reported done=true.")
                                        break 
                                except json.JSONDecodeError:
                                    logger.warning(f"[{stream_id}] Failed to decode JSON line from Ollama stream: {line}")
                                except Exception as e_parse:
                                     logger.error(f"[{stream_id}] Error processing Ollama stream chunk: {line}. Error: {e_parse}", exc_info=True)
                logger.info(f"[{stream_id}] Yielding final done status after {chunk_index} token chunks.")
                yield f"data: {json.dumps({'status': 'done'})}\n\n" 
            except httpx.RequestError as e_req:
                logger.error(f"[{stream_id}] HTTP request error connecting to Ollama: {e_req}", exc_info=True)
                yield f"data: {json.dumps({'error': f'Could not connect to Ollama service: {str(e_req)}'})}\n\n"
            except Exception as e_direct:
                logger.error(f"[{stream_id}] Unhandled exception during direct Ollama call: {e_direct}", exc_info=True)
                yield f"data: {json.dumps({'error': f'Server error during direct Ollama call: {str(e_direct)}'})}\n\n"
            finally:
                 logger.info(f"[{stream_id}] Direct Ollama event stream generator finished.")
        else:
            # --- LangChain RAG/Simple Chain Path ---
            logger.info(f"[{stream_id}] RAG is enabled. Using LangChain chain from get_rag_or_simple_chain.")
            try:
                chain_to_use = get_rag_or_simple_chain(use_rag_flag=True, specific_ollama_model_name=effective_model_name)
                chain_input = {"question": query, "chat_history": history} 
                logger.info(f"[{stream_id}] Streaming response via LangChain using model {effective_model_name}...")
                chunk_index = 0
                async for chunk in chain_to_use.astream(chain_input):
                    logger.debug(f"[{stream_id}] Yielding LangChain chunk {chunk_index}: '{chunk}'") 
                    sse_data = json.dumps({'token': chunk})
                    yield f"data: {sse_data}\n\n"
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

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Path is relative to the prefix defined in main.py (e.g., /api/chat_rag)
# Removed invalid 'prefix' argument from decorator
@router.post("/chat_rag", response_model=RAGChatResponse) 
async def rag_specific_chat_endpoint(request: RAGChatRequest):
    request_id = f"req_{int(time.time()*1000)}"
    logger.info(f"[{request_id}] RAG chat_rag: session {request.session_id}, use_rag={request.use_rag}, prompt: '{request.prompt[:50]}...'")
    try:
        chain = get_rag_or_simple_chain(use_rag_flag=request.use_rag) 
        response_content = await chain.ainvoke({"question": request.prompt, "chat_history": request.history})
        logger.info(f"[{request_id}] RAG chat_rag successful for session {request.session_id}")
        return RAGChatResponse(session_id=request.session_id, response=response_content)
    except HTTPException as e_http: 
        logger.error(f"[{request_id}] RAG chat_rag HTTP error for session {request.session_id}: {e_http.detail}")
        raise e_http 
    except Exception as e:
        logger.error(f"[{request_id}] RAG chat_rag unhandled error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")

# Path is relative to the prefix defined in main.py (e.g., /api/stream_rag)
# Removed invalid 'prefix' argument from decorator
@router.post("/stream_rag") 
async def rag_specific_stream_endpoint(request: RAGStreamRequest):
    request_id = f"req_{int(time.time()*1000)}"
    logger.info(f"[{request_id}] RAG stream_rag: session {request.session_id}, use_rag={request.use_rag}, prompt: '{request.prompt[:50]}...'")
    
    async def event_stream():
        stream_id = f"stream_{request_id}"
        logger.info(f"[{stream_id}] Starting explicit RAG event stream generation...")
        chunk_index = 0
        try:
            chain = get_rag_or_simple_chain(use_rag_flag=request.use_rag) 
            async for chunk in chain.astream({"question": request.prompt, "chat_history": request.history}):
                logger.debug(f"[{stream_id}] Yielding explicit RAG chunk {chunk_index}: '{chunk}'")
                sse_data = json.dumps({'token': chunk})
                yield f"data: {sse_data}\n\n"
                chunk_index += 1
            
            logger.info(f"[{stream_id}] Yielding explicit RAG done status after {chunk_index} chunks.")
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            logger.info(f"[{stream_id}] Explicit RAG Stream finished for session {request.session_id}")
        except HTTPException as e_http: 
            logger.error(f"[{stream_id}] HTTP Exception during explicit RAG stream generation: {e_http.detail}")
            yield f"data: {json.dumps({'error': e_http.detail})}\n\n"
        except Exception as e:
            logger.error(f"[{stream_id}] Unhandled exception during explicit RAG stream generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Failed to generate explicit RAG streaming response: {str(e)}'})}\n\n"
        finally:
             logger.info(f"[{stream_id}] Explicit RAG event stream generator finished.")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
