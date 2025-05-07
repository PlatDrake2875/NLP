# backend/routers/chat_router.py
import logging
import json
from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator, Optional

# Import schemas, config, and RAG components using direct absolute paths
from schemas import (
    LegacyChatRequest, RAGChatRequest, RAGChatResponse, 
    RAGStreamRequest, ChatResponseToken
)
from config import OLLAMA_MODEL_FOR_RAG, logger
from rag_components import get_rag_or_simple_chain


router = APIRouter(
    tags=["chat"],
    # Prefix is handled in main.py's include_router
)

@router.post("/chat") # Path relative to /api -> /api/chat
async def legacy_compatible_chat_stream(fastapi_req: FastAPIRequest): 
    try:
        actual_body = await fastapi_req.json()
    except json.JSONDecodeError:
        logger.error("/api/chat: Invalid JSON in request body.")
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON in request body."})

    logger.info(f"Received request for /api/chat. Raw body: {actual_body}")

    query = actual_body.get("query")
    model_name_from_request = actual_body.get("model")
    
    if not query:
        logger.error("/api/chat: 'query' field is missing or empty in request.")
        return JSONResponse(status_code=400, content={"detail": "'query' field is required."})
    
    effective_model_name = OLLAMA_MODEL_FOR_RAG 
    if model_name_from_request and model_name_from_request.strip() and model_name_from_request != "Unknown Model": 
        effective_model_name = model_name_from_request
        logger.info(f"/api/chat will use model from request: {effective_model_name}")
    else:
        logger.warning(f"/api/chat: 'model' field missing or invalid ('{model_name_from_request}'). Using default RAG model: {OLLAMA_MODEL_FOR_RAG}")

    logger.info(f"/api/chat final model: {effective_model_name}, query: '{query[:50]}...'")
    
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            chain_to_use = get_rag_or_simple_chain(use_rag_flag=True, specific_ollama_model_name=effective_model_name)
            chain_input = {"question": query, "chat_history": actual_body.get("history", [])} 
            
            logger.info(f"Streaming RAG-enabled response for /api/chat using model {effective_model_name}...")
            async for chunk in chain_to_use.astream(chain_input):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n" # Signal end of stream
            logger.info(f"Stream finished for /api/chat query: '{query[:50]}...'")
        except HTTPException as e_http: # Catch HTTPExceptions from get_rag_or_simple_chain
             logger.error(f"HTTP Exception during /api/chat stream: {e_http.detail}")
             yield f"data: {json.dumps({'error': e_http.detail})}\n\n"
        except Exception as e_stream:
            logger.error(f"Error during /api/chat stream: {e_stream}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Failed to generate streaming response: {str(e_stream)}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat_rag") # Path relative to /api -> /api/chat_rag
async def rag_specific_chat_endpoint(request: RAGChatRequest):
    logger.info(f"RAG chat_rag: session {request.session_id}, use_rag={request.use_rag}, prompt: '{request.prompt[:50]}...'")
    try:
        chain = get_rag_or_simple_chain(use_rag_flag=request.use_rag) 
        response_content = await chain.ainvoke({"question": request.prompt, "chat_history": request.history})
        return RAGChatResponse(session_id=request.session_id, response=response_content)
    except HTTPException as e_http: 
        raise e_http # Re-raise HTTPExceptions from chain creation
    except Exception as e:
        logger.error(f"RAG chat_rag error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")

@router.post("/stream_rag") # Path relative to /api -> /api/stream_rag
async def rag_specific_stream_endpoint(request: RAGStreamRequest):
    logger.info(f"RAG stream_rag: session {request.session_id}, use_rag={request.use_rag}, prompt: '{request.prompt[:50]}...'")
    async def event_stream():
        try:
            chain = get_rag_or_simple_chain(use_rag_flag=request.use_rag) 
            async for chunk in chain.astream({"question": request.prompt, "chat_history": request.history}):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            logger.info(f"RAG Stream finished for session {request.session_id}")
        except HTTPException as e_http: 
            yield f"data: {json.dumps({'error': e_http.detail})}\n\n"
        except Exception as e:
            logger.error(f"RAG stream_rag error for session {request.session_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Failed to generate RAG streaming response: {str(e)}'})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
