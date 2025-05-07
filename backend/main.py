import os
import logging
import aiohttp
import asyncio
import ollama 
import httpx 
import json
from datetime import datetime 
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request as FastAPIRequest 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma as LangchainChroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.responses import StreamingResponse, JSONResponse


# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL_FOR_RAG = os.getenv("OLLAMA_MODEL", "llama3") 
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="NLP Backend API with RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class OllamaModelInfo(BaseModel):
    name: str  
    modified_at: str
    size: int

    @classmethod
    def from_ollama(cls, raw: Dict[str, Any]) -> Optional["OllamaModelInfo"]:
        model_name = raw.get("name") or raw.get("model")
        if not model_name:
            logger.warning(f"Skipping model entry in from_ollama due to missing 'name'/'model': {raw}")
            return None

        modified_at_val = raw.get("modified_at")
        modified_at_str = ""
        if isinstance(modified_at_val, datetime):
            modified_at_str = modified_at_val.isoformat()
        elif isinstance(modified_at_val, str):
            modified_at_str = modified_at_val
        else:
            modified_at_str = "N/A"
            
        return cls(
            name=model_name,
            modified_at=modified_at_str,  
            size=raw.get("size", 0)
        )

class LegacyChatRequest(BaseModel):
    query: str
    model: str

class RAGChatRequest(BaseModel):
    session_id: str; prompt: str; history: List[Dict[str, str]] = []; use_rag: bool = True
class RAGChatResponse(BaseModel):
    session_id: str; response: str
class RAGStreamRequest(BaseModel):
    session_id: str; prompt: str; history: List[Dict[str, str]] = []; use_rag: bool = True


# --- RAG Components Setup ---
# (Keep this section as it was, assuming it initializes correctly based on previous logs)
chroma_client = None
embedding_function = None
vectorstore = None
retriever = None
ollama_chat_for_rag = None

try:
    logger.info(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} using chromadb-client 0.5.20 defaults.")
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
        logger.info("Attempting to get ChromaDB heartbeat...")
        chroma_client.heartbeat() 
        logger.info("Successfully connected to ChromaDB and got heartbeat.")
    except Exception as e_chroma_init:
        logger.critical(f"Failed to initialize ChromaDB client: {e_chroma_init}", exc_info=False)
        chroma_client = None

    if chroma_client:
        logger.info(f"Initializing HuggingFace embeddings with model: {EMBEDDING_MODEL_NAME}")
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False}
        )
        logger.info("HuggingFace embeddings initialized.")
        COLLECTION_NAME = "rag_documents"
        vectorstore = LangchainChroma(client=chroma_client, collection_name=COLLECTION_NAME, embedding_function=embedding_function)
        logger.info(f"LangChain Chroma vector store initialized for collection: {COLLECTION_NAME}")
        if vectorstore:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
            logger.info("Retriever initialized successfully.")
        else:
            logger.warning("Vectorstore object is None. Cannot initialize retriever.")
    else:
        logger.warning("ChromaDB client failed. Vectorstore and Retriever will be unavailable.")

    logger.info(f"Initializing ChatOllama (for RAG) with model: {OLLAMA_MODEL_FOR_RAG} at {OLLAMA_BASE_URL}")
    ollama_chat_for_rag = ChatOllama(model=OLLAMA_MODEL_FOR_RAG, base_url=OLLAMA_BASE_URL, temperature=0)
    logger.info("ChatOllama (for RAG) initialized.")

except Exception as e:
    logger.critical(f"Critical error during RAG components startup: {e}", exc_info=True)


# --- RAG Chain Definition ---
RAG_PROMPT_TEMPLATE = """CONTEXT:\n{context}\n\nCONVERSATION HISTORY:\n{chat_history}\n\nQUESTION:\n{question}\n\nAnswer the question based *only* on the provided context and conversation history. If the context doesn't contain the answer, say you don't know."""
rag_prompt_template_obj = ChatPromptTemplate.from_messages([("system", RAG_PROMPT_TEMPLATE), MessagesPlaceholder(variable_name="chat_history_messages"), ("human", "{question}")])
SIMPLE_PROMPT_TEMPLATE = """CONVERSATION HISTORY:\n{chat_history}\n\nQUESTION:\n{question}\n\nAnswer the question based on the conversation history and your general knowledge."""
simple_prompt_template_obj = ChatPromptTemplate.from_messages([("system", SIMPLE_PROMPT_TEMPLATE), MessagesPlaceholder(variable_name="chat_history_messages"), ("human", "{question}")])

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history: return "No history yet."
    return "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])

def format_history_for_lc(chat_history: List[Dict[str, str]]) -> list:
    messages = []
    for msg in chat_history:
        if msg.get("role") == "user": messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant": messages.append(AIMessage(content=msg.get("content", "")))
    return messages

def get_rag_or_simple_chain(use_rag_flag: bool, specific_ollama_model_name: Optional[str] = None):
    active_llm = ollama_chat_for_rag 
    if specific_ollama_model_name and specific_ollama_model_name != OLLAMA_MODEL_FOR_RAG and specific_ollama_model_name != "Unknown Model":
        logger.info(f"Chain will use specific Ollama model: {specific_ollama_model_name}")
        active_llm = ChatOllama(model=specific_ollama_model_name, base_url=OLLAMA_BASE_URL, temperature=0)
    elif specific_ollama_model_name == "Unknown Model":
        logger.warning(f"Received 'Unknown Model'. Chain will use default RAG model: {OLLAMA_MODEL_FOR_RAG}")
        if not ollama_chat_for_rag:
                 raise HTTPException(status_code=503, detail="Default RAG LLM is not available.")
        active_llm = ollama_chat_for_rag
    elif not active_llm: 
        logger.error("Default RAG LLM (ollama_chat_for_rag) is not available.")
        raise HTTPException(status_code=503, detail="Core LLM for RAG is not available.")
    else:
        logger.info(f"Chain will use default RAG model: {OLLAMA_MODEL_FOR_RAG}")

    if use_rag_flag and chroma_client and retriever:
        logger.info(f"Creating RAG chain with retriever, using LLM: {active_llm.model if hasattr(active_llm, 'model') else 'Default RAG LLM'}")
        return (RunnableParallel(context=(lambda x: x['question']) | retriever, question=RunnablePassthrough(), chat_history=(lambda x: format_chat_history(x['chat_history'])), chat_history_messages=(lambda x: format_history_for_lc(x['chat_history']))) | rag_prompt_template_obj | active_llm | StrOutputParser())
    else:
        if use_rag_flag: logger.warning("RAG requested but ChromaDB/retriever unavailable. Falling back to simple chain.")
        logger.info(f"Creating Simple chain, using LLM: {active_llm.model if hasattr(active_llm, 'model') else 'Default RAG LLM'}")
        return (RunnableParallel(question=(lambda x: x['question']), chat_history=(lambda x: format_chat_history(x['chat_history'])), chat_history_messages=(lambda x: format_history_for_lc(x['chat_history']))) | simple_prompt_template_obj | active_llm | StrOutputParser())

# --- API Endpoints ---
@app.get("/")
async def read_root(): return {"message": "Backend is running with RAG support"}

@app.get("/api/models", response_model=List[OllamaModelInfo])
async def list_ollama_models_endpoint():
    logger.info("/api/models endpoint called.")
    logger.info(f"Attempting to connect to Ollama at OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")

    try:
        client = ollama.AsyncClient(host=OLLAMA_BASE_URL)
        
        logger.info(f"Ollama AsyncClient created for host {OLLAMA_BASE_URL}. Calling client.list()...")
        # response_data is an ollama._types.ListResponse object
        ollama_list_response = await client.list() 
        logger.debug(f"Received raw response object from Ollama client.list(): {ollama_list_response}")

        # The ollama._types.ListResponse object should have a 'models' attribute
        # which is a list of ollama._types.Model objects.
        if not hasattr(ollama_list_response, 'models'):
            logger.error(f"Ollama ListResponse does not contain 'models' attribute. Full response object: {ollama_list_response}")
            return JSONResponse(
                status_code=500, 
                content={"detail": "Invalid response structure from Ollama (missing 'models' attribute in ListResponse)."}
            )

        models_from_ollama_lib = ollama_list_response.models

        if not isinstance(models_from_ollama_lib, list):
            logger.error(f"'models' attribute in Ollama ListResponse is not a list. Got {type(models_from_ollama_lib)}. Full response: {ollama_list_response}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Invalid data type for 'models' in Ollama ListResponse."}
            )
            
        logger.info(f"List of 'ollama.Model' objects from Ollama library contains {len(models_from_ollama_lib)} items.")
        
        parsed_models = []
        for ollama_model_obj in models_from_ollama_lib:
            # Convert the ollama.Model object to a dictionary
            # It's a Pydantic model, so .model_dump() (v2) or .dict() (v1) should work.
            # Let's try .model_dump() first, common for newer Pydantic.
            try:
                if hasattr(ollama_model_obj, 'model_dump'):
                    model_data_dict = ollama_model_obj.model_dump()
                elif hasattr(ollama_model_obj, 'dict'): # Fallback for older Pydantic
                    model_data_dict = ollama_model_obj.dict()
                else:
                    logger.warning(f"Skipping Ollama model object as it cannot be converted to dict: {ollama_model_obj}")
                    continue
            except Exception as e_dump:
                logger.error(f"Failed to convert ollama.Model object to dictionary: {ollama_model_obj}. Error: {e_dump}", exc_info=True)
                continue

            if not isinstance(model_data_dict, dict): # Should be a dict after conversion
                logger.warning(f"Skipping item as it did not convert to a dictionary: {model_data_dict}")
                continue
            
            try:
                model_info_instance = OllamaModelInfo.from_ollama(model_data_dict)
                if model_info_instance:
                    parsed_models.append(model_info_instance)
            except Exception as parse_exc:
                logger.error(f"Error parsing individual model data derived from {ollama_model_obj}. Dict was: {model_data_dict}. Exception: {parse_exc}", exc_info=True)

        logger.info(f"Successfully parsed {len(parsed_models)} models for API response.")
        
        if not parsed_models and models_from_ollama_lib:
             logger.warning("/api/models is returning an empty list because no models could be successfully parsed, though Ollama returned model data.")
        elif not models_from_ollama_lib:
             logger.info("/api/models received no models from Ollama (Ollama's 'models' list was empty in ListResponse).")
        
        return parsed_models

    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to Ollama at {OLLAMA_BASE_URL} for /api/models: {e}", exc_info=True)
        return JSONResponse(status_code=504, content={"detail": f"Timeout connecting to Ollama: {str(e)}"})
    except httpx.RequestError as e: 
        logger.error(f"httpx.RequestError connecting to Ollama at {OLLAMA_BASE_URL} for /api/models: {e}", exc_info=True)
        return JSONResponse(status_code=503, content={"detail": f"Could not connect to Ollama service: {str(e)}"})
    except ollama.ResponseError as e: 
        logger.error(f"Ollama API responded with an error for /api/models: {getattr(e, 'error', 'Unknown Ollama error')} (Status: {getattr(e, 'status_code', 'N/A')})", exc_info=True)
        status_code_to_return = getattr(e, 'status_code', 500)
        if not isinstance(status_code_to_return, int) or status_code_to_return < 100 or status_code_to_return > 599:
            status_code_to_return = 500
        return JSONResponse(status_code=status_code_to_return, 
                            content={"detail": f"Ollama API error: {getattr(e, 'error', 'Failed to communicate with Ollama')}"})
    except ollama.OllamaError as e: 
        logger.error(f"Ollama library error occurred in /api/models: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": f"Ollama library error: {str(e)}"})
    except Exception as e:
        logger.error(f"An unexpected error occurred in /api/models endpoint: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": f"An unexpected server error occurred while fetching models: {str(e)}"})



@app.post("/api/chat") 
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
    if model_name_from_request and model_name_from_request != "Unknown Model": 
        effective_model_name = model_name_from_request
        logger.info(f"/api/chat will use model from request: {effective_model_name}")
    else:
        logger.warning(f"/api/chat: 'model' field missing or invalid ('{model_name_from_request}'). Using default RAG model: {OLLAMA_MODEL_FOR_RAG}")

    logger.info(f"/api/chat final model: {effective_model_name}, query: '{query[:50]}...'")
    
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            chain_to_use = get_rag_or_simple_chain(use_rag_flag=True, specific_ollama_model_name=effective_model_name)
            chain_input = {"question": query, "chat_history": []} 
            
            logger.info(f"Streaming RAG-enabled response for /api/chat using model {effective_model_name}...")
            async for chunk in chain_to_use.astream(chain_input):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            logger.info(f"Stream finished for /api/chat query: '{query[:50]}...'")
        except HTTPException as e:
             logger.error(f"HTTP Exception during /api/chat stream: {e.detail}")
             yield f"data: {json.dumps({'error': e.detail})}\n\n"
        except Exception as e:
            logger.error(f"Error during /api/chat stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Failed to generate streaming response for /api/chat: {str(e)}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not vectorstore: raise HTTPException(status_code=503, detail="Vector store unavailable.")
    if not embedding_function: raise HTTPException(status_code=503, detail="Embedding function unavailable.")
    logger.info(f"Upload: {file.filename}, Type: {file.content_type}")
    temp_file_path = f"/tmp/{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer: await buffer.write(await file.read())
        logger.info(f"Saved to {temp_file_path}")
        if file.content_type == "application/pdf": loader = PyPDFLoader(temp_file_path)
        else: raise HTTPException(status_code=400, detail=f"Unsupported type: {file.content_type}")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} sections.")
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")
        if not chunks: return {"message": "No content found.", "filename": file.filename}
        vectorstore.add_documents(chunks)
        logger.info(f"Added chunks from {file.filename}.")
        return {"message": "Processed and added.", "filename": file.filename, "chunks_added": len(chunks)}
    except Exception as e:
        logger.error(f"Upload error {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.info(f"Removed {temp_file_path}")
            except OSError as e: logger.error(f"Error removing {temp_file_path}: {e}")

@app.post("/chat_rag", response_model=RAGChatResponse)
async def rag_specific_chat_endpoint(request: RAGChatRequest):
    logger.info(f"RAG chat_rag: session {request.session_id}, use_rag={request.use_rag}, prompt: '{request.prompt[:50]}...'")
    try:
        chain = get_rag_or_simple_chain(use_rag_flag=request.use_rag) 
        response_content = await chain.ainvoke({"question": request.prompt, "chat_history": request.history})
        return RAGChatResponse(session_id=request.session_id, response=response_content)
    except HTTPException as e: raise e
    except Exception as e:
        logger.error(f"RAG chat_rag error {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")

@app.post("/stream_rag")
async def rag_specific_stream_endpoint(request: RAGStreamRequest):
    logger.info(f"RAG stream_rag: session {request.session_id}, use_rag={request.use_rag}, prompt: '{request.prompt[:50]}...'")
    async def event_stream():
        try:
            chain = get_rag_or_simple_chain(use_rag_flag=request.use_rag) 
            async for chunk in chain.astream({"question": request.prompt, "chat_history": request.history}):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            logger.info(f"RAG Stream finished for session {request.session_id}")
        except HTTPException as e: yield f"data: {json.dumps({'error': e.detail})}\n\n"
        except Exception as e:
            logger.error(f"RAG stream_rag error {request.session_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Failed to generate RAG streaming response: {str(e)}'})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    ollama_ok, ollama_error = False, "ChatOllama (for RAG) component failed to initialize."
    if ollama_chat_for_rag:
        try:
            await ollama.AsyncClient(host=OLLAMA_BASE_URL).list()
            ollama_ok = True; ollama_error = None
        except Exception as e: ollama_error = f"Ollama connection or list models failed: {e}"; logger.warning(f"Ollama health check fail: {ollama_error}")
    
    chroma_ok, chroma_error = False, "Chroma client not initialized or connection failed."
    if chroma_client:
        try: chroma_client.heartbeat(); chroma_ok = True; chroma_error = None
        except Exception as e: chroma_error = f"ChromaDB heartbeat error: {e}"; logger.warning(f"ChromaDB health check fail: {chroma_error}")
    
    status_code = 200 if ollama_ok and chroma_ok else 503
    response_detail = {"status": "ok" if ollama_ok and chroma_ok else "error", "ollama": {"status": "connected" if ollama_ok else "disconnected", "details": ollama_error}, "chromadb": {"status": "connected" if chroma_ok else "disconnected", "details": chroma_error}}
    
    if status_code == 503: raise HTTPException(status_code=status_code, detail=response_detail)
    return response_detail

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally...")
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
