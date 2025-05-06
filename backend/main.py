import os
import logging
import aiohttp
import asyncio # Added for health check timeout
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator
import chromadb # Vector Store
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader # Example loader
# from langchain_community.document_loaders import UnstructuredFileLoader # More general loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use updated imports for LangChain 0.2+
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_ollama import ChatOllama # Updated import
from langchain_chroma import Chroma as LangchainChroma # Updated import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.responses import StreamingResponse # Import StreamingResponse


# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI()
# Hello
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAG Components Setup ---
chroma_client = None
embedding_function = None
vectorstore = None
llm = None
retriever = None

# Initialize components in a try block for better error handling during startup
try:
    logger.info(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(allow_reset=True))
    logger.info("Successfully connected to ChromaDB.")

    logger.info(f"Initializing HuggingFace embeddings with model: {EMBEDDING_MODEL_NAME}")
    # Use the updated HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Specify CPU device if needed
        encode_kwargs={'normalize_embeddings': False}
    )
    logger.info("HuggingFace embeddings initialized.")

    COLLECTION_NAME = "rag_documents"
    # Use the updated LangchainChroma import
    vectorstore = LangchainChroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    logger.info(f"LangChain Chroma vector store initialized for collection: {COLLECTION_NAME}")

    logger.info(f"Initializing ChatOllama with model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
    # Use the updated ChatOllama import
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    logger.info("ChatOllama initialized.")

    # --- Retriever Initialization with Debugging ---
    logger.info(f"Checking vectorstore object before retriever initialization: {vectorstore}") # <-- ADDED DEBUG LOG
    if vectorstore:
        logger.info("Vectorstore object confirmed, attempting to create retriever...") # <-- ADDED DEBUG LOG
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )
        logger.info("Retriever initialized successfully.") # <-- ADDED DEBUG LOG
    else:
        logger.warning("Vectorstore object is None or evaluates to False. Cannot initialize retriever.") # <-- Modified Log

except Exception as e:
    logger.critical(f"Failed to initialize RAG components during startup: {e}", exc_info=True)
    # Depending on severity, you might want to prevent the app from fully starting
    # For now, components might remain None, and endpoints should handle this


# --- RAG Chain Setup ---
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

Answer the question based *only* on the provided context and conversation history. If the context doesn't contain the answer, say you don't know.
"""
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history_messages"),
    ("human", "{question}"),
])

SIMPLE_PROMPT_TEMPLATE = """
CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

Answer the question based on the conversation history and your general knowledge.
"""
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", SIMPLE_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history_messages"),
    ("human", "{question}"),
])

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "No history yet."
    return "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])

def format_history_for_lc(chat_history: List[Dict[str, str]]) -> list:
    messages = []
    for msg in chat_history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
    return messages

def create_rag_chain(use_rag=True):
    if not llm:
         raise HTTPException(status_code=503, detail="LLM (Ollama) is not available. Initialization failed.")

    # Check retriever status *inside* the factory function
    current_retriever = retriever # Use the globally initialized retriever
    if current_retriever and use_rag:
        logger.info("Creating RAG chain with retriever.")
        rag_chain = (
            RunnableParallel(
                context=(lambda x: x['question']) | current_retriever,
                question=RunnablePassthrough(),
                chat_history=(lambda x: format_chat_history(x['chat_history'])),
                chat_history_messages=(lambda x: format_history_for_lc(x['chat_history']))
            )
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    else:
        if use_rag and not current_retriever:
             logger.warning("RAG requested but retriever is unavailable. Falling back to simple chain.")
        logger.info("Creating Simple chain (RAG disabled or retriever unavailable).")
        simple_chain = (
             RunnableParallel(
                question=(lambda x: x['question']),
                chat_history=(lambda x: format_chat_history(x['chat_history'])),
                chat_history_messages=(lambda x: format_history_for_lc(x['chat_history']))
             )
            | simple_prompt
            | llm
            | StrOutputParser()
        )
        return simple_chain

# --- API Models ---
class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    history: List[Dict[str, str]] = []
    use_rag: bool = True

class ChatResponse(BaseModel):
    session_id: str
    response: str

class StreamRequest(BaseModel):
    session_id: str
    prompt: str
    history: List[Dict[str, str]] = []
    use_rag: bool = True

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Backend is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Check component availability at the start of the endpoint
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store is not available (initialization failed?).")
    if not embedding_function:
         raise HTTPException(status_code=503, detail="Embedding function is not available (initialization failed?).")

    logger.info(f"Received file upload: {file.filename}, content type: {file.content_type}")
    temp_file_path = f"/tmp/{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"File saved temporarily to {temp_file_path}")

        if file.content_type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
             raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Currently only PDF is supported.")

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document pages/sections from {file.filename}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")

        if not chunks:
            logger.warning("No text chunks generated from the document.")
            return {"message": "No text content found or extracted from the document.", "filename": file.filename}

        logger.info(f"Adding {len(chunks)} chunks to Chroma collection '{COLLECTION_NAME}'...")
        # Use add_documents method
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully added chunks from {file.filename} to vector store.")

        return {"message": "Document processed and added to knowledge base.", "filename": file.filename, "chunks_added": len(chunks)}

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed temporary file: {temp_file_path}")
            except OSError as e:
                 logger.error(f"Error removing temporary file {temp_file_path}: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request for session {request.session_id}, use_rag={request.use_rag}")
    try:
        chain = create_rag_chain(use_rag=request.use_rag) # Can raise 503 if LLM is down
        chain_input = {"question": request.prompt, "chat_history": request.history}
        response_content = await chain.ainvoke(chain_input)
        logger.info(f"Generated response for session {request.session_id}")
        return ChatResponse(session_id=request.session_id, response=response_content)
    except HTTPException as e:
        raise e # Re-raise HTTP exceptions (e.g., 503 from create_rag_chain)
    except Exception as e:
        logger.error(f"Error during chat generation for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/stream")
async def stream_endpoint(request: StreamRequest):
    logger.info(f"Received stream request for session {request.session_id}, use_rag={request.use_rag}")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            chain = create_rag_chain(use_rag=request.use_rag) # Can raise 503
            chain_input = {"question": request.prompt, "chat_history": request.history}
            logger.info("Streaming response...")
            async for chunk in chain.astream(chain_input):
                yield chunk
            logger.info(f"Stream finished for session {request.session_id}")
        except HTTPException as e:
             logger.error(f"HTTP Exception during stream setup or generation: {e.detail}")
             yield f"Error: {e.detail}"
        except Exception as e:
            logger.error(f"Error during stream generation for session {request.session_id}: {e}", exc_info=True)
            yield f"Error: Failed to generate streaming response. {str(e)}"

    return StreamingResponse(event_stream(), media_type="text/plain")


# --- Health Check ---
@app.get("/health")
async def health_check():
    # Check Ollama connection
    ollama_ok = False
    ollama_error = "Unknown connection error"
    if llm:
        try:
            # Simple check: list local models via the Ollama client library
            # This requires the ollama library to be installed, which it should be
            await ollama.AsyncClient(host=OLLAMA_BASE_URL).list()
            ollama_ok = True
            ollama_error = None
        except Exception as e:
            ollama_error = f"Error connecting to Ollama at {OLLAMA_BASE_URL}: {e}"
            logger.warning(f"Health check failed for Ollama: {ollama_error}")
    else:
         ollama_error = "ChatOllama component failed to initialize."

    # Check ChromaDB connection
    chroma_ok = False
    chroma_error = "Chroma client not initialized."
    if chroma_client:
        try:
            chroma_client.heartbeat()
            chroma_ok = True
            chroma_error = None
        except Exception as e:
            chroma_error = f"Error connecting to ChromaDB: {e}"
            logger.warning(f"Health check failed for ChromaDB: {chroma_error}")
            chroma_ok = False

    status_code = 200 if ollama_ok and chroma_ok else 503
    response_detail = {
        "status": "ok" if ollama_ok and chroma_ok else "error",
        "ollama": {"status": "connected" if ollama_ok else "disconnected", "details": ollama_error},
        "chromadb": {"status": "connected" if chroma_ok else "disconnected", "details": chroma_error}
    }

    if status_code == 503:
         raise HTTPException(status_code=status_code, detail=response_detail)
    else:
         return response_detail


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally...")
    run_host = os.getenv("HOST", "0.0.0.0")
    run_port = int(os.getenv("PORT", "8000"))
    # Note: When running locally, ensure OLLAMA_BASE_URL and CHROMA_HOST/PORT point correctly
    # e.g., OLLAMA_BASE_URL=http://localhost:11434, CHROMA_HOST=localhost, CHROMA_PORT=8001
    uvicorn.run("main:app", host=run_host, port=run_port, reload=True) # Use reload for local dev
