# backend/rag_components.py
import logging
from typing import List, Dict, Optional, Any

import chromadb
from fastapi import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma as LangchainChroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, Runnable, 
    RunnableLambda, RunnableBranch, RunnableConfig 
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document 

# Import configurations from config.py using absolute path from root
try:
    from config import (
        CHROMA_HOST, CHROMA_PORT, EMBEDDING_MODEL_NAME, COLLECTION_NAME,
        OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG,
        RAG_PROMPT_TEMPLATE_STR, SIMPLE_PROMPT_TEMPLATE_STR, logger
    )
except ImportError:
    logger.critical("Failed to import from config. Ensure config.py exists and PYTHONPATH is correct.")
    raise


# --- Globals for RAG Components (initialized by setup_rag_components) ---
chroma_client: Optional[chromadb.HttpClient] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[LangchainChroma] = None
retriever: Optional[Runnable] = None 
ollama_chat_for_rag: Optional[ChatOllama] = None
rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None # Still needed if RAG is re-enabled


def setup_rag_components():
    """Initializes and sets up all RAG components."""
    global chroma_client, embedding_function, vectorstore, retriever, ollama_chat_for_rag
    global rag_prompt_template_obj, simple_prompt_template_obj

    logger.info("Setting up RAG components...")
    # ... (rest of the setup code remains the same) ...
    try:
        logger.info(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
            logger.info("Attempting to get ChromaDB heartbeat...")
            chroma_client.heartbeat()  
            logger.info("Successfully connected to ChromaDB and got heartbeat.")
        except Exception as e_chroma_init:
            logger.critical(f"Failed to initialize ChromaDB client: {e_chroma_init}", exc_info=True) 
            chroma_client = None 

        if chroma_client:
            logger.info(f"Initializing HuggingFace embeddings with model: {EMBEDDING_MODEL_NAME}")
            embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME, 
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': False} 
            )
            logger.info("HuggingFace embeddings initialized.")
            
            vectorstore = LangchainChroma(
                client=chroma_client, 
                collection_name=COLLECTION_NAME, 
                embedding_function=embedding_function
            )
            logger.info(f"LangChain Chroma vector store initialized for collection: {COLLECTION_NAME}")
            
            if vectorstore:
                retriever = vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={'k': 3} 
                )
                logger.info(f"Retriever initialized successfully (k=3).")
            else:
                logger.warning("Vectorstore object is None. Cannot initialize retriever.")
        else:
            logger.warning("ChromaDB client failed. Vectorstore and Retriever will be unavailable.")

        logger.info(f"Initializing ChatOllama (for RAG) with model: {OLLAMA_MODEL_FOR_RAG} at {OLLAMA_BASE_URL}")
        ollama_chat_for_rag = ChatOllama(
            model=OLLAMA_MODEL_FOR_RAG, 
            base_url=OLLAMA_BASE_URL, 
            temperature=0 
        )
        logger.info("ChatOllama (for RAG) initialized.")

    except Exception as e:
        logger.critical(f"Critical error during RAG components startup: {e}", exc_info=True)
    
    rag_prompt_template_obj = ChatPromptTemplate.from_messages([
        SystemMessage(content=RAG_PROMPT_TEMPLATE_STR), 
        MessagesPlaceholder(variable_name="chat_history_messages"), 
        HumanMessage(content="{question}")
    ])
    simple_prompt_template_obj = ChatPromptTemplate.from_messages([
        SystemMessage(content=SIMPLE_PROMPT_TEMPLATE_STR), 
        MessagesPlaceholder(variable_name="chat_history_messages"), 
        HumanMessage(content="{question}") 
    ])
    logger.info("Prompt templates initialized.")
    logger.info("RAG components setup finished.")


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history: 
        return "No previous conversation." 
    return "\n".join([f"{msg['role'].upper()}: {msg.get('content', '')}" for msg in chat_history])

def format_history_for_lc(chat_history: List[Dict[str, str]]) -> list:
    messages = []
    for msg in chat_history:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user": 
            messages.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ai": 
            messages.append(AIMessage(content=content))
    return messages

def combine_retrieved_documents(docs: List[Document]) -> str:
    if not docs:
        logger.info("[combine_retrieved_documents] No documents to combine.")
        return "No relevant context found." 
    
    combined = "\n\n---\n\n".join([doc.page_content for doc in docs])
    logger.info(f"[combine_retrieved_documents] Combined {len(docs)} documents. Length: {len(combined)} chars.")
    logger.debug(f"Combined context: {combined[:500]}...") 
    return combined


def get_rag_or_simple_chain(use_rag_flag: bool, specific_ollama_model_name: Optional[str] = None) -> Runnable:
    """
    Constructs a chain. If RAG is deactivated below, it always returns a simple conversational chain.
    Otherwise, it attempts RAG with fallback to simple chain if no docs are found.
    """
    global ollama_chat_for_rag, rag_prompt_template_obj, simple_prompt_template_obj, retriever

    # --- LLM Selection ---
    # ... (LLM selection logic remains the same) ...
    active_llm = ollama_chat_for_rag
    if specific_ollama_model_name and specific_ollama_model_name != OLLAMA_MODEL_FOR_RAG and specific_ollama_model_name != "Unknown Model":
        logger.info(f"Chain will use specific Ollama model: {specific_ollama_model_name}")
        try:
            active_llm = ChatOllama(model=specific_ollama_model_name, base_url=OLLAMA_BASE_URL, temperature=0)
        except Exception as e_llm_init:
            logger.error(f"Failed to initialize specific LLM '{specific_ollama_model_name}': {e_llm_init}. Falling back to default.")
            if not ollama_chat_for_rag:
                logger.critical("Default RAG LLM is not available after specific LLM failed.")
                raise HTTPException(status_code=503, detail="Core LLM components are not available.")
            active_llm = ollama_chat_for_rag 
    elif specific_ollama_model_name == "Unknown Model":
        logger.warning(f"Received 'Unknown Model'. Chain will use default RAG model: {OLLAMA_MODEL_FOR_RAG}")
        if not ollama_chat_for_rag:
            logger.critical("Default RAG LLM is not available for 'Unknown Model' case.")
            raise HTTPException(status_code=503, detail="Default RAG LLM is not available.")
        active_llm = ollama_chat_for_rag
    elif not ollama_chat_for_rag: 
        logger.error("Default RAG LLM (ollama_chat_for_rag) is not available.")
        raise HTTPException(status_code=503, detail="Core LLM for RAG is not available.")
    else:
        logger.info(f"Chain will use default RAG model: {OLLAMA_MODEL_FOR_RAG}")

    # --- Define the part of the chain that calls the LLM ---
    llm_chain_part = active_llm | StrOutputParser()
    
    # --- Simple Chain Definition (No RAG) ---
    if not simple_prompt_template_obj:
        logger.error("Simple prompt template is not initialized.")
        raise HTTPException(status_code=503, detail="Simple prompt template is not available.")

    # Use RunnablePassthrough.assign to explicitly create the input dict for the prompt
    # The input to this whole chain segment should be {"question": ..., "chat_history": ...}
    prepare_simple_input = RunnablePassthrough.assign(
        # Ensure the 'question' key exists and pass it through
        question=lambda x: x.get('question', 'MISSING QUESTION IN SIMPLE PREP'), 
        # Format the history and assign it to the key expected by MessagesPlaceholder
        chat_history_messages=lambda x: format_history_for_lc(x.get('chat_history', []))
    )

    # Log the prepared input right before it hits the prompt template
    def log_simple_input(prepared_input: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[log_simple_input] Input for Simple Prompt: {prepared_input}")
        return prepared_input

    simple_chain = (
        prepare_simple_input |
        RunnableLambda(log_simple_input) | # Log the dictionary
        simple_prompt_template_obj | 
        llm_chain_part 
    )

    # --- RAG Deactivated: Always use the simple chain for now ---
    logger.info("RAG functionality is currently DEACTIVATED. Forcing use of simple_chain.")
    return simple_chain
    # --- End of RAG Deactivated Section ---

    # --- RAG Chain Definition (Code below is bypassed) ---
    # ... (The RAG chain definition code remains here but is not executed) ...

