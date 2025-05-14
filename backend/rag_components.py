# backend/rag_components.py
import logging
from typing import List, Dict, Optional, Any

import chromadb
from fastapi import HTTPException, Depends # Ensure Depends is imported if used directly here, though typically in routers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma as LangchainChroma
from langchain_core.prompts import ChatPromptTemplate # Removed MessagesPlaceholder as it wasn't used directly
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, Runnable,
    RunnableLambda # Removed RunnableBranch, RunnableConfig as they are not used in the provided active code
)
from langchain_core.messages import AIMessage, HumanMessage # Removed SystemMessage as it wasn't used
from langchain_core.documents import Document

# Import configurations from config.py
try:
    from config import (
        CHROMA_HOST, CHROMA_PORT, EMBEDDING_MODEL_NAME, COLLECTION_NAME,
        OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG,
        RAG_ENABLED, # Imported from config
        RAG_PROMPT_TEMPLATE_STR, SIMPLE_PROMPT_TEMPLATE_STR,
        RAG_CONTEXT_PREFIX_TEMPLATE_STR,
        logger,
        # Attempt to import the new config for automation LLM
        OLLAMA_MODEL_FOR_AUTOMATION
    )
except ImportError:
    # Fallback if OLLAMA_MODEL_FOR_AUTOMATION is not in config.py
    logger.warning("OLLAMA_MODEL_FOR_AUTOMATION not found in config.py, defaulting to OLLAMA_MODEL_FOR_RAG or 'llama3'")
    OLLAMA_MODEL_FOR_AUTOMATION = OLLAMA_MODEL_FOR_RAG if 'OLLAMA_MODEL_FOR_RAG' in globals() else "llama3"


# --- Globals for RAG Components ---
chroma_client: Optional[chromadb.HttpClient] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[LangchainChroma] = None
retriever: Optional[Any] = None
ollama_chat_for_rag: Optional[ChatOllama] = None
ollama_chat_for_automation: Optional[ChatOllama] = None # New global for automation LLM
rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None
rag_context_prefix_prompt_template_obj: Optional[ChatPromptTemplate] = None

# --- Helper Functions (format_history_for_lc, format_docs remain the same) ---
def format_history_for_lc(history: List[Dict[str, str]]) -> List:
    """Converts custom history format to LangChain Message objects."""
    lc_messages = []
    for msg in history:
        role = msg.get("sender", "user")
        content = msg.get("text", "")
        if not isinstance(content, str): content = str(content)

        if role.lower() == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role.lower() == "bot":
            lc_messages.append(AIMessage(content=content))
        else:
            logger.warning(f"Unknown role '{role}' in history, treating as HumanMessage.")
            lc_messages.append(HumanMessage(content=content))
    return lc_messages

def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)

# --- Setup Function ---
def setup_rag_components():
    """Initializes all RAG components and stores them in globals."""
    global chroma_client, embedding_function, vectorstore, retriever, \
           ollama_chat_for_rag, ollama_chat_for_automation, \
           rag_prompt_template_obj, simple_prompt_template_obj, \
           rag_context_prefix_prompt_template_obj

    logger.info("Setting up RAG components...")

    # 1. Embedding Function (same as before)
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info(f"HuggingFace embeddings initialized with model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}", exc_info=True)
        embedding_function = None

    # 2. ChromaDB Client and Vectorstore (same as before)
    if embedding_function:
        try:
            logger.info(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
            temp_chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
            temp_chroma_client.heartbeat()
            logger.info("Successfully connected to ChromaDB and got heartbeat.")
            chroma_client = temp_chroma_client

            vectorstore = LangchainChroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_function,
            )
            logger.info(f"LangChain Chroma vector store initialized for collection: {COLLECTION_NAME}")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            logger.info("Retriever initialized successfully (k=3).")

        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB client/vectorstore/retriever: {e}", exc_info=True)
            chroma_client = vectorstore = retriever = None
    else:
        logger.warning("Skipping ChromaDB/Vectorstore/Retriever setup because embedding function failed.")
        chroma_client = vectorstore = retriever = None

    # 3. ChatOllama LLM (for RAG) (same as before)
    if OLLAMA_BASE_URL:
        try:
            logger.info(f"Initializing ChatOllama (for RAG) with model: {OLLAMA_MODEL_FOR_RAG} at {OLLAMA_BASE_URL}")
            ollama_chat_for_rag = ChatOllama(
                model=OLLAMA_MODEL_FOR_RAG,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1 # Example temperature
            )
            logger.info("ChatOllama (for RAG) initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama for RAG: {e}", exc_info=True)
            ollama_chat_for_rag = None
    else:
        logger.warning("OLLAMA_BASE_URL not set. ChatOllama for RAG cannot be initialized.")
        ollama_chat_for_rag = None

    # 4. ChatOllama LLM (for Automation Tasks) - NEW
    if OLLAMA_BASE_URL:
        try:
            logger.info(f"Initializing ChatOllama (for Automation) with model: {OLLAMA_MODEL_FOR_AUTOMATION} at {OLLAMA_BASE_URL}")
            ollama_chat_for_automation = ChatOllama(
                model=OLLAMA_MODEL_FOR_AUTOMATION,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2 # Example: Potentially different temperature for automation
                # You can add other parameters like num_predict, top_k, top_p if needed
            )
            logger.info(f"ChatOllama (for Automation) initialized with model {OLLAMA_MODEL_FOR_AUTOMATION}.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama for Automation: {e}", exc_info=True)
            ollama_chat_for_automation = None
    else:
        logger.warning("OLLAMA_BASE_URL not set. ChatOllama for Automation cannot be initialized.")
        ollama_chat_for_automation = None

    # 5. Prompt Templates (same as before)
    logger.info("Initializing prompt templates.")
    try:
        rag_prompt_template_obj = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)
        logger.info("RAG prompt template initialized.")
    except Exception as e:
        logger.error(f"Failed to create RAG prompt template: {e}", exc_info=True)
        rag_prompt_template_obj = None
    # ... (simple_prompt_template_obj and rag_context_prefix_prompt_template_obj initialization) ...
    try:
        simple_prompt_template_obj = ChatPromptTemplate.from_template(SIMPLE_PROMPT_TEMPLATE_STR)
        logger.info("Simple prompt template initialized.")
    except Exception as e:
        logger.error(f"Failed to create Simple prompt template: {e}", exc_info=True)
        simple_prompt_template_obj = None

    try:
        rag_context_prefix_prompt_template_obj = ChatPromptTemplate.from_template(RAG_CONTEXT_PREFIX_TEMPLATE_STR)
        logger.info("RAG context prefix prompt template initialized.")
    except Exception as e:
        logger.error(f"Failed to create RAG context prefix prompt template: {e}", exc_info=True)
        rag_context_prefix_prompt_template_obj = None

    logger.info("RAG components setup finished.")


# --- Dependency Functions (getters for initialized components) ---
# (get_chroma_client, get_vectorstore, get_embedding_function, get_retriever, get_ollama_chat_for_rag remain the same)
def get_chroma_client() -> chromadb.HttpClient:
    if chroma_client is None:
        logger.error("Global ChromaDB client is None. Setup might have failed or is incomplete.")
        raise HTTPException(status_code=503, detail="ChromaDB client is not available.")
    return chroma_client

def get_vectorstore() -> LangchainChroma:
    if vectorstore is None:
        logger.error("Global Vectorstore is None. Setup might have failed or is incomplete.")
        raise HTTPException(status_code=503, detail="Vector store is not available.")
    return vectorstore

def get_embedding_function() -> HuggingFaceEmbeddings:
    if embedding_function is None:
        logger.error("Global Embedding function is None. Setup might have failed.")
        raise HTTPException(status_code=503, detail="Embedding function is not available.")
    return embedding_function

def get_retriever() -> Any:
    if retriever is None:
        logger.error("Global Retriever is None. Setup might have failed or is incomplete.")
        raise HTTPException(status_code=503, detail="Retriever is not available.")
    return retriever

def get_ollama_chat_for_rag() -> ChatOllama:
    if ollama_chat_for_rag is None:
        logger.error("Global ChatOllama (for RAG) is None. Setup failed or OLLAMA_BASE_URL is not set.")
        raise HTTPException(status_code=503, detail="RAG chat model is not available.")
    return ollama_chat_for_rag

# --- Function that was missing ---
def get_llm_for_automation() -> ChatOllama:
    """
    Returns the initialized ChatOllama instance intended for automation tasks.
    This function is the one your automate_router.py is trying to import.
    """
    if ollama_chat_for_automation is None:
        logger.error("Global ChatOllama (for Automation) is None. Setup failed, OLLAMA_BASE_URL is not set, or OLLAMA_MODEL_FOR_AUTOMATION is problematic.")
        raise HTTPException(status_code=503, detail="Automation LLM is not available.")
    return ollama_chat_for_automation

# --- Optional Dependencies (for health check) ---
def get_optional_chroma_client() -> Optional[chromadb.HttpClient]:
    return chroma_client

def get_optional_ollama_chat_for_rag() -> Optional[ChatOllama]:
    return ollama_chat_for_rag

def get_optional_llm_for_automation() -> Optional[ChatOllama]: # New optional getter
    return ollama_chat_for_automation


# --- Function to Generate RAG Context Prefix (same as before) ---
async def get_rag_context_prefix(query: str) -> Optional[str]:
    if not RAG_ENABLED:
        logger.info("RAG is disabled globally. Skipping context prefix generation.")
        return None
    if not retriever or not rag_context_prefix_prompt_template_obj:
        logger.warning("Retriever or RAG context prefix prompt template not initialized. Cannot generate prefix.")
        return None

    try:
        logger.info(f"Invoking retriever for RAG context prefix with query: '{query[:50]}...'")
        retrieved_docs = await retriever.ainvoke(query)

        if not retrieved_docs:
            logger.info("No relevant documents found by retriever for RAG context prefix.")
            return None

        formatted_context = format_docs(retrieved_docs)
        logger.debug(f"Formatted context for prefix:\n{formatted_context}")

        # Ensure the template object is not None before calling format
        if rag_context_prefix_prompt_template_obj:
            formatted_prompt = rag_context_prefix_prompt_template_obj.format(
                context=formatted_context,
                question=query
            )
            logger.info(f"Generated RAG context prefix string (length: {len(formatted_prompt)}).")
            return formatted_prompt
        else:
            logger.error("rag_context_prefix_prompt_template_obj is None. Cannot format prompt.")
            return None

    except Exception as e:
        logger.error(f"Error generating RAG context prefix: {e}", exc_info=True)
        return None

