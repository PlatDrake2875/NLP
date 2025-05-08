# backend/rag_components.py
import logging
from typing import List, Dict, Optional, Any

import chromadb
from fastapi import HTTPException, Depends # Import Depends
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

# --- Globals for RAG Components ---
# Define globals at the module level, initially None
chroma_client: Optional[chromadb.HttpClient] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[LangchainChroma] = None
retriever: Optional[Runnable] = None
ollama_chat_for_rag: Optional[ChatOllama] = None
rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None

# --- Control Flag ---
RAG_ENABLED = False # Set to True to re-enable RAG functionality


# --- Dependency Functions ---
# Define these *after* the global variables they will access

def get_chroma_client() -> chromadb.HttpClient:
    """Dependency function to get the initialized ChromaDB client."""
    if chroma_client is None:
        logger.error("get_chroma_client dependency called but ChromaDB client is not available.")
        raise HTTPException(status_code=503, detail="ChromaDB client is not available. Setup might have failed.")
    return chroma_client

def get_vectorstore() -> LangchainChroma:
    """Dependency function to get the initialized vector store."""
    if vectorstore is None:
        logger.error("get_vectorstore dependency called but vector store is not available.")
        raise HTTPException(status_code=503, detail="Vector store is not available. Setup might have failed.")
    return vectorstore

def get_embedding_function() -> HuggingFaceEmbeddings:
    """Dependency function to get the initialized embedding function."""
    if embedding_function is None:
        logger.error("get_embedding_function dependency called but embedding function is not available.")
        raise HTTPException(status_code=503, detail="Embedding function is not available. Setup might have failed.")
    return embedding_function

def get_retriever() -> Runnable:
    """Dependency function to get the initialized retriever."""
    if retriever is None:
        logger.error("get_retriever dependency called but retriever is not available.")
        raise HTTPException(status_code=503, detail="Retriever is not available. Setup might have failed.")
    return retriever

def get_ollama_chat_for_rag() -> ChatOllama:
    """Dependency function to get the initialized ChatOllama instance for RAG."""
    if ollama_chat_for_rag is None:
        logger.error("get_ollama_chat_for_rag dependency called but ChatOllama is not available.")
        raise HTTPException(status_code=503, detail="Core LLM for RAG is not available. Setup might have failed.")
    return ollama_chat_for_rag

def get_rag_prompt_template() -> ChatPromptTemplate:
    """Dependency function to get the initialized RAG prompt template."""
    if rag_prompt_template_obj is None:
        logger.error("get_rag_prompt_template dependency called but RAG prompt template is not available.")
        raise HTTPException(status_code=503, detail="RAG prompt template is not available.")
    return rag_prompt_template_obj

def get_simple_prompt_template() -> ChatPromptTemplate:
    """Dependency function to get the initialized simple prompt template."""
    if simple_prompt_template_obj is None:
        logger.error("get_simple_prompt_template dependency called but simple prompt template is not available.")
        raise HTTPException(status_code=503, detail="Simple prompt template is not available.")
    return simple_prompt_template_obj

# --- Optional Dependency Functions for Health Check ---

def get_optional_chroma_client() -> Optional[chromadb.HttpClient]:
    """Optional dependency for health check, returns None if unavailable."""
    return chroma_client # Directly return the global variable

def get_optional_ollama_chat_for_rag() -> Optional[ChatOllama]:
    """Optional dependency for health check, returns None if unavailable."""
    return ollama_chat_for_rag # Directly return the global variable


# --- Component Setup Function ---
# This function populates the global variables defined above
def setup_rag_components():
    """Initializes and sets up all RAG components. Called once at startup."""
    global chroma_client, embedding_function, vectorstore, retriever, ollama_chat_for_rag
    global rag_prompt_template_obj, simple_prompt_template_obj

    # Prevent re-initialization if components already exist
    if vectorstore and ollama_chat_for_rag:
        logger.warning("RAG components already initialized. Skipping setup.")
        return

    logger.info("Setting up RAG components...")
    # Use temporary variables during setup to avoid partial global state
    temp_chroma_client = None
    temp_embedding_function = None
    temp_vectorstore = None
    temp_retriever = None
    temp_ollama_chat = None
    temp_rag_prompt = None
    temp_simple_prompt = None

    try:
        logger.info(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
        try:
            temp_chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
            temp_chroma_client.heartbeat()
            logger.info("Successfully connected to ChromaDB and got heartbeat.")
        except Exception as e_chroma_init:
            logger.critical(f"Failed to initialize ChromaDB client: {e_chroma_init}", exc_info=True)
            temp_chroma_client = None

        if temp_chroma_client:
            logger.info(f"Initializing HuggingFace embeddings with model: {EMBEDDING_MODEL_NAME}")
            try:
                temp_embedding_function = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': False}
                )
                logger.info("HuggingFace embeddings initialized.")
            except Exception as e_embed:
                 logger.critical(f"Failed to initialize HuggingFace embeddings: {e_embed}", exc_info=True)
                 temp_embedding_function = None

            if temp_embedding_function:
                try:
                    temp_vectorstore = LangchainChroma(
                        client=temp_chroma_client,
                        collection_name=COLLECTION_NAME,
                        embedding_function=temp_embedding_function
                    )
                    logger.info(f"LangChain Chroma vector store initialized for collection: {COLLECTION_NAME}")

                    temp_retriever = temp_vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={'k': 3}
                    )
                    logger.info(f"Retriever initialized successfully (k=3).")
                except Exception as e_vs:
                    logger.critical(f"Failed to initialize LangChain Chroma or Retriever: {e_vs}", exc_info=True)
                    temp_vectorstore = None
                    temp_retriever = None
            else:
                 logger.warning("Embedding function failed. Vectorstore/Retriever unavailable.")
                 temp_vectorstore = None
                 temp_retriever = None
        else:
            logger.warning("ChromaDB client failed. Vectorstore/Retriever unavailable.")
            temp_embedding_function = None
            temp_vectorstore = None
            temp_retriever = None

        logger.info(f"Initializing ChatOllama (for RAG) with model: {OLLAMA_MODEL_FOR_RAG} at {OLLAMA_BASE_URL}")
        try:
            temp_ollama_chat = ChatOllama(
                model=OLLAMA_MODEL_FOR_RAG,
                base_url=OLLAMA_BASE_URL,
                temperature=0
            )
            logger.info("ChatOllama (for RAG) initialized.")
        except Exception as e_ollama:
            logger.critical(f"Failed to initialize ChatOllama (for RAG): {e_ollama}", exc_info=True)
            temp_ollama_chat = None

    except Exception as e:
        logger.critical(f"Critical error during RAG components startup: {e}", exc_info=True)
        # Ensure all temps are None if a critical error occurred
        temp_chroma_client = None
        temp_embedding_function = None
        temp_vectorstore = None
        temp_retriever = None
        temp_ollama_chat = None

    # Initialize prompt templates
    try:
        temp_rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=RAG_PROMPT_TEMPLATE_STR),
            MessagesPlaceholder(variable_name="chat_history_messages"),
            HumanMessage(content="{question}")
        ])
        temp_simple_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SIMPLE_PROMPT_TEMPLATE_STR),
            MessagesPlaceholder(variable_name="chat_history_messages"),
            HumanMessage(content="{question}")
        ])
        logger.info("Prompt templates initialized.")
    except Exception as e_prompt:
        logger.critical(f"Failed to initialize prompt templates: {e_prompt}", exc_info=True)
        temp_rag_prompt = None
        temp_simple_prompt = None

    # --- Assign to Globals *after* all setup attempts ---
    chroma_client = temp_chroma_client
    embedding_function = temp_embedding_function
    vectorstore = temp_vectorstore
    retriever = temp_retriever
    ollama_chat_for_rag = temp_ollama_chat
    rag_prompt_template_obj = temp_rag_prompt
    simple_prompt_template_obj = temp_simple_prompt

    logger.info("RAG components setup finished.")


# --- Helper Functions ---
# (Keep format_chat_history, format_history_for_lc, combine_retrieved_documents as they are)
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


# --- Chain Construction Functions (Now use dependency functions internally) ---

def _select_llm(specific_ollama_model_name: Optional[str] = None) -> ChatOllama:
    """Selects the appropriate ChatOllama instance. Uses the RAG default if specific one fails."""
    # Use dependency function to ensure default is available
    rag_llm = get_ollama_chat_for_rag()
    active_llm = rag_llm

    if specific_ollama_model_name and specific_ollama_model_name != OLLAMA_MODEL_FOR_RAG and specific_ollama_model_name != "Unknown Model":
        logger.info(f"LLM Selection: Attempting to use specific model '{specific_ollama_model_name}'")
        try:
            active_llm = ChatOllama(model=specific_ollama_model_name, base_url=OLLAMA_BASE_URL, temperature=0)
        except Exception as e_llm_init:
            logger.error(f"Failed to initialize specific LLM '{specific_ollama_model_name}': {e_llm_init}. Falling back to default.")
            active_llm = rag_llm
    elif specific_ollama_model_name == "Unknown Model":
        logger.warning(f"LLM Selection: Received 'Unknown Model'. Using default RAG model: {OLLAMA_MODEL_FOR_RAG}")
        active_llm = rag_llm
    else:
        logger.info(f"LLM Selection: Using default RAG model: {OLLAMA_MODEL_FOR_RAG}")
        active_llm = rag_llm

    return active_llm

def _build_simple_chain(llm: ChatOllama) -> Runnable:
    """Builds the simple conversational chain (no RAG)."""
    # Get prompt via dependency function
    simple_prompt = get_simple_prompt_template()

    prepare_simple_input = RunnablePassthrough.assign(
        question=lambda x: x.get('question', 'MISSING QUESTION IN SIMPLE PREP'),
        chat_history_messages=lambda x: format_history_for_lc(x.get('chat_history', []))
    )

    def log_simple_input(prepared_input: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[log_simple_input] Input for Simple Prompt: {prepared_input}")
        return prepared_input

    simple_chain = (
        prepare_simple_input |
        RunnableLambda(log_simple_input) |
        simple_prompt |
        llm |
        StrOutputParser()
    )
    logger.info("Built simple_chain.")
    return simple_chain

def _build_rag_chain_with_fallback(llm: ChatOllama) -> Runnable:
    """Builds the RAG chain with fallback to the simple chain if no documents are found."""
    # Get components via dependency functions
    retriever_instance = get_retriever()
    rag_prompt = get_rag_prompt_template()
    simple_prompt = get_simple_prompt_template()

    retrieval_chain_part = (
        RunnablePassthrough.assign(
            docs=RunnableLambda(lambda x: logger.info(f"Retriever Input: '{x['question']}'") or x['question']) | retriever_instance
        ) |
        RunnablePassthrough.assign(
             context=lambda x: combine_retrieved_documents(x["docs"])
        )
    )

    def prepare_and_log_rag_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        question = input_dict.get('question', 'MISSING QUESTION')
        history = input_dict.get('chat_history', [])
        context = input_dict.get('context', 'MISSING CONTEXT')
        formatted_input = {
             "question": question,
             "chat_history_messages": format_history_for_lc(history),
             "context": context
        }
        logger.info(f"[prepare_and_log_rag_input] Input for RAG Prompt: {formatted_input}")
        return formatted_input

    rag_chain_main = (
        RunnableLambda(prepare_and_log_rag_input) |
        rag_prompt |
        llm |
        StrOutputParser()
    )

    prepare_simple_input_for_fallback = RunnablePassthrough.assign(
        question=lambda x: x.get('question', 'MISSING QUESTION IN FALLBACK PREP'),
        chat_history_messages=lambda x: format_history_for_lc(x.get('chat_history', []))
    )
    def log_simple_input_fallback(prepared_input: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[log_simple_input_fallback] Input for Simple Prompt (via RAG fallback): {prepared_input}")
        return prepared_input

    simple_fallback_chain = (
        prepare_simple_input_for_fallback |
        RunnableLambda(log_simple_input_fallback) |
        simple_prompt |
        llm |
        StrOutputParser()
    )

    def route_based_on_docs(info: Dict[str, Any]) -> Runnable:
        if info.get("docs"):
            logger.info(f"Routing to RAG chain because {len(info['docs'])} documents were retrieved.")
            return RunnableLambda(lambda x: x) | rag_chain_main
        else:
            logger.info("Routing to Simple chain (fallback) because no documents were retrieved.")
            return RunnableLambda(lambda x: x) | simple_fallback_chain

    full_rag_chain = retrieval_chain_part | RunnableLambda(route_based_on_docs)

    logger.info("Built final chain with RAG fallback logic.")
    return full_rag_chain


# --- Main Function to Get Chain ---
# This function now relies on the dependency functions being available when called
def get_rag_or_simple_chain(use_rag_flag: bool, specific_ollama_model_name: Optional[str] = None) -> Runnable:
    """
    Selects and returns the appropriate chain (RAG with fallback or Simple)
    based on the RAG_ENABLED flag and use_rag_flag.
    """
    # 1. Select the LLM instance (this uses get_ollama_chat_for_rag internally)
    active_llm = _select_llm(specific_ollama_model_name)

    # 2. Decide whether to use RAG or Simple based on global flag and request flag
    if RAG_ENABLED and use_rag_flag:
        logger.info("Attempting to build and return RAG chain with fallback.")
        # Check globals directly for availability before building
        if chroma_client and retriever and rag_prompt_template_obj and simple_prompt_template_obj:
             return _build_rag_chain_with_fallback(active_llm)
        else:
            logger.warning("RAG enabled, but core components missing. Falling back to simple chain.")
            # Need the simple prompt template even for fallback
            return _build_simple_chain(active_llm)
    else:
        if not RAG_ENABLED:
             logger.info("RAG_ENABLED is False. Building simple chain.")
        else:
             logger.info("use_rag_flag is False. Building simple chain.")
        return _build_simple_chain(active_llm)

