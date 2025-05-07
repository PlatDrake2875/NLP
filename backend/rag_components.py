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
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None

# --- Control Flag ---
# Set this to True to use the RAG chain (_build_rag_chain_with_fallback)
# Set this to False to always use the Simple chain (_build_simple_chain)
RAG_ENABLED = True # Set to True to re-enable RAG functionality

# --- Component Setup ---
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

# --- Helper Functions ---
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

def _select_llm(specific_ollama_model_name: Optional[str] = None) -> ChatOllama:
    """Selects the appropriate ChatOllama instance."""
    global ollama_chat_for_rag
    
    active_llm = ollama_chat_for_rag # Start with default

    if specific_ollama_model_name and specific_ollama_model_name != OLLAMA_MODEL_FOR_RAG and specific_ollama_model_name != "Unknown Model":
        logger.info(f"LLM Selection: Using specific model '{specific_ollama_model_name}'")
        try:
            active_llm = ChatOllama(model=specific_ollama_model_name, base_url=OLLAMA_BASE_URL, temperature=0)
        except Exception as e_llm_init:
            logger.error(f"Failed to initialize specific LLM '{specific_ollama_model_name}': {e_llm_init}. Falling back to default.")
            if not ollama_chat_for_rag: # Check if default exists before falling back
                logger.critical("Default RAG LLM is not available after specific LLM failed.")
                raise HTTPException(status_code=503, detail="Core LLM components are not available.")
            active_llm = ollama_chat_for_rag 
    elif specific_ollama_model_name == "Unknown Model":
        logger.warning(f"LLM Selection: Received 'Unknown Model'. Using default RAG model: {OLLAMA_MODEL_FOR_RAG}")
        if not ollama_chat_for_rag:
            logger.critical("Default RAG LLM is not available for 'Unknown Model' case.")
            raise HTTPException(status_code=503, detail="Default RAG LLM is not available.")
        active_llm = ollama_chat_for_rag
    elif not ollama_chat_for_rag: 
        logger.error("LLM Selection: Default RAG LLM (ollama_chat_for_rag) is not available.")
        raise HTTPException(status_code=503, detail="Core LLM for RAG is not available.")
    else:
        logger.info(f"LLM Selection: Using default RAG model: {OLLAMA_MODEL_FOR_RAG}")

    return active_llm

# --- Chain Construction Functions ---

def _build_simple_chain(llm: ChatOllama) -> Runnable:
    """Builds the simple conversational chain (no RAG)."""
    global simple_prompt_template_obj
    
    if not simple_prompt_template_obj:
        logger.error("Simple prompt template is not initialized for _build_simple_chain.")
        raise HTTPException(status_code=503, detail="Simple prompt template is not available.")

    # Prepare input for the simple prompt template
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
        simple_prompt_template_obj | 
        llm | # Use the passed LLM instance
        StrOutputParser()
    )
    logger.info("Built simple_chain.")
    return simple_chain

def _build_rag_chain_with_fallback(llm: ChatOllama) -> Runnable:
    """Builds the RAG chain with fallback to the simple chain if no documents are found."""
    global retriever, rag_prompt_template_obj, simple_prompt_template_obj

    if not retriever or not rag_prompt_template_obj or not simple_prompt_template_obj:
        logger.error("Cannot build RAG chain: Required components (retriever, RAG prompt, or Simple prompt) are missing.")
        raise HTTPException(status_code=503, detail="RAG components not fully initialized.")

    # --- Define parts specific to RAG ---
    retrieval_chain_part = (
        RunnablePassthrough.assign( 
            docs=RunnableLambda(lambda x: logger.info(f"Retriever Input: '{x['question']}'") or x['question']) | retriever 
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
        rag_prompt_template_obj |
        llm | # Use the passed LLM instance
        StrOutputParser()
    )

    # --- Define the simple chain again for the fallback path ---
    # (Could also call _build_simple_chain, but defining inline might be clearer for the branch)
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
        simple_prompt_template_obj | 
        llm | 
        StrOutputParser()
    )

    # --- Branching Logic ---
    def route_based_on_docs(info: Dict[str, Any]) -> Runnable:
        if info.get("docs"): 
            logger.info(f"Routing to RAG chain because {len(info['docs'])} documents were retrieved.")
            # Need to pass the input dict `info` to the chain
            return RunnableLambda(lambda x: x) | rag_chain_main
        else:
            logger.info("Routing to Simple chain (fallback) because no documents were retrieved.")
            # Need to pass the input dict `info` to the chain
            return RunnableLambda(lambda x: x) | simple_fallback_chain

    # --- Final Combined Chain ---
    full_rag_chain = retrieval_chain_part | RunnableLambda(route_based_on_docs)
    
    logger.info("Built final chain with RAG fallback logic.")
    return full_rag_chain


# --- Main Function to Get Chain ---
def get_rag_or_simple_chain(use_rag_flag: bool, specific_ollama_model_name: Optional[str] = None) -> Runnable:
    """
    Selects and returns the appropriate chain (RAG with fallback or Simple) 
    based on the RAG_ENABLED flag and use_rag_flag.
    """
    # 1. Select the LLM instance
    active_llm = _select_llm(specific_ollama_model_name)

    # 2. Decide whether to use RAG or Simple based on global flag and request flag
    if RAG_ENABLED and use_rag_flag:
        logger.info("Attempting to build and return RAG chain with fallback.")
        # Check if necessary RAG components are available before building
        if chroma_client and retriever and rag_prompt_template_obj and simple_prompt_template_obj:
             return _build_rag_chain_with_fallback(active_llm)
        else:
            logger.warning("RAG was enabled, but components are missing. Falling back to simple chain.")
            return _build_simple_chain(active_llm)
    else:
        if not RAG_ENABLED:
             logger.info("RAG_ENABLED is False. Building simple chain.")
        else: # RAG_ENABLED is True, but use_rag_flag is False
             logger.info("use_rag_flag is False. Building simple chain.")
        return _build_simple_chain(active_llm)

