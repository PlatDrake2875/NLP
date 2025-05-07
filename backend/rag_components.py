# backend/rag_components.py
import logging
from typing import List, Dict, Optional

import chromadb
from fastapi import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma as LangchainChroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.messages import AIMessage, HumanMessage

# Import configurations from config.py using direct absolute path
from config import (
    CHROMA_HOST, CHROMA_PORT, EMBEDDING_MODEL_NAME, COLLECTION_NAME,
    OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG,
    RAG_PROMPT_TEMPLATE_STR, SIMPLE_PROMPT_TEMPLATE_STR, logger
)

# --- Globals for RAG Components ---
chroma_client: Optional[chromadb.HttpClient] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[LangchainChroma] = None
retriever: Optional[Runnable] = None 
ollama_chat_for_rag: Optional[ChatOllama] = None
rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None


def setup_rag_components():
    """Initializes and sets up all RAG components."""
    global chroma_client, embedding_function, vectorstore, retriever, ollama_chat_for_rag
    global rag_prompt_template_obj, simple_prompt_template_obj

    logger.info("Setting up RAG components...")
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
                    search_kwargs={'k': 5} 
                )
                logger.info("Retriever initialized successfully.")
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
    
    # --- Prompt Templates ---
    rag_prompt_template_obj = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE_STR), 
        MessagesPlaceholder(variable_name="chat_history_messages"), 
        ("human", "{question}")
    ])
    simple_prompt_template_obj = ChatPromptTemplate.from_messages([
        ("system", SIMPLE_PROMPT_TEMPLATE_STR), 
        MessagesPlaceholder(variable_name="chat_history_messages"), 
        ("human", "{question}")
    ])
    logger.info("Prompt templates initialized.")
    logger.info("RAG components setup finished.")


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Formats chat history for inclusion in the prompt context."""
    if not chat_history: 
        return "No history yet."
    return "\n".join([f"{msg['role'].upper()}: {msg.get('content', '')}" for msg in chat_history])

def format_history_for_lc(chat_history: List[Dict[str, str]]) -> list:
    """Formats chat history into LangChain Message objects."""
    messages = []
    for msg in chat_history:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user": 
            messages.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ai": 
            messages.append(AIMessage(content=content))
    return messages

def get_rag_or_simple_chain(use_rag_flag: bool, specific_ollama_model_name: Optional[str] = None) -> Runnable:
    """
    Constructs and returns either a RAG-enabled chain or a simpler conversational chain.
    """
    global ollama_chat_for_rag, rag_prompt_template_obj, simple_prompt_template_obj, retriever

    active_llm = ollama_chat_for_rag
    
    # Determine the LLM to use
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

    # Select chain type
    if use_rag_flag and chroma_client and retriever and rag_prompt_template_obj:
        logger.info(f"Creating RAG chain with retriever, using LLM: {getattr(active_llm, 'model', 'N/A')}")
        rag_chain = (
            RunnableParallel(
                context=(lambda x: x['question']) | retriever, 
                question=RunnablePassthrough(), 
                chat_history=(lambda x: format_chat_history(x.get('chat_history', []))), 
                chat_history_messages=(lambda x: format_history_for_lc(x.get('chat_history', [])))
            ) | 
            rag_prompt_template_obj | 
            active_llm | 
            StrOutputParser()
        )
        return rag_chain
    else:
        if use_rag_flag: 
            logger.warning("RAG requested but ChromaDB/retriever unavailable or prompt template missing. Falling back to simple chain.")
        if not simple_prompt_template_obj:
            logger.error("Simple prompt template is not initialized.")
            raise HTTPException(status_code=503, detail="Simple prompt template is not available.")

        logger.info(f"Creating Simple chain, using LLM: {getattr(active_llm, 'model', 'N/A')}")
        simple_chain = (
            RunnableParallel(
                question=(lambda x: x['question']), 
                chat_history=(lambda x: format_chat_history(x.get('chat_history', []))),
                chat_history_messages=(lambda x: format_history_for_lc(x.get('chat_history', [])))
            ) | 
            simple_prompt_template_obj | 
            active_llm | 
            StrOutputParser()
        )
        return simple_chain
