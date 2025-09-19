# backend/rag_components.py
from typing import Any, Optional

import chromadb
from backend.config import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
    # Attempt to import the new config for automation LLM
    OLLAMA_MODEL_FOR_AUTOMATION,
    OLLAMA_MODEL_FOR_RAG,
    RAG_CONTEXT_PREFIX_TEMPLATE_STR,
    RAG_ENABLED,  # Imported from config
    RAG_PROMPT_TEMPLATE_STR,
    SIMPLE_PROMPT_TEMPLATE_STR,
    logger,
)
from fastapi import (
    HTTPException,  # Ensure Depends is imported if used directly here, though typically in routers
)
from langchain_chroma import Chroma as LangchainChroma
from langchain_core.documents import Document
from langchain_core.messages import (  # Removed SystemMessage as it wasn't used
    AIMessage,
    HumanMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,  # Removed MessagesPlaceholder as it wasn't used directly
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# --- Globals for RAG Components ---
chroma_client: Optional[chromadb.HttpClient] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[LangchainChroma] = None
retriever: Optional[Any] = None
ollama_chat_for_rag: Optional[ChatOllama] = None
ollama_chat_for_automation: Optional[ChatOllama] = None  # New global for automation LLM
rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None
rag_context_prefix_prompt_template_obj: Optional[ChatPromptTemplate] = None


# --- Helper Functions (format_history_for_lc, format_docs remain the same) ---
def format_history_for_lc(history: list[dict[str, str]]) -> list:
    """Converts custom history format to LangChain Message objects."""
    lc_messages = []
    for msg in history:
        role = msg.get("sender", "user")
        content = msg.get("text", "")
        if not isinstance(content, str):
            content = str(content)

        if role.lower() == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role.lower() == "bot":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def format_docs(docs: list[Document]) -> str:
    """Helper function to format retrieved documents."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)


# --- Setup Function ---
def setup_rag_components():
    """Initializes all RAG components and stores them in globals."""
    global \
        chroma_client, \
        embedding_function, \
        vectorstore, \
        retriever, \
        ollama_chat_for_rag, \
        ollama_chat_for_automation, \
        rag_prompt_template_obj, \
        simple_prompt_template_obj, \
        rag_context_prefix_prompt_template_obj

    # 1. Embedding Function (same as before)
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}", exc_info=True)
        embedding_function = None

    # 2. ChromaDB Client and Vectorstore (same as before)
    if embedding_function:
        try:
            temp_chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST, port=int(CHROMA_PORT)
            )
            temp_chroma_client.heartbeat()
            chroma_client = temp_chroma_client

            vectorstore = LangchainChroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_function,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        except Exception as e:
            logger.critical(
                f"Failed to initialize ChromaDB client/vectorstore/retriever: {e}",
                exc_info=True,
            )
            chroma_client = vectorstore = retriever = None
    else:
        chroma_client = vectorstore = retriever = None

    # 3. ChatOllama LLM (for RAG) (same as before)
    if OLLAMA_BASE_URL:
        try:
            ollama_chat_for_rag = ChatOllama(
                model=OLLAMA_MODEL_FOR_RAG,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,  # Example temperature
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama for RAG: {e}", exc_info=True)
            ollama_chat_for_rag = None
    else:
        ollama_chat_for_rag = None

    # 4. ChatOllama LLM (for Automation Tasks) - NEW
    if OLLAMA_BASE_URL:
        try:
            ollama_chat_for_automation = ChatOllama(
                model=OLLAMA_MODEL_FOR_AUTOMATION,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2,
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize ChatOllama for Automation: {e}", exc_info=True
            )
            ollama_chat_for_automation = None
    else:
        ollama_chat_for_automation = None

    # 5. Prompt Templates (same as before)
    try:
        rag_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_PROMPT_TEMPLATE_STR
        )
    except Exception as e:
        logger.error(f"Failed to create RAG prompt template: {e}", exc_info=True)
        rag_prompt_template_obj = None

    try:
        simple_prompt_template_obj = ChatPromptTemplate.from_template(
            SIMPLE_PROMPT_TEMPLATE_STR
        )
    except Exception as e:
        logger.error(f"Failed to create Simple prompt template: {e}", exc_info=True)
        simple_prompt_template_obj = None

    try:
        rag_context_prefix_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_CONTEXT_PREFIX_TEMPLATE_STR
        )
    except Exception as e:
        logger.error(
            f"Failed to create RAG context prefix prompt template: {e}", exc_info=True
        )
        rag_context_prefix_prompt_template_obj = None


# --- Dependency Functions (getters for initialized components) ---
def get_chroma_client() -> chromadb.HttpClient:
    if chroma_client is None:
        raise HTTPException(status_code=503, detail="ChromaDB client is not available.")
    return chroma_client


def get_vectorstore() -> LangchainChroma:
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store is not available.")
    return vectorstore


def get_embedding_function() -> HuggingFaceEmbeddings:
    if embedding_function is None:
        raise HTTPException(
            status_code=503, detail="Embedding function is not available."
        )
    return embedding_function


def get_retriever() -> Any:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not available.")
    return retriever


def get_ollama_chat_for_rag() -> ChatOllama:
    if ollama_chat_for_rag is None:
        raise HTTPException(status_code=503, detail="RAG chat model is not available.")
    return ollama_chat_for_rag


# --- Function that was missing ---
def get_llm_for_automation() -> ChatOllama:
    """
    Returns the initialized ChatOllama instance intended for automation tasks.
    This function is the one your automate_router.py is trying to import.
    """
    if ollama_chat_for_automation is None:
        raise HTTPException(status_code=503, detail="Automation LLM is not available.")
    return ollama_chat_for_automation


# --- Optional Dependencies (for health check) ---
def get_optional_chroma_client() -> Optional[chromadb.HttpClient]:
    return chroma_client


def get_optional_ollama_chat_for_rag() -> Optional[ChatOllama]:
    return ollama_chat_for_rag


def get_optional_llm_for_automation() -> Optional[ChatOllama]:
    return ollama_chat_for_automation


# --- Function to Generate RAG Context Prefix (same as before) ---
async def get_rag_context_prefix(query: str) -> Optional[str]:
    if not RAG_ENABLED:
        return None
    if not retriever or not rag_context_prefix_prompt_template_obj:
        return None

    try:
        retrieved_docs = await retriever.ainvoke(query)

        if not retrieved_docs:
            return None

        formatted_context = format_docs(retrieved_docs)

        if rag_context_prefix_prompt_template_obj:
            formatted_prompt = rag_context_prefix_prompt_template_obj.format(
                context=formatted_context, question=query
            )
            return formatted_prompt
        else:
            return None

    except Exception as e:
        logger.error(f"Error generating RAG context prefix: {e}", exc_info=True)
        return None
