# backend/rag_components.py
from typing import Any, Optional

import chromadb
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

from config import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_FOR_AUTOMATION,
    OLLAMA_MODEL_FOR_RAG,
    RAG_CONTEXT_PREFIX_TEMPLATE_STR,
    RAG_ENABLED,
    RAG_PROMPT_TEMPLATE_STR,
    SIMPLE_PROMPT_TEMPLATE_STR,
    logger,
)


class RAGComponents:
    """Singleton class to manage RAG components without global variables."""

    _instance: Optional["RAGComponents"] = None
    _initialized: bool = False

    def __new__(cls) -> "RAGComponents":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.chroma_client: Optional[chromadb.HttpClient] = None
            self.embedding_function: Optional[HuggingFaceEmbeddings] = None
            self.vectorstore: Optional[LangchainChroma] = None
            self.retriever: Optional[Any] = None
            self.ollama_chat_for_rag: Optional[ChatOllama] = None
            self.ollama_chat_for_automation: Optional[ChatOllama] = None
            self.rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.simple_prompt_template_obj: Optional[ChatPromptTemplate] = None
            self.rag_context_prefix_prompt_template_obj: Optional[
                ChatPromptTemplate
            ] = None
            RAGComponents._initialized = True

    def setup_components(self):
        """Initializes all RAG components."""
        # 1. Embedding Function
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # 2. ChromaDB Client and Vectorstore
        if self.embedding_function:
            try:
                temp_chroma_client = chromadb.HttpClient(
                    host=CHROMA_HOST, port=int(CHROMA_PORT)
                )
                temp_chroma_client.heartbeat()
                self.chroma_client = temp_chroma_client

                # Try to get existing collection first, create if it doesn't exist
                try:
                    self.chroma_client.get_collection(name=COLLECTION_NAME)
                    logger.info(
                        f"Using existing ChromaDB collection: {COLLECTION_NAME}"
                    )
                except ValueError:
                    # Collection doesn't exist, create it
                    self.chroma_client.create_collection(
                        name=COLLECTION_NAME,
                        metadata={"hnsw:space": "cosine"},  # Explicit metadata
                    )
                    logger.info(f"Created new ChromaDB collection: {COLLECTION_NAME}")

                self.vectorstore = LangchainChroma(
                    client=self.chroma_client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embedding_function,
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            except Exception as e:
                logger.critical(
                    "Failed to initialize ChromaDB client/vectorstore/retriever: %s",
                    e,
                    exc_info=True,
                )
                self.chroma_client = self.vectorstore = self.retriever = None
        else:
            self.chroma_client = self.vectorstore = self.retriever = None

        # 3. ChatOllama LLM (for RAG)
        self.ollama_chat_for_rag = ChatOllama(
            model=OLLAMA_MODEL_FOR_RAG,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,  # Example temperature
        )

        # 4. ChatOllama LLM (for Automation Tasks)
        self.ollama_chat_for_automation = ChatOllama(
            model=OLLAMA_MODEL_FOR_AUTOMATION,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

        # 5. Prompt Templates
        self.rag_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_PROMPT_TEMPLATE_STR
        )

        self.simple_prompt_template_obj = ChatPromptTemplate.from_template(
            SIMPLE_PROMPT_TEMPLATE_STR
        )

        self.rag_context_prefix_prompt_template_obj = ChatPromptTemplate.from_template(
            RAG_CONTEXT_PREFIX_TEMPLATE_STR
        )


def get_rag_components() -> RAGComponents:
    """Get the singleton RAG components instance."""
    return RAGComponents()


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
    """Initializes all RAG components using the singleton pattern."""
    rag_components = get_rag_components()
    rag_components.setup_components()


# --- Dependency Functions (getters for initialized components) ---
def get_chroma_client() -> chromadb.HttpClient:
    rag_components = get_rag_components()
    if rag_components.chroma_client is None:
        raise HTTPException(status_code=503, detail="ChromaDB client is not available.")
    return rag_components.chroma_client


def get_vectorstore() -> LangchainChroma:
    rag_components = get_rag_components()
    if rag_components.vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store is not available.")
    return rag_components.vectorstore


def get_embedding_function() -> HuggingFaceEmbeddings:
    rag_components = get_rag_components()
    if rag_components.embedding_function is None:
        raise HTTPException(
            status_code=503, detail="Embedding function is not available."
        )
    return rag_components.embedding_function


def get_retriever() -> Any:
    rag_components = get_rag_components()
    if rag_components.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not available.")
    return rag_components.retriever


def get_ollama_chat_for_rag() -> ChatOllama:
    rag_components = get_rag_components()
    if rag_components.ollama_chat_for_rag is None:
        raise HTTPException(status_code=503, detail="RAG chat model is not available.")
    return rag_components.ollama_chat_for_rag


# --- Function that was missing ---
def get_llm_for_automation() -> ChatOllama:
    rag_components = get_rag_components()
    if rag_components.ollama_chat_for_automation is None:
        raise HTTPException(status_code=503, detail="Automation LLM is not available.")
    return rag_components.ollama_chat_for_automation


# --- Optional Dependencies (for health check) ---
def get_optional_chroma_client() -> Optional[chromadb.HttpClient]:
    rag_components = get_rag_components()
    return rag_components.chroma_client


def get_optional_ollama_chat_for_rag() -> Optional[ChatOllama]:
    rag_components = get_rag_components()
    return rag_components.ollama_chat_for_rag


def get_optional_llm_for_automation() -> Optional[ChatOllama]:
    rag_components = get_rag_components()
    return rag_components.ollama_chat_for_automation


# --- Function to Generate RAG Context Prefix ---
async def get_rag_context_prefix(query: str) -> Optional[str]:
    if not RAG_ENABLED:
        return None

    rag_components = get_rag_components()
    if (
        not rag_components.retriever
        or not rag_components.rag_context_prefix_prompt_template_obj
    ):
        return None

    try:
        retrieved_docs = await rag_components.retriever.ainvoke(query)

        if not retrieved_docs:
            return None

        formatted_context = format_docs(retrieved_docs)

        if rag_components.rag_context_prefix_prompt_template_obj:
            formatted_prompt = (
                rag_components.rag_context_prefix_prompt_template_obj.format(
                    context=formatted_context, question=query
                )
            )
            return formatted_prompt
        else:
            return None

    except Exception as e:
        logger.error("Error generating RAG context prefix: %s", e, exc_info=True)
        return None
