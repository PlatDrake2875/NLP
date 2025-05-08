# backend/rag_components.py
import logging
from typing import List, Dict, Optional, Any

import chromadb
from fastapi import HTTPException, Depends
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

# Import configurations from config.py
try:
    from config import (
        CHROMA_HOST, CHROMA_PORT, EMBEDDING_MODEL_NAME, COLLECTION_NAME,
        OLLAMA_BASE_URL, OLLAMA_MODEL_FOR_RAG,
        # --- Import RAG_ENABLED from config ---
        RAG_ENABLED,
        RAG_PROMPT_TEMPLATE_STR, SIMPLE_PROMPT_TEMPLATE_STR,
        RAG_CONTEXT_PREFIX_TEMPLATE_STR,
        logger
    )
except ImportError as e:
    logger.critical(f"Failed to import from config: {e}. Ensure config.py exists and PYTHONPATH is correct.")
    raise

# --- Remove RAG_ENABLED definition from here ---
# RAG_ENABLED = True # Example: Set to True to enable RAG features

# --- Globals for RAG Components ---
chroma_client: Optional[chromadb.HttpClient] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[LangchainChroma] = None
retriever: Optional[Any] = None # Can be BaseRetriever or similar
ollama_chat_for_rag: Optional[ChatOllama] = None
rag_prompt_template_obj: Optional[ChatPromptTemplate] = None
simple_prompt_template_obj: Optional[ChatPromptTemplate] = None
rag_context_prefix_prompt_template_obj: Optional[ChatPromptTemplate] = None

# --- Helper Functions ---
def format_history_for_lc(history: List[Dict[str, str]]) -> List:
    """Converts custom history format to LangChain Message objects."""
    lc_messages = []
    for msg in history:
        role = msg.get("sender", "user")
        content = msg.get("text", "")
        if not isinstance(content, str): content = str(content) # Ensure content is string

        if role.lower() == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role.lower() == "bot":
            lc_messages.append(AIMessage(content=content))
        else: # Default to human if role is unclear
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
           ollama_chat_for_rag, rag_prompt_template_obj, simple_prompt_template_obj, \
           rag_context_prefix_prompt_template_obj # Add new global

    logger.info("Setting up RAG components...")

    # 1. Embedding Function
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info(f"HuggingFace embeddings initialized with model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}", exc_info=True)
        embedding_function = None # Ensure it's None on failure

    # 2. ChromaDB Client and Vectorstore (only if embeddings succeeded)
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

            # 3. Retriever (only if vectorstore succeeded)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            logger.info("Retriever initialized successfully (k=3).")

        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB client/vectorstore/retriever: {e}", exc_info=True)
            chroma_client = None
            vectorstore = None
            retriever = None
    else:
         logger.warning("Skipping ChromaDB/Vectorstore/Retriever setup because embedding function failed.")
         chroma_client = None
         vectorstore = None
         retriever = None


    # 4. ChatOllama LLM (for RAG)
    if OLLAMA_BASE_URL:
        try:
            logger.info(f"Initializing ChatOllama (for RAG) with model: {OLLAMA_MODEL_FOR_RAG} at {OLLAMA_BASE_URL}")
            ollama_chat_for_rag = ChatOllama(
                model=OLLAMA_MODEL_FOR_RAG,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            )
            logger.info("ChatOllama (for RAG) initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama for RAG: {e}", exc_info=True)
            ollama_chat_for_rag = None
    else:
        logger.warning("OLLAMA_BASE_URL not set. ChatOllama for RAG cannot be initialized.")
        ollama_chat_for_rag = None

    # 5. Prompt Templates
    logger.info("Initializing prompt templates.")
    try:
        rag_prompt_template_obj = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)
        logger.info("RAG prompt template initialized.")
    except Exception as e:
        logger.error(f"Failed to create RAG prompt template: {e}", exc_info=True)
        rag_prompt_template_obj = None

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


# --- Dependency Functions ---
def get_chroma_client() -> chromadb.HttpClient:
    if chroma_client is None:
        logger.error("Global ChromaDB client is None. Setup might have failed or is incomplete.")
        raise HTTPException(status_code=503, detail="ChromaDB client is not available. Service might be starting up or setup failed.")
    return chroma_client

def get_vectorstore() -> LangchainChroma:
    if vectorstore is None:
        logger.error("Global Vectorstore is None. Setup might have failed or is incomplete.")
        raise HTTPException(status_code=503, detail="Vector store is not available. Service might be starting up or setup failed.")
    return vectorstore

def get_embedding_function() -> HuggingFaceEmbeddings:
     if embedding_function is None:
        logger.error("Global Embedding function is None. Setup might have failed.")
        raise HTTPException(status_code=503, detail="Embedding function is not available.")
     return embedding_function

def get_retriever() -> Any: # Return type depends on as_retriever()
    if retriever is None:
        logger.error("Global Retriever is None. Setup might have failed or is incomplete.")
        raise HTTPException(status_code=503, detail="Retriever is not available. Service might be starting up or setup failed.")
    return retriever

def get_ollama_chat_for_rag() -> ChatOllama:
    if ollama_chat_for_rag is None:
        logger.error("Global ChatOllama (for RAG) is None. Setup might have failed or OLLAMA_BASE_URL is not set.")
        raise HTTPException(status_code=503, detail="RAG chat model is not available.")
    return ollama_chat_for_rag

# --- Optional Dependencies (for health check) ---
def get_optional_chroma_client() -> Optional[chromadb.HttpClient]:
    return chroma_client

def get_optional_ollama_chat_for_rag() -> Optional[ChatOllama]:
    return ollama_chat_for_rag

# --- Function to Generate RAG Context Prefix ---
async def get_rag_context_prefix(query: str) -> Optional[str]:
    """
    Retrieves context and formats it with the query using the prefix template.
    Returns the formatted prompt string or None if RAG is disabled/unavailable/finds no docs.
    """
    if not RAG_ENABLED: # Check the imported flag from config
        logger.info("RAG is disabled globally. Skipping context prefix generation.")
        return None
    if not retriever or not rag_context_prefix_prompt_template_obj:
        logger.warning("Retriever or RAG context prefix prompt template not initialized. Cannot generate prefix.")
        return None # Indicate components are missing

    try:
        logger.info(f"Invoking retriever for RAG context prefix with query: '{query[:50]}...'")
        retrieved_docs = await retriever.ainvoke(query)

        if not retrieved_docs:
            logger.info("No relevant documents found by retriever for RAG context prefix.")
            return None

        formatted_context = format_docs(retrieved_docs)
        logger.debug(f"Formatted context for prefix:\n{formatted_context}")

        formatted_prompt = rag_context_prefix_prompt_template_obj.format(
            context=formatted_context,
            question=query
        )
        logger.info(f"Generated RAG context prefix string (length: {len(formatted_prompt)}).")
        return formatted_prompt # Return the complete string ready for the LLM

    except Exception as e:
        logger.error(f"Error generating RAG context prefix: {e}", exc_info=True)
        return None # Indicate failure

# --- Original Chain Builders (Commented Out as Requested) ---
# def _select_llm(specific_ollama_model_name: Optional[str] = None) -> ChatOllama:
#     """Selects the LLM instance based on request or default RAG model."""
#     model_name = specific_ollama_model_name or OLLAMA_MODEL_FOR_RAG
#     logger.info(f"Selecting LLM: {model_name} at {OLLAMA_BASE_URL}")
#     # Assuming ollama_chat_for_rag is initialized with the default RAG model
#     # If a different model is requested, create a new instance
#     if specific_ollama_model_name and specific_ollama_model_name != OLLAMA_MODEL_FOR_RAG:
#         try:
#             return ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=0.1)
#         except Exception as e:
#             logger.error(f"Failed to create ChatOllama instance for {model_name}: {e}. Falling back to default RAG model.")
#             # Fallback to the globally initialized RAG LLM
#             if ollama_chat_for_rag:
#                 return ollama_chat_for_rag
#             else:
#                 logger.critical("Default RAG LLM is also not available.")
#                 raise HTTPException(status_code=503, detail="Chat model not available.")
#     elif ollama_chat_for_rag:
#         return ollama_chat_for_rag # Return the pre-initialized default RAG LLM
#     else:
#         logger.critical("Default RAG LLM was not initialized during setup.")
#         raise HTTPException(status_code=503, detail="Default chat model not available.")

# def _build_simple_chain(llm: ChatOllama) -> Runnable:
#     """Builds the simple chain (no RAG)."""
#     logger.info("Building simple chain.")
#     if not simple_prompt_template_obj:
#         logger.error("Simple prompt template object is None. Cannot build simple chain.")
#         raise ValueError("Simple prompt template not initialized.") # Raise internal error

#     return (
#         RunnablePassthrough.assign(
#             chat_history=RunnableLambda(lambda x: format_history_for_lc(x["chat_history"]))
#         )
#         | simple_prompt_template_obj
#         | llm
#         | StrOutputParser()
#     )

# def _build_rag_chain_with_fallback(llm: ChatOllama) -> Runnable:
#     """Builds the RAG chain with fallback to simple chain if context retrieval fails."""
#     logger.info("Building RAG chain with fallback.")
#     if not retriever or not rag_prompt_template_obj or not simple_prompt_template_obj:
#         logger.error("One or more components (retriever, RAG prompt, Simple prompt) are None. Cannot build RAG chain.")
#         raise ValueError("RAG components not fully initialized.") # Raise internal error

#     # Define the RAG part of the chain
#     rag_chain_part = (
#         RunnableParallel(
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         )
#         | rag_prompt_template_obj
#         | llm
#         | StrOutputParser()
#     )

#     # Define the simple part (used as fallback)
#     # Note: This assumes the input structure for simple chain is just the question
#     simple_chain_part = simple_prompt_template_obj | llm | StrOutputParser()

#     # Use RunnableBranch to decide the path
#     # Input to the branch is expected to be a dict with 'question' and 'chat_history'
#     # We need to adapt this slightly if the input structure differs
#     def route(info):
#         # This routing logic might need adjustment based on how context is handled
#         # For now, let's assume context is attempted first
#         # A more robust check might involve trying retrieval and seeing if it fails/returns empty
#         logger.debug(f"Routing decision based on input: {info}") # Add debug log
#         # If context is present and not the "No relevant context" message, use RAG
#         if info.get("context") and info["context"] != "No relevant context found.":
#              logger.info("Routing to RAG chain part.")
#              return rag_chain_part
#         else:
#              logger.info("Routing to simple chain part (fallback).")
#              return simple_chain_part

#     # The initial part retrieves context and prepares for branching
#     # Input expected: {"question": str, "chat_history": list}
#     full_rag_chain = RunnablePassthrough.assign(
#         context=lambda x: retriever.ainvoke(x["question"]) if retriever else [], # Attempt retrieval
#         chat_history_formatted=lambda x: format_history_for_lc(x["chat_history"]) # Format history
#     ).assign(
#         formatted_context=lambda x: format_docs(x["context"]) # Format the retrieved docs
#     ) | RunnableLambda(route) # Route based on formatted_context (needs adjustment)

#     # --- Potential Issue & Alternative Branching ---
#     # The above RunnableLambda(route) might not work as expected because the input
#     # to the lambda might not be the dictionary containing 'formatted_context'.
#     # A more standard LangChain approach for fallback:
#     # full_rag_chain = RunnableParallel(
#     #     {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
#     # ) | RunnablePassthrough.assign( # Pass history through
#     #      chat_history=lambda x: format_history_for_lc(x['chat_history']['chat_history']) # Adjust access if needed
#     # ) | RunnableBranch(
#     #      (lambda x: x["context"] != "No relevant context found.", rag_prompt_template_obj | llm | StrOutputParser()), # RAG path
#     #      simple_prompt_template_obj | llm | StrOutputParser() # Fallback path
#     # )
#     # logger.info("Using RunnableBranch for RAG fallback logic.")
#     # This alternative needs careful testing of the input structure at each step.
#     # For now, sticking with the simpler (potentially flawed) lambda routing for demonstration.

#     logger.info("Built RAG chain with fallback.")
#     return full_rag_chain


# def get_rag_or_simple_chain(use_rag_flag: bool, specific_ollama_model_name: Optional[str] = None) -> Runnable:
#     """
#     Selects and returns the appropriate chain (RAG with fallback or Simple)
#     based on the RAG_ENABLED flag and use_rag_flag.
#     """
#     active_llm = _select_llm(specific_ollama_model_name)

#     if RAG_ENABLED and use_rag_flag:
#         logger.info("Attempting to build and return RAG chain with fallback.")
#         if retriever and rag_prompt_template_obj and simple_prompt_template_obj:
#              # This is where the complex RAG chain with fallback is built
#              # For simplicity in this example, let's just return the RAG part
#              # A proper implementation needs the fallback logic from _build_rag_chain_with_fallback
#              rag_chain = (
#                  RunnableParallel(
#                      {"context": retriever | format_docs, "question": RunnablePassthrough()}
#                  )
#                  | rag_prompt_template_obj
#                  | active_llm
#                  | StrOutputParser()
#              )
#              logger.info("Returning RAG chain (fallback logic needs review).")
#              return rag_chain # Simplified return for now
#              # return _build_rag_chain_with_fallback(active_llm) # Use this when logic is confirmed
#         else:
#             logger.warning("RAG enabled, but core components missing. Falling back to simple chain.")
#             return _build_simple_chain(active_llm)
#     else:
#         if not RAG_ENABLED:
#              logger.info("RAG_ENABLED is False. Building simple chain.")
#         else: # RAG is enabled globally, but disabled for this request
#              logger.info(f"RAG disabled for this request (use_rag_flag={use_rag_flag}). Building simple chain.")
#         return _build_simple_chain(active_llm)

