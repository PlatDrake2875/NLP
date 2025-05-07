# backend/config.py
import os
import logging

# --- Environment Variable Based Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL_FOR_RAG = os.getenv("OLLAMA_MODEL", "llama3") 
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb") # Changed default to 'chromadb' for Docker networking
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Logging Setup ---
# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Get a logger instance for consistent naming
logger = logging.getLogger("nlp_backend") # Consistent logger name

# --- Constants ---
COLLECTION_NAME = "rag_documents"
RAG_PROMPT_TEMPLATE_STR = """CONTEXT:\n{context}\n\nCONVERSATION HISTORY:\n{chat_history}\n\nQUESTION:\n{question}\n\nAnswer the question based *only* on the provided context and conversation history. If the context doesn't contain the answer, say you don't know."""
SIMPLE_PROMPT_TEMPLATE_STR = """CONVERSATION HISTORY:\n{chat_history}\n\nQUESTION:\n{question}\n\nAnswer the question based on the conversation history and your general knowledge."""

# You can add other configurations here as needed
