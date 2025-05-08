# backend/main.py
import os
import logging
from fastapi import FastAPI, Request # Keep Request if needed elsewhere
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import configurations, routers, and setup functions
from config import logger
from routers import model_router, chat_router, upload_router, health_router, document_router
from rag_components import setup_rag_components # Import the setup function


# --- FastAPI Application Lifespan (for startup and shutdown events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    logger.info("FastAPI application startup...")
    # Call setup to initialize global components in rag_components.py
    setup_rag_components()
    logger.info("RAG components setup initiated.")
    yield
    # Shutdown event
    logger.info("FastAPI application shutdown...")
    # No explicit cleanup needed here if components don't hold external resources
    # that need closing (like file handles, specific network connections not handled by libraries)

# --- FastAPI App Setup ---
app = FastAPI(
    title="NLP Backend API with RAG",
    description="Backend API for NLP tasks with Retrieval Augmented Generation.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# --- Include Routers ---
@app.get("/")
async def read_root():
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the NLP Backend API with RAG support. See /docs for API documentation."}

app.include_router(model_router.router, prefix="/api")
app.include_router(chat_router.router, prefix="/api")
app.include_router(upload_router.router, prefix="/api")
app.include_router(health_router.router, prefix="/api")
app.include_router(document_router.router, prefix="/api")

# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "true").lower() == "true" # Allow disabling reload via env var

    logger.info(f"Starting Uvicorn server directly from main.py on {host}:{port} (Reload: {reload_flag})...")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_flag, # Use the flag
        log_level="info"
    )

