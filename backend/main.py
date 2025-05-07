# backend/main.py
import os 
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import configurations, routers, and setup functions using direct absolute imports
# (These should work correctly now)
from config import logger # Use the centralized logger
# Corrected import name: model_router instead of models_router
from routers import model_router, chat_router, upload_router, health_router 
from rag_components import setup_rag_components


# --- FastAPI Application Lifespan (for startup and shutdown events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    logger.info("FastAPI application startup...")
    setup_rag_components() # Initialize RAG components
    logger.info("RAG components setup complete during application startup.")
    yield
    # Shutdown event (if any cleanup is needed)
    logger.info("FastAPI application shutdown...")

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
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Include Routers ---
@app.get("/")
async def read_root():
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the NLP Backend API with RAG support. See /docs for API documentation."}

# Include routers with the /api prefix
# Corrected variable name: model_router instead of models_router
app.include_router(model_router.router, prefix="/api") 
app.include_router(chat_router.router, prefix="/api")   
app.include_router(upload_router.router, prefix="/api") 
app.include_router(health_router.router, prefix="/api") 

# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn
    from config import OLLAMA_BASE_URL # Use config value

    logger.info("Starting Uvicorn server directly from main.py...")
    uvicorn.run(
        "main:app", 
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", "8000")), 
        reload=True, 
        log_level="info" 
    )
