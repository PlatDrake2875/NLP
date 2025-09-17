# backend/main.py
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI  # Keep Request if needed elsewhere
from fastapi.middleware.cors import CORSMiddleware

# Import configurations, routers, and setup functions
from config import logger
from rag_components import setup_rag_components  # Import the setup function

# Updated to include automate_router
from routers import (
    automate_router,
    chat_router,
    document_router,
    health_router,
    model_router,
    upload_router,
)


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
    lifespan=lifespan,  # Use the lifespan context manager
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# --- Include Routers ---
@app.get("/")
async def read_root():
    logger.info("Root endpoint '/' accessed.")
    return {
        "message": "Welcome to the NLP Backend API with RAG support. See /docs for API documentation."
    }


app.include_router(health_router.router)
app.include_router(chat_router.router, prefix="/api", tags=["Chat Endpoints"])
app.include_router(model_router.router, prefix="/api", tags=["Model Endpoints"])
app.include_router(document_router.router, prefix="/api", tags=["Document Endpoints"])
app.include_router(upload_router.router, prefix="/api", tags=["Upload Endpoints"])
app.include_router(automate_router.router, prefix="/api", tags=["Automation Endpoints"])



# --- Main Method for Application Testing ---
async def test_chat_router(disable_guardrails_for_testing=False):
    """
    Test the chat router functionality with sample requests.
    This function demonstrates how to use the chat endpoint programmatically.

    Args:
        disable_guardrails_for_testing: If True, forces USE_GUARDRAILS=False for testing outside Docker
    """
    import os

    from fastapi.testclient import TestClient

    from routers.chat_router import router

    # Temporarily override USE_GUARDRAILS for testing if requested
    original_use_guardrails = None
    if disable_guardrails_for_testing:
        original_use_guardrails = os.environ.get("USE_GUARDRAILS")
        os.environ["USE_GUARDRAILS"] = "false"
        print("🔧 Temporarily disabled guardrails for testing outside Docker environment")

    # Create a test app with the chat router
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")

    print("=== Testing Chat Router ===")
    print("Configuration:")
    try:
        from config import (
            NEMO_GUARDRAILS_SERVER_URL,
            OLLAMA_BASE_URL,
            OLLAMA_MODEL_FOR_RAG,
            RAG_ENABLED,
            USE_GUARDRAILS,
        )
        print(f"  - OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
        print(f"  - OLLAMA_MODEL_FOR_RAG: {OLLAMA_MODEL_FOR_RAG}")
        print(f"  - RAG_ENABLED: {RAG_ENABLED}")
        print(f"  - USE_GUARDRAILS: {USE_GUARDRAILS}")
        print(f"  - NEMO_GUARDRAILS_SERVER_URL: {NEMO_GUARDRAILS_SERVER_URL}")
    except ImportError:
        print("  - Could not import configuration, using defaults")
    print()

    # Test cases with both guardrails and direct modes
    test_cases = [
        {
            "name": "Simple Hello Test",
            "data": {
                "query": "Hello, how are you?",
                "history": [],
                "use_rag": False
            }
        },
        {
            "name": "Follow-up Question",
            "data": {
                "query": "What's the weather like?",
                "history": [
                    {"sender": "user", "text": "Hello, how are you?"},
                    {"sender": "bot", "text": "Hello! I'm doing well, thank you for asking. How can I help you today?"}
                ],
                "use_rag": False
            }
        },
        {
            "name": "RAG-enabled Query",
            "data": {
                "query": "Tell me about machine learning",
                "history": [],
                "use_rag": True
            }
        }
    ]

    # Create test client
    with TestClient(test_app) as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['name']}")
            print(f"Request data: {test_case['data']}")

            try:
                # Make the request
                response = client.post("/api/chat", json=test_case['data'])

                print(f"Status Code: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")

                if response.status_code == 200:
                    print("Response content (first 500 chars):")
                    content = response.text[:500]
                    print(content)
                    if len(response.text) > 500:
                        print("... (truncated)")

                    # Check if we got a connection error (expected when testing outside Docker)
                    if "ConnectError" in content or "getaddrinfo failed" in content:
                        print("💡 Note: Connection error is expected when testing outside Docker environment")
                        if not disable_guardrails_for_testing:
                            print("💡 Hint: Run with disable_guardrails_for_testing=True to test direct Ollama connection")
                else:
                    print(f"Error response: {response.text}")

            except Exception as e:
                print(f"Error making request: {e}")

            print("-" * 50)
            print()

    # Restore original environment variable if it was changed
    if disable_guardrails_for_testing and original_use_guardrails is not None:
        os.environ["USE_GUARDRAILS"] = original_use_guardrails
        print("🔧 Restored original USE_GUARDRAILS setting")
    elif disable_guardrails_for_testing:
        # Remove the environment variable if it wasn't set originally
        os.environ.pop("USE_GUARDRAILS", None)

    print("=== Chat Router Testing Complete ===")


async def main():
    """
    Main method for testing the entire application functionality.
    This function can be used to test various components of the NLP backend.
    """
    from fastapi.testclient import TestClient

    print("=== NLP Backend Application Testing ===")
    print("Testing application components...")
    print()

    # Test the chat router (with guardrails disabled for local testing)
    print("1. Testing Chat Router:")
    await test_chat_router(disable_guardrails_for_testing=True)
    print()

    # Test the main application endpoints
    print("2. Testing Main Application:")
    with TestClient(app) as client:
        # Test root endpoint
        response = client.get("/")
        print(f"Root endpoint status: {response.status_code}")
        print(f"Root response: {response.json()}")
        print()

        # Test health endpoint
        response = client.get("/health")
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
        else:
            print(f"Health error: {response.text}")
        print()

        # Test models endpoint
        response = client.get("/api/models")
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            print(f"Available models: {len(models_data.get('models', []))} models found")
        else:
            print(f"Models error: {response.text}")

    print("=== Application Testing Complete ===")


# --- Alternative entry point for testing ---
async def test_application():
    """
    Alternative entry point specifically for testing.
    Call this with: python -c "import asyncio; from main import test_application; asyncio.run(test_application())"
    """
    await main()


# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    # import asyncio

    # # Run the test application
    # asyncio.run(test_application())

    # Uncomment below to start the uvicorn server instead
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = False  # As per your original code
    logger.info(
        f"Starting Uvicorn server directly from main.py on {host}:{port} (Reload: {reload_flag})..."
    )
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_flag,  # Explicitly set to False
        log_level="info",
    )
