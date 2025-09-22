"""
NeMo Guardrails service layer containing all guardrails business logic.
Separated from the web layer for better testability and maintainability.
"""

import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger("nemo_service")


class NemoService:
    """Service class handling all NeMo Guardrails business logic."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.rails: Optional[LLMRails] = None
        self._is_initialized = False

    def _get_default_config_path(self) -> str:
        """Get the default config path relative to the backend directory."""
        current_dir = Path(__file__).parent.parent  # Go up from services to backend
        config_dir = current_dir / "guardrails_config" / "mybot"
        return str(config_dir)

    def _get_ollama_base_url(self) -> str:
        """Determine the appropriate Ollama base URL based on environment."""
        base_url = "http://localhost:11434"
        logger.info("Using Ollama base URL: %s", base_url)
        return base_url

    async def initialize(self) -> bool:
        if self._is_initialized:
            return True

        try:
            logger.info("Initializing NeMo Guardrails with programmatic config")

            base_url = self._get_ollama_base_url()

            # Create config programmatically to avoid config parsing issues
            yaml_content = f"""
models:
  - type: main
    engine: ollama
    model: gemma3:latest
    parameters:
      base_url: {base_url}

instructions:
  - type: general
    content: "You are a helpful math AI assistant that must respond only in numbers and digits."
"""

            # Create rails configuration from YAML content
            rails_config = RailsConfig.from_content(
                colang_content="", yaml_content=yaml_content
            )

            logger.info("Rails config created programmatically")

            # Initialize the LLM Rails
            self.rails = LLMRails(rails_config)
            logger.info("LLM Rails initialized successfully")

            self._is_initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize NeMo Guardrails: %s", e)
            return await self._try_fallback_initialization()

    async def _try_fallback_initialization(self) -> bool:
        """Try fallback initialization with minimal config."""
        try:
            logger.info("Trying fallback initialization without guardrails config")
            base_url = self._get_ollama_base_url()

            # Create a minimal rails instance
            rails_config = RailsConfig.from_content(
                yaml_content=f"""
models:
  - type: main
    engine: ollama
    model: gemma3:latest
    parameters:
      base_url: {base_url}
""",
            )
            self.rails = LLMRails(rails_config)
            self._is_initialized = True
            logger.info("Fallback initialization successful")
            return True

        except Exception as fallback_error:
            logger.error("Fallback initialization also failed: %s", fallback_error)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize NeMo Guardrails: {fallback_error}",
            ) from fallback_error

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3",
        **kwargs,  # Accept but ignore unused parameters
    ) -> dict:
        """
        Process a chat completion request through NeMo Guardrails.

        Args:
            messages: List of chat messages
            model: Model name (passed through to underlying LLM)
            **kwargs: Additional parameters (stream, max_tokens, etc.)

        Returns:
            Dict: Chat completion response

        Raises:
            HTTPException: If service not initialized or processing fails.
        """
        if not self._is_initialized:
            raise HTTPException(
                status_code=500,
                detail="NeMo Guardrails not initialized. Call initialize() first.",
            )

        if not self.rails:
            raise HTTPException(status_code=500, detail="Rails not available")

        try:
            # Extract the user message (last message typically)
            user_message = self._extract_user_message(messages)

            logger.info(
                "Processing message through guardrails: %s...", user_message[:100]
            )

            result = await self.rails.generate_async(user_message)

            # Format response to match OpenAI-style response
            response = {
                "id": f"chatcmpl-local-{hash(user_message)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop",
                    }
                ],
                "usage": self._calculate_usage(user_message, result),
            }

            logger.info("Generated response: %s...", result[:100] if result else "None")
            return response

        except Exception as e:
            logger.error("Error in chat completion: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Chat completion failed: {e}"
            ) from e

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3",
        **kwargs,  # Accept but ignore unused parameters
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion response through NeMo Guardrails.

        Args:
            messages: List of chat messages
            model: Model name
            **kwargs: Additional parameters (max_tokens, etc.)

        Yields:
            str: Streaming response chunks in SSE format

        Raises:
            HTTPException: If service not initialized or streaming fails.
        """
        if not self._is_initialized:
            raise HTTPException(
                status_code=500,
                detail="NeMo Guardrails not initialized. Call initialize() first.",
            )

        try:
            # Get the full response first (TODO: implement true streaming if supported)
            response = await self.chat_completion(
                messages=messages,
                model=model,
                **kwargs,
            )

            # Simulate streaming by yielding the response
            content = response["choices"][0]["message"]["content"]

            # Yield the streaming response in SSE format
            chunk_data = {
                "id": response["id"],
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            }

            yield f"data: {self._json_dumps(chunk_data)}\n\n"

            # Final chunk to indicate completion
            final_chunk = {
                "id": response["id"],
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            yield f"data: {self._json_dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in streaming chat completion: %s", e)
            # Yield error in SSE format
            error_data = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {self._json_dumps(error_data)}\n\n"

    def _extract_user_message(self, messages: list[dict[str, str]]) -> str:
        """Extract the user message from the messages list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                if user_message:
                    return user_message

        raise HTTPException(status_code=400, detail="No user message found in messages")

    def _calculate_usage(self, user_message: str, result: Optional[str]) -> dict:
        """Calculate token usage for the response."""
        return {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(result.split()) if result else 0,
            "total_tokens": len(user_message.split())
            + (len(result.split()) if result else 0),
        }

    def _json_dumps(self, data: dict) -> str:
        """Helper method for JSON serialization."""
        import json

        return json.dumps(data)

    def is_available(self) -> bool:
        """Check if NeMo Guardrails is available and initialized."""
        return self._is_initialized and self.rails is not None

    async def health_check(self) -> dict:
        try:
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "message": "NeMo Guardrails not initialized",
                }

            # Try a simple test message
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat_completion(messages=test_messages)

            return {
                "status": "healthy",
                "message": "NeMo Guardrails is working correctly",
                "test_response_length": len(
                    response["choices"][0]["message"]["content"]
                ),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}",
            }


# Global instance for use throughout the application
_nemo_service_instance: Optional[NemoService] = None


async def get_nemo_service() -> NemoService:
    """
    Get or create the global NemoService instance.

    Returns:
        NemoService: The initialized service instance
    """
    global _nemo_service_instance  # noqa: PLW0603

    if _nemo_service_instance is None:
        _nemo_service_instance = NemoService()
        await _nemo_service_instance.initialize()

    return _nemo_service_instance


# Legacy function name for backward compatibility
async def get_local_nemo_instance() -> NemoService:
    """
    Legacy function name for backward compatibility.

    Returns:
        NemoService: The initialized service instance
    """
    return await get_nemo_service()


async def test_nemo_service():
    """Test function for the NeMo Guardrails service."""
    logger.info("Testing NeMo Guardrails service...")

    try:
        nemo = await get_nemo_service()

        if not nemo.is_available():
            logger.error("NeMo Guardrails service not available")
            return False

        # Test a simple message
        test_messages = [{"role": "user", "content": "Hello, this is a test message."}]

        response = await nemo.chat_completion(messages=test_messages)
        logger.info("Test response: %s", response)

        # Test health check
        health = await nemo.health_check()
        logger.info("Health check: %s", health)

        return True

    except Exception as e:
        logger.error("Test failed: %s", e)
        return False


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_nemo_service())
