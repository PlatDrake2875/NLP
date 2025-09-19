#!/usr/bin/env python3
"""
Local NeMo Guardrails integration module.

This module provides direct integration with NeMo Guardrails instead of using HTTP calls.
It handles the initialization and interaction with the guardrails system locally.
"""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger("nemo_local")


class LocalNemoGuardrails:
    """
    Local NeMo Guardrails integration that works directly with the library
    instead of making HTTP calls to a separate service.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the local NeMo Guardrails instance.

        Args:
            config_path: Path to the guardrails config directory.
                        Defaults to ./guardrails_config/mybot
        """
        self.config_path = config_path or self._get_default_config_path()
        self.rails: Optional[LLMRails] = None
        self._is_initialized = False

    def _get_default_config_path(self) -> str:
        """Get the default config path relative to this module."""
        current_dir = Path(__file__).parent
        config_dir = current_dir / "guardrails_config" / "mybot"
        return str(config_dir)

    async def initialize(self) -> bool:
        """
        Initialize the NeMo Guardrails system.

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            logger.info("Initializing NeMo Guardrails with programmatic config")

            # Determine the base URL based on environment
            # Use localhost for local development, host.docker.internal for Docker
            import os

            is_docker = os.path.exists("/.dockerenv")
            base_url = (
                "http://host.docker.internal:11434"
                if is_docker
                else "http://localhost:11434"
            )
            logger.info(f"Using Ollama base URL: {base_url}")

            # Create config programmatically instead of loading from file
            # This avoids the config parsing issues we're encountering
            config_dict = {
                "models": [
                    {
                        "type": "main",
                        "engine": "ollama",
                        "model": "gemma3:latest",
                        "parameters": {"base_url": base_url},
                    }
                ],
                "instructions": [
                    {"type": "general", "content": "You are a helpful AI assistant."}
                ],
            }

            # Create rails configuration from dict
            rails_config = RailsConfig.from_content(colang_content="", yaml_content="")
            rails_config.models = config_dict["models"]
            rails_config.instructions = config_dict["instructions"]

            logger.info("Rails config created programmatically")

            # Initialize the LLM Rails
            self.rails = LLMRails(rails_config)
            logger.info("LLM Rails initialized successfully")

            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize NeMo Guardrails: {e}")
            # Try fallback to minimal file-based config
            try:
                logger.info("Trying fallback initialization without guardrails config")
                # Determine the base URL for fallback too
                import os

                is_docker = os.path.exists("/.dockerenv")
                base_url = (
                    "http://host.docker.internal:11434"
                    if is_docker
                    else "http://localhost:11434"
                )

                # Create a minimal rails instance
                rails_config = RailsConfig.from_content(
                    colang_content="",
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
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {fallback_error}")
                return False

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3",
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Process a chat completion request through NeMo Guardrails.

        Args:
            messages: List of chat messages
            model: Model name (passed through to underlying LLM)
            stream: Whether to stream the response
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Dict: Chat completion response
        """
        if not self._is_initialized:
            raise RuntimeError(
                "NeMo Guardrails not initialized. Call initialize() first."
            )

        if not self.rails:
            raise RuntimeError("Rails not available")

        try:
            # Extract the user message (last message typically)
            user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            if not user_message:
                raise ValueError("No user message found in messages")

            logger.info(
                f"Processing message through guardrails: {user_message[:100]}..."
            )

            if stream:
                # For streaming, we'll need to handle it differently
                # For now, let's implement non-streaming and add streaming later
                result = await self.rails.generate_async(user_message)
            else:
                result = await self.rails.generate_async(user_message)

            # Format response to match OpenAI-style response
            response = {
                "id": f"chatcmpl-local-{hash(user_message)}",
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(result.split()) if result else 0,
                    "total_tokens": len(user_message.split())
                    + (len(result.split()) if result else 0),
                },
            }

            logger.info(f"Generated response: {result[:100] if result else 'None'}...")
            return response

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3",
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion response through NeMo Guardrails.

        Args:
            messages: List of chat messages
            model: Model name
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            str: Streaming response chunks in SSE format
        """
        if not self._is_initialized:
            raise RuntimeError(
                "NeMo Guardrails not initialized. Call initialize() first."
            )

        # For now, implement streaming by getting the full response and yielding it
        # TODO: Implement true streaming if NeMo Guardrails supports it
        try:
            response = await self.chat_completion(
                messages=messages,
                model=model,
                stream=False,
                max_tokens=max_tokens,
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

            yield f"data: {__import__('json').dumps(chunk_data)}\n\n"

            # Final chunk to indicate completion
            final_chunk = {
                "id": response["id"],
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            yield f"data: {__import__('json').dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            # Yield error in SSE format
            error_data = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {__import__('json').dumps(error_data)}\n\n"

    def is_available(self) -> bool:
        """Check if NeMo Guardrails is available and initialized."""
        return self._is_initialized and self.rails is not None


# Global instance for use throughout the application
_local_nemo_instance: Optional[LocalNemoGuardrails] = None


async def get_local_nemo_instance() -> LocalNemoGuardrails:
    """
    Get or create the global LocalNemoGuardrails instance.

    Returns:
        LocalNemoGuardrails: The initialized instance
    """
    global _local_nemo_instance

    if _local_nemo_instance is None:
        _local_nemo_instance = LocalNemoGuardrails()
        await _local_nemo_instance.initialize()

    return _local_nemo_instance


async def test_local_nemo():
    """Test function for the local NeMo integration."""
    logger.info("Testing local NeMo Guardrails integration...")

    try:
        nemo = await get_local_nemo_instance()

        if not nemo.is_available():
            logger.error("NeMo Guardrails not available")
            return False

        # Test a simple message
        test_messages = [{"role": "user", "content": "Hello, this is a test message."}]

        response = await nemo.chat_completion(messages=test_messages)
        logger.info(f"Test response: {response}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_local_nemo())
