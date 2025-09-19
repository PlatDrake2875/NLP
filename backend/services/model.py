"""
Model service layer containing all model-related business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Optional

import ollama
from fastapi import HTTPException

from config import OLLAMA_BASE_URL
from schemas import OllamaModelInfo


class ModelService:
    """Service class handling all model-related business logic."""

    def __init__(self):
        self.ollama_base_url = OLLAMA_BASE_URL

    async def list_available_models(self) -> list[OllamaModelInfo]:
        """
        Get list of available Ollama models.

        Returns:
            List of OllamaModelInfo objects

        Raises:
            HTTPException: If there's an error communicating with Ollama
        """
        client = ollama.AsyncClient(host=self.ollama_base_url)
        ollama_list_response = await client.list()

        if not hasattr(ollama_list_response, "models"):
            raise HTTPException(
                status_code=500,
                detail="Invalid response structure from Ollama (missing 'models' attribute in ListResponse).",
            )

        models_from_ollama_lib = ollama_list_response.models

        if not isinstance(models_from_ollama_lib, list):
            raise HTTPException(
                status_code=500,
                detail="Invalid data type for 'models' in Ollama ListResponse.",
            )

        parsed_models = []
        for ollama_model_obj in models_from_ollama_lib:
            model_info = self._parse_ollama_model(ollama_model_obj)
            if model_info:
                parsed_models.append(model_info)

        return parsed_models

    def _parse_ollama_model(self, ollama_model_obj) -> Optional[OllamaModelInfo]:
        """
        Parse an Ollama model object into OllamaModelInfo.

        Args:
            ollama_model_obj: The model object from Ollama

        Returns:
            OllamaModelInfo instance or None if parsing fails
        """

        model_data_dict = ollama_model_obj.model_dump()

        return OllamaModelInfo.from_ollama(model_data_dict)

    def _get_valid_status_code(self, status_code: int) -> int:
        """
        Ensure status code is valid, return 500 if not.

        Args:
            status_code: The status code to validate

        Returns:
            Valid HTTP status code
        """
        if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
            return 500
        return status_code
