"""
Dependency injection setup for the FastAPI application.
Provides service instances to endpoints.
"""

from functools import lru_cache

from backend.services.automate import AutomateService
from backend.services.chat import ChatService
from backend.services.document import DocumentService
from backend.services.health import HealthService
from backend.services.model import ModelService
from backend.services.upload import UploadService


@lru_cache
def get_chat_service() -> ChatService:
    """
    Get a ChatService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return ChatService()


@lru_cache
def get_model_service() -> ModelService:
    """
    Get a ModelService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return ModelService()


@lru_cache
def get_upload_service() -> UploadService:
    """
    Get an UploadService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return UploadService()


@lru_cache
def get_document_service() -> DocumentService:
    """
    Get a DocumentService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return DocumentService()


@lru_cache
def get_health_service() -> HealthService:
    """
    Get a HealthService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return HealthService()


@lru_cache
def get_automate_service() -> AutomateService:
    """
    Get an AutomateService instance.
    Using lru_cache to ensure singleton behavior during request lifecycle.
    """
    return AutomateService()
