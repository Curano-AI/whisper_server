"""Dependency provider functions for FastAPI."""

from fastapi import Request

from app.core.config import AppConfig
from app.services import HealthService, ModelManager, TranscriptionService


def get_settings(request: Request) -> AppConfig:
    """Return application settings from app state."""
    return request.app.state.settings


def get_model_manager(request: Request) -> ModelManager:
    """Return the shared :class:`ModelManager` instance."""
    return request.app.state.model_manager


def get_transcription_service(request: Request) -> TranscriptionService:
    """Return the shared :class:`TranscriptionService` instance."""
    return request.app.state.transcription_service


def get_health_service(request: Request) -> HealthService:
    """Return the shared :class:`HealthService` instance."""
    return request.app.state.health_service
