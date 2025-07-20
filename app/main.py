"""FastAPI application initialization and configuration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.api.v1 import models, transcriptions
from app.core.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import LoggingMiddleware
from app.core.security import SecurityHeadersMiddleware
from app.services import (
    AudioProcessor,
    LanguageDetector,
    ModelManager,
    TranscriptionService,
)
from app.services.health import HealthService

# Initialize logging
setup_logging()

# Get application settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger = logging.getLogger(__name__)

    # Startup
    logger.info("WhisperX FastAPI Server starting up...")
    logger.info(f"Configuration: {settings.model_dump()}")

    app.state.settings = settings
    app.state.model_manager = ModelManager()
    app.state.transcription_service = TranscriptionService(
        audio_processor=AudioProcessor(),
        language_detector=LanguageDetector(),
        model_manager=app.state.model_manager,
    )
    app.state.health_service = HealthService(app.state.model_manager)

    # Preload default model and perform health check
    try:
        app.state.model_manager.load_model(settings.default_model)
    except Exception as exc:
        logger.exception(
            "Failed to load default model '%s': %s", settings.default_model, exc
        )
        raise SystemExit(1) from exc

    try:
        app.state.health_service.get_health()
    except Exception as exc:
        logger.exception("Startup health check failed: %s", exc)
        raise SystemExit(1) from exc

    yield

    # Shutdown
    logger.info("WhisperX FastAPI Server shutting down...")
    manager: ModelManager | None = getattr(app.state, "model_manager", None)
    if manager:
        manager.clear()


# Create FastAPI application
app = FastAPI(
    title="WhisperX FastAPI Server",
    description="OpenAI-compatible audio transcription service using WhisperX",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Register middleware and exception handlers
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
register_exception_handlers(app)

# Include API routers
app.include_router(transcriptions.router, prefix="/v1/audio", tags=["transcriptions"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(health.router, tags=["health"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
