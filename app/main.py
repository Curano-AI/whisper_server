"""FastAPI application initialization and configuration."""

import logging

from fastapi import FastAPI

from app.api import health
from app.api.v1 import models, transcriptions
from app.core.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import LoggingMiddleware

# Initialize logging
setup_logging()

# Get application settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title="WhisperX FastAPI Server",
    description="OpenAI-compatible audio transcription service using WhisperX",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register middleware and exception handlers
app.add_middleware(LoggingMiddleware)
register_exception_handlers(app)

# Include API routers
app.include_router(transcriptions.router, prefix="/v1/audio", tags=["transcriptions"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(health.router, tags=["health"])


@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger = logging.getLogger(__name__)
    logger.info("WhisperX FastAPI Server starting up...")
    logger.info(f"Configuration: {settings.model_dump()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    logger = logging.getLogger(__name__)
    logger.info("WhisperX FastAPI Server shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
