"""Exception handler registration for FastAPI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

if TYPE_CHECKING:  # pragma: no cover - type checking imports
    from fastapi import FastAPI, Request

from app.core.exceptions import WhisperXAPIException

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the application."""

    @app.exception_handler(WhisperXAPIException)
    async def handle_whisperx_exception(
        _request: Request, exc: WhisperXAPIException
    ) -> JSONResponse:
        logger.error("%s: %s", exc.error_type, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        generic = WhisperXAPIException(
            "Internal server error", error_type="server_error", status_code=500
        )
        return JSONResponse(status_code=500, content=generic.to_dict())
