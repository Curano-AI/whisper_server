"""Application middleware components."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:  # pragma: no cover - type checking imports
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request details and response time."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        logger.info("%s %s", request.method, request.url.path)

        try:
            response = await call_next(request)
            duration = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s - %s %.2fms",
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )
            return response
        except Exception:
            duration = (time.perf_counter() - start) * 1000
            logger.exception("Request error")
            logger.info(
                "%s %s - %s %.2fms",
                request.method,
                request.url.path,
                500,
                duration,
            )
            raise
