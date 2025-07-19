"""Health check API endpoints."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.exceptions import ResourceError
from app.models.responses import ErrorDetail, ErrorResponse, HealthCheckResponse
from app.services import HealthService

router = APIRouter()

# Global service instance used by all requests
service = HealthService()


@router.get(
    "/healthcheck",
    response_model=HealthCheckResponse,
    responses={503: {"model": ErrorResponse}},
)
async def health_check() -> JSONResponse | HealthCheckResponse:
    """Return application health status."""
    try:
        return service.get_health()
    except ResourceError as exc:
        error = ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.error_code,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error.model_dump(),
        )
