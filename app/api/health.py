"""Health check API endpoints."""

from fastapi import APIRouter, Depends

from app.dependencies import get_health_service
from app.models.responses import ErrorResponse, HealthCheckResponse
from app.services import HealthService

router = APIRouter()


@router.get(
    "/healthcheck",
    response_model=HealthCheckResponse,
    responses={503: {"model": ErrorResponse}},
)
async def health_check(
    service: HealthService = Depends(get_health_service),
) -> HealthCheckResponse:
    """Return application health status."""
    return service.get_health()
