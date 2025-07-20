"""Health check API endpoints."""

from fastapi import APIRouter, Depends

from app.models.responses import ErrorResponse, HealthCheckResponse
from app.services import HealthService

router = APIRouter()


def get_health_service() -> HealthService:
    """Create a new :class:`HealthService` instance."""
    return HealthService()


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
