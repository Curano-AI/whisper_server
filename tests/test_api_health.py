from datetime import datetime
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api import health as health_api
from app.core.exceptions import ResourceError
from app.main import app
from app.models.responses import (
    HealthCheckResponse,
    HealthStatus,
    ServiceHealth,
    SystemResources,
)


def setup_function() -> None:
    """Override dependency with a mock service for each test."""
    mock_service = Mock()
    app.dependency_overrides[health_api.get_health_service] = lambda: mock_service
    health_api.mock_service = mock_service


def teardown_function() -> None:
    """Clear dependency overrides after each test."""
    app.dependency_overrides.clear()


def _dummy_response() -> HealthCheckResponse:
    health = HealthStatus(
        status="healthy", timestamp=datetime.utcnow(), version="1", uptime_seconds=1.0
    )
    service = ServiceHealth(name="transcription", status="healthy")
    system = SystemResources(
        cpu_usage_percent=10.0,
        memory_usage_percent=20.0,
        memory_available_gb=5.0,
        disk_usage_percent=30.0,
        gpu_available=False,
        gpu_memory_usage_percent=None,
    )
    return HealthCheckResponse(
        health=health, services=[service], system=system, loaded_models=["small"]
    )


def test_healthcheck_success() -> None:
    """GET /healthcheck returns health info."""
    dummy = _dummy_response()
    health_api.mock_service.get_health.return_value = dummy

    with TestClient(app) as client:
        response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json()["health"]["status"] == "healthy"
    health_api.mock_service.get_health.assert_called_once()


def test_healthcheck_error() -> None:
    """Service error results in 503 response."""
    health_api.mock_service.get_health.side_effect = ResourceError("boom")

    with TestClient(app) as client:
        response = client.get("/healthcheck")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "resource_error"
