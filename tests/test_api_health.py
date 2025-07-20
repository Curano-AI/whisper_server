from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
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


@pytest.fixture
def mock_health_service():
    """Create a mock health service for testing."""
    return Mock()


@pytest.fixture
def client(mock_health_service):
    """Create test client with mocked health service dependency."""
    app.dependency_overrides[health_api.get_health_service] = (
        lambda: mock_health_service
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _dummy_response() -> HealthCheckResponse:
    health = HealthStatus(
        status="healthy", timestamp=datetime.now(UTC), version="1", uptime_seconds=1.0
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


def test_healthcheck_success(client, mock_health_service) -> None:
    """GET /healthcheck returns health info."""
    dummy = _dummy_response()
    mock_health_service.get_health.return_value = dummy

    response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json()["health"]["status"] == "healthy"
    mock_health_service.get_health.assert_called_once()


def test_healthcheck_error(client, mock_health_service) -> None:
    """Service error results in 503 response."""
    mock_health_service.get_health.side_effect = ResourceError("boom")

    response = client.get("/healthcheck")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "resource_error"
