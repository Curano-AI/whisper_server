from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import transcriptions as trans_api
from app.core.exceptions import TranscriptionError
from app.main import app


@pytest.fixture
def mock_transcription_service():
    """Create a mock transcription service for testing."""
    return Mock()


@pytest.fixture
def client(mock_transcription_service):
    """Create test client with mocked transcription service dependency."""
    app.dependency_overrides[trans_api.get_transcription_service] = (
        lambda: mock_transcription_service
    )

    # Store and set test max file size
    original_size = app.state.settings.max_file_size
    app.state.settings.max_file_size = 1024

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    # Restore original settings and clear overrides
    app.state.settings.max_file_size = original_size
    app.dependency_overrides.clear()


def test_transcription_error_handled(client, mock_transcription_service) -> None:
    mock_transcription_service.transcribe.side_effect = TranscriptionError(
        "boom", error_code="failed"
    )
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "transcription_error"
    assert data["error"]["code"] == "failed"


def test_unhandled_exception_returns_server_error(
    client, mock_transcription_service
) -> None:
    mock_transcription_service.transcribe.side_effect = Exception("oops")
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "server_error"
