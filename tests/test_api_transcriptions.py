"""Tests for /v1/audio/transcriptions endpoint."""

from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import transcriptions as trans_api
from app.main import app
from app.models.responses import TranscriptionResponse


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

    with TestClient(app) as test_client:
        yield test_client

    # Restore original settings and clear overrides
    app.state.settings.max_file_size = original_size
    app.dependency_overrides.clear()


def test_transcription_json_success(client, mock_transcription_service) -> None:
    """POST returns JSON transcription output."""
    dummy = TranscriptionResponse(
        text="hello", segments=None, words=None, language="en"
    )
    mock_transcription_service.transcribe.return_value = dummy

    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    mock_transcription_service.transcribe.assert_called_once()


def test_transcription_text_format(client, mock_transcription_service) -> None:
    """Return plain text when requested."""
    mock_transcription_service.transcribe.return_value = "hello"
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    data = {"response_format": "text"}
    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    assert response.text == "hello"
    assert response.headers["content-type"].startswith("text/plain")


def test_transcription_file_too_large(client, mock_transcription_service) -> None:
    """Reject files exceeding max size."""
    app.state.settings.max_file_size = 1
    files = {"file": ("big.wav", BytesIO(b"ab"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "file_too_large"
    mock_transcription_service.transcribe.assert_not_called()
