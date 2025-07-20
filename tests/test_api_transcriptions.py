"""Tests for /v1/audio/transcriptions endpoint."""

from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.v1 import transcriptions as trans_api
from app.main import app
from app.models.responses import TranscriptionResponse


def setup_function() -> None:
    """Override dependencies with mocks for each test."""
    mock_service = Mock()
    app.dependency_overrides[trans_api.get_transcription_service] = lambda: mock_service
    setup_function.default_size = app.state.settings.max_file_size
    app.state.settings.max_file_size = 1024
    trans_api.mock_service = mock_service


def teardown_function() -> None:
    """Clear overrides after each test."""
    app.dependency_overrides.clear()
    app.state.settings.max_file_size = setup_function.default_size


def test_transcription_json_success() -> None:
    """POST returns JSON transcription output."""
    dummy = TranscriptionResponse(
        text="hello", segments=None, words=None, language="en"
    )
    trans_api.mock_service.transcribe.return_value = dummy

    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    trans_api.mock_service.transcribe.assert_called_once()


def test_transcription_text_format() -> None:
    """Return plain text when requested."""
    trans_api.mock_service.transcribe.return_value = "hello"
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    data = {"response_format": "text"}
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    assert response.text == "hello"
    assert response.headers["content-type"].startswith("text/plain")


def test_transcription_file_too_large() -> None:
    """Reject files exceeding max size."""
    app.state.settings.max_file_size = 1
    files = {"file": ("big.wav", BytesIO(b"ab"), "audio/wav")}
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "file_too_large"
    trans_api.mock_service.transcribe.assert_not_called()
