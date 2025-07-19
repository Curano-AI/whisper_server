"""Tests for /v1/audio/transcriptions endpoint."""

from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.v1 import transcriptions as trans_api
from app.main import app
from app.models.responses import TranscriptionResponse

client = TestClient(app)


def setup_function() -> None:
    """Reset service instance and settings before each test."""
    trans_api.service = Mock()
    trans_api.settings.max_file_size = 1024


def test_transcription_json_success() -> None:
    """POST returns JSON transcription output."""
    dummy = TranscriptionResponse(
        text="hello", segments=None, words=None, language="en"
    )
    trans_api.service.transcribe.return_value = dummy

    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    trans_api.service.transcribe.assert_called_once()


def test_transcription_text_format() -> None:
    """Return plain text when requested."""
    trans_api.service.transcribe.return_value = "hello"
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    data = {"response_format": "text"}
    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    assert response.text == "hello"
    assert response.headers["content-type"].startswith("text/plain")


def test_transcription_file_too_large() -> None:
    """Reject files exceeding max size."""
    trans_api.settings.max_file_size = 1
    files = {"file": ("big.wav", BytesIO(b"ab"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "file_too_large"
    trans_api.service.transcribe.assert_not_called()
