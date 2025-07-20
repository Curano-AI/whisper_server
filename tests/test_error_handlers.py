from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.v1 import transcriptions as trans_api
from app.core.exceptions import TranscriptionError
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function() -> None:
    trans_api.service = Mock()
    trans_api.settings.max_file_size = 1024


def test_transcription_error_handled() -> None:
    trans_api.service.transcribe.side_effect = TranscriptionError(
        "boom", error_code="failed"
    )
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "transcription_error"
    assert data["error"]["code"] == "failed"


def test_unhandled_exception_returns_server_error() -> None:
    trans_api.service.transcribe.side_effect = Exception("oops")
    files = {"file": ("test.wav", BytesIO(b"data"), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "server_error"
