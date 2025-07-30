import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)


def test_health_check_unauthenticated() -> None:
    """Health check should be accessible without authentication."""
    response = client.get("/healthcheck")
    assert response.status_code == 200


def test_transcription_unauthenticated_failure() -> None:
    """Transcription endpoint should fail without authentication when enabled."""
    settings = get_settings()
    if not settings.auth_enabled:
        pytest.skip("Authentication is not enabled")

    response = client.post("/v1/audio/transcriptions")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth_header_missing"


def test_transcription_with_invalid_key() -> None:
    """Transcription endpoint should fail with an invalid API key."""
    settings = get_settings()
    if not settings.auth_enabled or not settings.api_key:
        pytest.skip("Authentication is not enabled or API key is not set")

    headers = {"Authorization": "Bearer invalid-key"}
    # This requires a file, but we expect it to fail before processing the file.
    # The API will return a 401 before a 422 for the missing file.
    response = client.post("/v1/audio/transcriptions", headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_transcription_with_valid_key_but_no_file() -> None:
    """Test that a valid key allows access but fails on validation."""
    settings = get_settings()
    if not settings.auth_enabled or not settings.api_key:
        pytest.skip("Authentication is not enabled or API key is not set")

    api_key = settings.api_key.get_secret_value()
    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.post("/v1/audio/transcriptions", headers=headers)

    # We expect a 422 Unprocessable Entity error because no file is provided,
    # which confirms that authentication was successful.
    assert response.status_code == 422


def test_models_list_unauthenticated_failure():
    """Models endpoint should fail without authentication when enabled."""
    settings = get_settings()
    if not settings.auth_enabled:
        pytest.skip("Authentication is not enabled")

    response = client.get("/models/list")
    assert response.status_code == 401


def test_models_list_with_valid_key():
    """Models endpoint should succeed with a valid API key."""
    settings = get_settings()
    if not settings.auth_enabled or not settings.api_key:
        pytest.skip("Authentication is not enabled or API key is not set")

    api_key = settings.api_key.get_secret_value()
    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.get("/models/list", headers=headers)
    assert response.status_code == 200
