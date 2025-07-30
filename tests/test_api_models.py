"""Tests for model management API endpoints."""

from collections.abc import Generator
from unittest.mock import Mock

import pytest
import whisperx
from fastapi.testclient import TestClient

from app.api.v1 import models as models_api
from app.core.exceptions import ModelLoadError
from app.main import app
from app.services.model_manager import ModelManager


@pytest.fixture(autouse=True)
def setup_teardown() -> Generator[None, None, None]:
    """Override model manager dependency for each test and clear overrides."""
    mm = ModelManager()
    app.dependency_overrides[models_api.get_model_manager] = lambda: mm
    yield
    app.dependency_overrides.clear()


def test_list_models_empty(authenticated_client: TestClient) -> None:
    """GET /models/list returns empty when no models are loaded."""
    response = authenticated_client.get("/models/list")
    assert response.status_code == 200
    data = response.json()
    assert data["loaded_models"] == []
    assert data["total_memory_usage_mb"] is None


def test_load_model_success(
    authenticated_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /models/load loads a model and returns metadata."""
    mock_model = Mock()
    monkeypatch.setattr(whisperx, "load_model", lambda *_, **__: mock_model)
    response = authenticated_client.post("/models/load", json={"model_name": "small"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_info"]["model_name"] == "small"


def test_load_model_error(
    authenticated_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /models/load returns error when manager raises ModelLoadError."""

    def raise_error(*_args, **_kwargs):
        raise ModelLoadError("boom", model_name="bad", error_code="load_failed")

    mm = app.dependency_overrides[models_api.get_model_manager]()
    monkeypatch.setattr(mm, "load_model", raise_error)
    response = authenticated_client.post("/models/load", json={"model_name": "small"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "load_failed"


def test_unload_model_success(
    authenticated_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /models/unload unloads an existing model."""
    mm: ModelManager = app.dependency_overrides[models_api.get_model_manager]()
    # Load a dummy model using the public API instead of touching internals
    mock_model = Mock()
    monkeypatch.setattr(whisperx, "load_model", lambda *_, **__: mock_model)
    mm.load_model("small")

    response = authenticated_client.post("/models/unload", json={"model_name": "small"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "small" not in mm.list_models()
