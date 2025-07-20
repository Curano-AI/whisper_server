"""Tests for model management API endpoints."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.v1 import models as models_api
from app.core.exceptions import ModelLoadError
from app.main import app
from app.services.model_manager import ModelManager


def setup_function() -> None:
    """Override model manager dependency for each test."""
    mm = ModelManager()
    app.dependency_overrides[models_api.get_model_manager] = lambda: mm


def teardown_function() -> None:
    """Clear overrides."""
    app.dependency_overrides.clear()


def test_list_models_empty() -> None:
    """GET /models/list returns empty when no models are loaded."""
    with TestClient(app) as client:
        response = client.get("/models/list")
    assert response.status_code == 200
    data = response.json()
    assert data["loaded_models"] == []
    assert data["total_memory_usage_mb"] is None


def test_load_model_success(monkeypatch) -> None:
    """POST /models/load loads a model and returns metadata."""
    mock_model = Mock()
    monkeypatch.setattr(
        "app.services.model_manager.whisperx.load_model", lambda *_, **__: mock_model
    )

    with TestClient(app) as client:
        response = client.post("/models/load", json={"model_name": "small"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_info"]["model_name"] == "small"


def test_load_model_error(monkeypatch) -> None:
    """POST /models/load returns error when manager raises ModelLoadError."""

    def raise_error(*_args, **_kwargs):
        raise ModelLoadError("boom", model_name="bad", error_code="load_failed")

    mm = app.dependency_overrides[models_api.get_model_manager]()
    monkeypatch.setattr(mm, "load_model", raise_error)
    with TestClient(app) as client:
        response = client.post("/models/load", json={"model_name": "small"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "load_failed"


def test_unload_model_success() -> None:
    """POST /models/unload unloads an existing model."""
    mm = app.dependency_overrides[models_api.get_model_manager]()
    # Load a dummy model using the public API instead of touching internals
    mm.load_model("small")

    with TestClient(app) as client:
        response = client.post("/models/unload", json={"model_name": "small"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "small" not in mm.list_models()
