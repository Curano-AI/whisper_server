"""Tests for model management API endpoints."""

from datetime import datetime
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.v1 import models as models_api
from app.core.exceptions import ModelLoadError
from app.main import app
from app.services.model_manager import ModelManager

client = TestClient(app)


def setup_function() -> None:
    """Reset model manager before each test."""
    models_api.model_manager = ModelManager()


def test_list_models_empty() -> None:
    """GET /models/list returns empty when no models are loaded."""
    response = client.get("/models/list")
    assert response.status_code == 200
    data = response.json()
    assert data["loaded_models"] == []
    assert data["total_memory_usage_mb"] is None


def test_load_model_success(monkeypatch) -> None:
    """POST /models/load loads a model and returns metadata."""
    mm = models_api.model_manager

    def fake_load(model_name: str, device=None, compute_type=None):
        mm._models[model_name] = {
            "model": Mock(),
            "device": device or mm.settings.device,
            "compute_type": compute_type
            or mm.settings.get_compute_type(device or mm.settings.device),
            "load_time": datetime.utcnow(),
            "last_used": datetime.utcnow(),
        }
        return mm._models[model_name]["model"]

    monkeypatch.setattr(mm, "load_model", fake_load)

    response = client.post("/models/load", json={"model_name": "small"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_info"]["model_name"] == "small"


def test_load_model_error(monkeypatch) -> None:
    """POST /models/load returns error when manager raises ModelLoadError."""

    def raise_error(*_args, **_kwargs):
        raise ModelLoadError("boom", model_name="bad", error_code="load_failed")

    monkeypatch.setattr(models_api.model_manager, "load_model", raise_error)
    response = client.post("/models/load", json={"model_name": "small"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "load_failed"


def test_unload_model_success() -> None:
    """POST /models/unload unloads an existing model."""
    mm = models_api.model_manager
    # Load a dummy model using the public API instead of touching internals
    mm.load_model("small")

    response = client.post("/models/unload", json={"model_name": "small"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "small" not in mm._models
