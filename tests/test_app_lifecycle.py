from fastapi.testclient import TestClient

from app.main import app


def test_startup_loads_default_model() -> None:
    """Default model is loaded on startup and cleared on shutdown."""
    with TestClient(app) as client:
        assert app.state.model_manager.list_models() == [
            app.state.settings.default_model
        ]
    assert app.state.model_manager.list_models() == []
