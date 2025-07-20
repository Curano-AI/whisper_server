from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def test_env_overrides(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("PORT", "1234")
    settings = get_settings()
    assert settings.port == 1234
    get_settings.cache_clear()


def test_security_headers() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthcheck")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("Referrer-Policy") == "no-referrer"
    assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
    assert (
        resp.headers.get("Strict-Transport-Security")
        == "max-age=63072000; includeSubDomains"
    )
    assert resp.headers.get("Content-Security-Policy") == "default-src 'self'"
