"""Authentication and authorization dependencies."""

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import AppConfig
from app.core.exceptions import AuthenticationError

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

EXPECTED_AUTH_HEADER_PARTS = 2


def get_settings(request: Request) -> AppConfig:
    """Return application settings from app state."""
    return request.app.state.settings


def get_api_key(
    settings: AppConfig = Depends(get_settings),
    api_key: str = Security(api_key_header),
) -> str:
    """Validate API key from Authorization header."""
    if not settings.auth_enabled:
        return ""  # Skip auth if disabled

    if not api_key:
        raise AuthenticationError(
            "Authorization header is missing", error_code="auth_header_missing"
        )

    # Note: Authorization header is expected to be "Bearer <key>"
    key_parts = api_key.split()
    if len(key_parts) != EXPECTED_AUTH_HEADER_PARTS or key_parts[0].lower() != "bearer":
        raise AuthenticationError(
            "Invalid Authorization header format. Expected 'Bearer <key>'",
            error_code="invalid_auth_header_format",
        )

    token = key_parts[1]
    if not settings.api_key or token != settings.api_key.get_secret_value():
        raise AuthenticationError("Invalid API key", error_code="invalid_api_key")

    return token
