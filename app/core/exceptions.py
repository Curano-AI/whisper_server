"""Custom exception classes for error handling."""

from typing import Any


class WhisperXAPIException(Exception):
    """Base exception for WhisperX API."""

    def __init__(
        self,
        message: str,
        error_type: str = "api_error",
        error_code: str | None = None,
        param: str | None = None,
        status_code: int = 500,
    ):
        self.message = message
        self.error_type = error_type
        self.error_code = error_code
        self.param = param
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to OpenAI-compatible error dictionary."""
        error_dict = {
            "message": self.message,
            "type": self.error_type,
        }

        if self.error_code:
            error_dict["code"] = self.error_code

        if self.param:
            error_dict["param"] = self.param

        return {"error": error_dict}


class AudioProcessingError(WhisperXAPIException):
    """Audio file processing errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        param: str | None = None,
    ):
        super().__init__(
            message=message,
            error_type="audio_processing_error",
            error_code=error_code,
            param=param,
            status_code=400,
        )


class ModelLoadError(WhisperXAPIException):
    """Model loading/management errors."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        error_code: str | None = None,
    ):
        super().__init__(
            message=message,
            error_type="model_load_error",
            error_code=error_code,
            param="model" if model_name else None,
            status_code=500,
        )
        self.model_name = model_name


class TranscriptionError(WhisperXAPIException):
    """Transcription processing errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        param: str | None = None,
    ):
        super().__init__(
            message=message,
            error_type="transcription_error",
            error_code=error_code,
            param=param,
            status_code=500,
        )


class ValidationError(WhisperXAPIException):
    """Request validation errors."""

    def __init__(
        self,
        message: str,
        param: str | None = None,
        error_code: str | None = None,
    ):
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            error_code=error_code,
            param=param,
            status_code=422,
        )


class ResourceError(WhisperXAPIException):
    """System resource errors."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(
            message=message,
            error_type="resource_error",
            error_code=error_code,
            status_code=503,
        )


class AuthenticationError(WhisperXAPIException):
    """Authentication errors."""

    def __init__(
        self,
        message: str = "Invalid authentication credentials",
        error_code: str | None = None,
    ):
        super().__init__(
            message=message,
            error_type="authentication_error",
            error_code=error_code,
            status_code=401,
        )


class RateLimitError(WhisperXAPIException):
    """Rate limiting errors."""

    def __init__(
        self, message: str = "Rate limit exceeded", error_code: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="rate_limit_error",
            error_code=error_code,
            status_code=429,
        )
