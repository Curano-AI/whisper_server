"""Transcription API endpoints."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import Response

from app.core.exceptions import ValidationError
from app.dependencies import get_settings, get_transcription_service
from app.models.requests import TranscriptionRequest
from app.models.responses import ErrorResponse, TranscriptionOutput

if TYPE_CHECKING:  # AICODE-NOTE: avoid runtime import cost
    from app.core.config import AppConfig
    from app.services import TranscriptionService

router = APIRouter()


@router.post(
    "/transcriptions",
    response_model=TranscriptionOutput,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="whisper-1", description="ID of the model to use"),
    language: str | None = Form(
        default=None, description="Language of the input audio (ISO-639-1 format)"
    ),
    prompt: str | None = Form(
        default=None, max_length=244, description="Optional text to guide style"
    ),
    response_format: str = Form(
        default="json", description="Format of the transcript output"
    ),
    temperature: float = Form(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature 0-1"
    ),
    timestamp_granularities: list[str] = Form(
        default=["segment"], description="Timestamp granularities to populate"
    ),
    beam_size: int = Form(default=2, ge=1, le=10, description="Beam size for decoding"),
    suppress_tokens: list[str] | None = Form(
        default=None, description="List of phrases to suppress"
    ),
    vad_options: dict[str, Any] | None = Form(
        default=None, description="Voice Activity Detection options"
    ),
    service: TranscriptionService = Depends(get_transcription_service),
    settings: AppConfig = Depends(get_settings),
) -> Response | TranscriptionOutput:
    """Create transcription from uploaded audio file."""

    allowed_formats = ["json", "text", "srt", "verbose_json", "vtt"]
    if response_format not in allowed_formats:
        raise ValidationError(
            f"response_format must be one of {allowed_formats}",
            param="response_format",
            error_code="invalid_format",
        )

    allowed_models = [
        "whisper-1",
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
    ]
    if model not in allowed_models:
        raise ValidationError(
            f"Model must be one of {allowed_models}",
            param="model",
            error_code="invalid_model",
        )

    allowed_granularities = ["word", "segment"]
    for granularity in timestamp_granularities:
        if granularity not in allowed_granularities:
            raise ValidationError(
                f"timestamp_granularities must contain only {allowed_granularities}",
                param="timestamp_granularities",
                error_code="invalid_granularity",
            )

    # Create request object from parsed parameters
    request = TranscriptionRequest(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        beam_size=beam_size,
        suppress_tokens=suppress_tokens,
        vad_options=vad_options,
    )

    data = await request.file.read()
    if len(data) > settings.max_file_size:
        err = ValidationError(
            f"File too large. Limit {settings.max_file_size} bytes",
            param="file",
            error_code="file_too_large",
        )
        err.status_code = status.HTTP_400_BAD_REQUEST
        raise err

    suffix = Path(request.file.filename or "default.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        output = service.transcribe(tmp_path, request)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)

    if isinstance(output, str):
        media_type = "text/plain"
        if request.response_format == "srt":
            media_type = "text/srt"
        elif request.response_format == "vtt":
            media_type = "text/vtt"
        return Response(content=output, media_type=media_type)

    return output
