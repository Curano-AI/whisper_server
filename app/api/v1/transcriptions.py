"""Transcription API endpoints."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response

from app.core.config import AppConfig
from app.core.exceptions import ValidationError
from app.dependencies import get_settings, get_transcription_service
from app.models.requests import TranscriptionRequest  # noqa: TC001
from app.models.responses import ErrorResponse, TranscriptionOutput
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
    request: TranscriptionRequest = Depends(),
    service: TranscriptionService = Depends(get_transcription_service),
    settings: AppConfig = Depends(get_settings),
) -> Response | TranscriptionOutput:
    """Create transcription from uploaded audio file."""

    data = await request.file.read()
    if len(data) > settings.max_file_size:
        err = ValidationError(
            f"File too large. Limit {settings.max_file_size} bytes",
            param="file",
            error_code="file_too_large",
        )
        err.status_code = status.HTTP_400_BAD_REQUEST
        raise err

    suffix = Path(request.file.filename).suffix or ".wav"
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
