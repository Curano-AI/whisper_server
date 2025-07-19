"""Transcription API endpoints."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.exceptions import AudioProcessingError, TranscriptionError
from app.models.requests import TranscriptionRequest  # noqa: TCH001
from app.models.responses import ErrorDetail, ErrorResponse, TranscriptionOutput
from app.services import TranscriptionService

router = APIRouter()

# Global service instance used by all requests
service = TranscriptionService()
settings = get_settings()


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
) -> Response | JSONResponse | TranscriptionOutput:
    """Create transcription from uploaded audio file."""

    data = await request.file.read()
    if len(data) > settings.max_file_size:
        error = ErrorResponse(
            error=ErrorDetail(
                message=f"File too large. Limit {settings.max_file_size} bytes",
                type="invalid_request_error",
                param="file",
                code="file_too_large",
            )
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content=error.model_dump()
        )

    suffix = Path(request.file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        output = service.transcribe(tmp_path, request)
    except (AudioProcessingError, TranscriptionError) as exc:
        error = ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.error_code,
            )
        )
        return JSONResponse(status_code=exc.status_code, content=error.model_dump())
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
