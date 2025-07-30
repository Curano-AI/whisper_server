from io import BytesIO
from unittest.mock import Mock, patch

import openai
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1 import transcriptions as trans_api
from app.core.config import get_settings
from app.main import app
from app.models.responses import TranscriptionResponse


@pytest.mark.asyncio
async def test_openai_client_transcription() -> None:
    """OpenAI client is compatible with our API."""
    settings = get_settings()
    service_mock = Mock()
    service_mock.transcribe.return_value = TranscriptionResponse(
        text="hello", segments=None, words=None, language="en"
    )
    overrides = {trans_api.get_transcription_service: lambda: service_mock}
    with patch.dict(app.dependency_overrides, overrides):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as async_client:
            api_key = (
                settings.api_key.get_secret_value()
                if settings.auth_enabled and settings.api_key
                else "test"
            )
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="http://testserver/v1",
                http_client=async_client,
            )
            result = await client.audio.transcriptions.create(
                file=("test.wav", BytesIO(b"data")), model="whisper-1"
            )

    assert result.text == "hello"
    service_mock.transcribe.assert_called_once()
