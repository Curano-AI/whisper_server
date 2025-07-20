from io import BytesIO
from unittest.mock import Mock, patch

import openai
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1 import transcriptions as trans_api
from app.main import app
from app.models.responses import TranscriptionResponse


@pytest.mark.asyncio
async def test_openai_client_transcription() -> None:
    """OpenAI client is compatible with our API."""
    # Override service so we don't hit real model logic
    with patch.object(trans_api, "service", Mock()) as service_mock:
        service_mock.transcribe.return_value = TranscriptionResponse(
            text="hello", segments=None, words=None, language="en"
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as async_client:
            client = openai.AsyncOpenAI(
                api_key="test",
                base_url="http://testserver/v1",
                http_client=async_client,
            )
            result = await client.audio.transcriptions.create(
                file=("test.wav", BytesIO(b"data")), model="whisper-1"
            )

        assert result.text == "hello"
        service_mock.transcribe.assert_called_once()
