from io import BytesIO
from unittest.mock import Mock

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
    trans_api.service = Mock()
    trans_api.service.transcribe.return_value = TranscriptionResponse(
        text="hello", segments=None, words=None, language="en"
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as async_client:
        client = openai.AsyncOpenAI(
            api_key="test", base_url="http://testserver/v1", http_client=async_client
        )
        result = await client.audio.transcriptions.create(
            file=("test.wav", BytesIO(b"data")), model="whisper-1"
        )

    assert result.text == "hello"
    trans_api.service.transcribe.assert_called_once()
