import sys
import types

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

# Stub whisperx module
whisperx = types.ModuleType("whisperx")


def _load_model_stub(*_args, **_kwargs):
    return object()


whisperx.load_model = _load_model_stub  # type: ignore[attr-defined]
sys.modules.setdefault("whisperx", whisperx)

# Stub torch module with cuda property
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
    is_available=lambda: False, empty_cache=lambda: None
)
sys.modules.setdefault("torch", torch)

# Stub openai whisper tokenizer module
whisper = types.ModuleType("whisper")
whisper_tokenizer = types.ModuleType("whisper.tokenizer")


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 255 for c in text]


def _get_tokenizer_stub(*_args, **_kwargs):
    return _DummyTokenizer()


whisper_tokenizer.get_tokenizer = _get_tokenizer_stub  # type: ignore[attr-defined]

whisper.tokenizer = whisper_tokenizer  # type: ignore[attr-defined]
sys.modules.setdefault("whisper", whisper)
sys.modules.setdefault("whisper.tokenizer", whisper_tokenizer)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def authenticated_client() -> TestClient:
    settings = get_settings()
    client = TestClient(app, raise_server_exceptions=False)
    if settings.auth_enabled and settings.api_key:
        api_key = settings.api_key.get_secret_value()
        client.headers["Authorization"] = f"Bearer {api_key}"
    return client
