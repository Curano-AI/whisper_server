import sys
import types

# Stub whisperx module
whisperx = types.ModuleType("whisperx")


def _load_model_stub(*_args, **_kwargs):
    return object()


whisperx.load_model = _load_model_stub
sys.modules.setdefault("whisperx", whisperx)

# Stub torch module with cuda property
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
sys.modules.setdefault("torch", torch)

# Stub openai whisper tokenizer module
whisper = types.ModuleType("whisper")
whisper_tokenizer = types.ModuleType("whisper.tokenizer")


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 255 for c in text]


def _get_tokenizer_stub(*_args, **_kwargs):
    return _DummyTokenizer()


whisper_tokenizer.get_tokenizer = _get_tokenizer_stub

whisper.tokenizer = whisper_tokenizer
sys.modules.setdefault("whisper", whisper)
sys.modules.setdefault("whisper.tokenizer", whisper_tokenizer)
