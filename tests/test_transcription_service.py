from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from app.models.requests import TranscriptionRequest
from app.services.transcription import TranscriptionService


class DummyModel:
    def __init__(self, result: dict):
        self.result = result
        self.called_with = None

    def transcribe(self, *args, **kwargs):
        self.called_with = (args, kwargs)
        return self.result


@pytest.fixture
def service():
    return TranscriptionService(
        audio_processor=Mock(),
        language_detector=Mock(),
        model_manager=Mock(),
    )


def test_transcribe_json(service):
    service.audio_processor.process_audio_for_language_detection.return_value = (
        None,
        0,
        ["chunk1.wav", "chunk2.wav"],
    )
    service.language_detector.detect_from_samples.return_value = (
        "en",
        0.9,
        {"en": 2},
        {"en": 1.8},
    )
    dummy_result = {
        "text": "hello дима торжок world",
        "segments": [
            {
                "id": 0,
                "seek": 0.0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "tokens": [1, 2],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            },
            {
                "id": 1,
                "seek": 0.0,
                "start": 1.0,
                "end": 2.0,
                "text": "world",
                "tokens": [3, 4],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            },
        ],
        "word_segments": None,
        "duration": 2.0,
    }
    model = DummyModel(dummy_result)
    service.model_manager.load_model.return_value = model

    req = TranscriptionRequest(
        file=UploadFile(filename="test.wav", file=BytesIO(b"data"))
    )
    output = service.transcribe("test.wav", req)

    assert output.text == "hello world"
    assert isinstance(output.segments, list)
    assert model.called_with[1]["language"] == "en"


def test_transcribe_formats(service):
    service.audio_processor.process_audio_for_language_detection.return_value = (
        None,
        0,
        ["chunk.wav"],
    )
    service.language_detector.detect_from_samples.return_value = ("en", 1.0, {}, {})
    result = {
        "text": "hello world",
        "segments": [
            {
                "id": 0,
                "seek": 0.0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "tokens": [1],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            },
            {
                "id": 1,
                "seek": 0.0,
                "start": 1.0,
                "end": 2.0,
                "text": "world",
                "tokens": [2],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            },
        ],
        "duration": 2.0,
    }
    model = DummyModel(result)
    service.model_manager.load_model.return_value = model

    for fmt in ["text", "srt", "vtt", "verbose_json"]:
        req = TranscriptionRequest(
            file=UploadFile(filename="a.wav", file=BytesIO(b"data")),
            response_format=fmt,
        )
        out = service.transcribe("a.wav", req)
        assert out

    with pytest.raises(ValidationError):
        req = TranscriptionRequest(
            file=UploadFile(filename="a.wav", file=BytesIO(b"data")),
            response_format="bad",
        )
        service.transcribe("a.wav", req)
