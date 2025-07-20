import time
from io import BytesIO
from unittest.mock import Mock

from fastapi import UploadFile

from app.models.requests import TranscriptionRequest
from app.services.transcription import TranscriptionService


def test_transcription_speed() -> None:
    """Transcription pipeline completes quickly with mocks."""
    service = TranscriptionService(
        audio_processor=Mock(), language_detector=Mock(), model_manager=Mock()
    )
    service.audio_processor.process_audio_for_language_detection.return_value = (
        None,
        0,
        [],
    )
    service.language_detector.detect_from_samples.return_value = ("en", 1.0, {}, {})
    model = Mock()
    model.transcribe.return_value = {"text": "hi", "segments": [], "duration": 0.1}
    service.model_manager.load_model.return_value = model

    req = TranscriptionRequest(file=UploadFile(filename="a.wav", file=BytesIO(b"d")))

    start = time.perf_counter()
    service.transcribe("a.wav", req)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5
