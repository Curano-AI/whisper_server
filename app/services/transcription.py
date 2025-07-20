"""Core transcription orchestration service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.core.exceptions import TranscriptionError
from app.models.responses import (
    TranscriptionOutput,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
)
from app.services.audio_processor import AudioProcessor
from app.services.language_detector import LanguageDetector
from app.services.model_manager import ModelManager
from app.utils.transcribe_utils import clean, get_suppress_tokens, ts

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    from pathlib import Path

    from app.models.requests import TranscriptionRequest

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service orchestrating WhisperX transcription workflow."""

    def __init__(
        self,
        audio_processor: AudioProcessor | None = None,
        language_detector: LanguageDetector | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.settings = get_settings()
        self.audio_processor = audio_processor or AudioProcessor()
        self.language_detector = language_detector or LanguageDetector()
        self.model_manager = model_manager or ModelManager()

    # AICODE-NOTE: ASR and VAD option defaults come from transcribe.py via AppConfig
    def _build_asr_options(self, request: TranscriptionRequest) -> dict[str, Any]:
        opts = self.settings.get_asr_options()
        if request.beam_size is not None:
            opts["beam_size"] = request.beam_size
        if request.temperature is not None:
            opts["temperatures"] = [request.temperature]
        if request.suppress_tokens is not None:
            opts["suppress_tokens"] = get_suppress_tokens(request.suppress_tokens)
        else:
            opts["suppress_tokens"] = get_suppress_tokens(
                self.settings.suppress_phrases
            )
        return opts

    def _build_vad_options(self, request: TranscriptionRequest) -> dict[str, Any]:
        opts = self.settings.get_vad_options()
        if request.vad_options:
            opts.update(request.vad_options)
        return opts

    def _format_text(self, segments: list[dict[str, Any]]) -> str:
        return "\n".join(clean(seg["text"]) for seg in segments)

    def _format_srt(self, segments: list[dict[str, Any]]) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{ts(seg['start'])} --> {ts(seg['end'])}")
            lines.append(clean(seg["text"]))
            lines.append("")
        return "\n".join(lines)

    def _format_vtt(self, segments: list[dict[str, Any]]) -> str:
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(f"{ts(seg['start'])} --> {ts(seg['end'])}")
            lines.append(clean(seg["text"]))
            lines.append("")
        return "\n".join(lines)

    def _format_response(
        self,
        result: dict[str, Any],
        language: str,
        response_format: str,
    ) -> TranscriptionOutput:
        segments = result.get("segments", [])
        if response_format == "json":
            return TranscriptionResponse(
                text=clean(result.get("text", "")),
                segments=segments,
                words=result.get("word_segments"),
                language=language,
            )
        if response_format == "verbose_json":
            return VerboseTranscriptionResponse(
                task="transcribe",
                language=language,
                duration=result.get("duration", 0.0),
                text=clean(result.get("text", "")),
                segments=segments,
                words=result.get("word_segments"),
            )
        if response_format == "text":
            return self._format_text(segments)
        if response_format == "srt":
            return self._format_srt(segments)
        if response_format == "vtt":
            return self._format_vtt(segments)
        raise TranscriptionError(
            f"Unsupported response format: {response_format}",
            error_code="invalid_format",
            param="response_format",
        )

    def transcribe(
        self, file_path: str | Path, request: TranscriptionRequest
    ) -> TranscriptionOutput:
        """Run the full transcription pipeline."""
        try:
            # Audio preprocessing and language detection
            audio, _lead_ms, sample_paths = (
                self.audio_processor.process_audio_for_language_detection(file_path)
            )
            language, _, _votes, _prob_sum = self.language_detector.detect_from_samples(
                sample_paths
            )

            # Load model and run inference
            model = self.model_manager.load_model(request.model)

            asr_opts = self._build_asr_options(request)

            result = model.transcribe(
                str(file_path),
                batch_size=self.settings.batch_size,
                language=language,
                verbose=False,
                beam_size=asr_opts.get("beam_size", 2),
                condition_on_previous_text=asr_opts.get(
                    "condition_on_previous_text", False
                ),
                suppress_tokens=asr_opts.get("suppress_tokens", []),
                temperatures=asr_opts.get("temperatures", [0.0]),
                no_speech_threshold=asr_opts.get("no_speech_threshold", 0.6),
            )

            return self._format_response(
                result, language, request.response_format or "json"
            )
        except Exception as exc:  # pragma: no cover - safety net
            logger.error("Transcription failed: %s", exc)
            raise TranscriptionError(
                "Transcription failed", error_code="transcribe_failed"
            ) from exc
