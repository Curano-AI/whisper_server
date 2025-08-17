"""Speaker diarization service using pyannote.audio."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.core.exceptions import TranscriptionError

if TYPE_CHECKING:
    from pathlib import Path

    from app.models.requests import TranscriptionRequest

logger = logging.getLogger(__name__)


class DiarizationService:
    """Service for speaker diarization using pyannote.audio."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pipeline = None
        self._device = None

    def _get_hf_token(self, request: TranscriptionRequest) -> str:
        """Get HuggingFace token from request or environment."""
        token = (
            request.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        )
        if not token:
            raise TranscriptionError(
                "HuggingFace token required for diarization. "
                "Provide hf_token parameter or set HF_TOKEN environment variable.",
                error_code="missing_hf_token",
                param="hf_token",
            )
        return token

    def _load_pipeline(self, hf_token: str) -> Any:
        """Load pyannote speaker diarization pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch  # noqa: PLC0415, RUF100
            from pyannote.audio import Pipeline  # noqa: PLC0415, RUF100

            logger.info("Loading pyannote speaker diarization pipeline")

            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self._pipeline.to(device)
                self._device = "cuda"
                logger.info("Moved diarization pipeline to GPU")
            else:
                self._device = "cpu"
                logger.info("Using CPU for diarization pipeline")

            return self._pipeline

        except ImportError as exc:
            raise TranscriptionError(
                "pyannote.audio not installed. "
                "Install with: pip install pyannote-audio",
                error_code="missing_dependency",
            ) from exc
        except Exception as exc:
            logger.error("Failed to load diarization pipeline: %s", exc)
            raise TranscriptionError(
                f"Failed to load diarization pipeline: {exc}",
                error_code="pipeline_load_failed",
            ) from exc

    def _build_diarization_kwargs(
        self, request: TranscriptionRequest
    ) -> dict[str, Any]:
        """Build diarization parameters from request."""
        kwargs = {}

        if request.num_speakers is not None:
            kwargs["num_speakers"] = request.num_speakers
        else:
            if request.min_speakers is not None:
                kwargs["min_speakers"] = request.min_speakers
            if request.max_speakers is not None:
                kwargs["max_speakers"] = request.max_speakers

        return kwargs

    def _convert_to_wav(self, file_path: str | Path) -> str:
        """
        Convert audio file to WAV format for pyannote compatibility.

        Returns:
            Path to temporary WAV file
        """
        try:
            from pydub import AudioSegment  # noqa: PLC0415, RUF100

            # Load audio with pydub (supports many formats)
            audio = AudioSegment.from_file(str(file_path))

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                wav_path = tmp_wav.name

            # Export to WAV format (16kHz, mono, 16-bit)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")

            logger.info("Converted %s to WAV format: %s", file_path, wav_path)
            return wav_path

        except ImportError as exc:
            raise TranscriptionError(
                "pydub not available for audio conversion. "
                "Audio file must be in WAV format.",
                error_code="missing_dependency",
            ) from exc
        except Exception as exc:
            logger.error("Failed to convert audio to WAV: %s", exc)
            raise TranscriptionError(
                f"Failed to convert audio format: {exc}",
                error_code="audio_conversion_failed",
            ) from exc

    def diarize(self, file_path: str | Path, request: TranscriptionRequest) -> Any:
        """
        Perform speaker diarization on audio file.

        Returns:
            pyannote.core.Annotation: Diarization results with speaker labels
        """
        if not request.enable_diarization:
            return None

        wav_path = None
        try:
            hf_token = self._get_hf_token(request)
            pipeline = self._load_pipeline(hf_token)

            # Convert to WAV format for pyannote compatibility
            wav_path = self._convert_to_wav(file_path)

            logger.info("Running speaker diarization on %s", wav_path)

            # Build diarization parameters
            diarization_kwargs = self._build_diarization_kwargs(request)

            # Run diarization
            diarization = pipeline(wav_path, **diarization_kwargs)

            logger.info(
                "Diarization completed. Found %d speakers", len(diarization.labels())
            )

            return diarization

        except Exception as exc:
            logger.warning("Diarization failed: %s", exc)
            return None
        finally:
            if wav_path and wav_path != str(file_path):
                with contextlib.suppress(OSError):
                    os.unlink(wav_path)

    def assign_speakers_to_segments(
        self, segments: list[dict[str, Any]], diarization: Any
    ) -> list[dict[str, Any]]:
        """
        Assign speaker labels to transcription segments based on temporal overlap.

        Args:
            segments: List of transcription segments with start/end times
            diarization: pyannote.core.Annotation object from diarization

        Returns:
            Updated segments with speaker information
        """
        if diarization is None:
            return segments

        try:
            from pyannote.core import Segment  # noqa: PLC0415, RUF100

            updated_segments = []

            for segment in segments:
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)

                # Create segment for overlap calculation
                seg = Segment(start_time, end_time)

                # Find speaker with maximum overlap
                best_speaker = None
                max_overlap = 0.0

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_duration = (
                        seg.intersect(turn).duration if seg.overlaps(turn) else 0.0
                    )
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        best_speaker = speaker

                # Add speaker to segment
                updated_segment = segment.copy()
                updated_segment["speaker"] = best_speaker
                updated_segments.append(updated_segment)

            return updated_segments

        except Exception as exc:
            logger.error("Failed to assign speakers to segments: %s", exc)
            # Return original segments if speaker assignment fails
            return segments

    def assign_speakers_to_words(
        self, words: list[dict[str, Any]], diarization: Any
    ) -> list[dict[str, Any]]:
        """
        Assign speaker labels to word-level timestamps.

        Args:
            words: List of words with start/end times
            diarization: pyannote.core.Annotation object from diarization

        Returns:
            Updated words with speaker information
        """
        if diarization is None or not words:
            return words

        try:
            from pyannote.core import Segment  # noqa: PLC0415, RUF100

            updated_words = []

            for word in words:
                start_time = word.get("start", 0.0)
                end_time = word.get("end", 0.0)

                # Create segment for overlap calculation
                seg = Segment(start_time, end_time)

                # Find speaker with maximum overlap
                best_speaker = None
                max_overlap = 0.0

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_duration = (
                        seg.intersect(turn).duration if seg.overlaps(turn) else 0.0
                    )
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        best_speaker = speaker

                # Add speaker to word
                updated_word = word.copy()
                updated_word["speaker"] = best_speaker
                updated_words.append(updated_word)

            return updated_words

        except Exception as exc:
            logger.error("Failed to assign speakers to words: %s", exc)
            # Return original words if speaker assignment fails
            return words
