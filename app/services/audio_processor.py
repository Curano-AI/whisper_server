"""Audio processing service for WhisperX FastAPIerver.

This service handles audio file validation, loading, preprocessing,
and sample extraction for language detection. It implements the exact
audio processing logic from transcribe.py.
"""

import tempfile
from pathlib import Path
from typing import Any, ClassVar, cast

from pydub import AudioSegment, silence

from app.core.exceptions import AudioProcessingError
from app.utils.transcribe_utils import cleanup_temp_files, export_chunk


class AudioProcessor:
    """Service for audio file processing and preprocessing.

    Handles audio file validation, loading, silence trimming, and strategic
    sample extraction for language detection using the exact logic from transcribe.py.
    """

    # Exact parameters from transcribe.py
    SILENCE_MIN_DURATION = 300  # detect_nonsilent min_silence_len parameter
    SILENCE_THRESHOLD = -35  # detect_nonsilent silence_thresh parameter
    CHUNK_DURATION = 10_000  # 10 seconds for language detection samples
    CHUNK_OFFSET = 5_000  # 5 second offset for sample positioning
    SHORT_AUDIO_THRESHOLD = 15_000  # 15 seconds threshold for short audio handling

    # Supported audio formats (pydub supported formats)
    SUPPORTED_FORMATS: ClassVar[set[str]] = {
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".flac",
        ".ogg",
        ".wma",
        ".aac",
        ".webm",
        ".3gp",
        ".amr",
        ".au",
        ".aiff",
        ".opus",
    }

    def __init__(self) -> None:
        """Initialize AudioProcessor."""
        self._temp_files: list[str] = []

    def validate_audio_file(self, file_path: str | Path) -> None:
        """Validate audio file exists and has supported format.

        Args:
            file_path: Path to audio file

        Raises:
            AudioProcessingError: If file doesn't exist or format not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise AudioProcessingError(
                f"Audio file not found: {file_path}",
                error_code="file_not_found",
                param="file",
            )

        if not path.is_file():
            raise AudioProcessingError(
                f"Path is not a file: {file_path}",
                error_code="invalid_file",
                param="file",
            )

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise AudioProcessingError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}",
                error_code="unsupported_format",
                param="file",
            )

    def load_audio(self, file_path: str | Path) -> AudioSegment:
        """Load audio file using pydub.

        Args:
            file_path: Path to audio file

        Returns:
            AudioSegment object

        Raises:
            AudioProcessingError: If audio loading fails
        """
        try:
            self.validate_audio_file(file_path)
            audio = AudioSegment.from_file(str(file_path))

            if len(audio) == 0:
                raise AudioProcessingError(
                    "Audio file is empty or corrupted",
                    error_code="empty_audio",
                    param="file",
                )

            return audio

        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to load audio file: {e!s}",
                error_code="audio_load_failed",
                param="file",
            ) from e

    def trim_leading_silence(self, audio: AudioSegment) -> tuple[AudioSegment, int]:
        """Trim leading silence from audio using exact transcribe.py parameters.

        Uses pydub.silence.detect_nonsilent with parameters (300, -35) to find
        the first non-silent segment and returns the trimmed audio.

        Args:
            audio: AudioSegment to process

        Returns:
            Tuple of (trimmed_audio, leading_silence_ms)

        Raises:
            AudioProcessingError: If silence detection fails
        """
        try:
            # Exact logic from transcribe.py
            nonsilent_ranges = silence.detect_nonsilent(
                audio,
                min_silence_len=self.SILENCE_MIN_DURATION,
                silence_thresh=self.SILENCE_THRESHOLD,
            )

            # If no non-silent parts found, assume no leading silence
            lead_ms = nonsilent_ranges[0][0] if nonsilent_ranges else 0

            # Trim the audio from the first non-silent part
            trimmed_audio = cast("AudioSegment", audio[lead_ms:])

            return trimmed_audio, lead_ms

        except Exception as e:
            raise AudioProcessingError(
                f"Failed to trim leading silence: {e!s}",
                error_code="silence_trim_failed",
            ) from e

    def extract_language_samples(
        self, audio: AudioSegment, leading_silence_ms: int
    ) -> list[str]:
        """Extract strategic audio samples for language detection.

        Creates 3 audio samples at strategic positions using the exact logic
        from transcribe.py for language detection.

        Args:
            audio: Original audio (before trimming)
            leading_silence_ms: Amount of leading silence that was trimmed

        Returns:
            List of temporary WAV file paths

        Raises:
            AudioProcessingError: If sample extraction fails
        """
        try:
            dur_ms = len(audio)
            lead_ms = leading_silence_ms

            # Exact logic from transcribe.py for sample positions
            # Calculate the three positions, ensuring they're different
            pos1 = lead_ms
            pos2 = max(lead_ms + (dur_ms - lead_ms) // 3 - self.CHUNK_OFFSET, lead_ms)
            pos3 = max(
                lead_ms + 2 * (dur_ms - lead_ms) // 3 - self.CHUNK_OFFSET, lead_ms
            )

            # For very short audio, spread positions evenly
            if dur_ms <= self.SHORT_AUDIO_THRESHOLD:  # Less than 15 seconds
                available_duration = dur_ms - lead_ms
                if available_duration > 0:
                    pos2 = lead_ms + available_duration // 3
                    pos3 = lead_ms + 2 * available_duration // 3
                else:
                    pos2 = lead_ms
                    pos3 = lead_ms

            # Use list to maintain order and allow duplicates if needed
            positions = [pos1, pos2, pos3]

            # Create chunks at the positions
            chunks = []
            for chunk_start_ms in positions:
                # Ensure we don't go beyond audio duration
                actual_start_ms = chunk_start_ms
                if chunk_start_ms >= dur_ms:
                    actual_start_ms = max(0, dur_ms - self.CHUNK_DURATION)

                chunk_path = export_chunk(audio, actual_start_ms, self.CHUNK_DURATION)
                chunks.append(chunk_path)
                self._temp_files.append(chunk_path)

            return chunks

        except Exception as e:
            # Clean up any created files on error
            self.cleanup()
            raise AudioProcessingError(
                f"Failed to extract language samples: {e!s}",
                error_code="sample_extraction_failed",
            ) from e

    def process_audio_for_language_detection(
        self, file_path: str | Path
    ) -> tuple[AudioSegment, int, list[str]]:
        """Complete audio preprocessing pipeline for language detection.

        Loads audio, trims leading silence, and extracts samples for language detection.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (original_audio, leading_silence_ms, sample_file_paths)

        Raises:
            AudioProcessingError: If any processing step fails
        """
        try:
            # Load audio
            audio = self.load_audio(file_path)

            # Trim leading silence
            _, leading_silence_ms = self.trim_leading_silence(audio)

            # Extract language detection samples
            sample_paths = self.extract_language_samples(audio, leading_silence_ms)

            return audio, leading_silence_ms, sample_paths

        except AudioProcessingError:
            raise
        except Exception as e:
            self.cleanup()
            raise AudioProcessingError(
                f"Audio processing pipeline failed: {e!s}",
                error_code="processing_pipeline_failed",
            ) from e

    def create_temp_audio_file(self, audio: AudioSegment, format: str = "wav") -> str:
        """Create temporary audio file from AudioSegment.

        Args:
            audio: AudioSegment to export
            format: Audio format (default: wav)

        Returns:
            Path to temporary file

        Raises:
            AudioProcessingError: If file creation fails
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp:
                audio.export(tmp.name, format=format)
                self._temp_files.append(tmp.name)
                return tmp.name

        except Exception as e:
            raise AudioProcessingError(
                f"Failed to create temporary audio file: {e!s}",
                error_code="temp_file_creation_failed",
            ) from e

    def get_audio_info(self, audio: AudioSegment) -> dict[str, Any]:
        """Get audio file information.

        Args:
            audio: AudioSegment to analyze

        Returns:
            Dictionary with audio information
        """
        return {
            "duration_ms": len(audio),
            "duration_seconds": len(audio) / 1000.0,
            "frame_rate": audio.frame_rate,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_width": audio.frame_width,
            "frame_count": audio.frame_count(),
        }

    def cleanup(self) -> None:
        """Clean up all temporary files created by this processor."""
        if self._temp_files:
            files_to_cleanup = self._temp_files.copy()
            self._temp_files.clear()
            cleanup_temp_files(files_to_cleanup)
        else:
            cleanup_temp_files([])

    def __enter__(self) -> "AudioProcessor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()
