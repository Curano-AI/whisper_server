"""
Language detection service using WhisperX detector model.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import cast

import whisperx
from pydub import AudioSegment

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Language detection service using WhisperX detector model."""

    def __init__(self):
        """Initialize the language detector."""
        self.settings = get_settings()
        self._detector_model = None
        self._device = self.settings.device

    def _load_detector_model(self):
        """Load the detector model if not already loaded."""
        if self._detector_model is None:
            logger.info(f"Loading detector model: {self.settings.detector_model}")
            self._detector_model = whisperx.load_model(
                self.settings.detector_model,
                self._device,
                compute_type=self.settings.detector_compute_type,
            )
        return self._detector_model

    def detect_from_samples(
        self, audio_chunks: list[str], min_prob: float | None = None
    ) -> tuple[str, float, dict[str, int], dict[str, float]]:
        """Detect language from audio samples using exact transcribe.py logic.

        Args:
            audio_chunks: list of temporary WAV file paths
            min_prob: Minimum probability threshold (defaults to config value)

        Returns:
            tuple of (best_language, mean_probability, votes_dict, prob_sum_dict)
        """
        if min_prob is None:
            min_prob = self.settings.min_prob

        detector = self._load_detector_model()

        votes: dict[str, int] = {}
        prob_sum: dict[str, float] = {}

        # Process each audio chunk
        for wav_path in audio_chunks:
            try:
                logger.debug(f"Processing audio chunk: {wav_path}")

                # Transcribe chunk to detect language
                result = detector.transcribe(
                    wav_path,
                    batch_size=self.settings.detector_batch_size,
                    verbose=False,
                )

                lang = result["language"]
                # Get probability, default to 1.0 if field is missing
                prob = result.get("language_probs", {}).get(lang, 1.0)

                logger.debug(f"Chunk result: language={lang}, probability={prob:.3f}")

                # Filter by confidence threshold
                if prob < min_prob:
                    logger.debug(f"Filtering out {lang} (prob {prob:.3f} < {min_prob})")
                    continue

                # Accumulate votes and probability sums
                votes[lang] = votes.get(lang, 0) + 1
                prob_sum[lang] = prob_sum.get(lang, 0) + prob

            except Exception as e:
                logger.error(f"Error processing audio chunk {wav_path}: {e}")
                continue

        # If all languages were filtered out, run fallback detection
        if not votes:
            logger.info(
                "All languages filtered out, running "
                "fallback detection without threshold"
            )
            return self._fallback_detection(audio_chunks)

        # Select best language using vote count and confidence sum tie-breaking
        best_lang = max(votes, key=lambda lang: (votes[lang], prob_sum[lang]))
        mean_prob = prob_sum[best_lang] / votes[best_lang] if votes[best_lang] else 0

        logger.info(
            f"Language detection result: votes={votes}, "
            f"mean_prob={mean_prob:.2f} → '{best_lang}'"
        )

        return best_lang, mean_prob, votes, prob_sum

    def _fallback_detection(
        self, audio_chunks: list[str]
    ) -> tuple[str, float, dict[str, int], dict[str, float]]:
        """Fallback detection without probability filtering."""
        logger.info("Running fallback language detection without probability threshold")

        detector = self._load_detector_model()

        votes: dict[str, int] = {}
        prob_sum: dict[str, float] = {}

        # Process chunks again without probability filtering
        for wav_path in audio_chunks:
            try:
                result = detector.transcribe(
                    wav_path,
                    batch_size=self.settings.detector_batch_size,
                    verbose=False,
                )

                lang = result["language"]
                prob = result.get("language_probs", {}).get(lang, 1.0)

                votes[lang] = votes.get(lang, 0) + 1
                prob_sum[lang] = prob_sum.get(lang, 0) + prob

            except Exception as e:
                logger.error(f"Error in fallback detection for {wav_path}: {e}")
                continue

        if not votes:
            # Ultimate fallback - return English
            logger.warning(
                "No language detected even in fallback, defaulting to English"
            )
            return "en", 0.0, {"en": 1}, {"en": 0.0}

        # Select best language
        best_lang = max(votes, key=lambda lang: (votes[lang], prob_sum[lang]))
        mean_prob = prob_sum[best_lang] / votes[best_lang] if votes[best_lang] else 0

        logger.info(
            f"Fallback detection result: votes={votes}, "
            f"mean_prob={mean_prob:.2f} → '{best_lang}'"
        )

        return best_lang, mean_prob, votes, prob_sum

    def detect_language(
        self, audio: AudioSegment, lead_ms: int = 0, min_prob: float | None = None
    ) -> tuple[str, float, dict[str, int], dict[str, float]]:
        """Detect language from audio using strategic sampling.

        Args:
            audio: AudioSegment to analyze
            lead_ms: Leading silence offset in milliseconds
            min_prob: Minimum probability threshold

        Returns:
            tuple of (best_language, mean_probability, votes_dict, prob_sum_dict)
        """
        dur_ms = len(audio)

        # Calculate strategic sample positions
        starts = {
            lead_ms,  # After silence trimming
            max(lead_ms + (dur_ms - lead_ms) // 3 - 5_000, lead_ms),  # 1/3 position
            max(lead_ms + 2 * (dur_ms - lead_ms) // 3 - 5_000, lead_ms),  # 2/3 position
        }

        logger.info(
            f"Creating language detection samples at positions: {sorted(starts)}"
        )

        # Export audio chunks
        chunks = []
        try:
            for start_ms in sorted(starts):
                chunk_path = self._export_chunk(audio, start_ms)
                chunks.append(chunk_path)

            # Detect language from samples
            return self.detect_from_samples(chunks, min_prob)

        finally:
            # Clean up temporary files
            self._cleanup_chunks(chunks)

    def _export_chunk(
        self, audio: AudioSegment, start_ms: int, duration: int = 10_000
    ) -> str:
        """Export audio chunk to temporary WAV file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            segment = cast("AudioSegment", audio[start_ms : start_ms + duration])
            segment.export(tmp.name, format="wav")
            logger.debug(
                f"Exported audio chunk: {tmp.name} (start={start_ms}ms, "
                f"duration={duration}ms)"
            )
            return tmp.name

    def _cleanup_chunks(self, chunk_paths: list[str]) -> None:
        """Clean up temporary audio chunk files."""
        for chunk_path in chunk_paths:
            try:
                if Path(chunk_path).exists():
                    os.unlink(chunk_path)
                    logger.debug(f"Cleaned up chunk: {chunk_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up chunk {chunk_path}: {e}")

    def unload_model(self) -> None:
        """Unload the detector model to free memory."""
        if self._detector_model is not None:
            logger.info("Unloading detector model")
            del self._detector_model
            self._detector_model = None
