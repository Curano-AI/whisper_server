"""
Language detection service using WhisperX detector model
with robust handling for poor quality audio.

This implementation uses multiple strategies to ensure accurate language detection:
1. Multi-chunk sampling (6 chunks by default)
2. Dynamic confidence threshold adjustment
3. Confidence-aware ensemble voting
4. Progressive refinement with multiple passes
"""

import logging
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, cast

from pydub import AudioSegment

from app.core.config import get_settings
from app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Robust language detection service using WhisperX detector model."""

    def __init__(self):
        """Initialize the language detector."""
        self.settings = get_settings()
        self._detector_model = None
        self._device = self.settings.device

    def _load_detector_model(self):
        """Load the detector model if not already loaded."""
        import whisperx  # noqa: PLC0415, RUF100

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
        """Detect language from audio samples with robust handling for poor quality.

        Uses multiple passes with progressively refined thresholds to ensure
        accurate detection even with noisy or poor quality audio.

        Args:
            audio_chunks: list of temporary WAV file paths
            min_prob: Minimum probability threshold (defaults to config value)

        Returns:
            tuple of (best_language, mean_probability, votes_dict, prob_sum_dict)
        """
        if min_prob is None:
            min_prob = self.settings.min_prob

        # Filter chunks by quality before processing
        quality_chunks, filtering_stats = self._filter_chunks_by_quality(audio_chunks)

        logger.info(
            f"Quality filtering: using {len(quality_chunks)}/{len(audio_chunks)} chunks"
        )
        if filtering_stats.get("used_fallback"):
            logger.warning("Quality filtering used safety fallback - using all chunks")

        # First pass: Collect all predictions with lower threshold
        initial_threshold = self.settings.min_language_confidence
        all_predictions = self._collect_predictions(
            quality_chunks, threshold=initial_threshold
        )

        if not all_predictions:
            logger.warning(
                "No predictions collected in first pass, trying without threshold"
            )
            all_predictions = self._collect_predictions(quality_chunks, threshold=0.0)

        if not all_predictions:
            logger.error("No valid predictions from any chunk")
            return "en", 0.0, {"en": 1}, {"en": 0.0}

        # Analyze predictions and determine dynamic threshold
        dynamic_threshold = self._calculate_dynamic_threshold(all_predictions, min_prob)

        # Second pass: Filter predictions with dynamic threshold
        filtered_predictions = [
            p for p in all_predictions if p["confidence"] >= dynamic_threshold
        ]

        if not filtered_predictions:
            # If dynamic threshold filters everything, use top predictions
            logger.warning(
                f"Dynamic threshold {dynamic_threshold:.2f} too high, "
                f"using top predictions"
            )
            sorted_predictions = sorted(
                all_predictions, key=lambda x: x["confidence"], reverse=True
            )
            # Take predictions with confidence above median
            median_conf = sorted_predictions[len(sorted_predictions) // 2]["confidence"]
            filtered_predictions = [
                p for p in all_predictions if p["confidence"] >= median_conf
            ]

        # Apply ensemble voting with confidence weighting
        best_lang, mean_prob, votes, prob_sum = self._ensemble_voting(
            filtered_predictions
        )

        # Language consistency check
        consistency_score = self._check_language_consistency(filtered_predictions)

        logger.info(
            f"Language detection result: '{best_lang}' "
            f"(confidence={mean_prob:.2f}, consistency={consistency_score:.2f}, "
            f"votes={votes})"
        )

        # If consistency is low, run progressive refinement
        if consistency_score < self.settings.min_consensus_ratio:
            logger.info("Low consistency detected, running progressive refinement")
            best_lang, mean_prob, votes, prob_sum = self._progressive_refinement(
                quality_chunks, all_predictions, best_lang
            )

        return best_lang, mean_prob, votes, prob_sum

    def _collect_predictions(
        self, chunk_paths: list[str], threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """Collect all predictions from chunks with confidence above threshold."""
        detector = self._load_detector_model()
        predictions = []

        for i, wav_path in enumerate(chunk_paths):
            try:
                logger.debug(f"Processing chunk {i + 1}/{len(chunk_paths)}: {wav_path}")

                try:
                    from faster_whisper.audio import (  # noqa: PLC0415, RUF100
                        decode_audio,
                    )

                    audio = decode_audio(wav_path)

                    # Detect language with confidence scores
                    detection_result = detector.model.detect_language(audio)

                    EXPECTED_DETECTION_TUPLE_SIZE = 2
                    if (
                        isinstance(detection_result, tuple)
                        and len(detection_result) >= EXPECTED_DETECTION_TUPLE_SIZE
                    ):
                        lang, lang_probs = detection_result[0], detection_result[1]
                        logger.info(
                            f"Chunk {i + 1} - Direct detection: "
                            f"lang={lang}, probs={lang_probs}"
                        )
                    else:
                        # Fallback to transcribe method
                        result = detector.transcribe(
                            wav_path,
                            batch_size=self.settings.get_detector_batch_size(),
                            verbose=False,
                        )
                        lang = result.get("language")
                        lang_probs = result.get("language_probs", {})
                        logger.info(
                            f"Chunk {i + 1} - Transcribe result: "
                            f"lang={lang}, has probs={bool(lang_probs)}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Direct language detection failed: {e}, "
                        "falling back to transcribe"
                    )
                    # Fallback to transcribe method
                    result = detector.transcribe(
                        wav_path,
                        batch_size=self.settings.get_detector_batch_size(),
                        verbose=False,
                    )
                    lang = result.get("language")
                    lang_probs = result.get("language_probs", {})

                if not lang:
                    logger.warning(f"No language detected in chunk {i + 1}")
                    continue

                if not lang_probs and lang:
                    default_confidence = 0.9
                    lang_probs = {lang: default_confidence}
                    logger.info(
                        f"Chunk {i + 1}: {lang} detected without probs, "
                        f"using default confidence {default_confidence}"
                    )

                # Check if detected language has probability
                if lang in lang_probs:
                    confidence = lang_probs[lang]
                    if confidence >= threshold:
                        predictions.append(
                            {
                                "chunk_id": i,
                                "language": lang,
                                "confidence": confidence,
                                "all_probs": lang_probs,
                            }
                        )
                        logger.debug(f"Chunk {i + 1}: {lang} ({confidence:.3f})")
                    else:
                        logger.debug(
                            f"Chunk {i + 1}: {lang} filtered "
                            f"(conf={confidence:.3f} < {threshold:.3f})"
                        )
                else:
                    # This should rarely happen now with our fallback
                    logger.warning(
                        f"Chunk {i + 1}: {lang} detected but no confidence available"
                    )

            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                continue

        return predictions

    def _calculate_dynamic_threshold(
        self, predictions: list[dict[str, Any]], base_threshold: float
    ) -> float:
        """Calculate dynamic threshold based on prediction distribution."""
        if not predictions:
            return base_threshold

        confidences = [p["confidence"] for p in predictions]

        # Calculate statistics
        mean_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        min_conf = min(confidences)

        # Sort for percentile calculation
        sorted_conf = sorted(confidences)
        percentile_25 = sorted_conf[len(sorted_conf) // 4]

        # Dynamic threshold strategy
        variance_threshold = 0.5
        low_confidence_threshold = 0.5

        if max_conf - min_conf > variance_threshold:
            # High variance - use percentile-based threshold
            dynamic_threshold = percentile_25
        elif mean_conf < low_confidence_threshold:
            # Low overall confidence - be more lenient
            dynamic_threshold = min(base_threshold * 0.7, percentile_25)
        else:
            # Normal case - use base threshold or mean minus std
            std_dev = (
                sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            ) ** 0.5
            dynamic_threshold = max(mean_conf - std_dev, base_threshold * 0.8)

        # Ensure threshold is reasonable
        dynamic_threshold = max(0.2, min(dynamic_threshold, 0.8))

        logger.info(
            f"Dynamic threshold: {dynamic_threshold:.2f} "
            f"(base={base_threshold:.2f}, mean={mean_conf:.2f}, "
            f"range={min_conf:.2f}-{max_conf:.2f})"
        )

        return dynamic_threshold

    def _ensemble_voting(
        self, predictions: list[dict[str, Any]]
    ) -> tuple[str, float, dict[str, int], dict[str, float]]:
        """Apply ensemble voting with confidence weighting."""
        votes: dict[str, int] = {}
        prob_sum: dict[str, float] = {}
        weighted_scores: dict[str, float] = {}

        # Aggregate predictions
        for pred in predictions:
            lang = pred["language"]
            conf = pred["confidence"]

            votes[lang] = votes.get(lang, 0) + 1
            prob_sum[lang] = prob_sum.get(lang, 0) + conf

            # Calculate weighted score using confidence
            weight = conf**self.settings.confidence_weight
            weighted_scores[lang] = weighted_scores.get(lang, 0) + weight

        # Select best language using weighted scores
        if not weighted_scores:
            return "en", 0.0, {"en": 1}, {"en": 0.0}

        # Sort by weighted score, then by average confidence, then by votes
        def score_key(lang: str) -> tuple[float, float, int]:
            avg_conf = prob_sum[lang] / votes[lang] if votes[lang] else 0
            return (weighted_scores[lang], avg_conf, votes[lang])

        best_lang = max(weighted_scores.keys(), key=score_key)
        mean_prob = prob_sum[best_lang] / votes[best_lang] if votes[best_lang] else 0

        logger.debug(
            f"Ensemble voting: {votes}, weighted_scores={weighted_scores}, "
            f"selected='{best_lang}'"
        )

        return best_lang, mean_prob, votes, prob_sum

    def _check_language_consistency(self, predictions: list[dict[str, Any]]) -> float:
        """Check consistency of language predictions across chunks."""
        if not predictions:
            return 0.0

        languages = [p["language"] for p in predictions]
        language_counts = Counter(languages)

        # Most common language and its ratio
        most_common_lang, count = language_counts.most_common(1)[0]
        consistency_ratio = count / len(predictions)

        # Weighted consistency considering confidence
        weighted_consistency = 0.0
        total_weight = 0.0

        for pred in predictions:
            weight = pred["confidence"]
            if pred["language"] == most_common_lang:
                weighted_consistency += weight
            total_weight += weight

        if total_weight > 0:
            weighted_consistency /= total_weight

        # Combined consistency score
        consistency_score = (consistency_ratio + weighted_consistency) / 2

        logger.debug(
            f"Language consistency: {consistency_score:.2f} "
            f"(ratio={consistency_ratio:.2f}, weighted={weighted_consistency:.2f})"
        )

        return consistency_score

    def _progressive_refinement(
        self,
        chunk_paths: list[str],
        initial_predictions: list[dict[str, Any]],
        initial_best: str,  # noqa: ARG002
    ) -> tuple[str, float, dict[str, int], dict[str, float]]:
        """Progressively refine language detection with focused analysis."""
        logger.info("Running progressive refinement for language detection")

        # Identify top competing languages
        language_scores = {}
        for pred in initial_predictions:
            lang = pred["language"]
            score = pred["confidence"]
            language_scores[lang] = language_scores.get(lang, 0) + score

        # Get top 2-3 languages
        top_languages = sorted(
            language_scores.items(), key=lambda x: x[1], reverse=True
        )[:3]

        logger.info(f"Top competing languages: {top_languages}")

        # Re-analyze with focus on distinguishing between top languages
        refined_predictions = []
        detector = self._load_detector_model()

        for i, wav_path in enumerate(chunk_paths[:3]):  # Focus on first 3 chunks
            try:
                # Try multiple times with different parameters
                for attempt in range(2):
                    result = detector.transcribe(
                        wav_path,
                        batch_size=1,  # Smaller batch for more careful analysis
                        verbose=False,
                    )

                    lang_probs = result.get("language_probs", {})

                    # Check probabilities for top languages
                    min_refinement_prob = 0.1
                    for lang, _ in top_languages:
                        if (
                            lang in lang_probs
                            and lang_probs[lang] > min_refinement_prob
                        ):
                            refined_predictions.append(
                                {
                                    "chunk_id": f"{i}-{attempt}",
                                    "language": lang,
                                    "confidence": lang_probs[lang],
                                    "all_probs": lang_probs,
                                }
                            )

            except Exception as e:
                logger.error(f"Error in refinement for chunk {i}: {e}")

        # Combine with initial predictions
        all_predictions = initial_predictions + refined_predictions

        # Final ensemble voting
        return self._ensemble_voting(all_predictions)

    def _filter_chunks_by_quality(
        self, chunk_paths: list[str]
    ) -> tuple[list[str], dict[str, Any]]:
        """Filter audio chunks by quality using AudioProcessor."""
        processor = AudioProcessor()
        try:
            return processor.filter_chunks_by_quality(chunk_paths)
        finally:
            processor.cleanup()

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
        # Create more chunks for robust detection
        num_chunks = self.settings.num_language_chunks
        processor = AudioProcessor()
        chunks = []  # Initialize to avoid UnboundLocalError

        try:
            chunks = processor.extract_language_samples(audio, lead_ms, num_chunks)
            logger.info(f"Created {len(chunks)} chunks for language detection")

            # Detect language from samples
            return self.detect_from_samples(chunks, min_prob)

        finally:
            # Clean up temporary files
            processor.cleanup()
            if hasattr(self, "_cleanup_chunks") and chunks:
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
