"""Unit tests for LanguageDetector service."""

import contextlib
from unittest.mock import Mock, patch

from pydub import AudioSegment

from app.services.language_detector import LanguageDetector


class TestLanguageDetector:
    """Test cases for LanguageDetector service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.detector, "_detector_model") and self.detector._detector_model:
            self.detector.unload_model()

    def test_init(self):
        """Test LanguageDetector initialization."""
        detector = LanguageDetector()
        assert detector._detector_model is None
        assert hasattr(detector, "settings")
        assert hasattr(detector, "_device")

    @patch("app.services.language_detector.whisperx.load_model")
    def test_load_detector_model_first_time(self, mock_load_model):
        """Test loading detector model for the first time."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        result = self.detector._load_detector_model()

        assert result == mock_model
        assert self.detector._detector_model == mock_model
        mock_load_model.assert_called_once_with(
            self.detector.settings.detector_model,
            self.detector._device,
            compute_type=self.detector.settings.detector_compute_type,
        )

    @patch("app.services.language_detector.whisperx.load_model")
    def test_load_detector_model_already_loaded(self, mock_load_model):
        """Test loading detector model when already loaded."""
        mock_model = Mock()
        self.detector._detector_model = mock_model

        result = self.detector._load_detector_model()

        assert result == mock_model
        mock_load_model.assert_not_called()

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_success(self, mock_load_model):
        """Test successful language detection from samples."""
        # Mock detector model
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock transcription results
        mock_detector.transcribe.side_effect = [
            {"language": "en", "language_probs": {"en": 0.8}},
            {"language": "en", "language_probs": {"en": 0.9}},
            {"language": "es", "language_probs": {"es": 0.7}},
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav", "/tmp/chunk3.wav"]
        result = self.detector.detect_from_samples(audio_chunks, min_prob=0.6)

        best_lang, mean_prob, votes, prob_sum = result

        assert best_lang == "en"  # Most votes (2 vs 1)
        assert abs(mean_prob - 0.85) < 0.001  # (0.8 + 0.9) / 2
        assert votes == {"en": 2, "es": 1}
        assert abs(prob_sum["en"] - 1.7) < 0.001
        assert prob_sum["es"] == 0.7

        # Verify transcribe was called for each chunk
        assert mock_detector.transcribe.call_count == 3

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_confidence_filtering(self, mock_load_model):
        """Test confidence-based filtering in language detection."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock results with low confidence
        mock_detector.transcribe.side_effect = [
            {"language": "en", "language_probs": {"en": 0.5}},  # Below threshold
            {"language": "es", "language_probs": {"es": 0.4}},  # Below threshold
            {"language": "fr", "language_probs": {"fr": 0.3}},  # Below threshold
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav", "/tmp/chunk3.wav"]

        # Mock fallback detection
        with patch.object(self.detector, "_fallback_detection") as mock_fallback:
            mock_fallback.return_value = ("en", 0.5, {"en": 3}, {"en": 1.5})

            result = self.detector.detect_from_samples(audio_chunks, min_prob=0.6)

            # Should trigger fallback detection
            mock_fallback.assert_called_once_with(audio_chunks)
            assert result == ("en", 0.5, {"en": 3}, {"en": 1.5})

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_missing_language_probs(self, mock_load_model):
        """Test handling missing language_probs field."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock result without language_probs field
        mock_detector.transcribe.return_value = {"language": "en"}

        audio_chunks = ["/tmp/chunk1.wav"]
        result = self.detector.detect_from_samples(audio_chunks, min_prob=0.6)

        best_lang, mean_prob, votes, prob_sum = result

        assert best_lang == "en"
        assert mean_prob == 1.0  # Default probability
        assert votes == {"en": 1}
        assert prob_sum == {"en": 1.0}

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_transcription_error(self, mock_load_model):
        """Test handling transcription errors."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock transcription error for first chunk, success for second
        mock_detector.transcribe.side_effect = [
            Exception("Transcription failed"),
            {"language": "en", "language_probs": {"en": 0.8}},
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav"]
        result = self.detector.detect_from_samples(audio_chunks, min_prob=0.6)

        best_lang, mean_prob, votes, prob_sum = result

        # Should only process successful chunk
        assert best_lang == "en"
        assert mean_prob == 0.8
        assert votes == {"en": 1}
        assert prob_sum == {"en": 0.8}

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_tie_breaking(self, mock_load_model):
        """Test tie-breaking using confidence sum."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock equal votes but different confidence sums
        mock_detector.transcribe.side_effect = [
            {"language": "en", "language_probs": {"en": 0.9}},  # High confidence
            {"language": "es", "language_probs": {"es": 0.7}},  # Lower confidence
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav"]
        result = self.detector.detect_from_samples(audio_chunks, min_prob=0.6)

        best_lang, mean_prob, votes, prob_sum = result

        # Should choose 'en' due to higher confidence sum
        assert best_lang == "en"
        assert votes == {"en": 1, "es": 1}  # Equal votes
        assert prob_sum["en"] > prob_sum["es"]  # But en has higher confidence

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_fallback_detection_success(self, mock_load_model):
        """Test fallback detection without probability filtering."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock results with low confidence (would be filtered in normal detection)
        mock_detector.transcribe.side_effect = [
            {"language": "en", "language_probs": {"en": 0.3}},
            {"language": "en", "language_probs": {"en": 0.4}},
            {"language": "es", "language_probs": {"es": 0.2}},
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav", "/tmp/chunk3.wav"]
        result = self.detector._fallback_detection(audio_chunks)

        best_lang, mean_prob, votes, prob_sum = result

        assert best_lang == "en"  # Most votes
        assert mean_prob == 0.35  # (0.3 + 0.4) / 2
        assert votes == {"en": 2, "es": 1}
        assert prob_sum == {"en": 0.7, "es": 0.2}

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_fallback_detection_no_results(self, mock_load_model):
        """Test fallback detection when no results are obtained."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock transcription errors for all chunks
        mock_detector.transcribe.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav", "/tmp/chunk3.wav"]
        result = self.detector._fallback_detection(audio_chunks)

        best_lang, mean_prob, votes, prob_sum = result

        # Should default to English
        assert best_lang == "en"
        assert mean_prob == 0.0
        assert votes == {"en": 1}
        assert prob_sum == {"en": 0.0}

    def test_detect_language_strategic_sampling(self):
        """Test strategic sampling positions calculation."""
        # Create mock audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=60000)  # 60 seconds

        with (
            patch.object(self.detector, "_export_chunk") as mock_export,
            patch.object(self.detector, "detect_from_samples") as mock_detect,
            patch.object(self.detector, "_cleanup_chunks") as mock_cleanup,
        ):
            # Mock chunk export
            mock_export.side_effect = [
                "/tmp/chunk1.wav",
                "/tmp/chunk2.wav",
                "/tmp/chunk3.wav",
            ]

            # Mock detection result
            mock_detect.return_value = ("en", 0.8, {"en": 3}, {"en": 2.4})

            lang_result = self.detector.detect_language(mock_audio, lead_ms=2000)

            # Verify result is returned correctly
            assert lang_result == ("en", 0.8, {"en": 3}, {"en": 2.4})

            # Verify strategic sampling positions
            assert mock_export.call_count == 3

            # Get the start positions from export calls
            call_args = [call[0] for call in mock_export.call_args_list]
            start_positions = sorted([args[1] for args in call_args])

            # Expected positions for 60s audio with 2s lead:
            # - lead_ms = 2000
            # - 1/3 position = 2000 + (60000-2000)//3 - 5000 = 16333
            # - 2/3 position = 2000 + 2*(60000-2000)//3 - 5000 = 35666
            expected_positions = sorted([2000, 16333, 35666])
            assert start_positions == expected_positions

            # Verify cleanup was called
            mock_cleanup.assert_called_once()

    def test_detect_language_short_audio(self):
        """Test language detection with short audio."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=8000)  # 8 seconds

        with (
            patch.object(self.detector, "_export_chunk") as mock_export,
            patch.object(self.detector, "detect_from_samples") as mock_detect,
            patch.object(self.detector, "_cleanup_chunks"),
        ):
            mock_export.side_effect = ["/tmp/chunk1.wav"]
            mock_detect.return_value = ("en", 0.8, {"en": 1}, {"en": 0.8})

            lang_result = self.detector.detect_language(mock_audio, lead_ms=1000)

            # Verify result is returned correctly
            assert lang_result == ("en", 0.8, {"en": 1}, {"en": 0.8})

            # For short audio with lead_ms=1000, positions might collapse to 1
            # because: lead_ms=1000, 1/3 pos = max(1000 + 2333 - 5000, 1000) = 1000
            # 2/3 pos = max(1000 + 4666 - 5000, 1000) = 1000
            # So all positions collapse to lead_ms=1000
            assert mock_export.call_count >= 1

            # Verify positions don't go below lead_ms
            call_args = [call[0] for call in mock_export.call_args_list]
            start_positions = [args[1] for args in call_args]

            for pos in start_positions:
                assert pos >= 1000  # All positions should be >= lead_ms

    @patch("tempfile.NamedTemporaryFile")
    def test_export_chunk_success(self, mock_temp_file):
        """Test successful audio chunk export."""
        # Mock audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_chunk = Mock(spec=AudioSegment)
        mock_audio.__getitem__ = Mock(return_value=mock_chunk)

        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test_chunk.wav"
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_file)
        mock_temp_file.return_value.__exit__ = Mock(return_value=None)

        chunk_path = self.detector._export_chunk(mock_audio, 5000, 10000)

        assert chunk_path == "/tmp/test_chunk.wav"
        mock_audio.__getitem__.assert_called_once_with(slice(5000, 15000))
        mock_chunk.export.assert_called_once_with("/tmp/test_chunk.wav", format="wav")

    @patch("tempfile.NamedTemporaryFile")
    def test_export_chunk_default_duration(self, mock_temp_file):
        """Test chunk export with default duration."""
        mock_audio = Mock(spec=AudioSegment)
        mock_chunk = Mock(spec=AudioSegment)
        mock_audio.__getitem__ = Mock(return_value=mock_chunk)

        mock_file = Mock()
        mock_file.name = "/tmp/test_chunk.wav"
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_file)
        mock_temp_file.return_value.__exit__ = Mock(return_value=None)

        chunk_path = self.detector._export_chunk(mock_audio, 2000)

        # Verify the returned path
        assert chunk_path == "/tmp/test_chunk.wav"

        # Should use default duration of 10_000ms
        mock_audio.__getitem__.assert_called_once_with(slice(2000, 12000))

    @patch("app.services.language_detector.Path")
    @patch("app.services.language_detector.os.unlink")
    def test_cleanup_chunks_success(self, mock_unlink, mock_path):
        """Test successful cleanup of chunk files."""
        chunk_paths = ["/tmp/chunk1.wav", "/tmp/chunk2.wav", "/tmp/chunk3.wav"]

        # Mock Path.exists to return True
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        self.detector._cleanup_chunks(chunk_paths)

        # Verify Path was called for each chunk
        assert mock_path.call_count == 3
        mock_path.assert_any_call("/tmp/chunk1.wav")
        mock_path.assert_any_call("/tmp/chunk2.wav")
        mock_path.assert_any_call("/tmp/chunk3.wav")

        # Verify unlink was called for each existing file
        assert mock_unlink.call_count == 3

    @patch("app.services.language_detector.Path")
    @patch("app.services.language_detector.os.unlink")
    def test_cleanup_chunks_file_not_exists(self, mock_unlink, mock_path):
        """Test cleanup when some files don't exist."""
        chunk_paths = ["/tmp/chunk1.wav", "/tmp/chunk2.wav"]

        # Mock first file exists, second doesn't
        mock_path_instances = [Mock(), Mock()]
        mock_path_instances[0].exists.return_value = True
        mock_path_instances[1].exists.return_value = False
        mock_path.side_effect = mock_path_instances

        self.detector._cleanup_chunks(chunk_paths)

        # Should only unlink the existing file
        mock_unlink.assert_called_once_with("/tmp/chunk1.wav")

    @patch("app.services.language_detector.Path")
    @patch("app.services.language_detector.os.unlink")
    def test_cleanup_chunks_unlink_error(self, mock_unlink, mock_path):
        """Test cleanup with unlink error."""
        chunk_paths = ["/tmp/chunk1.wav"]

        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock unlink to raise exception
        mock_unlink.side_effect = OSError("Permission denied")

        # Should not raise exception, just log warning
        self.detector._cleanup_chunks(chunk_paths)

        mock_unlink.assert_called_once_with("/tmp/chunk1.wav")

    def test_unload_model_when_loaded(self):
        """Test unloading model when it's loaded."""
        # Mock loaded model
        mock_model = Mock()
        self.detector._detector_model = mock_model

        self.detector.unload_model()

        assert self.detector._detector_model is None

    def test_unload_model_when_not_loaded(self):
        """Test unloading model when it's not loaded."""
        assert self.detector._detector_model is None

        # Should not raise exception
        self.detector.unload_model()

        assert self.detector._detector_model is None

    def test_detect_from_samples_default_min_prob(self):
        """Test that default min_prob comes from settings."""
        with patch.object(self.detector, "_load_detector_model") as mock_load_model:
            mock_detector = Mock()
            mock_load_model.return_value = mock_detector
            mock_detector.transcribe.return_value = {
                "language": "en",
                "language_probs": {"en": 0.8},
            }

            # Call without min_prob parameter
            result = self.detector.detect_from_samples(["/tmp/chunk1.wav"])

            # Should use settings.min_prob for filtering and return result
            assert result[0] == "en"  # Language detected
            assert result[1] == 0.8  # Probability
            assert result[2] == {"en": 1}  # Votes
            assert result[3] == {"en": 0.8}  # Probability sum

    def test_detect_language_cleanup_on_exception(self):
        """Test that cleanup happens even when detection fails."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=30000)

        with (
            patch.object(self.detector, "_export_chunk") as mock_export,
            patch.object(self.detector, "detect_from_samples") as mock_detect,
            patch.object(self.detector, "_cleanup_chunks") as mock_cleanup,
        ):
            # Mock chunk export
            mock_export.side_effect = [
                "/tmp/chunk1.wav",
                "/tmp/chunk2.wav",
                "/tmp/chunk3.wav",
            ]

            # Mock detection to raise exception
            mock_detect.side_effect = Exception("Detection failed")

            # Should raise exception but still call cleanup
            with contextlib.suppress(Exception):
                self.detector.detect_language(mock_audio)

            # Verify cleanup was called even with exception
            mock_cleanup.assert_called_once()

    def test_detect_language_with_custom_min_prob(self):
        """Test language detection with custom min_prob parameter."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=30000)

        with (
            patch.object(self.detector, "_export_chunk") as mock_export,
            patch.object(self.detector, "detect_from_samples") as mock_detect,
            patch.object(self.detector, "_cleanup_chunks"),
        ):
            mock_export.side_effect = [
                "/tmp/chunk1.wav",
                "/tmp/chunk2.wav",
                "/tmp/chunk3.wav",
            ]
            mock_detect.return_value = ("es", 0.75, {"es": 3}, {"es": 2.25})

            result = self.detector.detect_language(
                mock_audio, lead_ms=1000, min_prob=0.7
            )

            # Verify custom min_prob was passed to detect_from_samples
            mock_detect.assert_called_once()
            call_args = mock_detect.call_args
            # min_prob is passed as positional argument (second parameter)
            assert call_args[0][1] == 0.7

            # Verify result
            assert result == ("es", 0.75, {"es": 3}, {"es": 2.25})

    def test_detect_language_zero_lead_ms(self):
        """Test language detection with zero lead_ms."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=30000)

        with (
            patch.object(self.detector, "_export_chunk") as mock_export,
            patch.object(self.detector, "detect_from_samples") as mock_detect,
            patch.object(self.detector, "_cleanup_chunks"),
        ):
            mock_export.side_effect = [
                "/tmp/chunk1.wav",
                "/tmp/chunk2.wav",
                "/tmp/chunk3.wav",
            ]
            mock_detect.return_value = ("en", 0.8, {"en": 3}, {"en": 2.4})

            self.detector.detect_language(mock_audio, lead_ms=0)

            # Get the start positions from export calls
            call_args = [call[0] for call in mock_export.call_args_list]
            start_positions = sorted([args[1] for args in call_args])

            # All positions should be >= 0
            for pos in start_positions:
                assert pos >= 0

            # First position should be exactly 0 (lead_ms)
            assert 0 in start_positions

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_empty_chunks_list(self, mock_load_model):
        """Test detection with empty chunks list."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock fallback detection since no chunks to process
        with patch.object(self.detector, "_fallback_detection") as mock_fallback:
            mock_fallback.return_value = ("en", 0.0, {"en": 1}, {"en": 0.0})

            result = self.detector.detect_from_samples([])

            # Should trigger fallback detection
            mock_fallback.assert_called_once_with([])
            assert result == ("en", 0.0, {"en": 1}, {"en": 0.0})

    @patch.object(LanguageDetector, "_load_detector_model")
    def test_detect_from_samples_all_errors(self, mock_load_model):
        """Test detection when all chunks fail to transcribe."""
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector

        # Mock transcription errors for all chunks
        mock_detector.transcribe.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
        ]

        audio_chunks = ["/tmp/chunk1.wav", "/tmp/chunk2.wav"]

        # Mock fallback detection
        with patch.object(self.detector, "_fallback_detection") as mock_fallback:
            mock_fallback.return_value = ("en", 0.0, {"en": 1}, {"en": 0.0})

            result = self.detector.detect_from_samples(audio_chunks)

            # Should trigger fallback detection
            mock_fallback.assert_called_once_with(audio_chunks)
            assert result == ("en", 0.0, {"en": 1}, {"en": 0.0})

    def test_detect_language_return_value_structure(self):
        """Test that detect_language returns the correct tuple structure."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=30000)

        with (
            patch.object(self.detector, "_export_chunk") as mock_export,
            patch.object(self.detector, "detect_from_samples") as mock_detect,
            patch.object(self.detector, "_cleanup_chunks"),
        ):
            mock_export.side_effect = [
                "/tmp/chunk1.wav",
                "/tmp/chunk2.wav",
                "/tmp/chunk3.wav",
            ]
            expected_result = ("fr", 0.85, {"fr": 2, "en": 1}, {"fr": 1.7, "en": 0.8})
            mock_detect.return_value = expected_result

            result = self.detector.detect_language(mock_audio)

            # Verify return value structure
            assert isinstance(result, tuple)
            assert len(result) == 4
            assert isinstance(result[0], str)  # language
            assert isinstance(result[1], float)  # mean probability
            assert isinstance(result[2], dict)  # votes
            assert isinstance(result[3], dict)  # probability sum
            assert result == expected_result
