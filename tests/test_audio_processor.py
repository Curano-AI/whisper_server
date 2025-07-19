"""Unit tests for AudioProcessor service."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from app.core.exceptions import AudioProcessingError
from app.services.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()

    def teardown_method(self):
        """Clean up after tests."""
        self.processor.cleanup()

    def test_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor()
        assert processor._temp_files == []
        assert processor.SILENCE_MIN_DURATION == 300
        assert processor.SILENCE_THRESHOLD == -35
        assert processor.CHUNK_DURATION == 10_000
        assert processor.CHUNK_OFFSET == 5_000

    def test_supported_formats(self):
        """Test supported audio formats."""
        expected_formats = {
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
        assert expected_formats == self.processor.SUPPORTED_FORMATS

    def test_validate_audio_file_success(self):
        """Test successful audio file validation."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Should not raise exception
            self.processor.validate_audio_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_validate_audio_file_not_found(self):
        """Test validation with non-existent file."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.validate_audio_file("/nonexistent/file.wav")

        assert exc_info.value.error_code == "file_not_found"
        assert exc_info.value.param == "file"
        assert "not found" in str(exc_info.value)

    def test_validate_audio_file_not_file(self):
        """Test validation with directory instead of file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.validate_audio_file(tmp_dir)

            assert exc_info.value.error_code == "invalid_file"
            assert exc_info.value.param == "file"
            assert "not a file" in str(exc_info.value)

    def test_validate_audio_file_unsupported_format(self):
        """Test validation with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.validate_audio_file(tmp_path)

            assert exc_info.value.error_code == "unsupported_format"
            assert exc_info.value.param == "file"
            assert "Unsupported audio format" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    @patch("app.services.audio_processor.AudioSegment.from_file")
    def test_load_audio_success(self, mock_from_file):
        """Test successful audio loading."""
        # Mock audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=5000)  # 5 seconds
        mock_from_file.return_value = mock_audio

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.processor.load_audio(tmp_path)
            assert result == mock_audio
            mock_from_file.assert_called_once_with(tmp_path)
        finally:
            os.unlink(tmp_path)

    @patch("app.services.audio_processor.AudioSegment.from_file")
    def test_load_audio_empty_file(self, mock_from_file):
        """Test loading empty audio file."""
        # Mock empty audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=0)
        mock_from_file.return_value = mock_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.load_audio(tmp_path)

            assert exc_info.value.error_code == "empty_audio"
            assert exc_info.value.param == "file"
            assert "empty or corrupted" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    @patch("app.services.audio_processor.AudioSegment.from_file")
    def test_load_audio_loading_error(self, mock_from_file):
        """Test audio loading with exception."""
        mock_from_file.side_effect = Exception("Loading failed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.load_audio(tmp_path)

            assert exc_info.value.error_code == "audio_load_failed"
            assert exc_info.value.param == "file"
            assert "Failed to load audio file" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    @patch("app.services.audio_processor.silence.detect_nonsilent")
    def test_trim_leading_silence_with_silence(self, mock_detect):
        """Test trimming leading silence when silence is detected."""
        # Mock audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__getitem__ = Mock(return_value=mock_audio)

        # Mock silence detection returning ranges
        mock_detect.return_value = [(1000, 5000), (6000, 10000)]

        result_audio, lead_ms = self.processor.trim_leading_silence(mock_audio)

        assert result_audio == mock_audio
        assert lead_ms == 1000
        mock_detect.assert_called_once_with(
            mock_audio, min_silence_len=300, silence_thresh=-35
        )
        mock_audio.__getitem__.assert_called_once_with(slice(1000, None))

    @patch("app.services.audio_processor.silence.detect_nonsilent")
    def test_trim_leading_silence_no_silence(self, mock_detect):
        """Test trimming when no silence is detected."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__getitem__ = Mock(return_value=mock_audio)

        # Mock no silence detected
        mock_detect.return_value = []

        result_audio, lead_ms = self.processor.trim_leading_silence(mock_audio)

        assert result_audio == mock_audio
        assert lead_ms == 0
        mock_audio.__getitem__.assert_called_once_with(slice(0, None))

    @patch("app.services.audio_processor.silence.detect_nonsilent")
    def test_trim_leading_silence_error(self, mock_detect):
        """Test trimming with silence detection error."""
        mock_audio = Mock(spec=AudioSegment)
        mock_detect.side_effect = Exception("Detection failed")

        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.trim_leading_silence(mock_audio)

        assert exc_info.value.error_code == "silence_trim_failed"
        assert "Failed to trim leading silence" in str(exc_info.value)

    @patch("app.services.audio_processor.export_chunk")
    def test_extract_language_samples_success(self, mock_export):
        """Test successful language sample extraction."""
        # Mock audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=60000)  # 60 seconds

        # Mock export_chunk to return file paths
        mock_export.side_effect = [
            "/tmp/chunk1.wav",
            "/tmp/chunk2.wav",
            "/tmp/chunk3.wav",
        ]

        result = self.processor.extract_language_samples(mock_audio, 2000)

        assert len(result) == 3
        assert result == ["/tmp/chunk1.wav", "/tmp/chunk2.wav", "/tmp/chunk3.wav"]
        assert len(self.processor._temp_files) == 3

        # Verify export_chunk was called with correct parameters
        assert mock_export.call_count == 3

        # Check the start positions (should be sorted)
        call_args = [call[0] for call in mock_export.call_args_list]
        start_positions = [args[1] for args in call_args]

        # Expected positions: lead_ms=2000, dur_ms=60000
        # Position calculation: {2000, 16333, 35666}
        expected_starts = sorted([2000, 16333, 35666])
        assert start_positions == expected_starts

    @patch("app.services.audio_processor.export_chunk")
    def test_extract_language_samples_short_audio(self, mock_export):
        """Test sample extraction with very short audio."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=5000)  # 5 seconds

        mock_export.side_effect = [
            "/tmp/chunk1.wav",
            "/tmp/chunk2.wav",
            "/tmp/chunk3.wav",
        ]

        result = self.processor.extract_language_samples(mock_audio, 1000)

        assert len(result) == 3
        # Should still create 3 samples even with short audio
        assert mock_export.call_count == 3

    @patch("app.services.audio_processor.export_chunk")
    def test_extract_language_samples_error(self, mock_export):
        """Test sample extraction with error."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=60000)

        mock_export.side_effect = Exception("Export failed")

        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.extract_language_samples(mock_audio, 2000)

        assert exc_info.value.error_code == "sample_extraction_failed"
        assert "Failed to extract language samples" in str(exc_info.value)

    @patch.object(AudioProcessor, "extract_language_samples")
    @patch.object(AudioProcessor, "trim_leading_silence")
    @patch.object(AudioProcessor, "load_audio")
    def test_process_audio_for_language_detection_success(
        self, mock_load, mock_trim, mock_extract
    ):
        """Test complete audio processing pipeline."""
        # Mock returns
        mock_audio = Mock(spec=AudioSegment)
        mock_load.return_value = mock_audio
        mock_trim.return_value = (mock_audio, 1500)
        mock_extract.return_value = ["/tmp/chunk1.wav", "/tmp/chunk2.wav"]

        result = self.processor.process_audio_for_language_detection("/test/audio.wav")

        assert result == (mock_audio, 1500, ["/tmp/chunk1.wav", "/tmp/chunk2.wav"])
        mock_load.assert_called_once_with("/test/audio.wav")
        mock_trim.assert_called_once_with(mock_audio)
        mock_extract.assert_called_once_with(mock_audio, 1500)

    @patch.object(AudioProcessor, "load_audio")
    def test_process_audio_for_language_detection_error(self, mock_load):
        """Test processing pipeline with error."""
        mock_load.side_effect = Exception("Load failed")

        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.process_audio_for_language_detection("/test/audio.wav")

        assert exc_info.value.error_code == "processing_pipeline_failed"
        assert "Audio processing pipeline failed" in str(exc_info.value)

    def test_create_temp_audio_file_success(self):
        """Test creating temporary audio file."""
        # Create mock audio segment
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.export = Mock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test_audio.wav"
            mock_temp.return_value.__enter__ = Mock(return_value=mock_file)
            mock_temp.return_value.__exit__ = Mock(return_value=None)

            result = self.processor.create_temp_audio_file(mock_audio, "wav")

            assert result == "/tmp/test_audio.wav"
            assert "/tmp/test_audio.wav" in self.processor._temp_files
            mock_audio.export.assert_called_once_with(
                "/tmp/test_audio.wav", format="wav"
            )

    def test_create_temp_audio_file_error(self):
        """Test temporary file creation with error."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.export = Mock(side_effect=Exception("Export failed"))

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test_audio.wav"
            mock_temp.return_value.__enter__ = Mock(return_value=mock_file)
            mock_temp.return_value.__exit__ = Mock(return_value=None)

            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.create_temp_audio_file(mock_audio)

            assert exc_info.value.error_code == "temp_file_creation_failed"
            assert "Failed to create temporary audio file" in str(exc_info.value)

    def test_get_audio_info(self):
        """Test getting audio information."""
        mock_audio = Mock(spec=AudioSegment)
        mock_audio.__len__ = Mock(return_value=30000)  # 30 seconds
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.sample_width = 2
        mock_audio.frame_width = 4
        mock_audio.frame_count = Mock(return_value=1323000)

        info = self.processor.get_audio_info(mock_audio)
        expected = {
            "duration_ms": 30000,
            "duration_seconds": 30.0,
            "frame_rate": 44100,
            "channels": 2,
            "sample_width": 2,
            "frame_width": 4,
            "frame_count": 1323000,
        }
        assert info == expected

    @patch("app.services.audio_processor.cleanup_temp_files")
    def test_cleanup(self, mock_cleanup):
        """Test cleanup of temporary files."""
        # Add some temp files
        self.processor._temp_files = ["/tmp/file1.wav", "/tmp/file2.wav"]

        self.processor.cleanup()

        mock_cleanup.assert_called_once_with(["/tmp/file1.wav", "/tmp/file2.wav"])
        assert self.processor._temp_files == []

    def test_context_manager(self):
        """Test AudioProcessor as context manager."""
        # Add some temp files to test cleanup
        self.processor._temp_files = ["/tmp/file1.wav", "/tmp/file2.wav"]

        with patch.object(self.processor, "cleanup") as mock_cleanup:
            with self.processor as processor:
                assert processor is self.processor
            mock_cleanup.assert_called_once()

    def test_context_manager_with_exception(self):
        """Test context manager cleanup on exception."""
        with patch.object(AudioProcessor, "cleanup") as mock_cleanup:
            try:
                with AudioProcessor():
                    raise ValueError("Test exception")
            except ValueError:
                pass

            mock_cleanup.assert_called_once()


class TestAudioProcessorIntegration:
    """Integration tests for AudioProcessor with real audio data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()

    def teardown_method(self):
        """Clean up after tests."""
        self.processor.cleanup()

    def create_test_audio(self, duration_ms: int = 5000) -> AudioSegment:
        """Create a test audio segment."""
        # Create a simple sine wave audio for testing
        # Generate 1 second of 440Hz sine wave
        tone = Sine(440).to_audio_segment(duration=1000)

        # Repeat to get desired duration
        audio = AudioSegment.empty()
        while len(audio) < duration_ms:
            audio += tone

        return audio[:duration_ms]

    def test_real_audio_processing_pipeline(self):
        """Test the complete pipeline with real audio data."""
        # Create test audio with some silence at the beginning
        silence = AudioSegment.silent(duration=2000)  # 2 seconds silence
        audio_content = self.create_test_audio(10000)  # 10 seconds content
        test_audio = silence + audio_content

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            test_audio.export(tmp.name, format="wav")
            tmp_path = tmp.name

        try:
            # Test the complete pipeline
            original_audio, leading_silence, samples = (
                self.processor.process_audio_for_language_detection(tmp_path)
            )

            # Verify results
            assert isinstance(original_audio, AudioSegment)
            assert len(original_audio) == 12000  # 2s silence + 10s content
            assert isinstance(leading_silence, int)
            assert leading_silence >= 0
            assert isinstance(samples, list)
            assert len(samples) == 3

            # Verify all sample files exist
            for sample_path in samples:
                assert Path(sample_path).exists()
                assert sample_path.endswith(".wav")

        finally:
            os.unlink(tmp_path)

    def test_audio_info_with_real_audio(self):
        """Test audio info extraction with real audio."""
        test_audio = self.create_test_audio(5000)

        info = self.processor.get_audio_info(test_audio)

        assert info["duration_ms"] == 5000
        assert info["duration_seconds"] > 0
        assert info["frame_rate"] > 0
        assert info["channels"] > 0
        assert info["sample_width"] > 0
        assert info["frame_count"] > 0
