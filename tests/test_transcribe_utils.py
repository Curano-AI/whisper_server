"""Unit tests for transcribe utility functions."""

from unittest.mock import Mock, patch

import pytest
from pydub import AudioSegment

from app.utils.transcribe_utils import (
    DEFAULT_SUPPRESS_PHRASES,
    clean,
    cleanup_temp_files,
    export_chunk,
    get_suppress_tokens,
    ts,
)


class TestTimestampFormatting:
    """Test cases for timestamp formatting function."""

    def test_ts_basic_formatting(self):
        """Test basic timestamp formatting."""
        assert ts(0.0) == "00:00:00,000"
        assert ts(1.0) == "00:00:01,000"
        assert ts(60.0) == "00:01:00,000"
        assert ts(3600.0) == "01:00:00,000"

    def test_ts_with_milliseconds(self):
        """Test timestamp formatting with milliseconds."""
        assert ts(1.123) == "00:00:01,123"
        assert ts(65.456) == "00:01:05,456"
        assert ts(3661.789) == "01:01:01,789"

    def test_ts_edge_cases(self):
        """Test timestamp formatting edge cases."""
        assert ts(0.001) == "00:00:00,001"
        assert ts(59.999) == "00:00:59,999"
        assert ts(3599.999) == "00:59:59,999"

    def test_ts_large_values(self):
        """Test timestamp formatting with large values."""
        assert ts(7200.0) == "02:00:00,000"  # 2 hours
        assert ts(10800.5) == "03:00:00,500"  # 3 hours with milliseconds


class TestTextCleaning:
    """Test cases for text cleaning function."""

    def test_clean_default_phrases(self):
        """Test cleaning with default suppress phrases."""
        text = "Hello дима торжок world"
        assert clean(text) == "Hello world"

    def test_clean_multiple_phrases(self):
        """Test cleaning multiple suppress phrases."""
        text = "Start дима торжок middle dima torzok end"
        assert clean(text) == "Start middle end"

    def test_clean_custom_phrases(self):
        """Test cleaning with custom suppress phrases."""
        text = "Hello unwanted world"
        result = clean(text, suppress_phrases=["unwanted"])
        assert result == "Hello world"

    def test_clean_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "  Hello   world  "
        assert clean(text) == "Hello world"

    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        assert clean("") == ""

    def test_clean_no_matches(self):
        """Test cleaning text with no suppress phrases."""
        text = "Clean text without issues"
        assert clean(text) == "Clean text without issues"

    def test_clean_case_sensitive(self):
        """Test that cleaning is case sensitive."""
        text = "Hello ДИМА ТОРЖОК world"
        # Should not remove uppercase version
        assert "ДИМА ТОРЖОК" in clean(text)


class TestAudioChunkExport:
    """Test cases for audio chunk export function."""

    @pytest.fixture
    def mock_audio(self):
        """Create a mock AudioSegment for testing."""
        audio = Mock(spec=AudioSegment)
        audio.__getitem__ = Mock(return_value=audio)
        audio.export = Mock()
        return audio

    @patch("tempfile.NamedTemporaryFile")
    def test_export_chunk_basic(self, mock_tempfile, mock_audio):
        """Test basic audio chunk export."""
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        result = export_chunk(mock_audio, 1000, 5000)

        assert result == "/tmp/test.wav"
        mock_audio.__getitem__.assert_called_once_with(slice(1000, 6000))
        mock_audio.export.assert_called_once_with("/tmp/test.wav", format="wav")

    @patch("tempfile.NamedTemporaryFile")
    def test_export_chunk_default_duration(self, mock_tempfile, mock_audio):
        """Test audio chunk export with default duration."""
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        export_chunk(mock_audio, 2000)

        # Default duration is 10_000ms
        mock_audio.__getitem__.assert_called_once_with(slice(2000, 12000))

    @patch("tempfile.NamedTemporaryFile")
    def test_export_chunk_file_suffix(self, mock_tempfile, mock_audio):
        """Test that temporary file has .wav suffix."""
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        export_chunk(mock_audio, 0)

        mock_tempfile.assert_called_once_with(delete=False, suffix=".wav")


class TestSuppressTokens:
    """Test cases for suppress token generation."""

    @patch("app.utils.transcribe_utils.get_tokenizer")
    def test_get_suppress_tokens_default(self, mock_get_tokenizer):
        """Test suppress token generation with default phrases."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # дима торжок
            [4, 5],  # dima torzok
            [6, 7, 8],  # dima torzhok
            [9, 10],  # субтитры подогнал
        ]
        mock_get_tokenizer.return_value = mock_tokenizer

        result = get_suppress_tokens()

        expected = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert result == expected
        mock_get_tokenizer.assert_called_once_with(multilingual=True)

    @patch("app.utils.transcribe_utils.get_tokenizer")
    def test_get_suppress_tokens_custom(self, mock_get_tokenizer):
        """Test suppress token generation with custom phrases."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [[100, 101], [200, 201]]  # hello  # world
        mock_get_tokenizer.return_value = mock_tokenizer

        result = get_suppress_tokens(["hello", "world"])

        expected = sorted([100, 101, 200, 201])
        assert result == expected

    @patch("app.utils.transcribe_utils.get_tokenizer")
    def test_get_suppress_tokens_duplicates(self, mock_get_tokenizer):
        """Test that duplicate tokens are removed."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # phrase1
            [2, 3, 4],  # phrase2 (overlapping tokens)
        ]
        mock_get_tokenizer.return_value = mock_tokenizer

        result = get_suppress_tokens(["phrase1", "phrase2"])

        expected = sorted([1, 2, 3, 4])  # No duplicates
        assert result == expected

    @patch("app.utils.transcribe_utils.get_tokenizer")
    def test_get_suppress_tokens_empty_list(self, mock_get_tokenizer):
        """Test suppress token generation with empty phrase list."""
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        result = get_suppress_tokens([])

        assert result == []


class TestCleanupTempFiles:
    """Test cases for temporary file cleanup."""

    @patch("pathlib.Path.exists")
    @patch("os.unlink")
    def test_cleanup_existing_files(self, mock_unlink, mock_exists):
        """Test cleanup of existing files."""
        mock_exists.return_value = True
        files = ["/tmp/file1.wav", "/tmp/file2.wav"]

        cleanup_temp_files(files)

        assert mock_exists.call_count == 2
        assert mock_unlink.call_count == 2
        mock_unlink.assert_any_call("/tmp/file1.wav")
        mock_unlink.assert_any_call("/tmp/file2.wav")

    @patch("pathlib.Path.exists")
    @patch("os.unlink")
    def test_cleanup_nonexistent_files(self, mock_unlink, mock_exists):
        """Test cleanup handles nonexistent files gracefully."""
        mock_exists.return_value = False
        files = ["/tmp/nonexistent.wav"]

        cleanup_temp_files(files)

        mock_exists.assert_called_once()
        mock_unlink.assert_not_called()

    @patch("pathlib.Path.exists")
    @patch("os.unlink")
    def test_cleanup_handles_os_error(self, mock_unlink, mock_exists):
        """Test cleanup handles OS errors gracefully."""
        mock_exists.return_value = True
        mock_unlink.side_effect = OSError("Permission denied")
        files = ["/tmp/protected.wav"]

        # Should not raise exception
        cleanup_temp_files(files)

        mock_unlink.assert_called_once_with("/tmp/protected.wav")

    def test_cleanup_empty_list(self):
        """Test cleanup with empty file list."""
        # Should not raise exception
        cleanup_temp_files([])


class TestDefaultSuppressPhrases:
    """Test cases for default suppress phrases constant."""

    def test_default_suppress_phrases_content(self):
        """Test that default suppress phrases match original transcribe.py."""
        expected = ["дима торжок", "dima torzok", "dima torzhok", "субтитры подогнал"]
        assert expected == DEFAULT_SUPPRESS_PHRASES

    def test_default_suppress_phrases_immutable(self):
        """Test that default suppress phrases list exists and is accessible."""
        assert isinstance(DEFAULT_SUPPRESS_PHRASES, list)
        assert len(DEFAULT_SUPPRESS_PHRASES) == 4
