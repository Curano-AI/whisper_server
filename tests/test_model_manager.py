"""Unit tests for ModelManager service."""

from unittest.mock import Mock, patch

import pytest

from app.core.exceptions import ModelLoadError
from app.services.model_manager import ModelManager


class TestModelManager:
    """Test cases for ModelManager service."""

    def setup_method(self):
        self.manager = ModelManager()

    def teardown_method(self):
        self.manager.clear()

    @patch("whisperx.load_model")
    def test_load_model_success(self, mock_load_model):
        """Load a model successfully."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        result = self.manager.load_model("small", device="cpu", compute_type="int8")

        assert result is mock_model
        assert "small" in self.manager.list_models()
        mock_load_model.assert_called_once_with("small", "cpu", compute_type="int8")

    @patch("whisperx.load_model", side_effect=Exception("boom"))
    def test_load_model_failure(self, mock_load_model):
        """Handle load failure with ModelLoadError."""
        with pytest.raises(ModelLoadError):
            self.manager.load_model("small")
        mock_load_model.assert_called_once()

    @patch("whisperx.load_model")
    def test_get_model(self, mock_load_model):
        """Retrieve previously loaded model."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        self.manager.load_model("small")

        retrieved = self.manager.get_model("small")
        assert retrieved is mock_model

    def test_get_model_not_loaded(self):
        """Error when model not loaded."""
        with pytest.raises(ModelLoadError):
            self.manager.get_model("missing")

    @patch("whisperx.load_model")
    def test_unload_model(self, mock_load_model):
        """Unload model and remove from cache."""
        mock_load_model.return_value = Mock()
        self.manager.load_model("small")

        self.manager.unload_model("small")
        assert "small" not in self.manager.list_models()
