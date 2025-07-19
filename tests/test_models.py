"""Tests for Pydantic request and response models."""

from datetime import datetime
from io import BytesIO

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from app.models.requests import (
    ModelLoadRequest,
    ModelUnloadRequest,
    TranscriptionRequest,
    TranslationRequest,
)
from app.models.responses import (
    ErrorDetail,
    ErrorResponse,
    HealthCheckResponse,
    HealthStatus,
    LoadedModelInfo,
    LoadedModelsResponse,
    ModelInfo,
    ModelListResponse,
    ModelLoadResponse,
    ModelUnloadResponse,
    ServiceHealth,
    SystemResources,
    TranscriptionResponse,
    TranscriptionSegment,
    TranslationResponse,
    VerboseTranscriptionResponse,
    Word,
)


class TestTranscriptionRequest:
    """Test TranscriptionRequest model validation."""

    def test_valid_transcription_request(self):
        """Test valid transcription request creation."""
        # Create a mock UploadFile
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        request = TranscriptionRequest(
            file=upload_file,
            model="whisper-1",
            language="en",
            temperature=0.5,
            response_format="json",
        )

        assert request.file == upload_file
        assert request.model == "whisper-1"
        assert request.language == "en"
        assert request.temperature == 0.5
        assert request.response_format == "json"

    def test_default_values(self):
        """Test default values are set correctly."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        request = TranscriptionRequest(file=upload_file)

        assert request.model == "whisper-1"
        assert request.language is None
        assert request.temperature == 0.0
        assert request.response_format == "json"
        assert request.timestamp_granularities == ["segment"]
        assert request.beam_size == 2

    def test_invalid_response_format(self):
        """Test validation error for invalid response format."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        with pytest.raises(ValidationError) as exc_info:
            TranscriptionRequest(file=upload_file, response_format="invalid_format")

        assert "response_format must be one of" in str(exc_info.value)

    def test_invalid_temperature(self):
        """Test validation error for invalid temperature."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        with pytest.raises(ValidationError):
            TranscriptionRequest(file=upload_file, temperature=1.5)  # > 1.0

        with pytest.raises(ValidationError):
            TranscriptionRequest(file=upload_file, temperature=-0.1)  # < 0.0

    def test_invalid_language(self):
        """Test validation error for invalid language code."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        with pytest.raises(ValidationError) as exc_info:
            TranscriptionRequest(file=upload_file, language="invalid_lang")

        assert "Unsupported language code" in str(exc_info.value)

    def test_invalid_model(self):
        """Test validation error for invalid model name."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        with pytest.raises(ValidationError) as exc_info:
            TranscriptionRequest(file=upload_file, model="invalid_model")

        assert "Model must be one of" in str(exc_info.value)

    def test_invalid_timestamp_granularities(self):
        """Test validation error for invalid timestamp granularities."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        with pytest.raises(ValidationError) as exc_info:
            TranscriptionRequest(
                file=upload_file, timestamp_granularities=["invalid_granularity"]
            )

        assert "timestamp_granularities must contain only" in str(exc_info.value)

    def test_beam_size_validation(self):
        """Test beam size validation."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        # Valid beam size
        request = TranscriptionRequest(file=upload_file, beam_size=5)
        assert request.beam_size == 5

        # Invalid beam sizes
        with pytest.raises(ValidationError):
            TranscriptionRequest(file=upload_file, beam_size=0)  # < 1

        with pytest.raises(ValidationError):
            TranscriptionRequest(file=upload_file, beam_size=11)  # > 10

    def test_prompt_max_length(self):
        """Test prompt maximum length validation."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        # Valid prompt
        request = TranscriptionRequest(file=upload_file, prompt="Short prompt")
        assert request.prompt == "Short prompt"

        # Invalid prompt (too long)
        long_prompt = "x" * 245  # > 244 characters
        with pytest.raises(ValidationError):
            TranscriptionRequest(file=upload_file, prompt=long_prompt)


class TestTranslationRequest:
    """Test TranslationRequest model validation."""

    def test_valid_translation_request(self):
        """Test valid translation request creation."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        request = TranslationRequest(
            file=upload_file, model="whisper-1", temperature=0.3, response_format="text"
        )

        assert request.file == upload_file
        assert request.model == "whisper-1"
        assert request.temperature == 0.3
        assert request.response_format == "text"

    def test_translation_default_values(self):
        """Test default values for translation request."""
        file_content = b"fake audio content"
        upload_file = UploadFile(filename="test.wav", file=BytesIO(file_content))

        request = TranslationRequest(file=upload_file)

        assert request.model == "whisper-1"
        assert request.temperature == 0.0
        assert request.response_format == "json"


class TestModelLoadRequest:
    """Test ModelLoadRequest model validation."""

    def test_valid_model_load_request(self):
        """Test valid model load request."""
        request = ModelLoadRequest(
            model_name="large-v3", device="cuda", compute_type="float16"
        )

        assert request.model_name == "large-v3"
        assert request.device == "cuda"
        assert request.compute_type == "float16"

    def test_invalid_model_name(self):
        """Test validation error for invalid model name."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLoadRequest(model_name="invalid_model")

        assert "Model must be one of" in str(exc_info.value)

    def test_invalid_device(self):
        """Test validation error for invalid device."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLoadRequest(model_name="large-v3", device="invalid_device")

        assert "Device must be one of" in str(exc_info.value)

    def test_invalid_compute_type(self):
        """Test validation error for invalid compute type."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLoadRequest(model_name="large-v3", compute_type="invalid_type")

        assert "Compute type must be one of" in str(exc_info.value)


class TestModelUnloadRequest:
    """Test ModelUnloadRequest model validation."""

    def test_valid_model_unload_request(self):
        """Test valid model unload request."""
        request = ModelUnloadRequest(model_name="large-v3")
        assert request.model_name == "large-v3"

    def test_invalid_model_name(self):
        """Test validation error for invalid model name."""
        with pytest.raises(ValidationError) as exc_info:
            ModelUnloadRequest(model_name="invalid_model")

        assert "Model must be one of" in str(exc_info.value)


class TestResponseModels:
    """Test response model creation and validation."""

    def test_word_model(self):
        """Test Word model creation."""
        word = Word(word="hello", start=1.0, end=1.5)
        assert word.word == "hello"
        assert word.start == 1.0
        assert word.end == 1.5

    def test_transcription_segment(self):
        """Test TranscriptionSegment model creation."""
        segment = TranscriptionSegment(
            id=1,
            seek=0.0,
            start=1.0,
            end=2.0,
            text="Hello world",
            tokens=[1, 2, 3],
            temperature=0.0,
            avg_logprob=-0.5,
            compression_ratio=1.2,
            no_speech_prob=0.1,
        )

        assert segment.id == 1
        assert segment.text == "Hello world"
        assert segment.tokens == [1, 2, 3]

    def test_transcription_response(self):
        """Test TranscriptionResponse model creation."""
        segment = TranscriptionSegment(
            id=1,
            seek=0.0,
            start=1.0,
            end=2.0,
            text="Hello world",
            tokens=[1, 2, 3],
            temperature=0.0,
            avg_logprob=-0.5,
            compression_ratio=1.2,
            no_speech_prob=0.1,
        )

        response = TranscriptionResponse(
            text="Hello world", segments=[segment], language="en"
        )

        assert response.text == "Hello world"
        assert response.segments is not None
        assert len(response.segments) == 1
        assert response.language == "en"

    def test_verbose_transcription_response(self):
        """Test VerboseTranscriptionResponse model creation."""
        segment = TranscriptionSegment(
            id=1,
            seek=0.0,
            start=1.0,
            end=2.0,
            text="Hello world",
            tokens=[1, 2, 3],
            temperature=0.0,
            avg_logprob=-0.5,
            compression_ratio=1.2,
            no_speech_prob=0.1,
        )

        response = VerboseTranscriptionResponse(
            task="transcribe",
            language="en",
            duration=10.5,
            text="Hello world",
            segments=[segment],
        )

        assert response.task == "transcribe"
        assert response.language == "en"
        assert response.duration == 10.5
        assert response.text == "Hello world"

    def test_model_info(self):
        """Test ModelInfo model creation."""
        model_info = ModelInfo(id="whisper-1", created=1677610602, root="whisper-1")

        assert model_info.id == "whisper-1"
        assert model_info.object == "model"
        assert model_info.owned_by == "whisperx"

    def test_loaded_model_info(self):
        """Test LoadedModelInfo model creation."""
        now = datetime.now()
        model_info = LoadedModelInfo(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            load_time=now,
            last_used=now,
            memory_usage_mb=1024.5,
        )

        assert model_info.model_name == "large-v3"
        assert model_info.device == "cuda"
        assert model_info.memory_usage_mb == 1024.5

    def test_health_check_response(self):
        """Test HealthCheckResponse model creation."""
        health = HealthStatus(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_seconds=3600.0,
        )

        service = ServiceHealth(
            name="transcription", status="healthy", response_time_ms=50.0
        )

        system = SystemResources(
            cpu_usage_percent=25.0,
            memory_usage_percent=60.0,
            memory_available_gb=8.0,
            disk_usage_percent=40.0,
            gpu_available=True,
            gpu_memory_usage_percent=45.0,
        )

        response = HealthCheckResponse(
            health=health, services=[service], system=system, loaded_models=["large-v3"]
        )

        assert response.health.status == "healthy"
        assert len(response.services) == 1
        assert response.system.cpu_usage_percent == 25.0

    def test_error_response(self):
        """Test ErrorResponse model creation."""
        error_detail = ErrorDetail(
            type="validation_error", message="Invalid input", param="temperature"
        )

        error_response = ErrorResponse(error=error_detail)

        assert error_response.error.type == "validation_error"
        assert error_response.error.message == "Invalid input"
        assert error_response.error.param == "temperature"

    def test_model_load_response(self):
        """Test ModelLoadResponse model creation."""
        now = datetime.now()
        model_info = LoadedModelInfo(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            load_time=now,
            last_used=now,
            memory_usage_mb=1024.5,
        )

        response = ModelLoadResponse(
            success=True, message="Model loaded successfully", model_info=model_info
        )

        assert response.success is True
        assert response.model_info is not None
        assert response.model_info.model_name == "large-v3"
        assert response.model_info.memory_usage_mb == 1024.5

    def test_model_unload_response(self):
        """Test ModelUnloadResponse model creation."""
        response = ModelUnloadResponse(
            success=True, message="Model unloaded successfully", freed_memory_mb=1024.5
        )

        assert response.success is True
        assert response.freed_memory_mb == 1024.5

    def test_translation_response(self):
        """Test TranslationResponse model creation."""
        response = TranslationResponse(text="Hello world")
        assert response.text == "Hello world"

    def test_model_list_response(self):
        """Test ModelListResponse model creation."""
        model1 = ModelInfo(id="whisper-1", created=1677610602, root="whisper-1")
        model2 = ModelInfo(id="large-v3", created=1677610603, root="large-v3")

        response = ModelListResponse(object="list", data=[model1, model2])

        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "whisper-1"

    def test_loaded_models_response(self):
        """Test LoadedModelsResponse model creation."""
        now = datetime.now()
        model_info = LoadedModelInfo(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            load_time=now,
            last_used=now,
            memory_usage_mb=1024.5,
        )

        response = LoadedModelsResponse(loaded_models=[model_info])

        assert len(response.loaded_models) == 1
        assert response.loaded_models[0].model_name == "large-v3"
