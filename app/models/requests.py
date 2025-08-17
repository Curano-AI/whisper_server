"""Pydantic request models for OpenAI-compatible transcription API."""

from typing import Any

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptionRequest(BaseModel):
    """OpenAI-compatible transcription request model."""

    file: UploadFile = Field(..., description="Audio file to transcribe")
    model: str | None = Field(default="whisper-1", description="ID of the model to use")
    language: str | None = Field(
        default=None, description="Language of the input audio (ISO-639-1 format)"
    )
    prompt: str | None = Field(
        default=None,
        max_length=244,
        description="Optional text to guide the model's style",
    )
    response_format: str | None = Field(
        default="json", description="Format of the transcript output"
    )
    temperature: float | None = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature between 0 and 1"
    )
    timestamp_granularities: list[str] | None = Field(
        default=["segment"], description="Timestamp granularities to populate"
    )

    # WhisperX-specific parameters
    beam_size: int | None = Field(
        default=2, ge=1, le=10, description="Beam size for decoding"
    )
    suppress_tokens: list[str] | None = Field(
        default=None, description="List of phrases to suppress in transcription"
    )
    vad_options: dict[str, Any] | None = Field(
        default=None, description="Voice Activity Detection options"
    )

    # Speaker diarization parameters
    enable_diarization: bool = Field(
        default=False, description="Enable speaker diarization"
    )
    min_speakers: int | None = Field(
        default=None, ge=1, le=20, description="Minimum number of speakers"
    )
    max_speakers: int | None = Field(
        default=None, ge=1, le=20, description="Maximum number of speakers"
    )
    num_speakers: int | None = Field(
        default=None, ge=1, le=20, description="Exact number of speakers (if known)"
    )
    hf_token: str | None = Field(
        default=None, description="HuggingFace token for diarization models"
    )

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        """Validate response format is supported."""
        allowed_formats = ["json", "text", "srt", "verbose_json", "vtt"]
        if v not in allowed_formats:
            raise ValueError(f"response_format must be one of {allowed_formats}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate language code format."""
        if v is None:
            return v

        # Common ISO-639-1 language codes supported by Whisper
        supported_languages = {
            "af",
            "am",
            "ar",
            "as",
            "az",
            "ba",
            "be",
            "bg",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jw",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "la",
            "lb",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sa",
            "sd",
            "si",
            "sk",
            "sl",
            "sn",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yo",
            "zh",
        }

        if v not in supported_languages:
            raise ValueError(f"Unsupported language code: {v}")
        return v

    @field_validator("timestamp_granularities")
    @classmethod
    def validate_timestamp_granularities(cls, v):
        """Validate timestamp granularities."""
        if v is None:
            return ["segment"]

        allowed_granularities = ["word", "segment"]
        for granularity in v:
            if granularity not in allowed_granularities:
                raise ValueError(
                    f"timestamp_granularities must contain only {allowed_granularities}"
                )
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        """Validate model name."""
        # OpenAI compatible model names and WhisperX model names
        allowed_models = [
            "whisper-1",  # OpenAI default
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
        ]

        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow UploadFile


class TranslationRequest(BaseModel):
    """OpenAI-compatible translation request model."""

    file: UploadFile = Field(..., description="Audio file to translate")
    model: str | None = Field(default="whisper-1", description="ID of the model to use")
    prompt: str | None = Field(
        default=None,
        max_length=244,
        description="Optional text to guide the model's style",
    )
    response_format: str | None = Field(
        default="json", description="Format of the transcript output"
    )
    temperature: float | None = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature between 0 and 1"
    )

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        """Validate response format is supported."""
        allowed_formats = ["json", "text", "srt", "verbose_json", "vtt"]
        if v not in allowed_formats:
            raise ValueError(f"response_format must be one of {allowed_formats}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        """Validate model name."""
        allowed_models = [
            "whisper-1",  # OpenAI default
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
        ]

        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow UploadFile


class ModelLoadRequest(BaseModel):
    """Request model for loading a specific model."""

    model_name: str = Field(..., description="Name of the model to load")
    device: str | None = Field(
        default=None, description="Device to load model on (cuda/cpu)"
    )
    compute_type: str | None = Field(
        default=None, description="Compute type for model (float16/int8)"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name."""
        allowed_models = [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
        ]

        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        """Validate device specification."""
        if v is None:
            return v

        allowed_devices = ["cuda", "cpu"]
        if v not in allowed_devices:
            raise ValueError(f"Device must be one of {allowed_devices}")
        return v

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v):
        """Validate compute type."""
        if v is None:
            return v

        allowed_types = ["float16", "int8", "float32"]
        if v not in allowed_types:
            raise ValueError(f"Compute type must be one of {allowed_types}")
        return v


class ModelUnloadRequest(BaseModel):
    """Request model for unloading a specific model."""

    model_name: str = Field(..., description="Name of the model to unload")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name."""
        allowed_models = [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
        ]

        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v
