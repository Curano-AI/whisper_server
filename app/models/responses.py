"""Pydantic response models for OpenAI-compatible transcription API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Word(BaseModel):
    """Word-level timestamp information."""

    word: str = Field(..., description="The text content of the word")
    start: float = Field(..., description="Start time of the word in seconds")
    end: float = Field(..., description="End time of the word in seconds")


class TranscriptionSegment(BaseModel):
    """Transcription segment with timing and metadata."""

    id: int = Field(..., description="Unique identifier of the segment")
    seek: float = Field(..., description="Seek offset of the segment")
    start: float = Field(..., description="Start time of the segment in seconds")
    end: float = Field(..., description="End time of the segment in seconds")
    text: str = Field(..., description="Text content of the segment")
    tokens: list[int] = Field(..., description="Token IDs of the segment")
    temperature: float = Field(..., description="Temperature used for the segment")
    avg_logprob: float = Field(
        ..., description="Average log probability of the segment"
    )
    compression_ratio: float = Field(
        ..., description="Compression ratio of the segment"
    )
    no_speech_prob: float = Field(..., description="Probability of no speech")
    words: list[Word] | None = Field(
        default=None, description="Word-level timestamps (if requested)"
    )


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response (JSON format)."""

    text: str = Field(..., description="The transcribed text")
    segments: list[TranscriptionSegment] | None = Field(
        default=None, description="Segments of the transcription with timestamps"
    )
    words: list[Word] | None = Field(
        default=None, description="Word-level timestamps (if requested)"
    )
    language: str | None = Field(
        default=None, description="Detected or specified language"
    )


class VerboseTranscriptionResponse(BaseModel):
    """OpenAI-compatible verbose transcription response."""

    task: str = Field(default="transcribe", description="The task performed")
    language: str = Field(..., description="Detected or specified language")
    duration: float = Field(..., description="Duration of the audio in seconds")
    text: str = Field(..., description="The transcribed text")
    segments: list[TranscriptionSegment] = Field(
        ..., description="Segments of the transcription with timestamps"
    )
    words: list[Word] | None = Field(
        default=None, description="Word-level timestamps (if requested)"
    )


class TranslationResponse(BaseModel):
    """OpenAI-compatible translation response."""

    text: str = Field(..., description="The translated text")
    segments: list[TranscriptionSegment] | None = Field(
        default=None, description="Segments of the translation with timestamps"
    )
    words: list[Word] | None = Field(
        default=None, description="Word-level timestamps (if requested)"
    )
    language: str | None = Field(default=None, description="Source language detected")


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(
        ..., description="Unix timestamp of when the model was created"
    )
    owned_by: str = Field(
        default="whisperx", description="Organization that owns the model"
    )
    permission: list[dict[str, Any]] = Field(
        default_factory=list, description="Model permissions"
    )
    root: str = Field(..., description="Root model identifier")
    parent: str | None = Field(default=None, description="Parent model")


class ModelListResponse(BaseModel):
    """Response for listing available models."""

    object: str = Field(default="list", description="Object type")
    data: list[ModelInfo] = Field(..., description="List of available models")


class LoadedModelInfo(BaseModel):
    """Information about a currently loaded model."""

    model_name: str = Field(..., description="Name of the model")
    device: str = Field(..., description="Device the model is loaded on")
    compute_type: str = Field(..., description="Compute type used")
    load_time: datetime = Field(..., description="When the model was loaded")
    last_used: datetime = Field(..., description="When the model was last used")
    memory_usage_mb: float | None = Field(
        default=None, description="Approximate memory usage in MB"
    )


class LoadedModelsResponse(BaseModel):
    """Response for listing currently loaded models."""

    loaded_models: list[LoadedModelInfo] = Field(
        ..., description="List of currently loaded models"
    )
    total_memory_usage_mb: float | None = Field(
        default=None, description="Total memory usage of all loaded models in MB"
    )


class ModelLoadResponse(BaseModel):
    """Response for model loading operation."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    model_info: LoadedModelInfo | None = Field(
        default=None, description="Information about the loaded model"
    )


class ModelUnloadResponse(BaseModel):
    """Response for model unloading operation."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    freed_memory_mb: float | None = Field(
        default=None, description="Amount of memory freed in MB"
    )


class HealthStatus(BaseModel):
    """Health check status information."""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Timestamp of the health check")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")


class ServiceHealth(BaseModel):
    """Individual service health information."""

    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    message: str | None = Field(default=None, description="Status message")
    response_time_ms: float | None = Field(
        default=None, description="Service response time in milliseconds"
    )


class SystemResources(BaseModel):
    """System resource information."""

    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    memory_available_gb: float = Field(..., description="Available memory in GB")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_memory_usage_percent: float | None = Field(
        default=None, description="GPU memory usage percentage"
    )


class HealthCheckResponse(BaseModel):
    """Complete health check response."""

    health: HealthStatus = Field(..., description="Overall health status")
    services: list[ServiceHealth] = Field(..., description="Individual service health")
    system: SystemResources = Field(..., description="System resource information")
    loaded_models: list[str] = Field(..., description="Currently loaded model names")


class ErrorDetail(BaseModel):
    """Error detail information."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: str | None = Field(
        default=None, description="Parameter that caused the error"
    )
    code: str | None = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""

    error: ErrorDetail = Field(..., description="Error details")


# Response format unions for different output types
TranscriptionOutput = (
    TranscriptionResponse
    | VerboseTranscriptionResponse
    | str  # For text, srt, vtt formats
)

TranslationOutput = TranslationResponse | str  # For text, srt, vtt formats
