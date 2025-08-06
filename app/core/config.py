"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


def get_device() -> str:
    """Get the device for Torch."""
    import torch  # noqa: PLC0415, RUF100

    return "cuda" if torch.cuda.is_available() else "cpu"


class AppConfig(BaseSettings):
    """Application configuration."""

    # Server configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    debug: bool = Field(default=False, alias="DEBUG")

    # Security configuration
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    api_key: SecretStr | None = Field(default=None, alias="API_KEY")

    # Model configuration
    default_model: str = Field(default="large-v3", alias="DEFAULT_MODEL")
    detector_model: str = Field(default="small", alias="DETECTOR_MODEL")
    device: str = Field(
        default_factory=get_device,
        alias="DEVICE",
    )

    # Processing configuration
    batch_size: int = Field(default=8, alias="BATCH_SIZE")
    min_prob: float = Field(
        default=0.6,
        alias="MIN_PROB",
        description="Language detection confidence threshold",
    )

    # Language detection weighting configuration
    confidence_weight: float = Field(
        default=2.0,
        alias="CONFIDENCE_WEIGHT",
        description="Confidence weighting factor for language detection scoring",
    )

    # Enhanced language detection configuration
    num_language_chunks: int = Field(
        default=6,
        alias="NUM_LANGUAGE_CHUNKS",
        description="Number of audio chunks to sample for language detection",
    )
    min_language_confidence: float = Field(
        default=0.3,
        alias="MIN_LANGUAGE_CONFIDENCE",
        description="Minimum confidence for initial language detection pass",
    )
    dynamic_threshold_enabled: bool = Field(
        default=True,
        alias="DYNAMIC_THRESHOLD_ENABLED",
        description="Enable dynamic confidence threshold adjustment",
    )
    min_consensus_ratio: float = Field(
        default=0.5,
        alias="MIN_CONSENSUS_RATIO",
        description="Minimum ratio of chunks agreeing on language",
    )

    # Audio quality filtering configuration
    enable_quality_filtering: bool = Field(
        default=True,
        alias="ENABLE_QUALITY_FILTERING",
        description="Enable audio chunk quality filtering for language detection",
    )
    min_chunk_rms: int = Field(
        default=100,
        alias="MIN_CHUNK_RMS",
        description="Minimum RMS level for chunk quality validation",
    )
    min_chunk_duration_ms: int = Field(
        default=1000,
        alias="MIN_CHUNK_DURATION_MS",
        description="Minimum chunk duration in milliseconds",
    )
    min_chunk_amplitude: int = Field(
        default=1000,
        alias="MIN_CHUNK_AMPLITUDE",
        description="Minimum peak amplitude for chunk quality validation",
    )
    quality_fallback_threshold: float = Field(
        default=0.5,
        alias="QUALITY_FALLBACK_THRESHOLD",
        description="Use all chunks if more than this fraction are filtered",
    )
    max_file_size: int = Field(
        default=200 * 1024 * 1024, alias="MAX_FILE_SIZE"
    )  # 200MB

    # Audio processing
    silence_min_duration: int = Field(
        default=300, description="detect_nonsilent min_silence_len parameter"
    )
    silence_threshold: int = Field(
        default=-35, description="detect_nonsilent silence_thresh parameter"
    )
    chunk_duration: int = Field(
        default=10_000, description="10 seconds for language detection samples"
    )
    chunk_offset: int = Field(
        default=5_000, description="5 second offset for sample positioning"
    )

    # Language detection
    detector_batch_size: int | None = Field(
        default=4,
        alias="DETECTOR_BATCH_SIZE",
        description="batch_size for language detection (transcribe.py: 4)",
    )
    detector_compute_type: str = Field(
        default="int8", description="compute_type for detector model"
    )

    # ASR options
    asr_beam_size: int = Field(default=2)
    asr_condition_on_previous_text: bool = Field(default=False)
    asr_temperatures: list[float] = Field(default=[0.0])
    asr_no_speech_threshold: float = Field(default=0.6)

    # VAD options
    vad_min_silence_duration_ms: int = Field(default=500)
    vad_speech_pad_ms: int = Field(default=200)

    # Suppression tokens
    suppress_phrases: list[str] = Field(
        default=["дима торжок", "dima torzok", "dima torzhok", "субтитры подогнал"]
    )

    # Logging configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_compute_type(self, device: str | None = None) -> str:
        """Get compute type based on device."""
        target_device = device or self.device
        return "float16" if target_device == "cuda" else "int8"

    def get_asr_options(self) -> dict[str, Any]:
        """Get ASR options dictionary."""
        return {
            "beam_size": self.asr_beam_size,
            "condition_on_previous_text": self.asr_condition_on_previous_text,
            "temperatures": self.asr_temperatures,
            "no_speech_threshold": self.asr_no_speech_threshold,
        }

    def get_vad_options(self) -> dict[str, Any]:
        """Get VAD options dictionary."""
        return {
            "min_silence_duration_ms": self.vad_min_silence_duration_ms,
            "speech_pad_ms": self.vad_speech_pad_ms,
        }

    def get_detector_batch_size(self) -> int:
        """Return effective detector batch size."""
        return self.detector_batch_size or self.batch_size


@lru_cache
def get_settings() -> AppConfig:
    """Get cached application settings."""
    return AppConfig()
