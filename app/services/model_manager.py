"""Model management service for WhisperX models."""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage WhisperX model lifecycle and caching."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._models: dict[str, dict[str, Any]] = {}

    def load_model(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        asr_options: dict[str, Any] | None = None,
        vad_options: dict[str, Any] | None = None,
    ) -> Any:
        """Load and cache a WhisperX model.

        Args:
            model_name: Name of the model to load (defaults to settings.default_model)
            device: Device to load model on (defaults to settings.device)
            compute_type: Precision type (defaults based on device)
            asr_options: ASR options to pass to whisperx.load_model
            vad_options: VAD options to pass to whisperx.load_model

        Returns:
            Loaded model instance.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        import whisperx  # noqa: PLC0415

        name = model_name or self.settings.default_model
        # Create cache key including options for proper caching
        cache_key = f"{name}_{device or self.settings.device}_{compute_type or 'auto'}"

        if cache_key in self._models:
            logger.info("Model %s already loaded", name)
            return self._models[cache_key]["model"]

        target_device = device or self.settings.device
        ctype = compute_type or self.settings.get_compute_type(target_device)

        try:
            logger.info("Loading model %s on %s with %s", name, target_device, ctype)

            # Load model with options
            if asr_options or vad_options:
                model = whisperx.load_model(
                    name,
                    target_device,
                    compute_type=ctype,
                    asr_options=asr_options or {},
                    vad_options=vad_options or {},
                )
            else:
                model = whisperx.load_model(name, target_device, compute_type=ctype)

        except Exception as exc:  # pragma: no cover - error path
            logger.error("Failed to load model %s: %s", name, exc)
            raise ModelLoadError(
                f"Failed to load model {name}: {exc}",
                model_name=name,
                error_code="load_failed",
            ) from exc

        self._models[cache_key] = {
            "model": model,
            "device": target_device,
            "compute_type": ctype,
            "load_time": datetime.now(UTC),
            "last_used": datetime.now(UTC),
        }
        return model

    def get_model(self, model_name: str | None = None) -> Any:
        """Retrieve a loaded model by name."""
        name = model_name or self.settings.default_model
        # Try to find any loaded variant of this model
        for cache_key, entry in self._models.items():
            if cache_key.startswith(f"{name}_"):
                entry["last_used"] = datetime.now(UTC)
                return entry["model"]
        raise ModelLoadError(
            f"Model not loaded: {name}", model_name=name, error_code="not_loaded"
        )

    def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """Return metadata dictionary for a loaded model."""
        name = model_name or self.settings.default_model
        # Try to find any loaded variant of this model
        for cache_key, entry in self._models.items():
            if cache_key.startswith(f"{name}_"):
                return entry
        raise ModelLoadError(
            f"Model not loaded: {name}", model_name=name, error_code="not_loaded"
        )

    def unload_model(self, model_name: str) -> None:
        """Unload a model and free associated memory."""
        import torch  # noqa: PLC0415

        # Find and remove all variants of this model
        keys_to_remove = [
            key for key in self._models if key.startswith(f"{model_name}_")
        ]
        for cache_key in keys_to_remove:
            entry = self._models.pop(cache_key, None)
            if entry and entry["device"] == "cuda":
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()

    def list_models(self) -> list[str]:
        """List currently loaded models."""
        # Extract unique model names from cache keys
        model_names = set()
        for cache_key in self._models:
            model_name = cache_key.split("_")[0]
            model_names.add(model_name)
        return list(model_names)

    def clear(self) -> None:
        """Unload all loaded models."""
        # Extract unique model names from cache keys first
        model_names = set()
        for cache_key in list(self._models.keys()):
            model_name = cache_key.split("_")[0]
            model_names.add(model_name)

        # Then unload each unique model
        for model_name in model_names:
            self.unload_model(model_name)
