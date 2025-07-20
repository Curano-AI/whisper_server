"""Model management service for WhisperX models."""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

import torch
import whisperx

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
    ) -> Any:
        """Load and cache a WhisperX model.

        Args:
            model_name: Name of the model to load (defaults to settings.default_model)
            device: Device to load model on (defaults to settings.device)
            compute_type: Precision type (defaults based on device)

        Returns:
            Loaded model instance.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        name = model_name or self.settings.default_model
        if name in self._models:
            logger.info("Model %s already loaded", name)
            return self._models[name]["model"]

        target_device = device or self.settings.device
        ctype = compute_type or self.settings.get_compute_type(target_device)

        try:
            logger.info("Loading model %s on %s with %s", name, target_device, ctype)
            model = whisperx.load_model(name, target_device, compute_type=ctype)
        except Exception as exc:  # pragma: no cover - error path
            logger.error("Failed to load model %s: %s", name, exc)
            raise ModelLoadError(
                f"Failed to load model {name}: {exc}",
                model_name=name,
                error_code="load_failed",
            ) from exc

        self._models[name] = {
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
        entry = self._models.get(name)
        if not entry:
            raise ModelLoadError(
                f"Model not loaded: {name}", model_name=name, error_code="not_loaded"
            )
        entry["last_used"] = datetime.now(UTC)
        return entry["model"]

    def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """Return metadata dictionary for a loaded model."""
        name = model_name or self.settings.default_model
        self.get_model(name)
        return self._models[name]

    def unload_model(self, model_name: str) -> None:
        """Unload a model and free associated memory."""
        entry = self._models.pop(model_name, None)
        if entry and entry["device"] == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def list_models(self) -> list[str]:
        """List currently loaded models."""
        return list(self._models.keys())

    def clear(self) -> None:
        """Unload all loaded models."""
        for name in list(self._models):
            self.unload_model(name)
