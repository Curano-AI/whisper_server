"""Health monitoring service."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import psutil

from app.core.exceptions import ResourceError
from app.models.responses import (
    HealthCheckResponse,
    HealthStatus,
    ServiceHealth,
    SystemResources,
)
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)


class HealthService:
    """Provide application and system health information."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self.model_manager = model_manager or ModelManager()
        self.start_time = time.time()
        self.version = "0.1.0"

    def _get_gpu_usage(self) -> float | None:
        """Return GPU memory usage percent if available."""
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        try:
            used = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return used / total * 100 if total else None
        except Exception:  # pragma: no cover - safety net
            return None

    def _is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()

    def _get_system_resources(self) -> SystemResources:
        """Collect system resource metrics."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
        except Exception as exc:  # pragma: no cover - psutil failure
            logger.error("Resource check failed: %s", exc)
            raise ResourceError("Failed to read system resources") from exc

        return SystemResources(
            cpu_usage_percent=cpu,
            memory_usage_percent=mem.percent,
            memory_available_gb=mem.available / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_available=self._is_gpu_available(),
            gpu_memory_usage_percent=self._get_gpu_usage(),
        )

    def get_health(self) -> HealthCheckResponse:
        """Return overall health information."""
        system = self._get_system_resources()
        health = HealthStatus(
            status="healthy",
            timestamp=datetime.now(UTC),
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
        )
        services = [ServiceHealth(name="transcription", status="healthy")]
        loaded = self.model_manager.list_models()
        return HealthCheckResponse(
            health=health,
            services=services,
            system=system,
            loaded_models=loaded,
        )
