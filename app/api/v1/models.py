"""Model management API endpoints."""

from fastapi import APIRouter

from app.core.exceptions import ModelLoadError
from app.models.requests import ModelLoadRequest, ModelUnloadRequest
from app.models.responses import (
    ErrorResponse,
    LoadedModelInfo,
    LoadedModelsResponse,
    ModelLoadResponse,
    ModelUnloadResponse,
)
from app.services import ModelManager

router = APIRouter()

# Global model manager instance for all requests
model_manager = ModelManager()


@router.get("/list", response_model=LoadedModelsResponse)
async def list_models() -> LoadedModelsResponse:
    """List currently loaded models."""
    loaded_models: list[LoadedModelInfo] = []
    total_mem = 0.0
    for name in model_manager.list_models():
        entry = model_manager.get_model_info(name)
        memory_usage = None  # AICODE-NOTE: memory tracking not implemented yet
        if memory_usage:
            total_mem += memory_usage
        loaded_models.append(
            LoadedModelInfo(
                model_name=name,
                device=entry["device"],
                compute_type=entry["compute_type"],
                load_time=entry["load_time"],
                last_used=entry["last_used"],
                memory_usage_mb=memory_usage,
            )
        )

    return LoadedModelsResponse(
        loaded_models=loaded_models,
        total_memory_usage_mb=total_mem if total_mem else None,
    )


@router.post(
    "/load",
    response_model=ModelLoadResponse,
    responses={500: {"model": ErrorResponse}},
)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """Load a WhisperX model."""
    model_manager.load_model(request.model_name, request.device, request.compute_type)
    entry = model_manager.get_model_info(request.model_name)
    model_info = LoadedModelInfo(
        model_name=request.model_name,
        device=entry["device"],
        compute_type=entry["compute_type"],
        load_time=entry["load_time"],
        last_used=entry["last_used"],
        memory_usage_mb=None,  # AICODE-TODO: calculate GPU/CPU memory usage
    )
    return ModelLoadResponse(
        success=True,
        message="Model loaded successfully",
        model_info=model_info,
    )


@router.post(
    "/unload",
    response_model=ModelUnloadResponse,
    responses={500: {"model": ErrorResponse}},
)
async def unload_model(
    request: ModelUnloadRequest,
) -> ModelUnloadResponse:
    """Unload a WhisperX model."""
    if request.model_name not in model_manager.list_models():
        raise ModelLoadError(
            f"Model not loaded: {request.model_name}",
            model_name=request.model_name,
            error_code="not_loaded",
        )

    model_manager.unload_model(request.model_name)
    return ModelUnloadResponse(
        success=True,
        message="Model unloaded successfully",
        freed_memory_mb=None,
    )
