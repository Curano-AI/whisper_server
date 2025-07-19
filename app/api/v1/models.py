"""Model management API endpoints."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.exceptions import ModelLoadError
from app.models.requests import ModelLoadRequest, ModelUnloadRequest
from app.models.responses import (
    ErrorDetail,
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
        entry = model_manager._models[name]
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
async def load_model(request: ModelLoadRequest) -> JSONResponse | ModelLoadResponse:
    """Load a WhisperX model."""
    try:
        model_manager.load_model(
            request.model_name, request.device, request.compute_type
        )
        entry = model_manager._models[request.model_name]
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
    except ModelLoadError as exc:
        error = ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.error_code,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error.model_dump(),
        )


@router.post(
    "/unload",
    response_model=ModelUnloadResponse,
    responses={500: {"model": ErrorResponse}},
)
async def unload_model(
    request: ModelUnloadRequest,
) -> JSONResponse | ModelUnloadResponse:
    """Unload a WhisperX model."""
    if request.model_name not in model_manager.list_models():
        error = ErrorResponse(
            error=ErrorDetail(
                message=f"Model not loaded: {request.model_name}",
                type="model_load_error",
                param="model",
                code="not_loaded",
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error.model_dump(),
        )

    model_manager.unload_model(request.model_name)
    return ModelUnloadResponse(
        success=True,
        message="Model unloaded successfully",
        freed_memory_mb=None,
    )
