"""Model management API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/list")
async def list_models():
    """List models endpoint - placeholder for task 10"""
    return {"message": "List models endpoint - to be implemented in task 10"}


@router.post("/load")
async def load_model():
    """Load model endpoint - placeholder for task 10"""
    return {"message": "Load model endpoint - to be implemented in task 10"}


@router.post("/unload")
async def unload_model():
    """Unload model endpoint - placeholder for task 10"""
    return {"message": "Unload model endpoint - to be implemented in task 10"}
