"""Health check API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthcheck")
async def health_check():
    """Health check endpoint - placeholder for task 11"""
    return {"message": "Health check endpoint - to be implemented in task 11"}
