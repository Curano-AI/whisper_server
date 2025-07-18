"""Transcription API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/transcriptions")
async def create_transcription():
    """Create transcription endpoint - placeholder for task 8"""
    return {"message": "Transcription endpoint - to be implemented in task 8"}
