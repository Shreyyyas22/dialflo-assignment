from fastapi import APIRouter, File, Form, UploadFile, status

from app.schemas.analysis import AnalyzeResponse
from app.services.analysis_service import analyze_audio_request

router = APIRouter(tags=["Analysis"])


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_audio(
    contact_id: str = Form(..., description="UUID identifier of the contact"),
    audio: UploadFile = File(  # noqa: B008
        ..., description="Audio file payload (WAV, MP3, OGG, FLAC, etc.)"
    ),
):
    audio_bytes = await audio.read()
    response = await analyze_audio_request(
        contact_id=contact_id, audio_bytes=audio_bytes
    )
    return response
