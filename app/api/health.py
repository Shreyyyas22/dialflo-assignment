from fastapi import APIRouter, Response, status

from app.ml.model_manager import is_model_loaded
from app.schemas.analysis import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health():
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def get_ready(response: Response):
    loaded = is_model_loaded()
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", model_loaded=False)
    return ReadyResponse(status="ready", model_loaded=True)
