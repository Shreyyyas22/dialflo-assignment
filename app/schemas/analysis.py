from typing import Literal

from pydantic import BaseModel, Field

GenderValue = Literal["male", "female", "unknown"]
AgeBracket = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
QualityClassification = Literal["good", "degraded", "insufficient"]


class GenderResult(BaseModel):
    value: GenderValue
    confidence: float = Field(..., ge=0.0, le=1.0)


class AgeResult(BaseModel):
    bracket: AgeBracket
    confidence: float = Field(..., ge=0.0, le=1.0)


class QualityMetrics(BaseModel):
    duration_seconds: float = Field(..., ge=0.0)
    rms: float = Field(..., ge=0.0)
    peak_amplitude: float = Field(..., ge=0.0)
    silence_ratio: float = Field(..., ge=0.0, le=1.0)
    clipping_ratio: float = Field(..., ge=0.0, le=1.0)
    estimated_snr_db: float


class QualityResult(BaseModel):
    classification: QualityClassification
    metrics: QualityMetrics


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderResult
    age: AgeResult
    quality: QualityResult
    processing_time_ms: float = Field(..., ge=0.0)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
