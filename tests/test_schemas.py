import pytest
from pydantic import ValidationError

from app.schemas.analysis import (
    AgeResult,
    AnalyzeResponse,
    GenderResult,
    QualityMetrics,
    QualityResult,
)


def test_gender_result_valid():
    res = GenderResult(value="female", confidence=0.92)
    assert res.value == "female"
    assert res.confidence == 0.92


def test_gender_result_invalid_confidence():
    with pytest.raises(ValidationError):
        GenderResult(value="male", confidence=1.5)


def test_age_result_valid():
    res = AgeResult(bracket="31-45", confidence=0.88)
    assert res.bracket == "31-45"
    assert res.confidence == 0.88


def test_quality_metrics_valid():
    qm = QualityMetrics(
        duration_seconds=5.0,
        rms=0.05,
        peak_amplitude=0.2,
        silence_ratio=0.1,
        clipping_ratio=0.0,
        estimated_snr_db=15.0,
    )
    assert qm.duration_seconds == 5.0
    assert qm.estimated_snr_db == 15.0


def test_analyze_response_valid():
    resp = AnalyzeResponse(
        contact_id="123e4567-e89b-12d3-a456-426614174000",
        gender=GenderResult(value="male", confidence=0.85),
        age=AgeResult(bracket="18-30", confidence=0.75),
        quality=QualityResult(
            classification="good",
            metrics=QualityMetrics(
                duration_seconds=3.0,
                rms=0.04,
                peak_amplitude=0.3,
                silence_ratio=0.1,
                clipping_ratio=0.0,
                estimated_snr_db=20.0,
            ),
        ),
        processing_time_ms=120.5,
    )
    assert resp.contact_id == "123e4567-e89b-12d3-a456-426614174000"
    assert resp.gender.value == "male"
