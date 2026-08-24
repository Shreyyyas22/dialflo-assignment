import numpy as np

from app.services.quality_service import analyze_audio_quality


def test_insufficient_audio_silent():
    # 2 seconds of pure silence
    signal = np.zeros(32000, dtype=np.float32)
    res = analyze_audio_quality(signal, sample_rate=16000)
    assert res.classification == "insufficient"
    assert res.metrics.rms == 0.0
    assert res.metrics.silence_ratio == 1.0


def test_insufficient_audio_too_short():
    # 0.1 second sine wave (below 0.5s minimum)
    t = np.linspace(0, 0.1, 1600, endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    res = analyze_audio_quality(signal, sample_rate=16000)
    assert res.classification == "insufficient"


def test_degraded_audio_clipping():
    # 2 seconds audio heavily clipped
    t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
    signal = np.clip(2.0 * np.sin(2 * np.pi * 440 * t), -1.0, 1.0)
    res = analyze_audio_quality(signal, sample_rate=16000)
    assert res.classification == "degraded"
    assert res.metrics.clipping_ratio > 0.05


def test_good_audio_clean():
    # 2 seconds clean audio with alternating speech/silence bursts
    t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t)
    # add small silent pause in middle
    signal[8000:12000] = 0.0001
    res = analyze_audio_quality(signal, sample_rate=16000)
    assert res.classification == "good"
    assert res.metrics.rms > 0.01
    assert res.metrics.clipping_ratio < 0.05
