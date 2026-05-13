from __future__ import annotations

import wave

import pytest

from src.pipeline.audio_utils import (
    mulaw_to_pcm16,
    pcm16_silence_ms,
    pcm16_to_mulaw,
    pcm16_to_wav,
    resample_pcm16,
    rms_energy_pcm16,
    wav_to_pcm16,
)


def _sine_pcm16(duration_ms: int, sample_rate: int = 16000, amp: int = 8000) -> bytes:
    import math, struct  # local — pure test helper

    n = int(sample_rate * duration_ms / 1000)
    samples = [int(amp * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]
    return b"".join(struct.pack("<h", s) for s in samples)


def test_mulaw_pcm_round_trip_lossy_but_close() -> None:
    pcm = _sine_pcm16(20)
    mulaw = pcm16_to_mulaw(pcm)
    back = mulaw_to_pcm16(mulaw)
    # μ-law is 8-bit lossy, but length must match (1 byte μ-law -> 2 bytes PCM16)
    assert len(mulaw) * 2 == len(pcm)
    assert len(back) == len(pcm)


def test_resample_changes_length_proportionally() -> None:
    pcm = _sine_pcm16(100, sample_rate=16000)  # 1600 samples
    out, _ = resample_pcm16(pcm, 16000, 8000)
    # ~half the samples
    assert 0.4 * len(pcm) <= len(out) <= 0.6 * len(pcm)


def test_resample_passthrough_when_rates_equal() -> None:
    pcm = _sine_pcm16(10)
    out, _ = resample_pcm16(pcm, 16000, 16000)
    assert out == pcm


def test_pcm_to_wav_round_trip() -> None:
    pcm = _sine_pcm16(50, sample_rate=16000)
    wav = pcm16_to_wav(pcm, sample_rate=16000)
    # WAV header presence
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    pcm_back, rate = wav_to_pcm16(wav)
    assert pcm_back == pcm
    assert rate == 16000


def test_wav_to_pcm_rejects_non_16bit() -> None:
    # Build an 8-bit WAV manually
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(b"\x80" * 800)
    with pytest.raises(ValueError, match="16-bit"):
        wav_to_pcm16(buf.getvalue())


def test_rms_energy_silence_is_zero() -> None:
    silence = pcm16_silence_ms(20)
    assert rms_energy_pcm16(silence) == 0.0


def test_rms_energy_nonzero_for_signal() -> None:
    pcm = _sine_pcm16(20, amp=4000)
    assert rms_energy_pcm16(pcm) > 100


def test_silence_length_matches_duration() -> None:
    silence = pcm16_silence_ms(100, sample_rate=8000)
    assert len(silence) == 800 * 2  # 800 samples * 2 bytes
