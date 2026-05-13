from __future__ import annotations

import math
import struct

import pytest

from src.pipeline.audio_utils import pcm16_silence_ms
from src.pipeline.interruption import InterruptionConfig, InterruptionWatcher
from src.pipeline.vad import (
    EndpointConfig,
    EndpointDetector,
    EnergyVAD,
    VADFrame,
)


def _tone(duration_ms: int, sample_rate: int = 16000, amp: int = 8000) -> bytes:
    n = int(sample_rate * duration_ms / 1000)
    samples = [int(amp * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]
    return b"".join(struct.pack("<h", s) for s in samples)


def test_energy_vad_silence_is_not_speech() -> None:
    vad = EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)
    frame = vad.detect(pcm16_silence_ms(30))
    assert frame.is_speech is False
    assert frame.energy == 0.0


def test_energy_vad_loud_signal_is_speech() -> None:
    vad = EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)
    frame = vad.detect(_tone(30, amp=8000))
    assert frame.is_speech is True
    assert frame.energy > 300.0


def test_energy_vad_quiet_signal_below_threshold() -> None:
    vad = EnergyVAD(rms_threshold=2000.0)
    frame = vad.detect(_tone(30, amp=500))
    assert frame.is_speech is False


def test_endpoint_detector_completes_after_speech_then_silence() -> None:
    det = EndpointDetector(frame_ms=30, cfg=EndpointConfig(min_speech_ms=60, min_silence_ms=120))
    # 90ms of speech (3 frames) => above min_speech
    for _ in range(3):
        assert det.feed(VADFrame(is_speech=True, energy=500)) is False
    # Two frames of silence (60ms) — not enough yet
    for _ in range(2):
        assert det.feed(VADFrame(is_speech=False, energy=0)) is False
    # Two more frames silence (cumulative 120ms) => completes
    assert det.feed(VADFrame(is_speech=False, energy=0)) is False
    assert det.feed(VADFrame(is_speech=False, energy=0)) is True


def test_endpoint_detector_silence_alone_does_not_complete() -> None:
    det = EndpointDetector(frame_ms=30, cfg=EndpointConfig(min_speech_ms=60, min_silence_ms=60))
    for _ in range(10):
        assert det.feed(VADFrame(is_speech=False, energy=0)) is False


def test_endpoint_detector_reset() -> None:
    det = EndpointDetector(frame_ms=30, cfg=EndpointConfig(min_speech_ms=30, min_silence_ms=60))
    det.feed(VADFrame(is_speech=True, energy=500))
    det.reset()
    # After reset, silence alone shouldn't fire
    for _ in range(5):
        assert det.feed(VADFrame(is_speech=False, energy=0)) is False


@pytest.mark.asyncio
async def test_interruption_watcher_fires_after_min_speech() -> None:
    fired_count = 0

    async def on_interrupt() -> None:
        nonlocal fired_count
        fired_count += 1

    w = InterruptionWatcher(
        InterruptionConfig(min_speech_ms=60, detection_interval_ms=20),
        frame_ms=20,
        on_interrupt=on_interrupt,
    )
    w.enable()
    # 2 frames of speech (40ms) — not enough
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is False
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is False
    # 3rd frame (60ms cumulative) — fires
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is True
    assert fired_count == 1
    # Subsequent frames don't re-fire
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is False
    assert fired_count == 1


@pytest.mark.asyncio
async def test_interruption_watcher_disabled_does_not_fire() -> None:
    w = InterruptionWatcher(
        InterruptionConfig(min_speech_ms=20, detection_interval_ms=20), frame_ms=20
    )
    # Not enabled
    for _ in range(10):
        assert await w.feed(VADFrame(is_speech=True, energy=500)) is False


@pytest.mark.asyncio
async def test_interruption_watcher_brief_silence_decays_counter() -> None:
    w = InterruptionWatcher(
        InterruptionConfig(min_speech_ms=80, detection_interval_ms=20),
        frame_ms=20,
    )
    w.enable()
    # 2 speech frames (40ms)
    await w.feed(VADFrame(is_speech=True, energy=500))
    await w.feed(VADFrame(is_speech=True, energy=500))
    # 1 silence — counter decays to 20ms
    await w.feed(VADFrame(is_speech=False, energy=0))
    # 2 speech frames (60ms cumulative — still not enough)
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is False
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is False
    # One more pushes us over
    assert await w.feed(VADFrame(is_speech=True, energy=500)) is True
