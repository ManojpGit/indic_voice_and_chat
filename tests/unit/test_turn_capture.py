from __future__ import annotations

from src.pipeline.turn_capture import accumulate_and_detect
from src.pipeline.vad import EnergyVAD, EndpointDetector, EndpointConfig


def _loud(n_frames: int, vad: EnergyVAD) -> bytes:
    # Max-amplitude PCM16 => high RMS => is_speech True.
    # frame_bytes is even (16-bit mono => 2 bytes/sample), so // 2 divides cleanly.
    return (b"\xff\x7f" * (vad.frame_bytes // 2)) * n_frames


def _silent(n_frames: int, vad: EnergyVAD) -> bytes:
    # frame_bytes is even (16-bit mono => 2 bytes/sample), so // 2 divides cleanly.
    return (b"\x00\x00" * (vad.frame_bytes // 2)) * n_frames


def test_returns_false_while_speech_accumulating():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    endpoint = EndpointDetector(vad.frame_ms, EndpointConfig())
    buf = bytearray()
    # A single loud frame mid-utterance must not signal end-of-turn.
    assert accumulate_and_detect(_loud(1, vad), vad, endpoint, buf) is False


def test_accumulates_pcm_into_buffer():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    endpoint = EndpointDetector(vad.frame_ms, EndpointConfig())
    buf = bytearray()
    pcm = _loud(1, vad)
    accumulate_and_detect(pcm, vad, endpoint, buf)
    assert bytes(buf) == pcm


def test_passed_frame_avoids_second_detect():
    from src.pipeline.vad import VADFrame, EndpointDetector, EndpointConfig

    class CountingVAD:
        sample_rate = 16000
        frame_ms = 30
        def __init__(self):
            self.calls = 0
        def detect(self, pcm16):
            self.calls += 1
            return VADFrame(is_speech=True, energy=9999.0)
        def reset(self):
            pass

    vad = CountingVAD()
    endpoint = EndpointDetector(vad.frame_ms, EndpointConfig())
    buf = bytearray()
    precomputed = vad.detect(b"\x00\x00")   # calls == 1
    accumulate_and_detect(b"\x00\x00", vad, endpoint, buf, frame=precomputed)
    assert vad.calls == 1                    # helper did NOT call detect again
    assert bytes(buf) == b"\x00\x00"         # still accumulated

    # And without a frame, the helper DOES detect.
    accumulate_and_detect(b"\x00\x00", vad, endpoint, buf)
    assert vad.calls == 2


def test_returns_true_at_end_of_utterance():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # 250ms speech then 600ms silence => endpoint fires (defaults).
    endpoint = EndpointDetector(vad.frame_ms, EndpointConfig())
    buf = bytearray()
    # Feed speech frames one at a time; should not fire yet.
    fired = False
    for _ in range(10):  # 300ms speech
        fired = accumulate_and_detect(_loud(1, vad), vad, endpoint, buf) or fired
    assert fired is False
    # Now feed silence frames until it fires.
    for _ in range(25):  # up to 750ms silence
        if accumulate_and_detect(_silent(1, vad), vad, endpoint, buf):
            fired = True
            break
    assert fired is True
