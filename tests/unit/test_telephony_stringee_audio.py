import pytest

from src.api.telephony_stringee_bridge import (
    AudioStore,
    BufferingAudioSink,
    pcm16_to_wav,
    resample_pcm16,
    wav_to_pcm16,
)


def test_pcm16_to_wav_roundtrips():
    pcm = b"\x01\x02\x03\x04" * 100
    wav = pcm16_to_wav(pcm, sample_rate=16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    back, rate = wav_to_pcm16(wav)
    assert back == pcm and rate == 16000


def test_resample_pcm16_8k_to_16k_doubles_length():
    pcm8 = b"\x00\x01" * 80  # 80 samples @ 8k
    pcm16 = resample_pcm16(pcm8, 8000, 16000)
    # 2x rate => ~2x samples (allow ratecv's boundary slack)
    assert abs(len(pcm16) - 2 * len(pcm8)) <= 4


@pytest.mark.asyncio
async def test_buffering_sink_collects_pcm():
    sink = BufferingAudioSink()
    await sink(b"ab")
    await sink(b"cd")
    assert sink.pcm == b"abcd"


def test_audio_store_put_get_and_token_is_opaque():
    store = AudioStore(ttl_seconds=60)
    token = store.put(b"wavbytes")
    assert isinstance(token, str) and len(token) >= 16
    assert store.get(token) == b"wavbytes"


def test_audio_store_evicts_expired(monkeypatch):
    import src.api.telephony_stringee_bridge as m
    t = {"now": 1000.0}
    monkeypatch.setattr(m.time, "monotonic", lambda: t["now"])
    store = AudioStore(ttl_seconds=10)
    token = store.put(b"x")
    t["now"] = 1011.0  # past TTL
    assert store.get(token) is None
