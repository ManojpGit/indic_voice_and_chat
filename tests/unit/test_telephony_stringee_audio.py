import pytest

from src.api.telephony_stringee_bridge import (
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
