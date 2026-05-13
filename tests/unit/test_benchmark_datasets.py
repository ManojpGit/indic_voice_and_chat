from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from src.benchmarks.datasets import (
    STTSample,
    load_rag_dataset,
    load_stt_dataset,
    load_task_dataset,
    load_tts_dataset,
    write_jsonl,
)


def test_load_stt_from_inline_records() -> None:
    samples = load_stt_dataset([
        {"id": "s1", "transcript": "Namaste", "language": "hi-IN"},
        {"id": "s2", "transcript": "Hello world", "language": "en-IN", "code_switch": True},
    ])
    assert len(samples) == 2
    assert samples[0].transcript == "Namaste"
    assert samples[1].code_switch is True


def test_load_stt_decodes_base64_audio() -> None:
    audio = b"\x00\x01\x02"
    samples = load_stt_dataset([{
        "id": "s1",
        "transcript": "hi",
        "audio_bytes_b64": base64.b64encode(audio).decode(),
    }])
    assert samples[0].audio_bytes == audio


def test_stt_sample_resolve_audio_from_path(tmp_path: Path) -> None:
    audio_file = tmp_path / "x.pcm"
    audio_file.write_bytes(b"abcd")
    sample = STTSample(id="s1", transcript="x", audio_path=str(audio_file))
    assert sample.resolve_audio() == b"abcd"


def test_stt_sample_resolve_audio_with_base_dir(tmp_path: Path) -> None:
    (tmp_path / "x.pcm").write_bytes(b"abcd")
    sample = STTSample(id="s1", transcript="x", audio_path="x.pcm")
    assert sample.resolve_audio(base_dir=tmp_path) == b"abcd"


def test_stt_sample_resolve_audio_missing_raises() -> None:
    sample = STTSample(id="s1", transcript="x")
    with pytest.raises(ValueError):
        sample.resolve_audio()


def test_load_tts() -> None:
    samples = load_tts_dataset([
        {"id": "t1", "text": "Namaste", "language": "hi-IN", "voice_id": "meera"},
    ])
    assert samples[0].voice_id == "meera"


def test_load_rag() -> None:
    samples = load_rag_dataset([
        {"id": "r1", "query": "What is plan B?", "expected_chunks": ["c1", "c2"], "expected_answer": "500GB"},
    ])
    assert samples[0].expected_chunks == ["c1", "c2"]
    assert samples[0].expected_answer == "500GB"


def test_load_task() -> None:
    samples = load_task_dataset([{
        "id": "scen-1",
        "user_turns": [
            {"role": "user", "content": "Yes I'm interested"},
            {"role": "user", "content": "Send WhatsApp"},
        ],
        "expected_disposition": "interested_callback",
        "required_slots": {"interest_level": "warm"},
    }])
    assert len(samples) == 1
    assert len(samples[0].user_turns) == 2
    assert samples[0].required_slots == {"interest_level": "warm"}


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "stt.jsonl"
    rows = [
        {"id": "s1", "transcript": "Namaste", "language": "hi"},
        {"id": "s2", "transcript": "Hello", "language": "en"},
    ]
    n = write_jsonl(path, rows)
    assert n == 2

    loaded = load_stt_dataset(path)
    assert [s.id for s in loaded] == ["s1", "s2"]


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "stt.jsonl"
    path.write_text(
        '{"id": "s1", "transcript": "a"}\n\n{"id": "s2", "transcript": "b"}\n',
        encoding="utf-8",
    )
    samples = load_stt_dataset(path)
    assert [s.id for s in samples] == ["s1", "s2"]


def test_load_jsonl_invalid_line_raises_with_context(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "s1"}\nthis is not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        load_stt_dataset(path)
