"""Dataset abstractions for benchmark inputs.

JSONL-backed (one record per line) so datasets are easy to diff and to
generate from CRM exports. The four canonical schemas:

STTSample           {id, audio_path | audio_bytes_b64, transcript, language, code_switch?}
TTSSample           {id, text, language, voice_id?}
RAGSample           {id, query, expected_chunks: [id, ...], expected_answer?}
TaskScenario        {id, user_turns: [...], expected_disposition, required_slots: {...}}

Each loader can either consume an inlined records list (for tests) or a
``Path`` to a JSONL file.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union


# --- Sample types --------------------------------------------------------


@dataclass
class STTSample:
    id: str
    transcript: str
    language: Optional[str] = None
    code_switch: bool = False
    audio_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None

    def resolve_audio(self, base_dir: Optional[Path] = None) -> bytes:
        if self.audio_bytes is not None:
            return self.audio_bytes
        if self.audio_path is None:
            raise ValueError(f"sample {self.id!r} has no audio data")
        p = Path(self.audio_path)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        return p.read_bytes()


@dataclass
class TTSSample:
    id: str
    text: str
    language: str = "hi-IN"
    voice_id: Optional[str] = None


@dataclass
class RAGSample:
    id: str
    query: str
    expected_chunks: list[str] = field(default_factory=list)
    expected_answer: Optional[str] = None


@dataclass
class TaskTurn:
    role: str         # "user" | "system_event"
    content: str = ""
    event: Optional[str] = None


@dataclass
class TaskScenario:
    id: str
    user_turns: list[TaskTurn] = field(default_factory=list)
    expected_disposition: str = ""
    required_slots: dict[str, Any] = field(default_factory=dict)
    language: str = "hi"


# --- Generic JSONL helpers ----------------------------------------------


SourceLike = Union[Path, str, Iterable[dict]]


def _read_records(source: SourceLike) -> list[dict]:
    """Resolve either a JSONL path or an iterable of dict records."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{line_num} invalid JSON: {e}") from e
        return records
    return list(source)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


# --- Concrete loaders ---------------------------------------------------


def load_stt_dataset(source: SourceLike) -> list[STTSample]:
    out: list[STTSample] = []
    for r in _read_records(source):
        audio_bytes = None
        if "audio_bytes_b64" in r and r["audio_bytes_b64"]:
            audio_bytes = base64.b64decode(r["audio_bytes_b64"])
        out.append(STTSample(
            id=str(r["id"]),
            transcript=str(r.get("transcript") or ""),
            language=r.get("language"),
            code_switch=bool(r.get("code_switch", False)),
            audio_path=r.get("audio_path"),
            audio_bytes=audio_bytes,
        ))
    return out


def load_tts_dataset(source: SourceLike) -> list[TTSSample]:
    return [
        TTSSample(
            id=str(r["id"]),
            text=str(r["text"]),
            language=r.get("language", "hi-IN"),
            voice_id=r.get("voice_id"),
        )
        for r in _read_records(source)
    ]


def load_rag_dataset(source: SourceLike) -> list[RAGSample]:
    return [
        RAGSample(
            id=str(r["id"]),
            query=str(r["query"]),
            expected_chunks=list(r.get("expected_chunks") or []),
            expected_answer=r.get("expected_answer"),
        )
        for r in _read_records(source)
    ]


def load_task_dataset(source: SourceLike) -> list[TaskScenario]:
    out: list[TaskScenario] = []
    for r in _read_records(source):
        turns = []
        for t in r.get("user_turns") or []:
            turns.append(TaskTurn(
                role=t.get("role", "user"),
                content=t.get("content") or "",
                event=t.get("event"),
            ))
        out.append(TaskScenario(
            id=str(r["id"]),
            user_turns=turns,
            expected_disposition=str(r.get("expected_disposition") or ""),
            required_slots=dict(r.get("required_slots") or {}),
            language=str(r.get("language") or "hi"),
        ))
    return out
