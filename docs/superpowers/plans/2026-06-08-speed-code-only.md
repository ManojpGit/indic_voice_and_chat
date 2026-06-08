# Speed (code-only, pure-safe) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-turn latency fully measurable, then determine (via a spike) whether Sarvam native streaming TTS can shave real time-to-first-word with zero quality risk.

**Architecture:** Lever A adds an `endpoint_gap_ms` metric (last STT interim → endpoint finalize) to the existing per-turn log so the whole chain `endpoint_gap → llm_ttft → tts_first → total` is visible. Lever B is a research spike on Sarvam's TTS API; its GO/NO-GO outcome decides whether a *separate* streaming-implementation plan gets written.

**Tech Stack:** Python 3.12, asyncio, pytest + pytest-asyncio. Existing: `src/api/browser_bridge.py` (dev-console bridge), Deepgram streaming STT, Sarvam TTS.

Spec: `docs/superpowers/specs/2026-06-08-speed-code-only-design.md`.

---

## File Structure
- **Modify** `src/api/browser_bridge.py` — track last-interim time in `_consume_stream_events`; pass `endpoint_gap_ms` into `_dispatch_text_turn`; add it to the `"browser turn (stream)"` log.
- **Modify** `tests/unit/test_browser_bridge_streaming.py` — test that `endpoint_gap_ms` is computed + logged.
- **Spike only (no src):** Sarvam streaming investigation → documented GO/NO-GO appended to `docs/latency-llm-stt-experiments.md`.

---

## Task 1: `endpoint_gap_ms` measurement (Lever A)

**Files:**
- Modify: `src/api/browser_bridge.py` (`_consume_stream_events`, `_dispatch_text_turn`)
- Test: `tests/unit/test_browser_bridge_streaming.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_browser_bridge_streaming.py`:

```python
@pytest.mark.asyncio
async def test_endpoint_gap_ms_logged(caplog):
    import logging
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="haan"),
        STTStreamEvent(type="endpoint", text="haan ji boliye"),
    ])
    with caplog.at_level(logging.INFO):
        await bridge._consume_stream_events(session)
    recs = [r for r in caplog.records if r.getMessage() == "browser turn (stream)"]
    assert recs, "no 'browser turn (stream)' log emitted"
    gap = getattr(recs[0], "endpoint_gap_ms", None)
    assert gap is not None and gap >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py::test_endpoint_gap_ms_logged -v`
Expected: FAIL — `endpoint_gap_ms` attribute is `None` (not in the log extra yet).

- [ ] **Step 3: Track last-interim time + compute the gap in `_consume_stream_events`**

In `src/api/browser_bridge.py`, replace the body of `_consume_stream_events` (the `try:` block) with:

```python
        last_interim_t = None
        try:
            async for ev in session.events():
                if self._stopped:
                    return
                if ev.type == "interim":
                    last_interim_t = time.monotonic()
                    await self._send_json(
                        {"type": "partial", "role": "user", "text": ev.text}
                    )
                elif ev.type == "endpoint":
                    if self._agent_busy or not ev.text.strip():
                        continue
                    gap_ms = (
                        int((time.monotonic() - last_interim_t) * 1000)
                        if last_interim_t is not None else None
                    )
                    last_interim_t = None
                    await self._dispatch_text_turn(ev.text, endpoint_gap_ms=gap_ms)
        except Exception:  # noqa: BLE001 - never let the consumer die silently
            log.exception("stream event consumer crashed")
```

(`time` is already imported at the top of the file.)

- [ ] **Step 4: Thread `endpoint_gap_ms` into `_dispatch_text_turn` and its log**

Change the `_dispatch_text_turn` signature:

```python
    async def _dispatch_text_turn(self, text: str, endpoint_gap_ms: int | None = None) -> None:
```

Then add `endpoint_gap_ms` to the `"browser turn (stream)"` log `extra` dict (the block at ~line 346). The `extra` becomes:

```python
            extra={
                "user_text": (outcome.pipeline.user_text or "")[:80],
                "endpoint_gap_ms": endpoint_gap_ms,
                "llm_ttft_ms": m.llm_ttft_ms,
                "llm_total_ms": m.llm_total_ms,
                "tts_first_ms": m.tts_first_chunk_ms,
                "total_ms": m.total_latency_ms,
                "action": outcome.response.action,
                "agent_text": (outcome.response.response_text or "")[:100],
                "error": outcome.response.parse_error or "",
            },
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -q`
Expected: PASS (all streaming tests, including the new one).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "feat(metrics): log endpoint_gap_ms per turn (STT interim -> endpoint finalize)"
```

- [ ] **Step 8: Capture a live BEFORE baseline**

Restart the server and run ~8–10 dev-console turns; collect the `"browser turn (stream)"` lines. Record median/mean of `endpoint_gap_ms`, `tts_first_ms`, `total_ms` — this is the baseline the Sarvam spike (and any future Mumbai deploy) is measured against. (Server: `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`.)

---

## Task 2: Sarvam native streaming TTS — spike (Lever B1, decision gate)

**Files:** none in `src/` (research). Output: a dated GO/NO-GO note appended to `docs/latency-llm-stt-experiments.md`.

This is a research task, not TDD — there is no code to keep unless the spike says GO (and that becomes a *separate* plan).

- [ ] **Step 1: Determine whether Sarvam offers native streaming TTS**

Check Sarvam's TTS API docs (`https://docs.sarvam.ai/`, text-to-speech section) for a **streaming** or **WebSocket** TTS endpoint for `bulbul:v2`/`v3` that returns incremental audio. Confirm: endpoint URL, protocol (WS vs chunked HTTP), request shape, audio format/encoding, and whether it works for `hi-IN`.

- [ ] **Step 2: Probe it with a throwaway script (only if docs indicate streaming exists)**

Write a throwaway `_spike_sarvam_stream.py` (delete after) that authenticates with `TENANT_DEV_SARVAM_KEY`, sends a short Hindi sentence, and prints: time-to-first-chunk vs the current one-shot `synthesize` time, number of chunks, and total bytes. Run it with `set -a && . ./.env && set +a && .venv/bin/python _spike_sarvam_stream.py`. Delete the script when done.

- [ ] **Step 3: Record the GO/NO-GO decision**

Append a dated section to `docs/latency-llm-stt-experiments.md`:
- If **streaming exists and first-chunk is meaningfully faster** than one-shot → **GO**: note the API details + measured first-chunk delta.
- If **no streaming API**, or first-chunk is not faster → **NO-GO**: document the negative result; code-only speed work concludes here (the remaining lever is the deferred Mumbai deploy).

- [ ] **Step 4: Commit the finding**

```bash
git add docs/latency-llm-stt-experiments.md
git commit -m "docs(latency): Sarvam streaming TTS spike — GO/NO-GO result"
```

- [ ] **Step 5: If GO — stop and write the Lever B2 plan**

Do NOT implement streaming ad hoc. Re-invoke the **superpowers:writing-plans** skill to produce a dedicated Lever B2 plan against the discovered API, covering: a streaming method on `SarvamTTSAdapter` (yield PCM chunks, strip WAV per chunk), wiring `engine.tts_worker` to push the first chunk on arrival (preserving `cancel_event`/barge paths), a one-shot fallback on streaming failure, adapter + engine unit tests, and a live before/after vs the Task 1 baseline. If NO-GO, this plan is complete.

---

## Self-Review (author)
- **Spec coverage:** Lever A → Task 1 (metric + baseline). Lever B1 spike → Task 2 (with GO→separate B2 plan, NO-GO→done). Validation/before-after → Task 1 Step 8 + Task 2. Out-of-scope items (Mumbai, Lever C, model/voice) are not in any task. Covered.
- **Placeholders:** none — Task 1 is fully specified TDD; Task 2 is a research task with concrete deliverables and a decision gate (B2 code is intentionally deferred to a post-spike plan rather than fabricated).
- **Type consistency:** `endpoint_gap_ms: int | None` used consistently in `_consume_stream_events`, `_dispatch_text_turn`, the log extra, and the test (`>= 0`, non-None).
