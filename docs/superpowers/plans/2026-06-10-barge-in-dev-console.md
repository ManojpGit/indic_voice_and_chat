# Barge-in v1 (dev console) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Server-side, headphones-required barge-in on the dev-console streaming path: the user speaking over the agent cancels the reply within ~0.5s and is answered as the next turn, with no false-barges on Hindi backchannels.

**Architecture:** Detect barge from the Deepgram recognizer itself — a *sustained* interim (~450ms) while the agent is audible fires a generalized `_handle_barge_in`. Enabler: run turns as a background `asyncio.Task` so the consumer loop stays free to read interims during the reply. Barge is a mode (the `bargeIn` toggle ⇒ headphones ⇒ full-duplex mic, no echo).

**Tech Stack:** Python 3.12, asyncio, pytest + pytest-asyncio. `src/api/browser_bridge.py` (dev-console bridge), Deepgram streaming STT, `static/dev_console.html` (client).

Spec: `docs/superpowers/specs/2026-06-10-barge-in-dev-console-design.md`.

---

## File Structure
- **Modify** `src/api/browser_bridge.py` — barge state in `__init__`; `config` message handling + `_turn_task` teardown in `run()`; generalized `_handle_barge_in`; new `_barge_on_interim()`; background-task dispatch + detection wiring in `_consume_stream_events`; conditional gate in `_on_pcm_frame`.
- **Modify** `tests/unit/test_browser_bridge_streaming.py` — detector + dispatch + gate tests; update the two existing turn tests for background-task dispatch.
- **Modify** `static/dev_console.html` — send `config` on barge toggle; full-duplex mic when barge on; delete client RMS detector; handle `{"type":"interrupt"}`.

`BARGE_SUSTAIN_MS` is a module constant in `browser_bridge.py`.

---

## Task 1: Barge state, config message, generalized barge guard

**Files:**
- Modify: `src/api/browser_bridge.py`
- Test: `tests/unit/test_browser_bridge_streaming.py`

- [ ] **Step 1: Write failing tests** — append to `tests/unit/test_browser_bridge_streaming.py`:

```python
import asyncio
import time as _time


@pytest.mark.asyncio
async def test_config_message_enables_barge():
    bridge, _ = _bridge([])
    # simulate the run() control-message branch
    bridge._apply_control({"type": "config", "barge": True})
    assert bridge._barge_enabled is True
    bridge._apply_control({"type": "config", "barge": False})
    assert bridge._barge_enabled is False


@pytest.mark.asyncio
async def test_barge_guard_fires_during_playback_only(monkeypatch):
    # Most interruptions land during playback: _agent_busy already False,
    # but now < _play_until. The generalized guard must still cancel.
    bridge, _ = _bridge([])
    bridge._agent_busy = False
    bridge._cancel_event = asyncio.Event()
    bridge._play_until = _time.monotonic() + 5
    bridge._handle_barge_in()
    assert bridge._cancel_event.is_set()
    assert bridge._play_until == 0.0


@pytest.mark.asyncio
async def test_barge_guard_noop_when_agent_silent(monkeypatch):
    bridge, _ = _bridge([])
    bridge._agent_busy = False
    bridge._play_until = 0.0
    bridge._cancel_event = asyncio.Event()
    bridge._handle_barge_in()
    assert not bridge._cancel_event.is_set()  # nothing to cancel
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k "config_message or barge_guard" -q`
Expected: FAIL — `_apply_control` missing; `_handle_barge_in` early-returns when `_agent_busy` is False (so it won't cancel during playback).

- [ ] **Step 3: Add barge state in `__init__`**

In `src/api/browser_bridge.py`, immediately after the line `self._cancel_event = None  # set per in-flight streaming turn; barge-in fires it` add:

```python
        self._barge_enabled = False     # set by the client's {"type":"config","barge":...}
        self._turn_task = None          # in-flight turn runs as a task so barge can interrupt it
        self._barge_start_t = None      # monotonic time the current interruption's speech began
        self._had_turn = False          # opening is not barge-able; arm only after the first turn
```

- [ ] **Step 4: Add `_apply_control` + wire it in `run()`**

Add this method (near `_handle_barge_in`):

```python
    def _apply_control(self, ctrl: dict) -> None:
        """Handle a client control message (called from the run() WS loop)."""
        if ctrl.get("type") == "config":
            self._barge_enabled = bool(ctrl.get("barge"))
        elif ctrl.get("type") == "barge_in":
            self._handle_barge_in()
```

In `run()`, replace the existing control-message branch:

```python
                    if ctrl.get("type") == "barge_in":
                        self._handle_barge_in()
                    elif ctrl.get("type") == "end":
```
with:
```python
                    if ctrl.get("type") == "end":
```
and, right before that `if`, add the generic apply:
```python
                    self._apply_control(ctrl)
```
So the block reads:
```python
                    try:
                        ctrl = json.loads(text)
                    except (ValueError, TypeError):
                        ctrl = {}
                    self._apply_control(ctrl)
                    if ctrl.get("type") == "end":
                        await self._emit_outcome()
                        self._stopped = True
                        exit_reason = "client end"
                        break
                    continue
```

- [ ] **Step 5: Generalize `_handle_barge_in`**

Replace the guard line in `_handle_barge_in`:
```python
        if not self._agent_busy:
            return
```
with:
```python
        # Fire whenever the agent is AUDIBLE — generating (_agent_busy) OR its
        # audio is still playing (now < _play_until). Most interruptions land
        # during playback, after generation finished and _agent_busy is False.
        if not (self._agent_busy or time.monotonic() < self._play_until):
            return
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k "config_message or barge_guard" -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Lint + commit**

```bash
.venv/bin/ruff check --fix src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "feat(barge): barge state + config message + audible-window cancel guard"
```

---

## Task 2: `_barge_on_interim()` detection helper

**Files:**
- Modify: `src/api/browser_bridge.py`
- Test: `tests/unit/test_browser_bridge_streaming.py`

Pure detection logic, unit-testable with a controlled clock.

- [ ] **Step 1: Write failing tests** — append:

```python
@pytest.fixture
def _clock(monkeypatch):
    import src.api.browser_bridge as bb
    t = {"now": 1000.0}
    monkeypatch.setattr(bb.time, "monotonic", lambda: t["now"])
    monkeypatch.setattr(bb, "BARGE_SUSTAIN_MS", 450)
    return t


def _armed_bridge():
    bridge, _ = _bridge([])
    bridge._barge_enabled = True
    bridge._had_turn = True
    bridge._agent_busy = True       # agent audible
    return bridge


def test_barge_on_interim_fires_when_sustained(_clock):
    bridge = _armed_bridge()
    assert bridge._barge_on_interim() is False          # first interim -> start timer
    _clock["now"] += 0.5                                 # 500ms > 450ms threshold
    assert bridge._barge_on_interim() is True            # sustained -> fire
    assert bridge._barge_start_t is None                 # reset so it can't re-fire


def test_barge_on_interim_no_fire_for_short_backchannel(_clock):
    bridge = _armed_bridge()
    assert bridge._barge_on_interim() is False
    _clock["now"] += 0.2                                 # 200ms < 450ms
    assert bridge._barge_on_interim() is False


def test_barge_on_interim_no_fire_when_not_audible(_clock):
    bridge = _armed_bridge()
    bridge._agent_busy = False
    bridge._play_until = 0.0                              # agent NOT audible
    assert bridge._barge_on_interim() is False
    _clock["now"] += 1.0
    assert bridge._barge_on_interim() is False
    assert bridge._barge_start_t is None


def test_barge_on_interim_disabled_or_no_turn(_clock):
    bridge = _armed_bridge()
    bridge._barge_enabled = False
    assert bridge._barge_on_interim() is False
    bridge._barge_enabled = True
    bridge._had_turn = False
    assert bridge._barge_on_interim() is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k barge_on_interim -q`
Expected: FAIL — `_barge_on_interim` and `BARGE_SUSTAIN_MS` don't exist.

- [ ] **Step 3: Add the constant + helper**

Near the top module constants (after `_SEND_CHUNK = 8192`), add:
```python
# Barge-in: required sustained recognized-speech (ms) while the agent is audible
# before we treat it as an interruption (vs a one-word "haan/hmm" backchannel).
BARGE_SUSTAIN_MS = 450
```

Add the method (near `_handle_barge_in`):
```python
    def _barge_on_interim(self) -> bool:
        """Per-interim barge detector. True iff the user's speech has sustained
        past BARGE_SUSTAIN_MS while the agent is audible. Resets its timer when
        barge is off / no turn yet / the agent isn't audible."""
        if not (self._barge_enabled and self._had_turn):
            self._barge_start_t = None
            return False
        now = time.monotonic()
        audible = self._agent_busy or now < self._play_until
        if not audible:
            self._barge_start_t = None
            return False
        if self._barge_start_t is None:
            self._barge_start_t = now
            return False
        if now - self._barge_start_t >= BARGE_SUSTAIN_MS / 1000:
            self._barge_start_t = None
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k barge_on_interim -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check --fix src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "feat(barge): _barge_on_interim sustained-speech detector"
```

---

## Task 3: Run turns as a background task

**Files:**
- Modify: `src/api/browser_bridge.py` (`_consume_stream_events`, `run()` teardown)
- Test: `tests/unit/test_browser_bridge_streaming.py` (update 2 existing tests)

The consumer must not `await` the whole turn, or it can't read interims during the reply. Dispatch the turn as `self._turn_task` and keep looping.

- [ ] **Step 1: Update the existing turn tests to await the task**

In `tests/unit/test_browser_bridge_streaming.py`, in `test_endpoint_event_dispatches_text_turn`, change:
```python
    await bridge._consume_stream_events(session)
    assert bridge._agent.text_turns == ["और कुछ benefits हैं"]
```
to:
```python
    await bridge._consume_stream_events(session)
    if bridge._turn_task is not None:
        await bridge._turn_task          # turn now runs as a background task
    assert bridge._agent.text_turns == ["और कुछ benefits हैं"]
```
(`test_endpoint_ignored_while_agent_busy` needs no change — no turn is dispatched, so `_turn_task` stays None.)

- [ ] **Step 2: Run to verify the updated test fails**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py::test_endpoint_event_dispatches_text_turn -q`
Expected: FAIL — `_turn_task` doesn't exist yet as a dispatched task (still None; the turn was awaited inline), so `text_turns` is already populated but `bridge._turn_task` is None — the `await` line is a no-op and the assert passes... so to make this a real failing test, first apply Step 3's dispatch change, OR treat Step 1 as forward-compatible. Run it after Step 3.

- [ ] **Step 3: Change the endpoint branch in `_consume_stream_events`**

Replace the `elif ev.type == "endpoint":` block:
```python
                elif ev.type == "endpoint":
                    if self._agent_busy or not ev.text.strip():
                        last_interim_t = None
                        continue
                    gap_ms = (
                        int((time.monotonic() - last_interim_t) * 1000)
                        if last_interim_t is not None else None
                    )
                    last_interim_t = None
                    await self._dispatch_text_turn(ev.text, endpoint_gap_ms=gap_ms)
```
with:
```python
                elif ev.type == "endpoint":
                    if self._agent_busy or not ev.text.strip():
                        last_interim_t = None
                        continue
                    gap_ms = (
                        int((time.monotonic() - last_interim_t) * 1000)
                        if last_interim_t is not None else None
                    )
                    last_interim_t = None
                    # Set _agent_busy BEFORE create_task to close the race with a
                    # second endpoint; run the turn as a task so this loop stays
                    # free to read interims (and detect a barge) during the reply.
                    self._agent_busy = True
                    self._had_turn = True
                    self._turn_task = asyncio.create_task(
                        self._dispatch_text_turn(ev.text, endpoint_gap_ms=gap_ms)
                    )
```

- [ ] **Step 4: Cancel `_turn_task` on teardown**

In `run()`'s `finally:` block, right after the `stream_task.cancel()` lines, add:
```python
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
```

- [ ] **Step 5: Run the streaming suite**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -q`
Expected: PASS (all, including the updated dispatch test which now awaits `_turn_task`).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check --fix src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "feat(barge): dispatch turns as a background task (consumer stays responsive)"
```

---

## Task 4: Wire detection into the consumer

**Files:**
- Modify: `src/api/browser_bridge.py` (`_consume_stream_events` interim branch)
- Test: `tests/unit/test_browser_bridge_streaming.py`

- [ ] **Step 1: Write the failing test** — append:

```python
@pytest.mark.asyncio
async def test_consumer_barges_on_sustained_interim(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "BARGE_SUSTAIN_MS", 0)   # any 2nd interim while audible fires
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="ru"),
        STTStreamEvent(type="interim", text="ruko"),
    ])
    bridge._barge_enabled = True
    bridge._had_turn = True
    bridge._agent_busy = True                          # a turn is "in flight"
    bridge._cancel_event = asyncio.Event()
    bridge._play_until = 0.0
    await bridge._consume_stream_events(session)
    assert bridge._cancel_event.is_set()               # barge cancelled the turn
    interrupts = [m for m in bridge._ws.sent_json if m.get("type") == "interrupt"]
    assert interrupts                                  # client told to stop playback


@pytest.mark.asyncio
async def test_consumer_no_barge_when_disabled(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "BARGE_SUSTAIN_MS", 0)
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="ru"),
        STTStreamEvent(type="interim", text="ruko"),
    ])
    bridge._barge_enabled = False                      # off
    bridge._had_turn = True
    bridge._agent_busy = True
    bridge._cancel_event = asyncio.Event()
    await bridge._consume_stream_events(session)
    assert not bridge._cancel_event.is_set()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k consumer_barge -q`
Expected: FAIL — the interim branch doesn't call `_barge_on_interim` / send `interrupt` yet.

- [ ] **Step 3: Wire the detector into the interim branch**

Replace the `if ev.type == "interim":` block:
```python
                if ev.type == "interim":
                    last_interim_t = time.monotonic()
                    await self._send_json(
                        {"type": "partial", "role": "user", "text": ev.text}
                    )
```
with:
```python
                if ev.type == "interim":
                    last_interim_t = time.monotonic()
                    await self._send_json(
                        {"type": "partial", "role": "user", "text": ev.text}
                    )
                    if self._barge_on_interim():
                        self._handle_barge_in()
                        await self._send_json({"type": "interrupt"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k consumer_barge -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check --fix src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "feat(barge): detect + cancel + interrupt on sustained interim in the consumer"
```

---

## Task 5: Feed mic during playback in barge mode

**Files:**
- Modify: `src/api/browser_bridge.py` (`_on_pcm_frame`)
- Test: `tests/unit/test_browser_bridge_streaming.py`

- [ ] **Step 1: Write the failing test** — append:

```python
@pytest.mark.asyncio
async def test_mic_fed_during_playback_when_barge_enabled():
    bridge, session = _bridge([])
    bridge._stream_session = session
    bridge._agent_busy = False
    bridge._play_until = _time.monotonic() + 5         # agent audio still playing
    bridge._barge_enabled = True
    await bridge._on_pcm_frame(b"\x01\x02" * 160)
    assert session.sent == [b"\x01\x02" * 160]         # fed (so Deepgram can hear the interruption)


@pytest.mark.asyncio
async def test_mic_gated_during_playback_when_barge_disabled():
    bridge, session = _bridge([])
    bridge._stream_session = session
    bridge._agent_busy = False
    bridge._play_until = _time.monotonic() + 5
    bridge._barge_enabled = False
    await bridge._on_pcm_frame(b"\x01\x02" * 160)
    assert session.sent == []                          # echo gate still drops it
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k "mic_fed or mic_gated" -q`
Expected: FAIL — the first test fails: the current gate drops the frame regardless of `_barge_enabled`.

- [ ] **Step 3: Make the gate conditional on barge mode**

In `_on_pcm_frame`, replace:
```python
            if self._agent_busy or time.monotonic() < self._play_until:
                return
```
with:
```python
            # Half-duplex echo gate — drop the agent's own audio. Skipped in
            # barge mode: headphones ⇒ no echo, and we need the user's audio
            # during playback to detect an interruption.
            if not self._barge_enabled and (
                self._agent_busy or time.monotonic() < self._play_until
            ):
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge_streaming.py -k "mic_fed or mic_gated" -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check --fix src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git add src/api/browser_bridge.py tests/unit/test_browser_bridge_streaming.py
git commit -m "feat(barge): feed mic to recognizer during playback in barge mode"
```

---

## Task 6: Client wiring (`static/dev_console.html`)

**Files:**
- Modify: `static/dev_console.html`

No unit tests (browser JS); validated live in Task 7. Keep edits minimal and follow the existing style.

- [ ] **Step 1: Send `config` on barge toggle + on connect**

Add a helper (near the other top-level functions) and an event listener:
```javascript
function sendBargeConfig() {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "config", barge: $("bargeIn").checked }));
}
$("bargeIn").addEventListener("change", sendBargeConfig);
```
Right after the existing `ws.send(JSON.stringify({ type: "hello", tenant: "dev" }));` line, add:
```javascript
    sendBargeConfig();
```

- [ ] **Step 2: Full-duplex mic when barge on; delete the client RMS detector**

Replace the whole `workletNode.port.onmessage` handler body (the block that does the `halfDuplex && speaking` RMS VAD and the `ws.send(int16.buffer)`) with:
```javascript
      workletNode.port.onmessage = (e) => {
        if (!running) return;
        const stillPlaying = playCursor > audioCtx.currentTime + 0.05;
        const speaking = agentSpeaking || stillPlaying;
        // Half-duplex (speakers): mute the mic while the agent speaks. Skipped
        // when barge-in is on — the server needs the mic to hear interruptions
        // (headphones, no echo). Server-side detection replaces the old client VAD.
        if ($("halfDuplex").checked && speaking && !$("bargeIn").checked) return;
        const int16 = downsampleToInt16(e.data, audioCtx.sampleRate);
        if (ws.readyState === WebSocket.OPEN) ws.send(int16.buffer);
      };
```
Then delete the now-unused declarations `BARGE_RMS`, `BARGE_MS`, `bargeMs`, and `bargeArmed` (search for each and remove its `let`/`const` and any remaining references).

- [ ] **Step 3: Handle the server `interrupt` message; drop the `barge` (armed) handler**

In `ws.onmessage`, replace the `else if (msg.type === "barge") { ... }` branch with:
```javascript
    } else if (msg.type === "interrupt") {
      stopPlayback();            // server detected a barge: flush buffered audio now
      agentSpeaking = false;
```

- [ ] **Step 4: Manual smoke (load the page)**

Start the dev server and open `http://localhost:8765/dev/voice`; confirm the page loads with no JS console errors and the two checkboxes render. (Functional check is Task 7.)
Run: `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`

- [ ] **Step 5: Commit**

```bash
git add static/dev_console.html
git commit -m "feat(barge): client config + full-duplex mic + interrupt handler; drop client RMS VAD"
```

---

## Task 7: Live validation + tune

**Files:** none (manual). Output: a tuned `BARGE_SUSTAIN_MS` and a recorded result.

- [ ] **Step 1: Run a live barge session with HEADPHONES**

Start the server (`VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`), open `/dev/voice`, **check "Allow interruptions"**, uncheck "Mute mic while agent speaks", wear headphones. Have a turn so the agent gives a long reply, then speak over it ("ruko / nahi nahi / a real question").
Confirm: the agent **cuts within ~0.5s** and answers the interruption; the server log shows `barge-in: cancelling current turn`.

- [ ] **Step 2: Backchannel check**

During a long reply, say a short "haan" / "hmm". Confirm the agent does **not** stop. If backchannels barge, raise `BARGE_SUSTAIN_MS` (e.g. 600); if real interruptions feel sluggish, lower it (e.g. 350). Re-run.

- [ ] **Step 3: Speaker-mode regression**

Uncheck "Allow interruptions", check "Mute mic while agent speaks", use speakers. Confirm normal half-duplex turn-taking still works (no stuck-in-listening, no self-answers) — the echo gate path is unchanged.

- [ ] **Step 4: Record the result + final commit (if the constant changed)**

If you tuned `BARGE_SUSTAIN_MS`, commit the change:
```bash
git add src/api/browser_bridge.py
git commit -m "tune(barge): BARGE_SUSTAIN_MS -> <value> from live testing"
```
Append a dated note to `docs/superpowers/specs/2026-06-10-barge-in-dev-console-design.md` with the chosen threshold and the live result.

---

## Self-Review (author)
- **Spec coverage:** turns-as-task → Task 3; server-side detector (sustained interim) → Tasks 2+4; generalized `_handle_barge_in` → Task 1; feed-mic-during-playback → Task 5; `_had_turn` opening-guard → Tasks 2/3; client config + full-duplex + delete-RMS + interrupt → Task 6; constant + tuning → Tasks 2/7; live validation → Task 7. Out-of-scope items (telephony, speakers, Stringee) appear in no task. Covered.
- **Type/name consistency:** `_barge_enabled`, `_turn_task`, `_barge_start_t`, `_had_turn`, `BARGE_SUSTAIN_MS`, `_barge_on_interim`, `_apply_control`, and the `{"type":"config"|"interrupt"}` messages are used identically across tasks and tests. `_handle_barge_in` stays sync (flag-setting); the consumer sends `interrupt` after calling it.
- **Placeholders:** none — every code/test step is complete. (Task 3 Step 2's note explicitly says run the fail-check after Step 3, since the test is forward-compatible.)
- **Ordering:** background-task dispatch (Task 3) precedes detector wiring (Task 4) which precedes feeding-the-mic (Task 5); each task leaves the suite green.
